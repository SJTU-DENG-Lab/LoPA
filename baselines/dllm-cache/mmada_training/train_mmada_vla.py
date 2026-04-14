# Copyright 2025 MMaDA Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union
import pickle
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.utils import get_config, flatten_omega_conf, image_transform
from robot_data import RobotDataset
from models import MAGVITv2, get_mask_schedule, MMadaModelVLA, MMadaVLAConfig
from training.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

SYSTEM_PROMPT_LEN = 28

from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "vq16":
        return VQ_16
    else:
        raise ValueError(f"model_type {model_type} not supported.")

def load_action_model(config):
    model_type = config.get("type")
    action_model_name = config.get("action_model_name")
    if model_type == "fast":
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(action_model_name, trust_remote_code=True, use_fast=False)
        return tokenizer

    elif model_type == "vq_vae":
        return None
    else:
        raise ValueError(f"model_type {model_type} not supported.")

def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size_per_gpu = config.training.batch_size

    total_batch_size = (config.training.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps)

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)

    print('special tokens : \n', uni_prompting.sptids_dict)

    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    action_model = load_action_model(config.model.action_model)
    if config.model.action_model.type == "fast":
        action_vocab_size = config.model.mmada.fast_bpe_vocab_size
    else: # TODO vqvae for action
        action_vocab_size = 0
        pass

    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize mmada in pretraining stage 
    base_config = AutoConfig.from_pretrained(config.model.mmada.pretrained_model_path).to_dict()
    mmada_config_dict = {k: v for k, v in config.model.mmada.items()}
    merged_config = {**base_config, **mmada_config_dict}
    mmada_config = MMadaVLAConfig(**merged_config)
    model = MMadaModelVLA.from_pretrained(config.model.mmada.pretrained_model_path, torch_dtype=torch.bfloat16, config=mmada_config)

    model.resize_token_embeddings(mmada_config.new_vocab_size + action_vocab_size)

    model.config.embedding_size = model.config.vocab_size
    model = model.to(accelerator.device)

    mask_id = model.config.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    ##################################
    #         DATALOADER             #
    ##################################
    logger.info("Creating dataloaders and lr_scheduler")

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # Data for generation
    if config.dataset.gen_type == "t2i":
        # dataset = Text2ImageDataset(
        #     train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
        #     tokenizer=None,  # we want to get raw texts
        #     max_seq_length=preproc_config.max_seq_length,
        #     num_train_examples=config.experiment.max_train_examples_t2i,
        #     per_gpu_batch_size=config.training.batch_size_t2i,
        #     global_batch_size=total_batch_size_t2i_without_accum,
        #     num_workers=dataset_config.num_workers,
        #     resolution=preproc_config.resolution,
        #     shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        #     pin_memory=dataset_config.pin_memory,
        #     persistent_workers=dataset_config.persistent_workers,
        #     external_caption_path=dataset_config.external_caption_path,
        #     external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
        #     external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
        #     external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
        # )
        # train_dataloader_t2i = dataset.train_dataloader
        # num_update_steps_per_epoch = math.ceil(
        #     train_dataloader_t2i.num_batches / config.training.gradient_accumulation_steps)
        # num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
        pass

    elif config.dataset.gen_type == "its2ita": # image text --> subgoal image, reasoning, action
        dataset_its2ita = RobotDataset(
            json_path=None,
            tokenizer=uni_prompting.text_tokenizer,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            resolution=preproc_config.resolution,
            max_length=preproc_config.max_seq_length,
            task_name=preproc_config.task_name,
            chunk_size=preproc_config.chunk_size,
            subgoal_gen=preproc_config.subgoal_gen,
            reasoning_gen=preproc_config.reasoning_gen,
            buffer_size=dataset_config.shuffle_buffer_size,
            action_model=action_model,
        )
        train_dataloader_its2ita = torch.utils.data.DataLoader(dataset_its2ita, batch_size=config.training.batch_size,
                                                       sampler=None, collate_fn=dataset_its2ita.collate_fn,
                                                       num_workers=0) #dataset_config.num_workers
        if accelerator.is_main_process:
            stats_path = os.path.join(config.experiment.output_dir, f'dataset_stats.pkl')
            with open(stats_path, 'wb') as f:
                pickle.dump(dataset_its2ita.norm_stats, f)
        num_train_examples = dataset_its2ita.cumulative_len[-1]
        global_batch_size=config.training.batch_size * accelerator.num_processes

        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * dataset_config.num_workers))  # per dataloader worker
        num_batches = num_worker_batches * dataset_config.num_workers

        num_update_steps_per_epoch = math.ceil(
            num_batches / config.training.gradient_accumulation_steps)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
    else:
        raise ValueError(f"Unsupported dataset type {config.dataset.type}")


    # Combine these dataloaders into a single iterable model
    iterables = {
        "its2ita_flow": train_dataloader_its2ita,
    }

    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        logger.info(f"dirs: {dirs}")
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        logger.info(f"path: {path}")
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)
            logger.info(f"Resuming from checkpoint: {path}")
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            if os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin'):
                state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
                del state_dict
            elif os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin.index.json'):
                from safetensors.torch import load_file
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, f'{path}/unwrapped_model/')
            # if safetensors sharded checkpoint exists
            elif os.path.exists(f'{path}/unwrapped_model/model.safetensors.index.json'):
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(
                    model, 
                    f'{path}/unwrapped_model/',
                    # weight_map=None, 
                    # load_state_dict_fn="safetensors"
                )
            else:
                raise FileNotFoundError(f"Checkpoint {path}/unwrapped_model/pytorch_model.bin not found")
    else:
        logger.info("Not resuming from checkpoint")

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    vq_model.to(device=accelerator.device)

    mask_dtype = model.get_input_embeddings().weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    
    @torch.no_grad()
    def prepare_inputs_and_labels_for_its2ita(
        pixel_values, 
        input_text_ids,
        subgoal_pixel_values, 
        action_tokens,
        eps=1e-3,
        # states,
    ):
        ori_shape = pixel_values.shape
        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])

        image_tokens = vq_model.get_code(pixel_values)
        image_tokens = image_tokens.view(ori_shape[0], ori_shape[1], ori_shape[-1]) # bsz, view, length

        image_tokens = image_tokens.flatten(start_dim=1, end_dim=-1)
        image_tokens = image_tokens + len(uni_prompting.text_tokenizer)

        # TODO process padded action
        # action_tokens = action_model(actions.cpu())

        action_tokens = [torch.tensor(each) + len(uni_prompting.text_tokenizer) + vq_model.quantize.codebook_size for each in action_tokens] # 8192
        action_tokens = [each.tolist() for each in action_tokens]
        input_ids_its2ita, prompt_masks, labels_its2ita, attention_masks = uni_prompting((image_tokens, input_text_ids, action_tokens), 'its2ita')

        b, l = input_ids_its2ita.shape
        t = torch.rand(b, device=input_ids_its2ita.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_its2ita.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_its2ita)
        masked_indices = noisy_batch == mask_id
        noisy_batch[prompt_masks.bool()] = input_ids_its2ita[prompt_masks.bool()]  # keep the prompt tokens without mask
        masked_indices = noisy_batch == mask_id

        prompt_masks = prompt_masks.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_masks), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
        prompt_lengths = torch.sum(prompt_masks, dim=-1, keepdim=True)
        # state_tokens = state_tokens + len(uni_prompting.text_tokenizer) + vq_model.quantize.codebook_size #  8192

        if subgoal_pixel_values is not None:
            subgoal_image_tokens = vq_model.get_code(subgoal_pixel_values.squeeze(1))
            subgoal_image_tokens = subgoal_image_tokens + len(uni_prompting.text_tokenizer)
            # mask subgoal image tokens
            subgoal_input_ids, subgoal_labels, _, subgoal_mask_prob = mask_or_random_replace_tokens(
                subgoal_image_tokens, mask_id, config, mask_schedule=mask_schedule, is_train=True
            )
            prompt_indexs = torch.sum(prompt_masks, dim=-1, keepdim=True)
            temp_input_ids = []
            temp_labels = []
            temp_attn = []
            for i in range(noisy_batch.shape[0]):
                idx = prompt_indexs[i]
                temp_input_ids.append(torch.cat([
                    noisy_batch[i,:idx],
                    uni_prompting.sptids_dict['<|soi|>'].to('cuda'),
                    subgoal_input_ids[i],
                    uni_prompting.sptids_dict['<|eoi|>'].to('cuda'),
                    noisy_batch[i,idx:]
                ], dim=-1))
                temp_labels.append(torch.cat([
                    labels_its2ita[i,:idx],
                    uni_prompting.sptids_dict['<|soi|>'].to('cuda'),
                    subgoal_labels[i],
                    uni_prompting.sptids_dict['<|eoi|>'].to('cuda'),
                    labels_its2ita[i,idx:]
                ], dim=-1))
                temp_attn.append(torch.cat([torch.ones((1, subgoal_input_ids.shape[1]+2)), attention_masks[i]], dim=-1))
            noisy_batch = torch.stack(temp_input_ids, dim=0)
            labels_its2ita = torch.stack(temp_labels, dim=0)
            attention_masks = torch.cat(temp_attn, dim=0).to(torch.int)
            answer_lengths = torch.cat([torch.zeros((b, subgoal_input_ids.shape[1]+2)).to(answer_lengths.device), answer_lengths], dim=1)
            p_mask = torch.cat([torch.zeros((b, subgoal_input_ids.shape[1]+2)).to(answer_lengths.device), p_mask], dim=1)
        else:
            attention_masks = None
        return noisy_batch, labels_its2ita, p_mask, answer_lengths, prompt_lengths, attention_masks

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch, batch_idx, dataloader_idx in combined_dataloader:

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for image+text-to-text+image+action generation
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values = batch["its2ita_flow"]["images"]
            input_text_ids = batch["its2ita_flow"]["input_ids"]
            subgoal_pixel_values = batch["its2ita_flow"]["subgoal_images"]
            action_is_pad = batch["its2ita_flow"]["action_is_pad"]
            # reasoning_ids = batch["its2ita_flow"]["reasoning_ids"]
            actions = batch["its2ita_flow"]["actions"]
            states = batch["its2ita_flow"]["states"]
            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)

            if subgoal_pixel_values is not None:
                subgoal_pixel_values = subgoal_pixel_values.to(accelerator.device, non_blocking=True)
            # reasoning_ids = reasoning_ids.to(accelerator.device, non_blocking=True)
            # actions = actions.to(accelerator.device, non_blocking=True)
            # states = states.to(accelerator.device, non_blocking=True)
            mask_prob = None
            (
                input_ids_its2ita,
                labels_its2ita,
                p_mask_its2ita,
                answer_lengths,
                prompt_lengths,
                attention_masks
            ) = prepare_inputs_and_labels_for_its2ita(pixel_values, input_text_ids, subgoal_pixel_values, actions)

            
            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids_its2ita))
                logger.info("Labels: {}".format(labels_its2ita))

            with accelerator.accumulate(model):
                logits, loss_subgoal_gen, loss_lm, loss_action_gen = model.forward_process(
                    input_ids=input_ids_its2ita,
                    labels=labels_its2ita,
                    # action_length=config.dataset.preprocessing.action_length,
                    p_mask_its2ita=p_mask_its2ita,
                    answer_lengths_its2ita=answer_lengths, # reasoning text length
                    prompt_lengths=prompt_lengths,
                    attention_mask=attention_masks
                )
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss_subgoal_gen = accelerator.gather(loss_subgoal_gen.repeat(config.training.batch_size)).mean()
                avg_loss_lm = accelerator.gather(loss_lm.repeat(config.training.batch_size)).mean()
                avg_loss_action_gen = accelerator.gather(loss_action_gen.repeat(config.training.batch_size)).mean()
                loss = config.training.t2i_coeff * loss_subgoal_gen + \
                       config.training.lm_coeff * loss_lm + \
                       config.training.act_coeff * loss_action_gen

                if mask_prob is not None:
                    avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size)).mean()
                else:
                    avg_masking_rate = torch.tensor(-1)
                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "step_loss_subgoal_gen": avg_loss_subgoal_gen.item(),
                        "step_loss_action_gen": avg_loss_action_gen.item(),
                        "step_loss_lm": avg_loss_lm.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_subgoal_gen: {avg_loss_subgoal_gen.item():0.4f} "
                        f"Loss_action_gen: {avg_loss_action_gen.item():0.4f} "
                        f"Loss_lm: {avg_loss_lm.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                if ((global_step + 1) % config.experiment.generate_every == 0 or global_step == 0) and accelerator.is_main_process:
                    pass
                    # generate_images(
                    #     model,
                    #     vq_model,
                    #     uni_prompting,
                    #     accelerator,
                    #     config,
                    #     global_step + 1,
                    #     mask_schedule=mask_schedule,
                    # )

                    # visualize_predictions(
                    #     model,
                    #     vq_model,
                    #     uni_prompting,
                    #     config,
                    #     global_step + 1,
                    #     input_ids_its2ita,
                    #     image_tokens_ori,
                    #     batch["t2i_flow"]["images"],
                    #     input_text_ids,
                    #     logits,
                    #     accelerator
                    # )
                    #
                    # understanding_images(
                    #     model,
                    #     vq_model,
                    #     uni_prompting,
                    #     accelerator,
                    #     config,
                    #     global_step + 1,
                    # )

                global_step += 1

            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=True)

    accelerator.end_training()


@torch.no_grad()
def visualize_predictions(
        model,
        vq_model,
        uni_prompting,
        config,
        global_step,
        input_ids,
        image_tokens_ori,
        ori_images,
        texts,
        logits,
        accelerator
):
    logger.info("Visualizing predictions...")
    model.eval()

    recons_images = vq_model.decode_code(image_tokens_ori - len(uni_prompting.text_tokenizer))
    recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
    recons_images *= 255.0
    recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    images = torch.clamp((ori_images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    predictions = logits[:, -(config.model.mmada.num_vq_tokens + 1):-1:, len(uni_prompting.text_tokenizer) + config.model.mmada.num_new_special_tokens: len(uni_prompting.text_tokenizer) + config.model.mmada.num_new_special_tokens + config.model.mmada.codebook_size]
    
    predictions = predictions.argmax(axis=-1)
    mask_token_id = accelerator.unwrap_model(model).config.mask_token_id - len(uni_prompting.text_tokenizer)
    input_ids = input_ids[:, -(config.model.mmada.num_vq_tokens + 1):-1:] - len(uni_prompting.text_tokenizer)
    mask_ratio = list((torch.where(input_ids == mask_token_id, 1, 0).sum(
        dim=-1) / config.model.mmada.num_vq_tokens).cpu().numpy())
    predicted_images = torch.where(input_ids == mask_token_id, predictions, input_ids)
    predicted_images = vq_model.decode_code(predicted_images)
    predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
    predicted_images *= 255.0
    predicted_images = predicted_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    predicted_images = np.concatenate((images, recons_images, predicted_images), 2)
    pil_images = [Image.fromarray(image) for image in predicted_images]

    # Log images
    wandb_images = [wandb.Image(image, caption=f'mask ratio: {r:0.2f} \n caption: {texts[i]}') for i, (image, r) in
                    enumerate(zip(pil_images, mask_ratio))]
    wandb.log({"Original images v.s. Reconstructed images v.s. Predicted images": wandb_images}, step=global_step)

    model.train()


@torch.no_grad()
def generate_images(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        mask_schedule,
):
    logger.info("Generating images...")
    model.eval()

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()


    mask_dtype = model.get_input_embeddings().weight.dtype
    mask_token_id = accelerator.unwrap_model(model).config.mask_token_id
    image_tokens = torch.ones((len(validation_prompts), config.model.mmada.num_vq_tokens), dtype=torch.long,
                              device=accelerator.device) * mask_token_id
    input_ids, attention_mask = uni_prompting((validation_prompts, image_tokens), 't2i_gen')
    if config.training.guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
    else:
        uncond_input_ids = None
        uncond_attention_mask = None
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
        # Generate images
        gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            uncond_attention_mask=uncond_attention_mask,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            predict_all_tokens=config.training.get("predict_all_tokens", False),
            seq_len=config.model.mmada.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)

    model.train()

    if config.training.get("pre_encode", False):
        del vq_model

    # Convert to PIL images
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
    wandb.log({"Generated images": wandb_images}, step=global_step)
    
    

@torch.no_grad()
def understanding_images(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
):
    logger.info("Understanding images...")
    model.eval()
        
    file_list = os.listdir(config.dataset.params.mmu_image_root)
    responses = ['' for i in range(len(file_list))]
    images = []
    
    device = accelerator.device
    
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    for i, file_name in enumerate(file_list):
        image_path = os.path.join(config.dataset.params.mmu_image_root, file_name)
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
        image = image.unsqueeze(0)
        images.append(image)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        batch_size = 1
        
        input_ids = uni_prompting.text_tokenizer(['<|start_header_id|>user<|end_header_id|>\n' + "Please describe this image in detail."  +'<eot_id><|start_header_id|>assistant<|end_header_id|>\n'])['input_ids']
        input_ids = torch.tensor(input_ids).to(device)

        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
            input_ids
        ], dim=1).long()
        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            output_ids = accelerator.unwrap_model(model).mmu_generate(input_ids)
        # output_ids = torch.stack(output_ids).squeeze()[None]

        text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        responses[i] += text[0]
    model.train()
    images = torch.cat(images, dim=0)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images = [wandb.Image(image, caption=responses[i]) for i, image in enumerate(pil_images)]
    wandb.log({"Understanding images": wandb_images}, step=global_step)


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
