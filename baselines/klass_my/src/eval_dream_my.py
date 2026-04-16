import logging
import gc
import time  # 添加时间模块
import json
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union
import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm
import numpy as np  # 添加numpy导入

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")
import random

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("dream_lora")  # 将注册名修改为 dream
class DreamModel(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        lora_path: Optional[str] = None, # 保留参数以防命令行传入报错，但内部不再使用
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 128,
        max_length: Optional[int] = 2048,  
        add_bos_token: Optional[bool] = False,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        mc_num: Optional[int] = 128,
        classifier_free_guidance: Optional[float] = 1.0,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 128,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        temperature: Optional[float] = 0.2,  
        top_p: Optional[float] = None,  
        top_k: Optional[float] = None,
        alg: Optional[str] = "klass",
        unmask_strategy: Optional[str] = "all", 
        conf_threshold: Optional[float] = 0.9, 
        kl_threshold: Optional[float] = 0.01, 
        history_length: Optional[int] = 2, 
        alg_temp: Optional[float] = 0.0,
        escape_until: Optional[bool] = False,
        block_size: Optional[int] = 4,  
        mask_token_id: Optional[int] = 151666,  
        block_add_threshold: Optional[float] = 0.5,  
        decoded_token_threshold: Optional[int] = 0.9,  
        skip_threshold: Optional[float] = 1.0,  
        sampling_strategy: Optional[str] = "default",  
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # prepare for parallelism
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        # using one process with no model parallelism
        if not (parallelize or accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:  # Parallelism managed by accelerate
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        
        self.block_size = block_size
        self.block_add_threshold = block_add_threshold  
        self.skip_threshold = skip_threshold  
        self.sampling_strategy = sampling_strategy  
        self.decoded_token_threshold = decoded_token_threshold  
        self.save_dir = save_dir
        
        # 性能与生成统计
        self.total_generated_tokens = 0
        self.total_valid_tokens = 0  # 新增：用于统计不含 EOS 的有效 Token 数
        self.total_used_steps = 0  # 核心修改：新增实际step统计
        
        # 保存 target_dtype 以便后续使用
        self.target_dtype = get_dtype(dtype)
        
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                    if self.accelerator.is_local_main_process:
                        eval_logger.info(
                            f"Using {gpus} devices with data parallelism"
                        )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    self._rank = 0
                    self._world_size = 1
        else:
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        self.max_length = max_length
        self.add_bos_token = add_bos_token
        # generation params
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.unmask_strategy = unmask_strategy
        self.conf_threshold = conf_threshold
        self.kl_threshold = kl_threshold
        self.history_length = history_length
        self.alg_temp = alg_temp
        self.escape_until = escape_until
        self.block_size = block_size
        self.mask_token_id = mask_token_id

        # loglikelihood params
        self.nll_type = nll_type
        self.log_type = log_type
        self.mc_num = mc_num
        self.classifier_free_guidance = classifier_free_guidance
        self.sampling_eps = sampling_eps

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        target_dtype = get_dtype(dtype)
        print(target_dtype)
        # 核心修改：直接使用 transformers.AutoModel，不再使用 Peft 和自建类
        self.model = transformers.AutoModel.from_pretrained(
            pretrained,
            torch_dtype=target_dtype,
            trust_remote_code=trust_remote_code,
        ).eval()
        
        self.model = self.model.to(self.device)
        
        self.model.requires_grad_(False)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids
    
    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        if self.add_bos_token:
            prompts = [self.tokenizer.bos_token + p for p in prompts]
        
        responses = []
        
        for i, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer.encode(prompt)
            prompt_tensor = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
            
            if len(prompt_ids) > self.max_length - self.max_new_tokens:
                eval_logger.warning(f"Prompt length {len(prompt_ids)} is larger than {self.max_length-self.max_new_tokens}, cutoff on the left side")
                prompt_tensor = prompt_tensor[:, -(self.max_length-self.max_new_tokens):]

            response = self._generate_block_single(prompt_tensor)
            responses.append(response)
        
        return responses
    
    @torch.no_grad()
    def _generate_block_single(self, prompt):
        """
        使用 diffusion_generate 方式为单个prompt生成响应。
        """
        self.model.eval()
        attention_mask = torch.ones_like(prompt, device=self.device)
        
        with torch.inference_mode():
            output = self.model.diffusion_generate(
                prompt,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                alg=self.alg,
                unmask_strategy=self.unmask_strategy,
                alg_temp=self.alg_temp,
                conf_threshold=self.conf_threshold,
                kl_threshold=self.kl_threshold,
                kl_history_length=self.history_length,
                save_steps=False
            )
            
            # 核心修改：保存模型实际的 step 次数
            used_steps = getattr(output, 'used_steps', self.diffusion_steps)
            if hasattr(self, 'total_used_steps'):
                self.total_used_steps += used_steps
            else:
                self.total_used_steps = used_steps
            
            generated_sequence = output.sequences[0]
            
            # 核心修改：统计所有生成的token（包括EOS），即序列总长度减去prompt长度
            all_generated_tokens_count = generated_sequence.shape[-1] - prompt.shape[1]
            if hasattr(self, 'total_generated_tokens'):
                self.total_generated_tokens += all_generated_tokens_count
            else:
                self.total_generated_tokens = all_generated_tokens_count

            # ====== 新增：统计遇到 EOS 截断前的有效 Token 数量 ======
            gen_tokens_list = generated_sequence[prompt.shape[1]:].tolist()
            if self.tokenizer.eos_token_id in gen_tokens_list:
                valid_tokens_count = gen_tokens_list.index(self.tokenizer.eos_token_id)
            else:
                valid_tokens_count = len(gen_tokens_list)
                
            if hasattr(self, 'total_valid_tokens'):
                self.total_valid_tokens += valid_tokens_count
            else:
                self.total_valid_tokens = valid_tokens_count
            # ====================================================
            
            # 解码与截断（这里保留按EOS截断文本的逻辑以便评估准确率，但统计已将EOS计入）
            generations = [
                self.tokenizer.decode(g[len(p) :].tolist())
                for p, g in zip(prompt, output.sequences)
            ]
            response = generations[0].split(self.tokenizer.eos_token)[0]
            
            return response

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        
        # 初始化统计计数器
        self.total_generated_tokens = 0
        self.total_valid_tokens = 0  # 新增初始化
        self.total_used_steps = 0

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        
        start_time = time.time()

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(contexts)
            if not self.escape_until:
                for i, r in enumerate(responses):
                    for s in gen_args[0]['until']:
                        r = r.split(s)[0]
                    responses[i] = r

            res.extend(responses)
            pbar.update(len(contexts))
            del responses
            torch.cuda.empty_cache()
            gc.collect()

        end_time = time.time()
        total_time = end_time - start_time
        
        # 使用真实的累积统计数据
        num_tokens_with_eos = self.total_generated_tokens
        num_tokens_without_eos = self.total_valid_tokens  # 新增
        actual_steps = self.total_used_steps
        
        final_stats = {
            'processed_samples': len(requests),
            'total_samples': len(requests),
            'total_tokens_with_eos': num_tokens_with_eos,           
            'total_tokens_without_eos': num_tokens_without_eos,     
            'total_used_steps': actual_steps,
            'total_time': total_time,
            'tokens_per_second_with_eos': num_tokens_with_eos / total_time if total_time > 0 else 0,
            'tokens_per_second_without_eos': num_tokens_without_eos / total_time if total_time > 0 else 0, 
            'steps_per_sample': actual_steps / len(requests) if len(requests) > 0 else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 每个Rank保存自己的统计结果
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            
            save_path = os.path.join(self.save_dir, f'rank_{self.rank}_responses.jsonl')
            with open(save_path, 'w', encoding='utf-8') as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
            stats_path = os.path.join(self.save_dir, f'rank_{self.rank}_final_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)
        
        # 打印最终统计结果
        print("\n" + "="*60)
        print(f"=== [Rank {self.rank}] 最终统计结果 ===")
        print("="*60)
        print(f"处理样本数: {final_stats['processed_samples']}")
        print(f"总生成Token数(含EOS及Padding): {final_stats['total_tokens_with_eos']}")
        print(f"总生成Token数(截断至EOS前): {final_stats['total_tokens_without_eos']}")
        print(f"总实际使用Step数: {final_stats['total_used_steps']}")
        print(f"总时间: {final_stats['total_time']:.4f}秒")
        print(f"Token/秒(含EOS): {final_stats['tokens_per_second_with_eos']:.2f}")
        print(f"Token/秒(截断): {final_stats['tokens_per_second_without_eos']:.2f}")
        print(f"平均Step/样本: {final_stats['steps_per_sample']:.2f}")
        print(f"完成时间: {final_stats['timestamp']}")
        print("="*60)

        return res

    # --------------- 下方的 NLL 等代码保持不变，为节省空间我做了适当折叠 ---------------
    
    def _forward_process(self, batch):
        b, l = batch.shape
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1
        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, l)
        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False
        noisy_batch = torch.where(mask_indices, self.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.classifier_free_guidance > 1.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_token_id
            batch = torch.cat([batch, un_batch])
        input = batch
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
        if self.classifier_free_guidance > 1.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.classifier_free_guidance * (logits - un_logits) # 修复 cfg 引用错误
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        
        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        for _ in range(max(self.mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.mask_token_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) 
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() 
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous() 

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.mask_token_id
        if self.log_type == 'ftb':
            perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)
        else:
            perturbed_seq = torch.cat([perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False
        if self.log_type == 'ftb':
            logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        else:
            logits_index = torch.cat([mask_index, torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)], dim=-1)

        if self.log_type == 'ftb':
            loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().item()
        else:
            loss = F.cross_entropy(logits[logits_index], prefix[0], reduction='sum').cpu().item()
        return loss

    def _encode_pair(self, context, continuation):
        if self.add_bos_token:
            context = self.tokenizer.bos_token + context
            
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [self.tokenizer.eos_token_id]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        cutoff_length = max(len(whole_enc) - self.max_length, 0)
        if cutoff_length > 0:
            eval_logger.warning(f"Text length {len(whole_enc)} is larger than {self.max_length}, cutoff on the left side")
            context_remain = context_enc_len-cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                eval_logger.warning(f"All context (prompt) is truncated.")
                context_enc = ""
                continuation_enc = whole_enc[-self.max_length:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type == 'ar_ftb' or self.nll_type == 'ar_btf':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                is_target_greedy_dec = False
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()