import logging
import json
import time
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union, Dict
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm
import numpy as np
import os
import jinja2

# 导入LLaDA模型相关模块
from model_cache.llada.modeling_llada import LLaDAModelLM
from model_cache.llada.configuration_llada import LLaDAConfig

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="TemplateLM")

import random
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _create_base_block_causal_mask(
    seq_length: int, 
    prompt_length: int, 
    block_size: int, 
    device, 
    dtype
) -> torch.Tensor:
    """
    Creates a base block-causal attention mask for a single, contiguous sequence.
    This is a helper function to establish the correct local attention pattern.

    Rules:
    1. All tokens can attend to the entire prompt.
    2. A token after the prompt can attend to all tokens within its own block and all preceding blocks.
    """
    # 1. Initialize mask, blocking all attention by default. Shape is [query_len, key_len].
    mask = torch.full((seq_length, seq_length), -torch.inf, device=device, dtype=dtype)

    # 2. Rule 1: Allow all query tokens (rows) to attend to the full prompt (columns).
    mask[:, :prompt_length] = 0

    # 3. Rule 2: Process the block-causal attention for tokens after the prompt.
    for q_pos in range(prompt_length, seq_length):
        # Determine the block ID for the current query token.
        # Blocks are numbered starting from 0 *after* the prompt.
        pos_after_prompt = q_pos - prompt_length
        q_block_id = pos_after_prompt // block_size

        # A query can attend to all keys from block 0 up to its own block.
        for b_id in range(q_block_id + 1):
            # Calculate the absolute start and end positions of this key block.
            block_start = prompt_length + b_id * block_size
            block_end = min(prompt_length + (b_id + 1) * block_size, seq_length)
            
            # Allow the current query at q_pos to attend to all keys in this block.
            mask[q_pos, block_start:block_end] = 0
            
    return mask


## 移除未使用的多分支 attention mask 构造函数

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, sampling_strategy="default"):
    """
    统一的token采样和置信度计算函数
    
    Args:
        logits: 输入的logits张量
        temperature: 温度参数
        top_p: nucleus采样参数
        top_k: top-k采样参数
        sampling_strategy: 置信度计算策略
            - "default": 使用token概率作为置信度
            - "margin": 使用top1和top2的差值作为置信度
            - "neg_entropy": 使用负熵作为置信度
    
    Returns:
        confidence: 根据strategy计算的置信度
        x0: 采样得到的token
        initial_confidence: 原始token概率（保留用于向后兼容）
    """
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            initial_confidence, x0 = probs.max(dim=-1)
    else:
        initial_confidence, x0 = probs.max(dim=-1)

    # 根据sampling_strategy计算最终置信度
    if sampling_strategy == "margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        confidence = top1_probs - top2_probs
    elif sampling_strategy == "neg_entropy":
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    else:  # "default"
        confidence = initial_confidence.clone()

    return confidence, x0, initial_confidence
@register_model("dream_lora")
class DreamLoRA(TemplateLM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        lora_path: str,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 128,
        max_length: Optional[int] = 4096,
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
        alg: Optional[str] = "entropy",
        alg_temp: Optional[float] = 0.0,
        escape_until: Optional[bool] = False,
        block_size: Optional[int] = 4,
        mask_token_id: Optional[int] = 126336,
        block_add_threshold: Optional[float] = 0.5,
        decoded_token_threshold: Optional[float] = 0.9,
        skip_threshold: Optional[float] = 1.0,
        sampling_strategy: Optional[str] = "default",
        save_dir: Optional[str] = None,
        show_speed: Optional[bool] = True,
        use_uncertainty_logic: Optional[bool] = True,
        max_branches_kept: Optional[int] = 1,
        branching_factor: Optional[int] = 2,
        branch_confidence_decay: Optional[float] = 0.8,
        # [NEW PARAMETER] 控制是否保留基础分支
        branch_verification_mode: Optional[bool] = True,
        # [NEW PARAMETER] 控制基础分支是否参与竞争
        base_branch_competition: Optional[bool] = True,
        # [NEW PARAMETER] 验证开关：是否在验证中强制基础分支胜出（默认开启）
        verification_force_base_winner: Optional[bool] = False,
        branch_topp: Optional[float] = 0.5,
        selection_conf_alpha: Optional[float] = 0.5,
        # [NEW PARAMETER] 控制是否使用全注意力机制
        use_full_attention: Optional[bool] = True,
        
        **kwargs,
    ) -> None:
        super().__init__()
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
        if not (parallelize or accelerator.num_processes > 1):
            device_list = set(["cuda", "cpu"] + [f"cuda:{i}" for i in range(gpus)] + ["mps", "mps:0"] + [f"npu:{i}" for i in range(gpus)])
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(torch.__version__) < version.parse("2.1"):
                    raise RuntimeError(f"mps requires torch >= 2.1. You have {torch.__version__}")
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            if device != "cuda":
                eval_logger.info(f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model.")
            self._device = self.accelerator.device if hasattr(self, "accelerator") else torch.device(device)

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str): 
            self.batch_size_per_gpu = int(batch_size)

        self.lora_path = lora_path
        self.block_size = block_size
        self.skip_threshold = skip_threshold
        self.sampling_strategy = sampling_strategy
        self.target_dtype = get_dtype(dtype)
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    try: self.model.to(self.device)
                    except ValueError: 
                        eval_logger.debug("Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore.")
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize: 
                        eval_logger.warning("You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available.")
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. If you would like to use data parallelism, please launch the script with 'accelerate launch *script*'. Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(f"Using {gpus} devices with data parallelism")
                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator
                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    self._rank = 0
                    self._world_size = 1
        else:
            eval_logger.warning("Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration")
            self._rank = 0
            self._world_size = 1

        self.max_length = max_length
        self.add_bos_token = add_bos_token
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp
        self.escape_until = escape_until
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.nll_type = nll_type
        self.log_type = log_type
        self.mc_num = mc_num
        self.classifier_free_guidance = classifier_free_guidance
        self.sampling_eps = sampling_eps
        self.backend = "causal"
        self.truncation = False

        self.debug_print = kwargs.get("debug_print", False)

        self.save_dir = save_dir
        self.show_speed = show_speed
        self.use_uncertainty_logic = False

    @property
    def batch_size(self): return self.batch_size_per_gpu
    @property
    def eot_token_id(self): return self.tokenizer.eos_token_id
    @property
    def device(self): return self._device
    @property
    def rank(self): return self._rank
    @property
    def world_size(self): return self._world_size

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        target_dtype = get_dtype(dtype)
        config = LLaDAConfig.from_pretrained(pretrained)
        self.model = LLaDAModelLM.from_pretrained(pretrained, config=config, torch_dtype=target_dtype, trust_remote_code=False).eval()
        if target_dtype is not None and target_dtype != "auto":
            self.model = self.model.to(target_dtype)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        special_tokens_kwargs = {}
        
        if add_special_tokens is None:
            if self.backend == "causal": 
                special_tokens_kwargs = {"add_special_tokens": False or self.add_bos_token}
        else: 
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)
        if left_truncate_len: 
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(self, strings: List[str], padding_side: str = "left", left_truncate_len: int = None, truncation: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        add_special_tokens = {}
        if self.backend == "causal": 
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}
        encoding = self.tokenizer(strings, truncation=truncation, padding="longest", return_tensors="pt", **add_special_tokens)
        if left_truncate_len:
            original_lengths = encoding["input_ids"].size(1)
            if original_lengths > left_truncate_len:
                eval_logger.warn(f"Left truncation applied. Original sequence length was {original_lengths}, truncating to last {left_truncate_len} tokens. Some content will be lost.")
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def tok_decode(self, tokens, skip_special_tokens=True): return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _compute_generation_token_stats(
        self,
        generated_ids: List[int],
        block_size: int,
    ) -> Tuple[List[int], int, int]:
        trimmed_ids = list(generated_ids)
        while trimmed_ids and trimmed_ids[-1] == self.mask_token_id:
            trimmed_ids.pop()

        if self.eot_token_id in trimmed_ids:
            eos_idx = trimmed_ids.index(self.eot_token_id)
            actual_ids = trimmed_ids[:eos_idx]
            generated_tokens_including_eos = ((eos_idx // block_size) + 1) * block_size
        else:
            actual_ids = trimmed_ids
            generated_tokens_including_eos = len(trimmed_ids)

        actual_tokens_excluding_eos = len(actual_ids)
        return actual_ids, actual_tokens_excluding_eos, generated_tokens_including_eos

    def _count_tokens_after_truncation(self, response_text: str, until_terms: List[str] = None) -> int:
        truncated_text = response_text
        if until_terms and not self.escape_until:
            for term in until_terms:
                if len(term) > 0:
                    truncated_text = truncated_text.split(term)[0]
        generated_answer_ids = torch.tensor(self.tokenizer(truncated_text)["input_ids"])
        return int((generated_answer_ids != 126081).sum())

    @classmethod
    def create_from_arg_string(cls: Type[T], arg_string: str, additional_config: Optional[dict] = None) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        try:
            chat_templated = self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=add_generation_prompt, continue_final_message=not add_generation_prompt)
        except jinja2.exceptions.TemplateError:
            eval_logger.warning("Failed to apply chat template. removing the system role in chat history.")
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=add_generation_prompt, continue_final_message=not add_generation_prompt)
        return chat_templated

    @property
    def tokenizer_name(self) -> str: 
        return self.tokenizer.name_or_path.replace("/", "__")

    def _generate_blockwise_confidence_single_branch(self, prompt: torch.Tensor) -> Tuple[List[int], Dict]:
        self.model.eval()
        prompt_length = prompt.shape[1]
        x_t = prompt.clone().to(self.device)
        threshold = self.skip_threshold
        eos_token_id = self.eot_token_id
        stats = {
            "steps_taken": 0,
            "total_filled_original": 0,
            "original_fallback_triggers": 0,
        }
        gen_length = self.max_new_tokens
        block_length = self.block_size
        assert gen_length % block_length == 0, "max_new_tokens must be divisible by block_size"
        total_length = prompt_length + gen_length
        num_blocks = gen_length // block_length

        full_x_t = torch.full(
            (1, total_length),
            self.mask_token_id,
            dtype=prompt.dtype,
            device=self.device,
        )
        full_x_t[:, :prompt_length] = x_t

        with torch.inference_mode():
            for block_idx in range(num_blocks):
                block_start = prompt_length + block_idx * block_length
                block_end = block_start + block_length
                while True:
                    mask_region = full_x_t[:, block_start:block_end]
                    mask_index = mask_region == self.mask_token_id
                    mask_indices_in_block = mask_index[0].nonzero(as_tuple=True)[0]
                    if len(mask_indices_in_block) == 0:
                        break

                    stats["steps_taken"] += 1
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        outputs = self.model(full_x_t)
                    logits = outputs.logits
                    logits_region = logits[:, block_start:block_end]
                    block_mask_logits = logits_region[mask_index]

                    confidence, x0, initial_confidence = sample_tokens(
                        block_mask_logits,
                        self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        sampling_strategy=self.sampling_strategy,
                    )
                    high_conf_indices = (initial_confidence > threshold).nonzero(as_tuple=True)[0]

                    indices_to_fill = high_conf_indices
                    if len(indices_to_fill) == 0 and len(confidence) > 0:
                        stats["original_fallback_triggers"] += 1
                        indices_to_fill = torch.argmax(confidence).view(1)

                    if len(indices_to_fill) == 0:
                        break

                    positions = block_start + mask_indices_in_block[indices_to_fill]
                    values = x0[indices_to_fill]
                    full_x_t[0, positions] = values
                    stats["total_filled_original"] += len(indices_to_fill)
                    del positions, values

                    if stats["steps_taken"] > 1000:
                        eval_logger.warning("Generation stopped due to exceeding 1000 steps.")
                        generated_sequence_ids = full_x_t[0, prompt_length:].tolist()
                        (
                            generated_sequence_ids,
                            actual_tokens_excluding_eos,
                            generated_tokens_including_eos,
                        ) = self._compute_generation_token_stats(generated_sequence_ids, self.block_size)
                        stats["actual_tokens_excluding_eos"] = actual_tokens_excluding_eos
                        stats["generated_tokens_including_eos"] = generated_tokens_including_eos
                        return generated_sequence_ids, stats

                if eos_token_id is not None:
                    generated_part = full_x_t[0, prompt_length:block_end]
                    if (generated_part == eos_token_id).any():
                        break

        generated_sequence_ids = full_x_t[0, prompt_length:].tolist()
        (
            generated_sequence_ids,
            actual_tokens_excluding_eos,
            generated_tokens_including_eos,
        ) = self._compute_generation_token_stats(generated_sequence_ids, self.block_size)
        stats["actual_tokens_excluding_eos"] = actual_tokens_excluding_eos
        stats["generated_tokens_including_eos"] = generated_tokens_including_eos
        return generated_sequence_ids, stats

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        start_time = time.time()
        total_generated_tokens_including_eos = 0
        total_actual_tokens_excluding_eos = 0
        total_steps = 0
        bar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)), desc="Running generate_until requests")

        for i, req in enumerate(requests):
            question = req.args[0]
            gen_kwargs = req.args[1]
            contexts = [question]
            if self.add_bos_token:
                contexts = [self.tokenizer.bos_token + p for p in contexts]
            context_enc, _ = self.tok_batch_encode(contexts, truncation=self.truncation)
            input_ids = context_enc[0].unsqueeze(0)

            if input_ids.shape[1] > self.max_length - self.max_new_tokens:
                eval_logger.warning(f"Prompt length {input_ids.shape[1]} > {self.max_length - self.max_new_tokens}, cutoff on the left side")
                input_ids = input_ids[:, -(self.max_length - self.max_new_tokens):]

            generated_answer, stats = self._generate_original_single_branch(input_ids)

            total_steps += stats.get("steps_taken", 0)
            total_generated_tokens_including_eos += stats.get(
                "generated_tokens_including_eos", len(generated_answer)
            )
            total_actual_tokens_excluding_eos += stats.get(
                "actual_tokens_excluding_eos", len(generated_answer)
            )

            cont_toks_list = self.tokenizer.batch_decode([generated_answer], skip_special_tokens=True)
            s = cont_toks_list[0]

            if not self.escape_until:
                for term in gen_kwargs.get("until", []):
                    if len(term) > 0:
                        s = s.split(term)[0]

            res.append(s)
            bar.update(1)

        bar.close()

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            total_time = time.time() - start_time
            final_stats = {
                "processed_samples": len(res), "total_samples": len(requests),
                "total_generated_tokens_including_eos": int(total_generated_tokens_including_eos),
                "total_actual_tokens_excluding_eos": int(total_actual_tokens_excluding_eos),
                "total_parallel_steps": int(total_steps),
                "total_time": total_time,
                "generated_tokens_including_eos_per_second": float(total_generated_tokens_including_eos) / total_time if total_time > 0 else 0.0,
                "actual_tokens_excluding_eos_per_second": float(total_actual_tokens_excluding_eos) / total_time if total_time > 0 else 0.0,
                "generated_tokens_including_eos_per_step": float(total_generated_tokens_including_eos) / float(total_steps) if total_steps > 0 else 0.0,
                "actual_tokens_excluding_eos_per_step": float(total_actual_tokens_excluding_eos) / float(total_steps) if total_steps > 0 else 0.0,
                "timestamp": time.time(),
                "rank": self.rank,
                "world_size": self.world_size,
            }
            with open(os.path.join(self.save_dir, f'rank_{self.rank}_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)

        if self.show_speed and len(res) > 0:
            total_time = time.time() - start_time
            avg_tokens = total_generated_tokens_including_eos / len(res)
            avg_steps = total_steps / len(res)
            avg_tok_per_step = total_generated_tokens_including_eos / total_steps if total_steps > 0 else 0

            mode = "Single-Branch"

            print(f"\n==================== FINAL SUMMARY ({mode}, Corrected Stats) ====================")
            print(f"  - Total Samples Processed: {len(res)}")
            print(f"  - Total Generated Tokens Including EOS (sum of best paths): {total_generated_tokens_including_eos}")
            print(f"  - Total Actual Tokens Excluding EOS (sum of best paths): {total_actual_tokens_excluding_eos}")
            print(f"  - Total Steps (sum of best paths): {total_steps}")
            print(f"  - Total Time: {total_time:.2f} seconds")
            print("--------------------------------------------------------------------")
            print(f"  - Average Tokens per Sample (best path): {avg_tokens:.2f}")
            print(f"  - Average Steps per Sample (best path): {avg_steps:.2f}")
            print(f"  - Overall Effective Tokens/Step Ratio: {avg_tok_per_step:.2f}")
            print(f"  - Overall Throughput (Generated Tokens/Sec): {total_generated_tokens_including_eos / total_time:.2f}")
            print("==================================================================================\n")

        return res

    def _forward_process(self, batch):
        b, length = batch.shape
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1
        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, length)
        mask_indices = torch.rand((b, length), device=batch.device) < p_mask
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

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(batch).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        if self.classifier_free_guidance > 1.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.cfg * (logits - un_logits)

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
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end].to(self.device)
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
            eval_logger.warning(f"Text length {len(whole_enc)} > {self.max_length}, cutoff on the left side")
            context_remain = context_enc_len - cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                eval_logger.warning("All context (prompt) is truncated.")
                context_enc = ""
                continuation_enc = whole_enc[-self.max_length:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {"prefix_text": e["prefix"], "target_text": e["target"], "prefix": prefix, "target": target}
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize).with_format("torch")
        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix, target = elem["prefix"], elem["target"]
                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type in ['ar_ftb', 'ar_btf']:
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)
                is_target_greedy_dec = False
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]: raise NotImplementedError
    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]: raise NotImplementedError

    def _generate_original_single_branch(self, prompt: torch.Tensor) -> Tuple[List[int], Dict]:
        return self._generate_blockwise_confidence_single_branch(prompt)

if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
