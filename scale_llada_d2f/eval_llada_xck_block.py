import logging
import gc
import json
import time
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union, Dict, Set
import torch
import torch.nn.functional as F
import torch.distributions as dists
import transformers
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm
from peft import PeftConfig, PeftModel
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

# [MODIFIED] Branch 类增加了用于精确统计的属性
class Branch:
    """表示一个生成分支的状态"""
    def __init__(self, branch_id: int, x_t: torch.Tensor, block_states: Dict,
                 confidence: float = 1.0, past_key_values: Optional[Tuple] = None,
                 prompt_length: int = 0, is_base: bool = False,
                 creation_token_confidence: float = 1.0):
        self.branch_id = branch_id
        self.x_t = x_t.clone()  # 当前序列状态
        self.block_states = {k: v.copy() for k, v in block_states.items()}  # 块状态
        self.confidence = confidence  # 分支整体置信度
        self.step_confidences = []  # 记录每步的置信度
        self.is_active = True  # 分支是否活跃
        self.eos_detected = False  # 是否检测到结束符
        self.past_key_values = past_key_values

        # 新增统计属性
        self.prompt_length = prompt_length
        self.steps_completed = -1  # 标记完成时的并行轮数，-1表示未完成
        self.is_base = is_base
        # 分支创建时（fork 时）对应采样 token 的置信度
        self.creation_token_confidence = creation_token_confidence

    @property
    def generated_token_count(self) -> int:
        """动态计算生成的token数量"""
        return self.x_t.shape[1] - self.prompt_length



    def copy(self):
        """创建分支的优化拷贝，减少显存占用"""
        new_branch = Branch(
            self.branch_id,
            self.x_t.clone(),
            {k: v.copy() for k, v in self.block_states.items()},
            self.confidence,
            None,
            self.prompt_length,
            self.is_base,
            creation_token_confidence=self.creation_token_confidence,
        )
        new_branch.step_confidences = self.step_confidences.copy()
        new_branch.is_active = self.is_active
        new_branch.eos_detected = self.eos_detected
        new_branch.steps_completed = self.steps_completed
        return new_branch

## 移除未使用的分支剪枝与位置选择函数（未被调用），减少解析与维护开销

def create_full_block_attention_mask(prompt_length, max_length, block_size, device=None, dtype=None):
    if dtype is None:
        dtype = torch.bfloat16
    attention_mask = torch.full((1, 1, max_length, max_length), -torch.inf, device=device, dtype=dtype)
    attention_mask[:, :, :prompt_length, :prompt_length] = 0
    remaining_length = max_length - prompt_length
    num_blocks = (remaining_length + block_size - 1) // block_size
    for b in range(num_blocks):
        block_start = prompt_length + b * block_size
        block_end = min(prompt_length + (b + 1) * block_size, max_length)
        attention_mask[:, :, block_start:block_end, :prompt_length] = 0
        for prev_b in range(b):
            prev_start = prompt_length + prev_b * block_size
            prev_end = min(prompt_length + (prev_b + 1) * block_size, max_length)
            attention_mask[:, :, block_start:block_end, prev_start:prev_end] = 0
        attention_mask[:, :, block_start:block_end, block_start:block_end] = 0
    return attention_mask

def create_full_attention_mask(max_length, device=None, dtype=None):
    """创建全注意力掩码（全0矩阵）"""
    if dtype is None:
        dtype = torch.bfloat16
    attention_mask = torch.zeros((1, 1, max_length, max_length), device=device, dtype=dtype)
    return attention_mask

def extract_attention_mask(full_mask, start_pos, input_length, cache_length, use_full_attention=False):
    end_pos = start_pos + input_length
    total_length = cache_length + input_length
    
    if use_full_attention:
        # 使用全注意力机制，创建全0掩码
        extracted_mask = torch.zeros((1, 1, input_length, total_length),
                                   device=full_mask.device, dtype=full_mask.dtype)
    else:
        # 使用原有的块注意力机制
        extracted_mask = torch.full((1, 1, input_length, total_length), -torch.inf,
                                   device=full_mask.device, dtype=full_mask.dtype)
        if cache_length > 0:
            extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
        extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
    
    return extracted_mask

def find_shared_prefix_end(branches: List[Branch]) -> int:
    """找到所有分支的最长共享前缀位置（基于完成的块）"""
    if not branches:
        return 0

    # 找到所有分支中最小的完成块数量
    min_completed_blocks = float('inf')
    for branch in branches:
        completed_blocks = 0
        for block_id in sorted(branch.block_states.keys()):
            if (branch.block_states[block_id]['state'] in ['in_cache', 'to_cache'] and
                branch.block_states[block_id]['mask_count'] == 0):
                completed_blocks += 1
            else:
                break
        min_completed_blocks = min(min_completed_blocks, completed_blocks)

    if min_completed_blocks == float('inf') or min_completed_blocks == 0:
        # 返回prompt长度作为最小共享前缀
        return branches[0].prompt_length

    # 找到第min_completed_blocks个完成块的结束位置
    first_branch = branches[0]
    completed_count = 0
    for block_id in sorted(first_branch.block_states.keys()):
        if (first_branch.block_states[block_id]['state'] in ['in_cache', 'to_cache'] and
            first_branch.block_states[block_id]['mask_count'] == 0):
            completed_count += 1
            if completed_count == min_completed_blocks:
                return first_branch.block_states[block_id]['end_pos']
        else:
            break

    return branches[0].prompt_length





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
            x0 = dists.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
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

def evaluate_branch_confidence(
    logits: torch.Tensor,
    branch: Branch,
    branch_start_in_input: int,
    branch_length: int,
    shared_prefix_end: int,
    mask_token_id: int,
    sampling_strategy: str = "default",
    branch_topp: float = 0.5,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[float] = None,
    selection_conf_alpha: float = 0.5,
) -> float:
    """评估分支置信度（融合分支创建 token 置信度与未来 mask 区域置信度）

    最终置信度 = alpha * creation_token_confidence + (1 - alpha) * future_mask_confidence
    """
    # --- 未来 mask 区域置信度 ---
    # 保持原逻辑：branch_length==0 时 future_conf 视作 1.0（无需再评估未来区域）
    if branch_length == 0:
        future_conf = 1.0
    else:
        branch_logits = logits[0, branch_start_in_input:branch_start_in_input + branch_length, :]
        branch_sequence = branch.x_t[0, shared_prefix_end:]
        mask_positions = (branch_sequence == mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            future_conf = 1.0
        else:
            mask_logits = branch_logits[mask_positions, :]
            confidences, _, _ = sample_tokens(
                mask_logits, temperature, top_p, top_k, sampling_strategy
            )
            num_positions = len(confidences)
            if num_positions == 1:
                future_conf = confidences.item()
            else:
                # 用户需求：取置信度最低的 bottom 比例，衡量薄弱区域
                bottom_cnt = max(1, int(num_positions * branch_topp))
                sorted_confidences, _ = torch.sort(confidences, descending=False)
                future_conf = sorted_confidences[:bottom_cnt].mean().item()

    # --- 创建时 token 置信度 ---
    creation_conf = float(getattr(branch, "creation_token_confidence", 1.0))
    alpha = float(max(0.0, min(1.0, selection_conf_alpha)))
    combined_conf = alpha * creation_conf + (1.0 - alpha) * future_conf
    return combined_conf

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
        self.block_add_threshold = block_add_threshold
        self.skip_threshold = skip_threshold
        self.sampling_strategy = sampling_strategy
        self.decoded_token_threshold = decoded_token_threshold
        self.branch_topp = branch_topp
        self.selection_conf_alpha = selection_conf_alpha
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
        self.use_uncertainty_logic = use_uncertainty_logic

        self.max_branches_kept = max_branches_kept
        self.branching_factor = branching_factor
        self.branch_confidence_decay = branch_confidence_decay
        self.branch_verification_mode = branch_verification_mode
        self.base_branch_competition = base_branch_competition
        self.verification_force_base_winner = verification_force_base_winner
        self.use_full_attention = use_full_attention
        # self.keep_base_branch = keep_base_branch
        # 共享KV缓存（仅维护一套）
        self.shared_past_key_values = None

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
        peft_config = PeftConfig.from_pretrained(self.lora_path)
        self.model = PeftModel.from_pretrained(self.model, self.lora_path)
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

    # [MODIFIED] 返回值是一个元组，包含最终分支的详细统计信息
    def _generate_enhanced_speculative(self, prompt: torch.Tensor) -> Tuple[List[int], int, int, Dict]:
        self.model.eval()
        prompt_length = prompt.shape[1]

        if not self.use_uncertainty_logic:
            generated_ids, stats = self._generate_original_single_branch(prompt)
            (
                generated_ids,
                actual_tokens_excluding_eos,
                generated_tokens_including_eos,
            ) = self._compute_generation_token_stats(generated_ids, self.block_size)
            stats["actual_tokens_excluding_eos"] = actual_tokens_excluding_eos
            stats["generated_tokens_including_eos"] = generated_tokens_including_eos
            return generated_ids, stats.get('steps_taken', 0), generated_tokens_including_eos, stats

        # --- Initialization ---
        initial_x_t = prompt.clone().to(self.device)
        # 重置共享KV缓存
        self.shared_past_key_values = None
        # 手动维护缓存长度，与单分支代码保持一致
        shared_cache_length = 0

        if self.use_full_attention:
            full_attention_mask = create_full_attention_mask(
                max_length=self.max_length,
                device=self.device, dtype=self.target_dtype if self.target_dtype not in [None, "auto"] else torch.bfloat16
            )
        else:
            full_attention_mask = create_full_block_attention_mask(
                prompt_length=prompt_length, max_length=self.max_length, block_size=self.block_size,
                device=self.device, dtype=self.target_dtype if self.target_dtype not in [None, "auto"] else torch.bfloat16
            )

        initial_block_states = {
            0: {'start_pos': 0, 'end_pos': prompt_length, 'mask_count': 0, 'total_masks': prompt_length, 'state': 'to_cache', 'is_complete': True},
        }
        branches = [Branch(0, initial_x_t, initial_block_states, confidence=1.0, past_key_values=None, prompt_length=prompt_length, is_base=True, creation_token_confidence=1.0)]

        # General stats about the whole process
        run_stats = {
            "parallel_steps": 0, "total_filled_original": 0, "total_filled_new": 0, "original_fallback_triggers": 0,
            "branches_created": 1, "branches_pruned": 0, "max_active_branches": 0, "branch_processing_count": 0, "candidates_generated_this_round": 0
        }

        with torch.inference_mode():
            while any(b.is_active for b in branches):
                run_stats["parallel_steps"] += 1
                parallel_step_count = run_stats["parallel_steps"]

                active_branches_this_round = [b for b in branches if b.is_active]
                run_stats["branch_processing_count"] += len(active_branches_this_round)
                run_stats["max_active_branches"] = max(run_stats["max_active_branches"], len(active_branches_this_round))

                next_generation_branches = []

                # 预处理分支，添加新块和更新状态
                for branch in active_branches_this_round:
                    x_t = branch.x_t
                    block_states = branch.block_states

                    if len(block_states)-1 < (self.max_new_tokens // self.block_size) and not branch.eos_detected:
                        last_block_id = len(block_states) - 1
                        progress = ((block_states[last_block_id]['total_masks'] - block_states[last_block_id]['mask_count']) / block_states[last_block_id]['total_masks']) if block_states[last_block_id]['total_masks'] > 0 else 1.0
                        if progress >= self.block_add_threshold:
                            new_block_id = len(block_states)
                            new_start_pos = x_t.shape[1]
                            mask_block = torch.tensor([[self.mask_token_id] * self.block_size], device=self.device)
                            x_t = torch.cat([x_t, mask_block], dim=1)
                            block_states[new_block_id] = {'start_pos': new_start_pos, 'end_pos': new_start_pos + self.block_size, 'mask_count': self.block_size, 'total_masks': self.block_size, 'state': 'active', 'is_complete': False}
                            branch.x_t = x_t

                    self._update_block_completion_states(block_states, self.decoded_token_threshold)

                    is_generation_done = (x_t == self.mask_token_id).sum() == 0 and all(s['state'] != 'active' for s in block_states.values())
                    if is_generation_done:
                        branch.is_active = False
                        if branch.steps_completed == -1:
                            branch.steps_completed = parallel_step_count
                        next_generation_branches.append(branch)

                # 过滤出仍然活跃的分支
                remaining_active_branches = [b for b in active_branches_this_round if b.is_active]

                if not remaining_active_branches:
                    next_generation_branches.extend([b for b in active_branches_this_round if not b.is_active])
                    continue

                # 找到共享前缀和缓存信息
                shared_prefix_end = find_shared_prefix_end(remaining_active_branches)

                # 使用手动维护的缓存长度，与单分支代码保持一致
                current_cache_length = shared_cache_length

                # 检查是否需要更新缓存
                blocks_to_cache = []
                update_kvcache_len = 0
                # 选择用于布局计算的参考分支：优先基础分支
                layout_branch = None
                for b in remaining_active_branches:
                    if getattr(b, 'is_base', False):
                        layout_branch = b
                        break
                if layout_branch is None:
                    layout_branch = remaining_active_branches[0]

                for bid, state in layout_branch.block_states.items():
                    if (state['state'] == 'to_cache' and
                        state['end_pos'] <= shared_prefix_end):
                        blocks_to_cache.append(bid)

                if blocks_to_cache:
                    earliest_pos = min(layout_branch.block_states[bid]['start_pos'] for bid in blocks_to_cache)
                    latest_pos = max(layout_branch.block_states[bid]['end_pos'] for bid in blocks_to_cache)
                    update_kvcache_len = latest_pos - earliest_pos

                # 确定需要处理的输入范围
                if update_kvcache_len > 0:
                    # 如果有缓存更新，从需要缓存的最早位置开始
                    input_start_pos = min(layout_branch.block_states[bid]['start_pos'] for bid in blocks_to_cache)
                else:
                    # 否则从共享前缀结束位置开始
                    input_start_pos = shared_prefix_end

                # 在验证强制基础分支模式下，确保基础分支的后缀位于最前，保持其位置编码与单分支一致
                # if self.verification_force_base_winner:
                #     remaining_active_branches = sorted(remaining_active_branches, key=lambda b: (not getattr(b, 'is_base', False)))
                # 在基础分支参与竞争时，将基础分支放到最后，便于验证分支不受处理顺序影响
                if self.base_branch_competition:
                    # 非基础分支在前，基础分支在后；同时用 branch_id 保持稳定顺序
                    remaining_active_branches = sorted(
                        remaining_active_branches,
                        key=lambda b: (getattr(b, 'is_base', False), getattr(b, 'branch_id', 0))
                    )

                # [SIMPLIFIED] 现在每个分支独立通过模型，而不是并行处理
                branch_results = []  # 收集本轮各活跃分支前向结果
                
                for branch_idx, branch in enumerate(remaining_active_branches):
                    # [MEMORY OPTIMIZED] 直接计算输入范围，避免创建多余的tensor
                    needs_shared_prefix = input_start_pos < shared_prefix_end
                    needs_branch_suffix = branch.x_t.shape[1] > shared_prefix_end
                    
                    # 如果没有需要处理的输入，跳过这个分支
                    if not needs_shared_prefix and not needs_branch_suffix:
                        branch.is_active = False
                        if branch.steps_completed == -1:
                            branch.steps_completed = parallel_step_count
                        branch_results.append((branch, None, None))
                        continue
                    
                    # [MEMORY OPTIMIZED] 直接构建输入tensor，避免list和多次cat
                    if needs_shared_prefix and needs_branch_suffix:
                        branch_input = torch.cat([
                            branch.x_t[0, input_start_pos:shared_prefix_end],
                            branch.x_t[0, shared_prefix_end:]
                        ]).unsqueeze(0)
                    elif needs_shared_prefix:
                        branch_input = branch.x_t[0, input_start_pos:shared_prefix_end].unsqueeze(0)
                    else:  # needs_branch_suffix
                        branch_input = branch.x_t[0, shared_prefix_end:].unsqueeze(0)                    # 使用简单的块级因果attention mask（与单分支相同）
                    attention_mask = extract_attention_mask(
                        full_mask=full_attention_mask, 
                        start_pos=input_start_pos, 
                        input_length=branch_input.shape[1], 
                        cache_length=current_cache_length,
                        use_full_attention=self.use_full_attention
                    )

                    # if self.debug_print and branch_idx == 0:  # 只为第一个分支打印调试信息
                    #     print(f"[Round {parallel_step_count}] Branch {branch.branch_id} (base={getattr(branch,'is_base',False)})")
                    #     print(f"  cache_len={current_cache_length}, shared_prefix_end={shared_prefix_end}")
                    #     print(f"  input_start_pos={input_start_pos}, branch_input_len={branch_input.shape[1]}")
                    #     print(f"  update_kvcache_len={update_kvcache_len}")

                    # 前向传播（单个分支）
                    outputs = self.model(
                        branch_input,
                        attention_bias=attention_mask,
                        past_key_values=self.shared_past_key_values,
                        use_cache=True,
                        update_kvcache=update_kvcache_len + current_cache_length
                    )

                    if outputs.past_key_values is None:
                        eval_logger.error(f"Model did not return past_key_values for branch {branch.branch_id}.")
                        branch.is_active = False
                        if branch.steps_completed == -1:
                            branch.steps_completed = parallel_step_count
                        branch_results.append((branch, None, None))
                        continue

                    logits, new_pkv = outputs.logits, outputs.past_key_values
                    branch_results.append((branch, logits, new_pkv))

                # 评估每个分支的置信度
                branch_confidences = []
                for branch_idx, (branch, logits, new_pkv) in enumerate(branch_results):
                    if logits is not None:
                        branch_length = branch.x_t.shape[1] - shared_prefix_end if branch.x_t.shape[1] > shared_prefix_end else 0
                        if branch_length > 0:
                            shared_prefix_in_input_length = max(0, shared_prefix_end - input_start_pos)
                            branch_start_in_input = shared_prefix_in_input_length
                            confidence = evaluate_branch_confidence(
                                logits,
                                branch,
                                branch_start_in_input,
                                branch_length,
                                shared_prefix_end,
                                self.mask_token_id,
                                sampling_strategy=self.sampling_strategy,
                                branch_topp=self.branch_topp,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                top_k=self.top_k,
                                selection_conf_alpha=self.selection_conf_alpha,
                            )
                            branch_confidences.append((confidence, branch_idx, branch))
                        else:
                            branch_confidences.append((1.0, branch_idx, branch))
                    else:
                        # 无 logits 时，退化为仅使用创建置信度
                        fallback_conf = float(getattr(branch, "creation_token_confidence", 0.0))
                        branch_confidences.append((fallback_conf, branch_idx, branch))

                # 选择最佳分支
                branch_confidences.sort(key=lambda x: x[0], reverse=True)
                if self.verification_force_base_winner:
                    base_branch_result = next(((b, l, p) for (b, l, p) in branch_results if getattr(b, 'is_base', False)), None)
                    if base_branch_result is not None:
                        best_branch, best_logits, best_new_pkv = base_branch_result
                        best_confidence = 1.0
                    else:
                        best_confidence, best_idx, best_branch = branch_confidences[0]
                        _, best_logits, best_new_pkv = branch_results[best_idx]
                else:
                    best_confidence, best_idx, best_branch = branch_confidences[0]
                    _, best_logits, best_new_pkv = branch_results[best_idx]

                # [MEMORY OPTIMIZED] 更新共享KV缓存为最佳分支的KV
                if update_kvcache_len > 0 and best_new_pkv is not None:
                    self.shared_past_key_values = best_new_pkv
                    shared_cache_length += update_kvcache_len

                # [MEMORY OPTIMIZED] 重用变量，避免重复计算和tensor创建
                tokens_to_update = {}
                
                # 更新最佳分支的tokens
                if best_logits is not None:
                    branch_length = best_branch.x_t.shape[1] - shared_prefix_end if best_branch.x_t.shape[1] > shared_prefix_end else 0
                    if branch_length > 0:
                        # 找到分支在输入序列中的起始位置
                        shared_prefix_in_input_length = max(0, shared_prefix_end - input_start_pos)
                        branch_start_in_input = shared_prefix_in_input_length

                        # [MEMORY OPTIMIZED] 避免创建新的tensor切片，直接使用索引
                        active_blocks_ids = [bid for bid, state in best_branch.block_states.items() if state['state'] == 'active']

                        for block_id in active_blocks_ids:
                            block_start, block_end = best_branch.block_states[block_id]['start_pos'], best_branch.block_states[block_id]['end_pos']
                            mask_indices_in_block = (best_branch.x_t[0, block_start:block_end] == self.mask_token_id).nonzero(as_tuple=True)[0]
                            if len(mask_indices_in_block) == 0:
                                continue

                            mask_indices_abs = mask_indices_in_block + block_start
                            # 计算相对于分支起始位置的索引
                            mask_indices_rel_local = mask_indices_abs - shared_prefix_end

                            # 确保索引在有效范围内
                            valid_mask_indices = mask_indices_rel_local[(mask_indices_rel_local >= 0) & (mask_indices_rel_local < branch_length)]
                            if len(valid_mask_indices) == 0:
                                continue

                            # [MEMORY OPTIMIZED] 直接使用索引访问logits，避免创建新tensor
                            logits_indices = branch_start_in_input + valid_mask_indices
                            block_mask_logits = best_logits[0, logits_indices, :]
                            confidence, x0, initial_confidence = sample_tokens(
                                block_mask_logits, self.temperature, 
                                top_p=self.top_p, top_k=self.top_k, 
                                sampling_strategy=self.sampling_strategy
                            )
                            high_conf_indices = (initial_confidence > self.skip_threshold).nonzero(as_tuple=True)[0]
                            is_complete = best_branch.block_states[block_id]['is_complete']

                            indices_to_fill = []
                            if is_complete:
                                if len(high_conf_indices) > 0:
                                    indices_to_fill = high_conf_indices
                                else:
                                    run_stats["original_fallback_triggers"] += 1
                                    if len(confidence) > 0:
                                        _, most_conf_idx = torch.topk(confidence, 1)
                                        indices_to_fill = most_conf_idx
                            else:
                                if len(high_conf_indices) > 0:
                                    indices_to_fill = high_conf_indices

                            if len(indices_to_fill) > 0:
                                for idx in indices_to_fill:
                                    rel_local = valid_mask_indices[idx.item()]
                                    # 映射回分支绝对位置
                                    abs_pos = rel_local + shared_prefix_end
                                    token = x0[idx.item()].item()
                                    tokens_to_update[abs_pos.item()] = token
                                    # 去除与最终决策无关的 step_confidences 统计
                                    if token == self.eot_token_id:
                                        best_branch.eos_detected = True

                        run_stats["total_filled_original"] += len(tokens_to_update)

                        # [MEMORY OPTIMIZED] 只在有更新时执行tensor操作
                        if tokens_to_update:
                            for pos, token in tokens_to_update.items():
                                best_branch.x_t[0, pos] = token

                        # 移除中间平均置信度(稍后用 best_confidence 覆盖, 行为等价)

                # [MODIFIED] Set the branch's confidence to the new model-validated score for this round.
                best_branch.confidence = best_confidence

                # 生成新的k个分支用于下一轮
                newly_spawned_branches = []

                # === 预计算基础分支在“分支前”完成的块，用于后续安全更新缓存状态 ===
                # 说明：需求是块的 to_cache 转换只能依据基础逻辑（即当前轮 winner 的原始填充结果）
                # 不能因为生成竞争分支（其额外采样的一个 token 可能补齐块）而提前把块标记为 to_cache。
                base_completed_blocks = []  # 仅记录在 best_branch 当前序列中已真实完成的块
                base_active_blocks_snapshot = []  # 记录当前仍处于 active 的块（用于后续遍历）
                for _bid, _state in best_branch.block_states.items():
                    if _state['state'] == 'active':
                        base_active_blocks_snapshot.append(_bid)
                        start_, end_ = _state['start_pos'], _state['end_pos']
                        new_mask_cnt_ = (best_branch.x_t[0, start_:end_] == self.mask_token_id).sum().item()
                        # 同步基础分支上的 mask_count（这是基础逻辑完成后的客观结果）
                        _state['mask_count'] = new_mask_cnt_
                        if new_mask_cnt_ == 0:
                            # 该块在基础逻辑下已经填满，可以视为“候选可缓存”
                            base_completed_blocks.append(_bid)

                if self.branch_verification_mode:
                    if self.base_branch_competition:
                        # 基础分支竞争模式：基础分支也参与竞争
                        # [MEMORY OPTIMIZED] 直接使用best_branch，避免副本创建
                        base_branch = best_branch.copy()
                        base_branch.branch_id = len(branches)
                        base_branch.is_base = True
                        base_branch.creation_token_confidence = 1.0
                        base_branch.confidence = 1.0

                        # 生成其他竞争分支（如需要）。注意：基础分支将在最后追加。
                        if self.branching_factor > 1:
                            chosen_positions = set(tokens_to_update.keys()) if locals().get('tokens_to_update') else set()
                            self._generate_additional_branches(
                                newly_spawned_branches, best_branch, best_logits, 
                                active_blocks_ids, shared_prefix_end, chosen_positions, run_stats, branches
                            )

                        # 最后追加基础分支，确保它在竞争序列的末尾
                        newly_spawned_branches.append(base_branch)
                    else:
                        # 验证模式：直接使用经过基础逻辑更新的序列作为下一轮分支
                        best_branch.is_base = True
                        best_branch.creation_token_confidence = 1.0
                        best_branch.confidence = 1.0
                        newly_spawned_branches.append(best_branch)
                elif self.branching_factor > 1:
                    chosen_positions = set(tokens_to_update.keys()) if locals().get('tokens_to_update') else set()
                    self._generate_additional_branches(
                        newly_spawned_branches, best_branch, best_logits, 
                        active_blocks_ids, shared_prefix_end, chosen_positions, run_stats, branches
                    )
                else:
                    best_branch.is_base = True
                    best_branch.creation_token_confidence = 1.0
                    best_branch.confidence = 1.0
                    newly_spawned_branches.append(best_branch)

                # 更新缓存和块状态
                for nb in newly_spawned_branches:
                    if update_kvcache_len > 0:
                        for block_id in blocks_to_cache:
                            if block_id in nb.block_states:
                                nb.block_states[block_id]['state'] = 'in_cache'
                    # === 基于基础分支快照更新：只允许基础分支本轮已完成的块进入 to_cache ===
                    # 1. 重新同步各 active 块的 mask_count（不同分支的该值理论上只会 <= 基础分支，对我们逻辑不重要）
                    nb_active_blocks = [bid for bid, state in nb.block_states.items() if state['state'] == 'active']
                    for block_id in nb_active_blocks:
                        start, end = nb.block_states[block_id]['start_pos'], nb.block_states[block_id]['end_pos']
                        nb.block_states[block_id]['mask_count'] = (nb.x_t[0, start:end] == self.mask_token_id).sum().item()

                    # 2. 仅当该块在基础分支完成，且其前序块没有仍 active 的，才标记为 to_cache。
                    for block_id in nb_active_blocks:
                        if block_id in base_completed_blocks:
                            can_deactivate = all(prev_bid not in nb.block_states or nb.block_states[prev_bid]['state'] != 'active' for prev_bid in range(block_id))
                            if can_deactivate and nb.block_states[block_id]['mask_count'] == 0:
                                nb.block_states[block_id]['state'] = 'to_cache'

                # [MODIFIED] Replace old pruning logic with the new branch lifecycle logic.
                # The branches for the next round consist of all previously inactive branches
                # plus the new branches spawned from this round's single winner.
                branches = [b for b in branches if not b.is_active] + newly_spawned_branches

                # Update statistics based on the new logic.
                # We started with N active branches, selected 1 to be the parent, and discarded the other N-1.
                run_stats["branches_pruned"] += len(active_branches_this_round) - 1
                run_stats["candidates_generated_this_round"] = len(newly_spawned_branches)

                # [MEMORY OPTIMIZED] 强制垃圾回收，释放被剪枝分支的显存
                if len(active_branches_this_round) > 1:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # if parallel_step_count > 500:
                #     eval_logger.warning(f"Generation stopped due to exceeding 500 parallel steps.")
                #     # 强制结束所有分支
                #     for b in branches:
                #         b.is_active = False
                #         if b.steps_completed == -1: b.steps_completed = parallel_step_count
                #     break

        # [MODIFIED] Select the best branch based on completion speed, not the abandoned confidence metric.
        if not branches:
            eval_logger.warning("No branches available, returning original prompt")
            return [], 0, 0, run_stats

        # Filter for branches that actually completed their generation.
        completed_branches = [b for b in branches if not b.is_active and b.steps_completed != -1]

        if completed_branches:
            # Among completed branches, choose the one that finished in the fewest parallel steps.
            best_branch = min(completed_branches, key=lambda x: x.steps_completed)
            selection_reason = "Fastest completion (fewest parallel steps)"
        else:
            # If no branch completed (e.g., all hit the step limit), fall back to the one
            # that managed to generate the most tokens before stopping.
            eval_logger.warning("No branches completed generation. Selecting the one with the most generated tokens.")
            best_branch = max(branches, key=lambda x: x.generated_token_count)
            selection_reason = "Most tokens generated (fallback)"

        generated_sequence_ids = best_branch.x_t[0, prompt_length:].tolist()
        best_branch_steps = best_branch.steps_completed if best_branch.steps_completed != -1 else run_stats["parallel_steps"]
        (
            generated_sequence_ids,
            actual_tokens_excluding_eos,
            generated_tokens_including_eos,
        ) = self._compute_generation_token_stats(generated_sequence_ids, self.block_size)
        run_stats["actual_tokens_excluding_eos"] = actual_tokens_excluding_eos
        run_stats["generated_tokens_including_eos"] = generated_tokens_including_eos

        return generated_sequence_ids, best_branch_steps, generated_tokens_including_eos, run_stats

    def _generate_additional_branches(self, newly_spawned_branches, best_branch_template, best_logits, 
                                    active_blocks_ids, shared_prefix_end, chosen_positions, run_stats, 
                                    branches):
        """生成额外的分支用于竞争"""
        if best_logits is None:
            # 如果没有logits，只保留基础分支
            if not newly_spawned_branches:
                best_branch_template.is_base = True
                best_branch_template.creation_token_confidence = 1.0
                best_branch_template.confidence = 1.0
                newly_spawned_branches.append(best_branch_template)
            return

        # [MEMORY OPTIMIZED] 计算分支长度，避免重复计算
        branch_length = best_branch_template.x_t.shape[1] - shared_prefix_end if best_branch_template.x_t.shape[1] > shared_prefix_end else 0
        
        if branch_length > 0:
            # [MEMORY OPTIMIZED] 直接切片logits，避免多次创建副本
            if best_logits.shape[1] >= branch_length:
                branch_logits = best_logits[0, -branch_length:, :]
            else:
                branch_logits = best_logits[0, :, :]

            # [MEMORY OPTIMIZED] 预计算候选位置，避免在循环中重复计算
            candidate_positions = []
            for block_id in active_blocks_ids:
                block_start, block_end = best_branch_template.block_states[block_id]['start_pos'], best_branch_template.block_states[block_id]['end_pos']
                mask_indices_in_block = (best_branch_template.x_t[0, block_start:block_end] == self.mask_token_id).nonzero(as_tuple=True)[0]

                if len(mask_indices_in_block) > 0:
                    mask_indices_abs = mask_indices_in_block + block_start
                    for abs_pos_tensor in mask_indices_abs:
                        abs_pos = abs_pos_tensor.item()
                        if abs_pos not in chosen_positions and abs_pos >= shared_prefix_end:
                            rel_pos_local = abs_pos - shared_prefix_end
                            if 0 <= rel_pos_local < branch_length:
                                candidate_positions.append((abs_pos, rel_pos_local))

            top_candidates = []
            if candidate_positions:
                # [MEMORY OPTIMIZED] 批量计算置信度，避免逐个创建tensor
                confidences = []
                for abs_pos, rel_pos_local in candidate_positions:
                    if rel_pos_local < branch_logits.shape[0]:
                        pos_logits = branch_logits[rel_pos_local, :].unsqueeze(0)
                        # 使用统一的sample_tokens函数计算置信度
                        conf, _, _ = sample_tokens(
                            pos_logits, self.temperature, 
                            top_p=self.top_p, top_k=self.top_k,
                            sampling_strategy=self.sampling_strategy
                        )
                        confidences.append((conf.item(), abs_pos, rel_pos_local))

                # 选择前k个最高置信度的位置
                confidences.sort(key=lambda x: x[0], reverse=True)
                # 在竞争模式下，减少1个位置给基础分支
                effective_branching_factor = self.branching_factor - 1 if self.base_branch_competition and len(newly_spawned_branches) > 0 else self.branching_factor
                num_to_select = min(effective_branching_factor, len(confidences))
                
                for i in range(num_to_select):
                    confidence, abs_pos, rel_pos = confidences[i]
                    top_candidates.append((abs_pos, rel_pos, confidence))
        else:
            top_candidates = []

        if not top_candidates and not newly_spawned_branches:
            best_branch_template.is_base = True
            best_branch_template.creation_token_confidence = 1.0
            best_branch_template.confidence = 1.0
            newly_spawned_branches.append(best_branch_template)
        else:
            for abs_pos, rel_pos, confidence_score in top_candidates:
                # [MEMORY OPTIMIZED] 只在确实需要时才创建新分支副本
                new_branch = best_branch_template.copy()
                new_branch.branch_id = len(branches) + len(newly_spawned_branches)
                # 单一KV缓存，不在分支上存储KV
                new_branch.is_base = False

                # 对该位置进行采样
                if rel_pos < branch_logits.shape[0]:
                    pos_logits = branch_logits[rel_pos, :].unsqueeze(0)
                    conf_val, new_token, initial_conf = sample_tokens(
                        pos_logits, self.temperature, 
                        top_p=self.top_p, top_k=self.top_k,
                        sampling_strategy=self.sampling_strategy
                    )

                    token = new_token.item()
                    new_branch.x_t[0, abs_pos] = token
                    # 用户需求：创建置信度统一使用 sampling_strategy 变换后的 conf_val
                    creation_conf = float(conf_val.item())
                    new_branch.creation_token_confidence = creation_conf
                    new_branch.confidence = creation_conf

                    if token == self.eot_token_id:
                        new_branch.eos_detected = True

                    newly_spawned_branches.append(new_branch)
                    run_stats["total_filled_new"] += 1
                    run_stats["branches_created"] += 1

            # 如果没有成功生成任何新分支（包括基础分支），保留基础分支
            if not newly_spawned_branches:
                best_branch_template.is_base = True
                best_branch_template.creation_token_confidence = 1.0
                best_branch_template.confidence = 1.0
                newly_spawned_branches.append(best_branch_template)

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

            generated_answer, best_branch_steps, best_branch_tokens, stats = self._generate_enhanced_speculative(input_ids)

            total_steps += best_branch_steps
            total_generated_tokens_including_eos += best_branch_tokens
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

            if self.use_uncertainty_logic:
                if self.branch_verification_mode and self.base_branch_competition:
                    mode = "Multi-Branch (Base Branch Competition Mode)"
                elif self.branch_verification_mode:
                    mode = "Multi-Branch (Verification Mode)"
                else:
                    mode = "Multi-Branch"
            else:
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

    def _update_block_completion_states(self, block_states, decoded_token_threshold):
        for block_id in sorted(block_states.keys()):
            if block_states[block_id]['total_masks'] > 0:
                decoded_tokens = block_states[block_id]['total_masks'] - block_states[block_id]['mask_count']
                decode_ratio = decoded_tokens / block_states[block_id]['total_masks']
                if decode_ratio >= decoded_token_threshold:
                    next_block_id = block_id + 1
                    if next_block_id in block_states:
                        block_states[next_block_id]['is_complete'] = True

    def _generate_original_single_branch(self, prompt: torch.Tensor) -> Tuple[List[int], Dict]:
        self.model.eval()
        prompt_length = prompt.shape[1]

        x_t = prompt.clone().to(self.device)

        if self.use_full_attention:
            full_attention_mask = create_full_attention_mask(
                max_length=self.max_length,
                device=self.device, dtype=self.target_dtype if self.target_dtype not in [None, "auto"] else torch.bfloat16
            )
        else:
            full_attention_mask = create_full_block_attention_mask(
                prompt_length=prompt_length, max_length=self.max_length, block_size=self.block_size,
                device=self.device, dtype=self.target_dtype if self.target_dtype not in [None, "auto"] else torch.bfloat16
            )

        block_states = {
            0: {'start_pos': 0, 'end_pos': prompt_length, 'mask_count': 0, 'total_masks': prompt_length, 'state': 'to_cache', 'is_complete': True},
        }

        past_key_values = None
        cache_length = 0
        current_blocks = 0
        step = 0
        eos_detected = False
        eos_token_id = self.eot_token_id

        stats = {
            "steps_taken": 0, "total_filled_original": 0, "original_fallback_triggers": 0,
        }

        with torch.inference_mode():
            while True:
                step += 1

                if len(block_states)-1 < (self.max_new_tokens // self.block_size) and not eos_detected:
                    last_block_id = len(block_states) - 1
                    if block_states[last_block_id]['total_masks'] > 0:
                        progress = (block_states[last_block_id]['total_masks'] - block_states[last_block_id]['mask_count']) / block_states[last_block_id]['total_masks']
                    else:
                        progress = 1.0
                    if progress >= self.block_add_threshold:
                        new_block_id = len(block_states)
                        new_start_pos = x_t.shape[1]
                        mask_block = torch.tensor([[self.mask_token_id] * self.block_size], device=self.device)
                        x_t = torch.cat([x_t, mask_block], dim=1)
                        block_states[new_block_id] = {'start_pos': new_start_pos, 'end_pos': new_start_pos + self.block_size, 'mask_count': self.block_size, 'total_masks': self.block_size, 'state': 'active', 'is_complete': False}
                        current_blocks += 1

                self._update_block_completion_states(block_states, self.decoded_token_threshold)

                if (x_t == self.mask_token_id).sum() == 0 and current_blocks == 0:
                    break

                blocks_to_cache = [bid for bid, state in block_states.items() if state['state'] == 'to_cache']
                update_kvcache = 0
                if blocks_to_cache:
                    earliest_pos = min(block_states[bid]['start_pos'] for bid in blocks_to_cache)
                    latest_pos = max(block_states[bid]['end_pos'] for bid in blocks_to_cache)
                    update_kvcache = latest_pos - earliest_pos

                process_start_pos = cache_length
                active_blocks_ids = [bid for bid, state in block_states.items() if state['state'] == 'active']

                if update_kvcache > 0:
                    earliest_block_to_cache = min(blocks_to_cache)
                    process_start_pos = block_states[earliest_block_to_cache]['start_pos']
                    input_seq = x_t[:, process_start_pos:]
                elif active_blocks_ids:
                    process_start_pos = min(block_states[bid]['start_pos'] for bid in active_blocks_ids)
                    input_seq = x_t[:, process_start_pos:]
                else:
                    if blocks_to_cache:
                        continue
                    else:
                        break

                if input_seq.shape[1] == 0:
                    if (x_t == self.mask_token_id).any():
                        continue
                    else:
                        break

                attention_mask = extract_attention_mask(full_mask=full_attention_mask, start_pos=process_start_pos, input_length=input_seq.shape[1], cache_length=cache_length, use_full_attention=self.use_full_attention)

                outputs = self.model(
                    input_seq,
                    attention_bias=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True, update_kvcache=update_kvcache + cache_length
                )

                if outputs.past_key_values is None:
                    eval_logger.error(f"Model did not return past_key_values at step {step}.")
                    stats['steps_taken'] = step
                    return x_t[0, prompt_length:].tolist(), stats

                logits = outputs.logits

                if update_kvcache > 0:
                    past_key_values = outputs.past_key_values

                tokens_to_update = {}

                for block_id in active_blocks_ids:
                    block_start, block_end = block_states[block_id]['start_pos'], block_states[block_id]['end_pos']
                    mask_indices_in_block = (x_t[0, block_start:block_end] == self.mask_token_id).nonzero(as_tuple=True)[0]
                    if len(mask_indices_in_block) == 0:
                        continue

                    mask_indices_abs = mask_indices_in_block + block_start
                    mask_indices_rel = mask_indices_abs - process_start_pos
                    block_mask_logits = logits[0, mask_indices_rel, :]

                    confidence, x0, initial_confidence = sample_tokens(
                        block_mask_logits, self.temperature, 
                        top_p=self.top_p, top_k=self.top_k,
                        sampling_strategy=self.sampling_strategy
                    )
                    high_conf_indices = (initial_confidence > self.skip_threshold).nonzero(as_tuple=True)[0]

                    is_complete = block_states[block_id]['is_complete']

                    indices_to_fill = []
                    if is_complete:
                        if len(high_conf_indices) > 0:
                            indices_to_fill = high_conf_indices
                        else:
                            stats["original_fallback_triggers"] += 1
                            if len(confidence) > 0:
                                _, most_conf_idx = torch.topk(confidence, 1)
                                indices_to_fill = most_conf_idx
                    else:
                        if len(high_conf_indices) > 0:
                            indices_to_fill = high_conf_indices

                    if len(indices_to_fill) > 0:
                        for idx in indices_to_fill:
                            pos = mask_indices_abs[idx.item()].item()
                            token = x0[idx.item()].item()
                            tokens_to_update[pos] = token
                            if token == eos_token_id:
                                eos_detected = True

                stats["total_filled_original"] += len(tokens_to_update)

                # [MEMORY OPTIMIZED] 只在有更新时创建tensor，并立即清理
                if tokens_to_update:
                    positions = torch.tensor(list(tokens_to_update.keys()), device=self.device)
                    values = torch.tensor(list(tokens_to_update.values()), device=self.device)
                    x_t[0, positions] = values
                    # 立即清理临时tensor
                    del positions, values

                if update_kvcache > 0:
                    cache_length += update_kvcache
                    for block_id in blocks_to_cache:
                        block_states[block_id]['state'] = 'in_cache'

                blocks_to_deactivate = []
                for block_id in active_blocks_ids:
                    start, end = block_states[block_id]['start_pos'], block_states[block_id]['end_pos']
                    new_mask_count = (x_t[0, start:end] == self.mask_token_id).sum().item()
                    block_states[block_id]['mask_count'] = new_mask_count
                    if new_mask_count == 0:
                        blocks_to_deactivate.append(block_id)

                for block_id in blocks_to_deactivate:
                    if block_states[block_id]['state'] == 'active':
                        can_deactivate = all(prev_bid not in block_states or block_states[prev_bid]['state'] != 'active' for prev_bid in range(block_id))
                        if can_deactivate:
                            block_states[block_id]['state'] = 'to_cache'
                            current_blocks -= 1

                if step > 1000:
                    eval_logger.warning("Generation stopped due to exceeding 1000 steps.")
                    break

        generated_sequence_ids = x_t[0, prompt_length:].tolist()
        stats['steps_taken'] = step
        (
            generated_sequence_ids,
            actual_tokens_excluding_eos,
            generated_tokens_including_eos,
        ) = self._compute_generation_token_stats(generated_sequence_ids, self.block_size)
        stats["actual_tokens_excluding_eos"] = actual_tokens_excluding_eos
        stats["generated_tokens_including_eos"] = generated_tokens_including_eos

        return generated_sequence_ids, stats

if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
