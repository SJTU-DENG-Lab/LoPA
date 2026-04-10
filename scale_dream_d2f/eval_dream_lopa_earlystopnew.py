import logging
import gc
import json
import time
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union, Dict, Set
import torch
import torch.nn.functional as F
import torch.distributions as dists
import transformers
from accelerate import Accelerator
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import random

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Branch:
    def __init__(self, branch_id: int, x_t: torch.Tensor,
                 prompt_length: int = 0, is_base: bool = False,
                 creation_token_confidence: float = 1.0):
        self.branch_id = branch_id
        self.x_t = x_t.clone()
        self.is_active = True
        self.prompt_length = prompt_length
        self.is_base = is_base
        self.creation_token_confidence = creation_token_confidence

    def copy(self):
        return Branch(self.branch_id, self.x_t, self.prompt_length, self.is_base, self.creation_token_confidence)

def top_p_logits(logits, top_p=None):
    if top_p is None or top_p >= 1.0: return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

def top_k_logits(logits, top_k=None):
    if top_k is None or top_k == 0: return logits
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, sampling_strategy="default"):
    if temperature > 0: logits = logits / temperature
    logits = top_k_logits(logits, top_k)
    logits = top_p_logits(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0 and temperature != 0.0:
        x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
        initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        initial_confidence, x0 = probs.max(dim=-1)
    if sampling_strategy == "margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[..., 0] - sorted_probs[..., 1]
    elif sampling_strategy == "neg_entropy":
        confidence = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    else:
        confidence = initial_confidence.clone()
    return confidence, x0, initial_confidence

def evaluate_branch_confidence(logits: torch.Tensor, branch: Branch, mask_token_id: int, sampling_strategy: str,
                               branch_topp: float, temperature: float, top_p: Optional[float], top_k: Optional[int],
                               selection_conf_alpha: float) -> float:
    mask_positions = (branch.x_t[0, branch.prompt_length:] == mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        future_conf = 1.0
    else:
        shifted_logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        mask_logits = shifted_logits[0, branch.prompt_length + mask_positions, :]
        
        confidences, _, _ = sample_tokens(mask_logits, temperature, top_p, top_k, sampling_strategy)
        if len(confidences) == 1:
            future_conf = confidences.item()
        else:
            bottom_cnt = max(1, int(len(confidences) * branch_topp))
            sorted_confidences, _ = torch.sort(confidences, descending=False)
            future_conf = sorted_confidences[:bottom_cnt].mean().item()
    creation_conf = float(getattr(branch, "creation_token_confidence", 1.0))
    alpha = float(max(0.0, min(1.0, selection_conf_alpha)))
    return alpha * creation_conf + (1.0 - alpha) * future_conf

@register_model("dream")
class Dream(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        max_new_tokens: Optional[int] = 128,
        max_length: Optional[int] = 2048,
        add_bos_token: Optional[bool] = False,
        use_uncertainty_logic: Optional[bool] = True,
        base_branch_competition: Optional[bool] = True,
        verification_force_base_winner: Optional[bool] = False,
        branching_factor: Optional[int] = 2,
        branch_topp: Optional[float] = 0.5,
        selection_conf_alpha: Optional[float] = 0.5,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        accelerator = Accelerator()
        if accelerator.num_processes > 1: self.accelerator = accelerator
        
        if not hasattr(self, "accelerator"):
            self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self._device = accelerator.device

        if isinstance(batch_size, str): self.batch_size_per_gpu = int(batch_size)
        else: self.batch_size_per_gpu = batch_size
        
        self._create_model_and_tokenizer(pretrained, dtype, kwargs.get("trust_remote_code", True))

        if hasattr(self, "accelerator"):
            self._rank, self._world_size = self.accelerator.local_process_index, self.accelerator.num_processes
        else:
            self._rank, self._world_size = 0, 1

        model_device = next(self.model.parameters()).device
        if str(self.device) != str(model_device): self._device = model_device
            
        self.max_length = max_length
        self.add_bos_token = add_bos_token
        self.max_new_tokens = max_new_tokens
        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", None)
        self.top_k = kwargs.get("top_k", None)
        self.use_uncertainty_logic = use_uncertainty_logic
        self.base_branch_competition = base_branch_competition
        self.verification_force_base_winner = verification_force_base_winner
        self.branching_factor = branching_factor
        self.branch_topp = branch_topp
        self.selection_conf_alpha = selection_conf_alpha
        self.sampling_strategy = kwargs.get("sampling_strategy", "default")
        self.show_speed = kwargs.get("show_speed", True)
        self.escape_until = kwargs.get("escape_until", False)
        self.save_dir = save_dir
        self.nll_type=kwargs.get("nll_type", "mc")
        self.log_type=kwargs.get("log_type", "ftb")
        
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
        self.model = transformers.AutoModel.from_pretrained(
            pretrained, torch_dtype=get_dtype(dtype), trust_remote_code=trust_remote_code
        ).eval().to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids

    def _compute_generation_token_stats(
        self,
        generated_ids: List[int],
        block_length: int,
    ) -> Tuple[List[int], int, int]:
        trimmed_ids = list(generated_ids)
        mask_token_id = self.tokenizer.mask_token_id

        while trimmed_ids and trimmed_ids[-1] == mask_token_id:
            trimmed_ids.pop()

        actual_tokens_excluding_eos = len(trimmed_ids)
        if self.eot_token_id in trimmed_ids:
            generated_tokens_including_eos = (
                (trimmed_ids.index(self.eot_token_id) // block_length) + 1
            ) * block_length
            trimmed_ids = trimmed_ids[: trimmed_ids.index(self.eot_token_id)]
            actual_tokens_excluding_eos = len(trimmed_ids)
        else:
            generated_tokens_including_eos = len(trimmed_ids)
        return trimmed_ids, actual_tokens_excluding_eos, generated_tokens_including_eos
    
    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        args.update(args2)
        return cls(**args)

    @torch.no_grad()
    def _generate_enhanced_speculative(self, prompt: torch.Tensor) -> Tuple[List[int], int, int, Dict]:
        prompt_length = prompt.shape[1]
        mask_token_id = self.tokenizer.mask_token_id
        block_length = 32
        gen_length = self.max_new_tokens
        total_length = prompt_length + gen_length
        num_blocks = gen_length // block_length
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length"

        initial_x_t = torch.full((1, total_length), mask_token_id, dtype=torch.long, device=self.device)
        initial_x_t[:, :prompt_length] = prompt
        
        branches = [Branch(0, initial_x_t, prompt_length=prompt_length, is_base=True)]
        
        total_parallel_steps = 0
        masks_decoded_per_step = []  # 核心新增：记录每步实际解码的mask数量

        for block_idx in range(num_blocks):
            block_start = prompt_length + block_idx * block_length
            block_end = block_start + block_length
            
            while True:
                active_branches = [b for b in branches if b.is_active]
                
                base_branch_for_check = next((b for b in active_branches if b.is_base), None)
                if not base_branch_for_check or (base_branch_for_check.x_t[0, block_start:block_end] == mask_token_id).sum() == 0:
                    break
                
                # 核心新增：记录本次迭代前的mask数量
                masks_before = (base_branch_for_check.x_t[0, block_start:block_end] == mask_token_id).sum().item()

                winner_branch = None
                winner_logits = None
                best_confidence = -float('inf')
                base_branch_info = None
                branch_scores = []

                total_parallel_steps += 1
                
                for branch in active_branches:
                    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.model(branch.x_t)
                        logits = outputs.logits
                    
                    confidence = evaluate_branch_confidence(logits, branch, mask_token_id, self.sampling_strategy, self.branch_topp, self.temperature, self.top_p, self.top_k, self.selection_conf_alpha)
                    branch_scores.append((confidence, branch))
                    
                    if branch.is_base:
                        base_branch_info = (confidence, branch)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        winner_branch = branch
                        if winner_logits is not None:
                            del winner_logits
                        winner_logits = logits.clone()
                        del logits
                    else:
                        del logits
                
                if self.verification_force_base_winner and self.base_branch_competition and base_branch_info:
                    conf, base_branch = base_branch_info
                    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.model(base_branch.x_t)
                    winner_logits = outputs.logits.clone()
                    winner_branch = base_branch
                    best_confidence = conf

                branch_confidences = [(c, b, winner_logits if b is winner_branch else None) for c, b in branch_scores]
                branch_confidences.sort(key=lambda x: x[0], reverse=True)
                
                shifted_logits = torch.cat([winner_logits[:, :1], winner_logits[:, :-1]], dim=1)
                
                base_sequence = winner_branch.x_t.clone()
                threshold = 0.9
                
                mask_indices_in_block = (base_sequence[0, block_start:block_end] == mask_token_id).nonzero(as_tuple=True)[0]
                filled_positions_relative = set()

                if mask_indices_in_block.numel() > 0:
                    logits_region = shifted_logits[0, block_start:block_end]
                    mask_logits = logits_region[mask_indices_in_block]
                    
                    confidence, x0, _ = sample_tokens(mask_logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, sampling_strategy=self.sampling_strategy)
                    
                    high_conf_indices = (confidence > threshold).nonzero(as_tuple=True)[0]
                    
                    if high_conf_indices.numel() > 0:
                        indices_to_update_in_block = mask_indices_in_block[high_conf_indices]
                        tokens_to_insert = x0[high_conf_indices]
                        base_sequence[0, block_start + indices_to_update_in_block] = tokens_to_insert
                        filled_positions_relative.update(indices_to_update_in_block.tolist())
                    else:
                        if confidence.numel() > 0:
                            max_conf_idx = torch.argmax(confidence).item()
                            index_to_update_in_block = mask_indices_in_block[max_conf_idx]
                            token_to_insert = x0[max_conf_idx]
                            base_sequence[0, block_start + index_to_update_in_block] = token_to_insert
                            filled_positions_relative.add(index_to_update_in_block.item())
                
                newly_spawned_branches = []
                
                if self.base_branch_competition:
                    base_branch = Branch(len(branches) + len(newly_spawned_branches), base_sequence, prompt_length=prompt_length, is_base=True)
                    newly_spawned_branches.append(base_branch)

                self._generate_additional_branches(
                    newly_spawned_branches, 
                    winner_branch,
                    base_sequence, 
                    shifted_logits, 
                    block_start, 
                    block_end, 
                    filled_positions_relative, 
                    branches
                )
                
                if not newly_spawned_branches:
                    fallback_branch = Branch(len(branches), base_sequence, prompt_length=prompt_length, is_base=True)
                    newly_spawned_branches.append(fallback_branch)

                # 核心新增：计算本次迭代实际解码的mask数量
                final_base_branch = next((b for b in newly_spawned_branches if b.is_base), newly_spawned_branches[0])
                masks_after = (final_base_branch.x_t[0, block_start:block_end] == mask_token_id).sum().item()
                masks_decoded_this_step = masks_before - masks_after
                if masks_decoded_this_step > 0:  # 只记录有实际解码的步骤
                    masks_decoded_per_step.append(masks_decoded_this_step)

                old_branches = branches
                branches = newly_spawned_branches
                for branch in old_branches:
                    del branch.x_t
                del old_branches
                gc.collect()

            # 修复：在block生成完成后检查EOS并决定是否继续
            base_branch_final = next((b for b in branches if b.is_base), branches[0])
            generated_ids_check = base_branch_final.x_t[0, prompt_length:].tolist()
            if self.eot_token_id in generated_ids_check:
                break

        final_branch = next((b for b in branches if b.is_base), branches[0])
        generated_ids = final_branch.x_t[0, prompt_length:].tolist()
        (
            generated_ids,
            actual_tokens_excluding_eos,
            generated_tokens_including_eos,
        ) = self._compute_generation_token_stats(generated_ids, block_length)
        
        # 返回解码效率统计
        stats = {
            "parallel_steps": total_parallel_steps,
            "masks_decoded_per_step": masks_decoded_per_step,
            "actual_tokens_excluding_eos": actual_tokens_excluding_eos,
            "generated_tokens_including_eos": generated_tokens_including_eos,
        }
        return generated_ids, total_parallel_steps, generated_tokens_including_eos, stats

    def _generate_additional_branches(self, newly_spawned_branches, original_winner_branch, completed_sequence, shifted_logits, block_start, block_end, already_filled_relative, branches):
        mask_token_id = self.tokenizer.mask_token_id
        
        original_mask_indices_relative = (original_winner_branch.x_t[0, block_start:block_end] == mask_token_id).nonzero(as_tuple=True)[0]
        
        explorable_indices_relative = [idx.item() for idx in original_mask_indices_relative if idx.item() not in already_filled_relative]
        
        if not explorable_indices_relative: return

        explorable_indices_tensor = torch.tensor(explorable_indices_relative, device=self.device, dtype=torch.long)
        
        logits_region = shifted_logits[0, block_start:block_end, :]
        mask_logits = logits_region[explorable_indices_tensor]
        
        confidence, x0, _ = sample_tokens(mask_logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, sampling_strategy=self.sampling_strategy)

        all_potential_fills = []
        for i in range(len(explorable_indices_relative)):
            all_potential_fills.append({
                'conf': confidence[i].item(),
                'pos_relative': explorable_indices_relative[i],
                'token': x0[i].item()
            })
        
        all_potential_fills.sort(key=lambda x: x['conf'], reverse=True)

        num_to_generate = self.branching_factor
        if self.base_branch_competition: num_to_generate -= 1
        if num_to_generate <= 0: return
        
        num_to_select = min(num_to_generate, len(all_potential_fills))

        for i in range(num_to_select):
            fill_op = all_potential_fills[i]
            
            new_branch_sequence = completed_sequence.clone()
            absolute_pos = block_start + fill_op['pos_relative']
            new_branch_sequence[0, absolute_pos] = fill_op['token']
            
            new_branch = Branch(
                branch_id=len(branches) + len(newly_spawned_branches),
                x_t=new_branch_sequence,
                prompt_length=original_winner_branch.prompt_length,
                is_base=False,
                creation_token_confidence=fill_op['conf']
            )
            newly_spawned_branches.append(new_branch)
            
    def _iteratively_fill_block(self, current_sequence: torch.Tensor, block_start: int, block_end: int) -> Tuple[torch.Tensor, int]:
        x = current_sequence.clone()
        steps_taken = 0
        mask_token_id = self.tokenizer.mask_token_id
        threshold = 0.9

        for _ in range(block_end - block_start + 1):
            mask_index_in_block = (x[0, block_start:block_end] == mask_token_id)
            if mask_index_in_block.sum() == 0: break
            
            steps_taken += 1
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                model_output = self.model(x)
            
            logits = model_output.logits
            shifted_logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            logits_region = shifted_logits[0, block_start:block_end]
            mask_logits = logits_region[mask_index_in_block]
            
            probs = torch.softmax(mask_logits / (self.temperature if self.temperature > 0 else 1.0), dim=-1)
            confidence, x0 = probs.max(dim=-1)
            
            temp_filled_tokens = torch.full_like(x[0, block_start:block_end], mask_token_id)
            temp_filled_tokens[mask_index_in_block] = x0
            
            full_confidence = torch.full_like(x[0, block_start:block_end], -float('inf'), dtype=logits.dtype)
            full_confidence[mask_index_in_block] = confidence
            
            transfer_index = (full_confidence > threshold)
            
            updates_made = False
            if transfer_index.sum() > 0:
                x[0, block_start:block_end][transfer_index] = temp_filled_tokens[transfer_index]
                updates_made = True
            else:
                if full_confidence.numel() > 0 and (full_confidence > -float('inf')).any():
                    max_idx_val = torch.argmax(full_confidence).item()
                    x[0, block_start + max_idx_val] = temp_filled_tokens[max_idx_val]
                    updates_made = True
            
            if not updates_made: break
        return x, steps_taken

    def _generate_original_single_branch(self, prompt: torch.Tensor) -> Tuple[List[int], int, int, Dict]:
        prompt_length = prompt.shape[1]
        block_length = 32
        gen_length = self.max_new_tokens
        total_length = prompt_length + gen_length
        num_blocks = gen_length // block_length
        assert gen_length % block_length == 0
        
        x = torch.full((1, total_length), self.tokenizer.mask_token_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = prompt
        
        total_steps = 0
        for block_idx in range(num_blocks):
            block_start = prompt_length + block_idx * block_length
            block_end = block_start + block_length
            x, steps_this_block = self._iteratively_fill_block(x, block_start, block_end)
            total_steps += steps_this_block
            
            # 修复：在block生成完成后检查EOS
            generated_part = x[0, prompt_length:].tolist()
            if self.eot_token_id in generated_part:
                break
        
        generated_sequence_ids = x[0, prompt_length:].tolist()
        stats = {"steps_taken": total_steps}
        (
            generated_sequence_ids,
            actual_tokens_excluding_eos,
            generated_tokens_including_eos,
        ) = self._compute_generation_token_stats(generated_sequence_ids, block_length)
        stats["actual_tokens_excluding_eos"] = actual_tokens_excluding_eos
        stats["generated_tokens_including_eos"] = generated_tokens_including_eos
        return generated_sequence_ids, total_steps, generated_tokens_including_eos, stats

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        start_time = time.time()
        total_generated_tokens_including_eos = 0
        total_actual_tokens_excluding_eos = 0
        total_steps = 0
        all_masks_decoded_per_step = []  # 新增：全局记录所有案例的解码效率

        bar_desc = f"Running generate_until requests (Rank {self.rank}/{self.world_size - 1})"
        bar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)), desc=bar_desc)

        for req in requests:
            context, gen_kwargs = req.args[0], req.args[1]
            if self.add_bos_token: context = self.tokenizer.bos_token + context
            input_ids = self.tok_encode(context, add_special_tokens=False).to(self.device)
            if input_ids.shape[1] > self.max_length - self.max_new_tokens:
                input_ids = input_ids[:, -(self.max_length - self.max_new_tokens):]

            if self.use_uncertainty_logic:
                generated_answer, model_calls, tokens, stats = self._generate_enhanced_speculative(input_ids)
                parallel_steps = stats.get("parallel_steps", model_calls)
                masks_decoded_per_step = stats.get("masks_decoded_per_step", [])
                all_masks_decoded_per_step.extend(masks_decoded_per_step)
                total_steps += parallel_steps
            else:
                generated_answer, model_calls, tokens, stats = self._generate_original_single_branch(input_ids)
                total_steps += model_calls

            total_generated_tokens_including_eos += tokens
            total_actual_tokens_excluding_eos += stats.get(
                "actual_tokens_excluding_eos", len(generated_answer)
            )
            
            s = self.tok_decode(generated_answer, skip_special_tokens=True)
            if not self.escape_until:
                for term in gen_kwargs.get("until", []):
                    if len(term) > 0 and term in s: s = s.split(term)[0]
            if self.rank == 0: print(f"\nContext:\n{context}\nResponse:\n{s}")
            res.append(s)
            bar.update(1)
            
            # 核心新增：每个案例结束后打印解码效率
            if self.use_uncertainty_logic and masks_decoded_per_step:
                avg_masks_this_case = sum(masks_decoded_per_step) / len(masks_decoded_per_step)
                print(f"[Rank {self.rank}] Case {len(res)}: Avg Masks Decoded Per Step = {avg_masks_this_case:.2f}")
        bar.close()

        # 计算全局平均解码效率
        avg_masks_per_step = (sum(all_masks_decoded_per_step) / len(all_masks_decoded_per_step)) if all_masks_decoded_per_step else 0.0

        # 修复：移除all_reduce，每个rank独立保存
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            total_time = time.time() - start_time
            
            final_stats = {
                "processed_samples": len(res),
                "total_samples": len(requests),
                "total_generated_tokens_including_eos": int(total_generated_tokens_including_eos),
                "total_actual_tokens_excluding_eos": int(total_actual_tokens_excluding_eos),
                "total_parallel_steps": int(total_steps),      # 本rank的并行步数
                "avg_masks_decoded_per_step": float(avg_masks_per_step),  # 新增：全局平均解码效率
                "total_time": total_time,
                "generated_tokens_including_eos_per_second": float(total_generated_tokens_including_eos) / total_time if total_time > 0 else 0.0,
                "actual_tokens_excluding_eos_per_second": float(total_actual_tokens_excluding_eos) / total_time if total_time > 0 else 0.0,
                "generated_tokens_including_eos_per_step": float(total_generated_tokens_including_eos) / float(total_steps) if total_steps > 0 else 0.0,
                "actual_tokens_excluding_eos_per_step": float(total_actual_tokens_excluding_eos) / float(total_steps) if total_steps > 0 else 0.0,
                "timestamp": time.time(),
                "rank": self.rank,
                "world_size": self.world_size
            }
            
            # 明确标注是本地统计
            with open(os.path.join(self.save_dir, f'rank_{self.rank}_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)

        if self.show_speed and self.rank == 0 and res:
            total_time = time.time() - start_time
            mode = "Multi-Branch" if self.use_uncertainty_logic else "Single-Branch"
            avg_tok_per_step = (
                total_generated_tokens_including_eos / total_steps if total_steps > 0 else 0
            )
            
            print(f"\n==================== FINAL SUMMARY ({mode}) ====================")
            print(f"  - Total Samples: {len(res)}, Total Time: {total_time:.2f}s")
            print(f"  - Total Generated Tokens Including EOS: {total_generated_tokens_including_eos}, Total Parallel Steps: {total_steps}")
            print(f"  - Total Actual Tokens Excluding EOS: {total_actual_tokens_excluding_eos}")
            print(f"  - Avg Generated Tokens/Sample: {(total_generated_tokens_including_eos / len(res)):.2f}, Avg Parallel Steps/Sample: {(total_steps / len(res)):.2f}")
            print(f"  - Overall Throughput (Generated Tokens/Sec): {(total_generated_tokens_including_eos / total_time if total_time > 0 else 0):.2f}")
            print(f"  - Effective Tokens/Step Ratio: {avg_tok_per_step:.2f}")
            print(f"  - Avg Masks Decoded Per Step: {avg_masks_per_step:.2f}")  # 新增：打印全局平均解码效率
            print("====================================================================\n")
            
        return res
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented in this version.")
    
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError

if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
