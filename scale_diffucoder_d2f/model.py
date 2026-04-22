
import os
import gc
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./hf_home")

import torch
import torch.nn.functional as F
import time
import random
import numpy as np
import transformers

from stop_sequencer import StopSequencer
# [FIX] Import DynamicCache, which is essential for the correct algorithm
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - optional dependency
    LLM = None
    SamplingParams = None
from peft import PeftConfig, PeftModel


EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n#"
]


logger = logging.getLogger(__name__)


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        direct_completion: bool = True,
        dtype: str = "bfloat16",  # default
        trust_remote_code: bool = False,
        dataset: str = None,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.direct_completion = direct_completion
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        self.reset_statistics()

        if direct_completion:
            if dataset and dataset.lower() == "humaneval":
                self.eos += ["\ndef", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
            elif dataset and dataset.lower() == "mbpp":
                self.eos += ['\n"""', "\nassert"]

    @abstractmethod
    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def reset_statistics(self) -> None:
        self._total_forward_passes = 0
        self._total_generated_tokens = 0
        self._total_actual_tokens_excluding_eos = 0
        self._total_generation_time = 0.0
        self._total_samples = 0

    def _record_forward_pass(self, count: int = 1) -> None:
        self._total_forward_passes += max(count, 0)

    def _record_generation(
        self,
        generated_tokens: int,
        elapsed_time: float,
        actual_tokens_excluding_eos: Optional[int] = None,
    ) -> None:
        self._total_generated_tokens += max(generated_tokens, 0)
        if actual_tokens_excluding_eos is None:
            actual_tokens_excluding_eos = generated_tokens
        self._total_actual_tokens_excluding_eos += max(actual_tokens_excluding_eos, 0)
        self._total_generation_time += max(elapsed_time, 0.0)
        self._total_samples += 1

    def get_statistics(self) -> Dict[str, float]:
        total_tokens = self._total_generated_tokens
        total_actual_tokens = self._total_actual_tokens_excluding_eos
        total_time = self._total_generation_time
        total_forward = self._total_forward_passes
        total_samples = self._total_samples

        avg_tokens = total_tokens / total_samples if total_samples else 0.0
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        forward_ratio = total_forward / total_tokens if total_tokens > 0 else 0.0

        return {
            "total_samples": total_samples,
            "total_generated_tokens": total_tokens,
            "total_actual_tokens_excluding_eos": total_actual_tokens,
            "total_forward_passes": total_forward,
            "total_generation_time": total_time,
            "avg_tokens_per_sample": avg_tokens,
            "tokens_per_second": throughput,
            "forward_per_token": forward_ratio,
        }



class DiffuCoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        # [CRITICAL FIX] Pop arguments specific to DiffuCoder BEFORE calling super().__init__
        # otherwise DecoderBase will raise TypeError for unexpected kwargs.
        
        self.block_size = kwargs.pop("block_size", 32)
        self.block_add_threshold = kwargs.pop("block_add_threshold", 0.3)
        self.skip_threshold = kwargs.pop("skip_threshold", 0.95)
        self.decoded_token_threshold = kwargs.pop("decoded_token_threshold", 0.95)
        
        requested_device = kwargs.pop("device", "auto")
        self.max_length = kwargs.pop("max_length", 2048)
        
        # Now kwargs is clean of DiffuCoder-specific args
        super().__init__(name, **kwargs)

        if ',' in name:
            self.base_model_path, self.lora_path = name.split(',', 1)
        else:
            self.base_model_path = name
            self.lora_path = None

        self.device = self._resolve_device(requested_device)
        self.target_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        if self.device.startswith("cpu") and self.target_dtype in (torch.float16, torch.bfloat16):
            self.target_dtype = torch.float32
        
        self.mask_token_id = 151666
        self.sampling_strategy = "default"
        
        self.model, self.tokenizer = self._load_dream_model()

    def _build_prompt(self, prompt: str) -> str:
        return f"""<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
<|im_start|>user
Can you complete the following Python function?
```python
{prompt}
```
<|im_end|>
<|im_start|>assistant
```python
"""

    def _resolve_device(self, device: str) -> str:
        if device is None or device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_dream_model(self):
        print(f"Loading base model from: {self.base_model_path}")
        if self.lora_path:
            print(f"Loading LoRA from: {self.lora_path}")
        
        from model_cache.dream.model_dream import DreamModel
        from model_cache.dream.configuration_dream import DreamConfig
        
        model_config = DreamConfig.from_pretrained(self.base_model_path)
        model = DreamModel.from_pretrained(
            self.base_model_path, 
            config=model_config,
            torch_dtype=self.target_dtype,
            trust_remote_code=True,
        ).eval()
        
        if self.lora_path:
            config = PeftConfig.from_pretrained(self.lora_path)
            model = PeftModel.from_pretrained(model, self.lora_path)
        
        if self.target_dtype is not None:
            model = model.to(self.target_dtype)
        model = model.to(self.device)
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model_path, trust_remote_code=True
        )
        
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer

    def _create_full_block_attention_mask(self, prompt_length, max_length, block_size, device=None, dtype=None):
        if dtype is None:
            dtype = self.target_dtype
        
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

    def _extract_attention_mask(self, full_mask, start_pos, input_length, cache_length):
        end_pos = start_pos + input_length
        total_length = cache_length + input_length
        extracted_mask = torch.full((1, 1, input_length, total_length), -torch.inf, 
                                       device=full_mask.device, dtype=full_mask.dtype)
        if cache_length > 0:
            extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
        extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
        return extracted_mask

    def _sample_tokens(self, logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
        import torch.distributions as dists
        if temperature > 0: logits = logits / temperature
        if top_p is not None and top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
            mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        
        probs = torch.softmax(logits, dim=-1)
        if temperature > 0:
            try:
                x0 = dists.Categorical(probs=probs).sample()
                initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            except:
                initial_confidence, x0 = probs.max(dim=-1)
        else:
            initial_confidence, x0 = probs.max(dim=-1)
        
        confidence = initial_confidence.clone()
        if margin_confidence:
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            top1_probs = sorted_probs[:, 0]; top2_probs = sorted_probs[:, 1]
            confidence = top1_probs - top2_probs
        if neg_entropy:
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            confidence = torch.sum(probs * log_probs, dim=-1)
        return confidence, x0, initial_confidence

    def _generation_eos_token_ids(self):
        eos_ids = []
        for token_id in (self.tokenizer.pad_token_id, self.tokenizer.eos_token_id):
            if token_id is not None and token_id not in eos_ids:
                eos_ids.append(token_id)
        return eos_ids

    def _compute_generation_token_stats(self, generated_sequence, prompt_length=0):
        generated_tokens = generated_sequence[prompt_length:]
        generated_ids = generated_tokens.tolist() if hasattr(generated_tokens, 'tolist') else list(generated_tokens)

        generated_ids_including_eos = [
            token_id for token_id in generated_ids if token_id != self.mask_token_id
        ]

        eos_token_ids = self._generation_eos_token_ids()
        eos_positions = [
            generated_ids.index(token_id)
            for token_id in eos_token_ids
            if token_id in generated_ids
        ]
        actual_generated_ids = generated_ids[:min(eos_positions)] if eos_positions else generated_ids
        actual_ids = [
            token_id for token_id in actual_generated_ids if token_id != self.mask_token_id
        ]

        return actual_ids, len(actual_ids), len(generated_ids_including_eos)

    def _count_non_eos_tokens_before_truncation(self, generated_sequence, prompt_length, pad_token_id):
        _, _, generated_tokens_including_eos = self._compute_generation_token_stats(
            generated_sequence,
            prompt_length,
        )
        return generated_tokens_including_eos

    def _update_block_completion_states(self, block_states, decoded_token_threshold):
        for block_id in sorted(block_states.keys()):
            if block_states[block_id]['total_masks'] > 0:
                decoded_tokens = block_states[block_id]['total_masks'] - block_states[block_id]['mask_count']
                decode_ratio = decoded_tokens / block_states[block_id]['total_masks']
                if decode_ratio >= decoded_token_threshold:
                    next_block_id = block_id + 1
                    if next_block_id in block_states:
                        block_states[next_block_id]['is_complete'] = True

    def _shift_logits(self, logits, last_logit=None):
        if logits.shape[1] == 0: raise Exception("logits sequence length is 0")
        shifted_logits = torch.zeros_like(logits)
        shifted_logits[:, 1:, :] = logits[:, :-1, :]
        if last_logit is not None:
            shifted_logits[:, 0, :] = last_logit.squeeze(1)
            return shifted_logits
        shifted_logits[:, 0, :] = 1.0
        return shifted_logits

    def _generate_block_single(self, prompt_tensor, max_length=None, max_new_tokens=None):
        self.model.eval()
        start_time = time.time()

        if max_length is None:
            max_length = self.max_length
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        mask_id = self.mask_token_id
        block_size = self.block_size
        block_add_threshold = self.block_add_threshold
        skip_threshold = self.skip_threshold
        decoded_token_threshold = self.decoded_token_threshold

        prompt_tensor = prompt_tensor.clone()
        prompt_length = prompt_tensor.shape[1]

        full_attention_mask = self._create_full_block_attention_mask(
            prompt_length=prompt_length,
            max_length=max_length,
            block_size=block_size,
            device=self.device,
            dtype=self.target_dtype
        )

        with torch.inference_mode():
            x_t = prompt_tensor.to(self.device)
            block_states = {
                0: {
                    'start_pos': 0,
                    'end_pos': prompt_length,
                    'mask_count': 0,
                    'total_masks': prompt_length,
                    'state': 'to_cache',
                    'is_complete': True,
                },
            }

            past_key_values = None
            last_logits = None
            current_blocks = 0
            step = 0
            eos_detected = False

            while current_blocks >= 0:
                step += 1

                if len(block_states) - 1 < (max_new_tokens // block_size) and not eos_detected:
                    last_block_id = len(block_states) - 1
                    total_masks = block_states[last_block_id]['total_masks']
                    current_progress = ((total_masks - block_states[last_block_id]['mask_count']) / total_masks) if total_masks > 0 else 1.0
                    if current_progress >= block_add_threshold:
                        new_block_id = len(block_states)
                        new_start_pos = x_t.shape[1]
                        x_t = torch.cat([x_t, torch.tensor([[mask_id] * block_size], device=self.device)], dim=1)
                        block_states[new_block_id] = {
                            'start_pos': new_start_pos,
                            'end_pos': new_start_pos + block_size,
                            'mask_count': block_size,
                            'total_masks': block_size,
                            'state': 'active',
                            'is_complete': False,
                        }
                        current_blocks += 1

                self._update_block_completion_states(block_states, decoded_token_threshold)

                mask_index = (x_t == mask_id)
                if mask_index.sum() == 0 and current_blocks == 0:
                    break

                blocks_to_cache = [bid for bid, state in block_states.items() if state['state'] == 'to_cache']
                cache_length = 0 if past_key_values is None else past_key_values.get_seq_length()

                update_kvcache = 0
                if blocks_to_cache:
                    earliest_block_id = min(blocks_to_cache)
                    earliest_pos = block_states[earliest_block_id]['start_pos']
                    latest_block_id = max(blocks_to_cache)
                    latest_pos = block_states[latest_block_id]['end_pos']
                    update_kvcache = latest_pos - earliest_pos

                process_start_pos = cache_length

                if update_kvcache > 0:
                    earliest_block_to_cache = min(blocks_to_cache)
                    input_seq = x_t[:, block_states[earliest_block_to_cache]['start_pos']:]
                    process_start_pos = block_states[earliest_block_to_cache]['start_pos']
                else:
                    active_blocks = [bid for bid, state in block_states.items() if state['state'] == 'active']
                    if active_blocks:
                        earliest_active_after_cache = float('inf')
                        for bid in active_blocks:
                            if block_states[bid]['start_pos'] >= cache_length:
                                earliest_active_after_cache = min(earliest_active_after_cache, block_states[bid]['start_pos'])

                        if earliest_active_after_cache < float('inf'):
                            input_seq = x_t[:, earliest_active_after_cache:]
                            process_start_pos = earliest_active_after_cache
                        else:
                            input_seq = x_t[:, cache_length:]
                            if cache_length >= x_t.shape[1]:
                                print(f"Cache length ({cache_length}) >= sequence length ({x_t.shape[1]}) at step {step}. Exiting generation loop.")
                                raise Exception("Cache length >= sequence length")
                    else:
                        break

                if input_seq.shape[1] == 0:
                    print(f"Warning: input_seq is empty at step {step}. Breaking generation loop.")
                    raise Exception("input_seq is empty")

                input_length = input_seq.shape[1]
                attention_mask = self._extract_attention_mask(
                    full_mask=full_attention_mask,
                    start_pos=process_start_pos,
                    input_length=input_length,
                    cache_length=cache_length
                )

                outputs = self.model(
                    input_seq,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    update_kvcache=update_kvcache,
                )
                self._record_forward_pass()

                if update_kvcache > 0:
                    cache_end_idx = update_kvcache - 1
                    last_logits = outputs.logits[:, cache_end_idx, :].unsqueeze(1)
                    past_key_values = outputs.past_key_values
                    for block_id in blocks_to_cache:
                        block_states[block_id]['state'] = 'in_cache'

                logits = self._shift_logits(outputs.logits, last_logit=last_logits)

                blocks_to_deactivate = []

                for block_id in sorted(block_states.keys()):
                    if block_states[block_id]['state'] != 'active':
                        continue

                    block_start = block_states[block_id]['start_pos']
                    block_end = block_states[block_id]['end_pos']
                    block_mask_index = mask_index.clone()
                    block_mask_index[:, :block_start] = False
                    block_mask_index[:, block_end:] = False

                    if block_mask_index.sum() == 0:
                        blocks_to_deactivate.append(block_id)
                        continue

                    logit_offset = block_start - process_start_pos
                    block_rel_positions = torch.where(block_mask_index[0, block_start:block_end])[0]

                    if block_rel_positions.size(0) > 0:
                        block_mask_logits = logits[:, logit_offset + block_rel_positions, :]
                        confidence, x0, initial_confidence = self._sample_tokens(
                            block_mask_logits.squeeze(0),
                            self.temperature
                        )

                        is_complete = block_states[block_id]['is_complete']

                        if is_complete:
                            high_conf_indices = torch.where(initial_confidence > skip_threshold)[0]
                            if len(high_conf_indices) == 0:
                                number_transfer_tokens = 1
                                if len(confidence) > 0:
                                    _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                                else:
                                    transfer_index = torch.tensor([], device=self.device, dtype=torch.long)
                            else:
                                transfer_index = torch.tensor([], device=self.device, dtype=torch.long)

                            all_indices = torch.unique(torch.cat([transfer_index, high_conf_indices]))
                        else:
                            all_indices = torch.where(initial_confidence > skip_threshold)[0]

                        if len(all_indices) > 0:
                            x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_id
                            x0_[all_indices] = x0[all_indices].clone()

                            for idx in all_indices:
                                abs_pos = block_start + block_rel_positions[idx]
                                x_t[0, abs_pos] = x0_[idx]

                            block_states[block_id]['mask_count'] -= len(all_indices)

                            if self.tokenizer.pad_token_id is not None:
                                for idx in all_indices:
                                    if x0[idx].item() == self.tokenizer.pad_token_id:
                                        eos_detected = True
                                        break

                    mask_index = (x_t == mask_id)
                    block_mask_index = mask_index.clone()
                    block_mask_index[:, :block_start] = False
                    block_mask_index[:, block_end:] = False
                    if block_mask_index.sum() == 0:
                        blocks_to_deactivate.append(block_id)

                for block_id in blocks_to_deactivate:
                    if block_states[block_id]['state'] == 'active':
                        can_deactivate = True
                        for prev_block_id in range(block_id):
                            if prev_block_id in block_states and block_states[prev_block_id]['state'] == 'active':
                                can_deactivate = False
                                break

                        if can_deactivate:
                            block_states[block_id]['state'] = 'to_cache'
                            current_blocks -= 1

                if step > 10000:
                    print(f"WARNING: Hit safety check at step {step}. Exiting generation loop.")
                    break

        generated_sequence_ids = x_t[0, prompt_length:].tolist()
        _, actual_tokens_excluding_eos, generated_tokens_including_eos = self._compute_generation_token_stats(
            generated_sequence_ids,
            0,
        )
        self._record_generation(
            generated_tokens_including_eos,
            time.time() - start_time,
            actual_tokens_excluding_eos,
        )

        decoded_text = self.tokenizer.decode(generated_sequence_ids)
        pad_token = self.tokenizer.pad_token
        response = decoded_text.split(pad_token)[0] if pad_token else decoded_text
        return response

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample: self._set_seed(1234)
        input_text = self._build_prompt(prompt)
        prompt_ids = self.tokenizer.encode(input_text)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)

        max_prompt_length = max(0, self.max_length - self.max_new_tokens)
        if prompt_tensor.shape[1] > max_prompt_length:
            prompt_tensor = prompt_tensor[:, -max_prompt_length:]

        results = []
        for i in range(min(num_samples, self.batch_size)):
            try:
                response = self._generate_block_single(
                    prompt_tensor=prompt_tensor,
                    max_length=self.max_length,
                    max_new_tokens=self.max_new_tokens
                )
                print(f"Sample {i+1}:\n{response}\n{'-'*40}")
                results.append(response)
            except Exception as e:
                print(f"Error generating sample {i}: {str(e)}"); results.append("")
        return results


class DiffuCoderBasic(DiffuCoder):
    def __init__(self, name: str, **kwargs) -> None:
        # Basic variant should only load the base model (no LoRA).
        requested_device = kwargs.pop("device", "auto")
        self.max_length = kwargs.pop("max_length", 2048)
        token_per_step = kwargs.pop("token_per_step", 1)
        
        # [FIX] Also pop DiffuCoder-specific args here to prevent them reaching DecoderBase
        # even if DiffuCoderBasic might not use all of them, generating via make_model passes them.
        kwargs.pop("block_size", None)
        kwargs.pop("block_add_threshold", None)
        kwargs.pop("skip_threshold", None)
        kwargs.pop("decoded_token_threshold", None)
        
        # call DecoderBase init to set common fields
        DecoderBase.__init__(self, name, **kwargs)

        # treat name as base model path only
        self.base_model_path = name
        self.lora_path = None

        self.device = self._resolve_device(requested_device)
        self.target_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        if self.device.startswith("cpu") and self.target_dtype in (torch.float16, torch.bfloat16):
            self.target_dtype = torch.float32

        # reuse diffusion-related defaults from DiffuCoder
        self.mask_token_id = 151666
        self.sampling_strategy = "default"

        # basic generation hyperparams
        self.basic_token_per_step = max(1, int(token_per_step))
        self.basic_alg = kwargs.pop("basic_alg", "entropy")
        self.basic_alg_temp = kwargs.pop("basic_alg_temp", 0.0)
        self.basic_top_p = kwargs.pop("basic_top_p", 0.95)
        self.basic_temp = kwargs.pop("basic_temp", 0.4)
        self.basic_return_history = kwargs.pop("basic_return_history", True)

        # load base model only
        self.model, self.tokenizer = self._load_dream_model()

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        # Match infer_diffucoder: always call diffusion_generate once; loop for multiple samples.
        if do_sample:
            self._set_seed(1234)

        formatted_prompt = self._build_prompt(prompt)
        results: List[str] = []

        # Compute steps as max_new_tokens // token_per_step (at least 1).
        steps = max(1, int(self.max_new_tokens // self.basic_token_per_step))

        for _ in range(num_samples):
            tokenized = self.tokenizer(formatted_prompt, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device=self.device)
            attention_mask = tokenized["attention_mask"].to(device=self.device)

            self.model.eval()
            start_time = time.time()
            outputs = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                output_history=self.basic_return_history,
                return_dict_in_generate=True,
                steps=steps,
                temperature=self.basic_temp,
                top_p=self.basic_top_p,
                alg=self.basic_alg,
                alg_temp=self.basic_alg_temp,
            )
            elapsed = time.time() - start_time
            # Record one forward pass (conservatively count steps).
            self._record_forward_pass(steps)

            seq = outputs.sequences[0].detach().cpu()
            prompt_len = input_ids[0].shape[-1]
            gen_ids = seq[prompt_len:].tolist()

            _, actual_tokens_excluding_eos, _ = self._compute_generation_token_stats(
                gen_ids,
                0,
            )
            generated_tokens_including_eos = self.max_new_tokens
            self._record_generation(
                generated_tokens_including_eos,
                elapsed,
                actual_tokens_excluding_eos,
            )

            decoded_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            pad_token = self.tokenizer.pad_token
            if pad_token:
                decoded_text = decoded_text.split(pad_token)[0]
            decoded_text = decoded_text.split("<|dlm_pad|>")[0]
            print(f"Sample:\n{decoded_text}\n{'-'*40}")
            results.append(decoded_text)

        return results

    def _load_dream_model(self):
        # Use the remote implementation to stay consistent with infer_diffucoder and avoid local mask mismatches.
        from transformers import AutoModel
        print(f"Loading base model (basic) from: {self.base_model_path}")
        model = AutoModel.from_pretrained(
            self.base_model_path,
            torch_dtype=self.target_dtype,
            trust_remote_code=True,
        ).eval()

        # Apply dtype/device placement last.
        if self.target_dtype is not None:
            model = model.to(self.target_dtype)
        model = model.to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        print("Basic model and tokenizer loaded successfully (remote code)!")
        return model, tokenizer

# ==============================================================================
# Correct implementation for parallel decoding, 1:1 copy from the trusted script
# ==============================================================================
class Branch:
    def __init__(self, branch_id: int, x_t: torch.Tensor, block_states: Dict,
                 confidence: float = 1.0, past_key_values: Optional[Tuple] = None,
                 prompt_length: int = 0, is_base: bool = False,
                 creation_token_confidence: float = 1.0):
        self.branch_id = branch_id
        self.x_t = x_t.clone()
        self.block_states = {k: v.copy() for k, v in block_states.items()}
        self.confidence = confidence
        self.step_confidences: List[float] = []
        self.is_active = True
        self.eos_detected = False
        self.past_key_values = past_key_values
        self.prompt_length = prompt_length
        self.steps_completed = -1
        self.is_base = is_base
        self.creation_token_confidence = creation_token_confidence

    @property
    def generated_token_count(self) -> int:
        return self.x_t.shape[1] - self.prompt_length

    def copy(self):
        new_branch = Branch(
            self.branch_id, self.x_t.clone(), {k: v.copy() for k, v in self.block_states.items()},
            self.confidence, None, self.prompt_length, self.is_base,
            creation_token_confidence=self.creation_token_confidence,
        )
        new_branch.step_confidences = self.step_confidences.copy()
        new_branch.is_active = self.is_active
        new_branch.eos_detected = self.eos_detected
        new_branch.steps_completed = self.steps_completed
        return new_branch

def find_shared_prefix_end(branches: List[Branch]) -> int:
    if not branches: return 0
    min_completed_blocks = float('inf')
    for branch in branches:
        completed_blocks = 0
        for block_id in sorted(branch.block_states.keys()):
            if (branch.block_states[block_id]['state'] in ['in_cache', 'to_cache'] and
                branch.block_states[block_id]['mask_count'] == 0):
                completed_blocks += 1
            else: break
        min_completed_blocks = min(min_completed_blocks, completed_blocks)
    if min_completed_blocks == float('inf') or min_completed_blocks == 0: return branches[0].prompt_length
    first_branch = branches[0]; completed_count = 0
    for block_id in sorted(first_branch.block_states.keys()):
        if (first_branch.block_states[block_id]['state'] in ['in_cache', 'to_cache'] and
            first_branch.block_states[block_id]['mask_count'] == 0):
            completed_count += 1
            if completed_count == min_completed_blocks: return first_branch.block_states[block_id]['end_pos']
        else: break
    return branches[0].prompt_length

def spec_top_p_logits(logits: torch.Tensor, top_p: Optional[float] = None) -> torch.Tensor:
    if top_p is None or top_p >= 1: return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

def spec_top_k_logits(logits: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
    if top_k is None: return logits
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)

def spec_sample_tokens(
    logits: torch.Tensor, temperature: float = 0.0, top_p: Optional[float] = None,
    top_k: Optional[int] = None, sampling_strategy: str = "default",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch.distributions as dists
    if temperature > 0: logits = logits / temperature
    if top_p is not None and top_p < 1: logits = spec_top_p_logits(logits, top_p)
    if top_k is not None: logits = spec_top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception: initial_confidence, x0 = probs.max(dim=-1)
    else: initial_confidence, x0 = probs.max(dim=-1)
    if sampling_strategy == "margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[..., 0] - sorted_probs[..., 1]
    elif sampling_strategy == "neg_entropy":
        log_probs = torch.log(probs + 1e-10)
        confidence = torch.sum(probs * log_probs, dim=-1)
    else: confidence = initial_confidence.clone()
    return confidence, x0, initial_confidence

def evaluate_branch_confidence(
    logits: torch.Tensor, branch: Branch, branch_start_in_input: int, branch_length: int, shared_prefix_end: int,
    mask_token_id: int, sampling_strategy: str = "default", branch_topp: float = 0.5, temperature: float = 0.0,
    top_p: Optional[float] = None, top_k: Optional[int] = None, selection_conf_alpha: float = 0.5,
) -> float:
    if branch_length == 0: future_conf = 1.0
    else:
        branch_logits = logits[0, branch_start_in_input:branch_start_in_input + branch_length, :]
        mask_positions = (branch.x_t[0, shared_prefix_end:] == mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0: future_conf = 1.0
        else:
            mask_logits = branch_logits[mask_positions, :]
            confidences, _, _ = spec_sample_tokens(mask_logits, temperature, top_p, top_k, sampling_strategy)
            num_positions = len(confidences)
            if num_positions == 1: future_conf = confidences.item()
            else:
                bottom_cnt = max(1, int(num_positions * branch_topp))
                sorted_confidences, _ = torch.sort(confidences, descending=False)
                future_conf = sorted_confidences[:bottom_cnt].mean().item()
    creation_conf = float(getattr(branch, "creation_token_confidence", 1.0))
    alpha = float(max(0.0, min(1.0, selection_conf_alpha)))
    return alpha * creation_conf + (1.0 - alpha) * future_conf

class DiffuCoderParallel(DiffuCoder):
    def __init__(self, name: str, **kwargs) -> None:
        self.max_length = kwargs.pop("max_length", 4096)
        self.use_uncertainty_logic = kwargs.pop("use_uncertainty_logic", True)
        self.branching_factor = kwargs.pop("branching_factor", 2)
        self.branch_verification_mode = kwargs.pop("branch_verification_mode", True)
        self.base_branch_competition = kwargs.pop("base_branch_competition", True)
        self.verification_force_base_winner = kwargs.pop("verification_force_base_winner", False)
        self.branch_topp = kwargs.pop("branch_topp", 1)
        self.selection_conf_alpha = kwargs.pop("selection_conf_alpha", 0)
        self.top_p = kwargs.pop("top_p", None)
        self.top_k = kwargs.pop("top_k", None)
        self.use_full_attention = kwargs.pop("use_full_attention", True)
        self.forcing_sample = kwargs.pop("forcing_sample", True)
        
        # Super init will call DiffuCoder init, which will handle the rest
        # including extracting block_size etc.
        super().__init__(name, **kwargs)
        self.shared_past_key_values = None
        self.shared_last_logits = None

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.tokenizer.pad_token_id

    def _create_full_attention_mask(self, max_length, device=None, dtype=None):
        if dtype is None: dtype = self.target_dtype
        return torch.zeros((1, 1, max_length, max_length), device=device, dtype=dtype)
        
    def _extract_attention_mask_spec(self, full_mask, start_pos, input_length, cache_length):
        total_length = cache_length + input_length
        if self.use_full_attention:
            return torch.zeros((1, 1, input_length, total_length), device=full_mask.device, dtype=full_mask.dtype)
        else:
            end_pos = start_pos + input_length
            extracted_mask = torch.full((1, 1, input_length, total_length), -torch.inf, device=full_mask.device, dtype=full_mask.dtype)
            if cache_length > 0:
                extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
            extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
            return extracted_mask

    def _generate_original_single_branch(self, prompt: torch.Tensor) -> Tuple[List[int], Dict]:
        generated_ids_str, _ = super()._generate_block_single(prompt)
        # Compatibility wrapper
        return self.tokenizer.encode(generated_ids_str), {}

    def _generate_additional_branches(
        self, newly_spawned_branches: List[Branch], best_branch_template: Branch, best_logits: Optional[torch.Tensor],
        active_blocks_ids: List[int], shared_prefix_end: int, chosen_positions: set, run_stats: Dict, branches: List[Branch]
    ) -> None:
        if best_logits is None:
            if not newly_spawned_branches:
                best_branch_template.is_base = True; best_branch_template.creation_token_confidence = 1.0; best_branch_template.confidence = 1.0
                newly_spawned_branches.append(best_branch_template)
            return

        branch_length = best_branch_template.x_t.shape[1] - shared_prefix_end if best_branch_template.x_t.shape[1] > shared_prefix_end else 0
        if branch_length > 0:
            branch_logits = best_logits[0, -branch_length:, :] if best_logits.shape[1] >= branch_length else best_logits[0, :, :]
            candidate_positions = []
            for block_id in active_blocks_ids:
                bs, be = best_branch_template.block_states[block_id]["start_pos"], best_branch_template.block_states[block_id]["end_pos"]
                mask_idx_local = (best_branch_template.x_t[0, bs:be] == self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_idx_local) > 0:
                    for abs_pos_tensor in mask_idx_local + bs:
                        abs_pos = abs_pos_tensor.item()
                        if abs_pos not in chosen_positions and abs_pos >= shared_prefix_end:
                            rel_pos_local = abs_pos - shared_prefix_end
                            if 0 <= rel_pos_local < branch_length: candidate_positions.append((abs_pos, rel_pos_local))
            top_candidates = []
            if candidate_positions:
                confidences = []
                for abs_pos, rel_pos_local in candidate_positions:
                    if rel_pos_local < branch_logits.shape[0]:
                        pos_logits = branch_logits[rel_pos_local, :].unsqueeze(0)
                        conf, _, _ = spec_sample_tokens(pos_logits, self.temperature, self.top_p, self.top_k, self.sampling_strategy)
                        confidences.append((conf.item(), abs_pos, rel_pos_local))
                confidences.sort(key=lambda x: x[0], reverse=True)
                effective_branching_factor = self.branching_factor - 1 if self.base_branch_competition and len(newly_spawned_branches) > 0 else self.branching_factor
                num_to_select = min(effective_branching_factor, len(confidences))
                for i in range(num_to_select): top_candidates.append((confidences[i][1], confidences[i][2], confidences[i][0]))
        else: top_candidates = []

        if not top_candidates and not newly_spawned_branches:
            best_branch_template.is_base = True; best_branch_template.creation_token_confidence = 1.0; best_branch_template.confidence = 1.0
            newly_spawned_branches.append(best_branch_template)
        else:
            for abs_pos, rel_pos, conf_score in top_candidates:
                new_branch = best_branch_template.copy()
                new_branch.branch_id = len(branches) + len(newly_spawned_branches)
                new_branch.is_base = False
                if rel_pos < branch_logits.shape[0]:
                    pos_logits = branch_logits[rel_pos, :].unsqueeze(0)
                    conf_val, new_token, _ = spec_sample_tokens(pos_logits, self.temperature, self.top_p, self.top_k, self.sampling_strategy)
                    token = new_token.item()
                    new_branch.x_t[0, abs_pos] = token
                    new_branch.creation_token_confidence = float(conf_val.item())
                    new_branch.confidence = float(conf_val.item())
                    if token == self.pad_token_id: new_branch.eos_detected = True
                    newly_spawned_branches.append(new_branch)
                    run_stats["total_filled_new"] += 1; run_stats["branches_created"] += 1
            if not newly_spawned_branches:
                best_branch_template.is_base = True; best_branch_template.creation_token_confidence = 1.0; best_branch_template.confidence = 1.0
                newly_spawned_branches.append(best_branch_template)

    def _generate_enhanced_speculative(self, prompt: torch.Tensor) -> Tuple[List[int], int, int, Dict]:
        self.model.eval(); start_time = time.time()
        prompt_length = prompt.shape[1]
        
        if not self.use_uncertainty_logic:
            generated_ids, stats = self._generate_original_single_branch(prompt)
            _, actual_tokens_excluding_eos, generated_tokens_including_eos = self._compute_generation_token_stats(
                generated_ids,
                0,
            )
            self._record_generation(
                generated_tokens_including_eos,
                time.time() - start_time,
                actual_tokens_excluding_eos,
            )
            return generated_ids, stats.get("steps_taken", 0), len(generated_ids), stats

        self.shared_past_key_values = None; self.shared_last_logits = None
        
        full_attention_mask = self._create_full_attention_mask(self.max_length, self.device, self.target_dtype) if self.use_full_attention else self._create_full_block_attention_mask(prompt_length, self.max_length, self.block_size, self.device, self.target_dtype)
        
        initial_block_states = {0: {'start_pos': 0, 'end_pos': prompt_length, 'mask_count': 0, 'total_masks': prompt_length, 'state': 'to_cache', 'is_complete': True}}
        branches = [Branch(0, prompt.clone().to(self.device), initial_block_states, prompt_length=prompt_length, is_base=True, creation_token_confidence=1.0)]
        run_stats = {"parallel_steps": 0, "total_filled_original": 0, "total_filled_new": 0, "original_fallback_triggers": 0, "branches_created": 1, "branches_pruned": 0, "max_active_branches": 0}

        with torch.inference_mode():
            while any(b.is_active for b in branches):
                run_stats["parallel_steps"] += 1; parallel_step_count = run_stats["parallel_steps"]
                active_branches = [b for b in branches if b.is_active]
                run_stats["max_active_branches"] = max(run_stats["max_active_branches"], len(active_branches))

                for branch in active_branches:
                    if len(branch.block_states) - 1 < (self.max_new_tokens // self.block_size) and not branch.eos_detected:
                        last_id = len(branch.block_states) - 1; total = branch.block_states[last_id]['total_masks']
                        progress = ((total - branch.block_states[last_id]['mask_count']) / total) if total > 0 else 1.0
                        if progress >= self.block_add_threshold:
                            new_id = len(branch.block_states); start = branch.x_t.shape[1]
                            branch.x_t = torch.cat([branch.x_t, torch.tensor([[self.mask_token_id] * self.block_size], device=self.device)], dim=1)
                            branch.block_states[new_id] = {'start_pos': start, 'end_pos': start + self.block_size, 'mask_count': self.block_size, 'total_masks': self.block_size, 'state': 'active', 'is_complete': False}
                    self._update_block_completion_states(branch.block_states, self.decoded_token_threshold)
                    if (branch.x_t == self.mask_token_id).sum() == 0 and all(s['state'] != 'active' for s in branch.block_states.values()):
                        branch.is_active = False
                        if branch.steps_completed == -1: branch.steps_completed = parallel_step_count
                
                remaining_active = [b for b in active_branches if b.is_active]
                if not remaining_active: continue

                shared_prefix_end = find_shared_prefix_end(remaining_active)
                cache_len = 0 if self.shared_past_key_values is None else self.shared_past_key_values.get_seq_length()
                
                blocks_to_cache = []; update_kvcache_len = 0
                layout_branch = next((b for b in remaining_active if getattr(b, 'is_base', False)), remaining_active[0])
                for bid, state in layout_branch.block_states.items():
                    if state['state'] == 'to_cache' and state['end_pos'] <= shared_prefix_end: blocks_to_cache.append(bid)
                if blocks_to_cache:
                    update_kvcache_len = max(layout_branch.block_states[b]['end_pos'] for b in blocks_to_cache) - min(layout_branch.block_states[b]['start_pos'] for b in blocks_to_cache)
                
                input_start = min(layout_branch.block_states[b]['start_pos'] for b in blocks_to_cache) if update_kvcache_len > 0 else shared_prefix_end
                if self.base_branch_competition: remaining_active.sort(key=lambda b: (getattr(b, 'is_base', False), getattr(b, 'branch_id', 0)))

                branch_results = []; forward_called = False
                
                shared_cache_snapshot = None
                if self.shared_past_key_values is not None:
                    try:
                        shared_cache_snapshot = self.shared_past_key_values.to_legacy_cache()
                    except AttributeError:
                        shared_cache_snapshot = None

                for branch in remaining_active:
                    needs_prefix = input_start < shared_prefix_end; needs_suffix = branch.x_t.shape[1] > shared_prefix_end
                    if not needs_prefix and not needs_suffix:
                        branch.is_active = False; branch.steps_completed = parallel_step_count if branch.steps_completed == -1 else branch.steps_completed
                        branch_results.append((branch, None, None, None)); continue
                    
                    if needs_prefix and needs_suffix: branch_input = torch.cat([branch.x_t[0, input_start:shared_prefix_end], branch.x_t[0, shared_prefix_end:]]).unsqueeze(0)
                    elif needs_prefix: branch_input = branch.x_t[0, input_start:shared_prefix_end].unsqueeze(0)
                    else: branch_input = branch.x_t[0, shared_prefix_end:].unsqueeze(0)

                    branch_cache = None if shared_cache_snapshot is None else DynamicCache.from_legacy_cache(shared_cache_snapshot)
                    if branch_cache is None: branch_cache = self.shared_past_key_values
                    
                    attention_mask = self._extract_attention_mask_spec(full_attention_mask, input_start, branch_input.shape[1], cache_len)
                    outputs = self.model(branch_input, attention_mask=attention_mask, past_key_values=branch_cache, use_cache=True, update_kvcache=update_kvcache_len)
                    forward_called = True
                    
                    if outputs.past_key_values is None:
                        logger.error(f"Branch {branch.branch_id} failed."); branch.is_active=False; branch.steps_completed = parallel_step_count if branch.steps_completed == -1 else branch.steps_completed
                        branch_results.append((branch, None, None, None)); continue

                    raw_logits, new_pkv = outputs.logits, outputs.past_key_values
                    last_logit = raw_logits[:, update_kvcache_len - 1, :].unsqueeze(1) if update_kvcache_len > 0 else self.shared_last_logits
                    shifted_logits = self._shift_logits(raw_logits, last_logit=last_logit)
                    branch_results.append((branch, shifted_logits, new_pkv, last_logit))

                branch_confidences = []
                for idx, (branch, logits, _, _) in enumerate(branch_results):
                    if logits is not None:
                        branch_len = branch.x_t.shape[1] - shared_prefix_end if branch.x_t.shape[1] > shared_prefix_end else 0
                        if branch_len > 0:
                            branch_start_in_input = max(0, shared_prefix_end - input_start)
                            confidence = evaluate_branch_confidence(logits, branch, branch_start_in_input, branch_len, shared_prefix_end, self.mask_token_id, self.sampling_strategy, self.branch_topp, self.temperature, self.top_p, self.top_k, self.selection_conf_alpha)
                            branch_confidences.append((confidence, idx, branch))
                        else: branch_confidences.append((1.0, idx, branch))
                    else: branch_confidences.append((float(getattr(branch, "creation_token_confidence", 0.0)), idx, branch))
                
                branch_confidences.sort(key=lambda x: x[0], reverse=True)
                if self.verification_force_base_winner and (base_res := next(((b, l, p, last) for b, l, p, last in branch_results if getattr(b, 'is_base', False)), None)):
                    best_branch, best_logits, best_pkv, best_last_logit = base_res; best_conf = 1.0
                else:
                    best_conf, best_idx, best_branch = branch_confidences[0]; _, best_logits, best_pkv, best_last_logit = branch_results[best_idx]
                
                if update_kvcache_len > 0 and best_pkv is not None:
                    self.shared_past_key_values = best_pkv
                    if best_last_logit is not None: self.shared_last_logits = best_last_logit
                elif self.shared_last_logits is None and best_last_logit is not None: self.shared_last_logits = best_last_logit

                tokens_to_update = {}; active_block_ids = []
                if best_logits is not None:
                    branch_len = best_branch.x_t.shape[1] - shared_prefix_end if best_branch.x_t.shape[1] > shared_prefix_end else 0
                    if branch_len > 0:
                        branch_start_in_input = max(0, shared_prefix_end - input_start)
                        active_block_ids = [bid for bid, st in best_branch.block_states.items() if st['state'] == 'active']
                        for bid in active_block_ids:
                            bs, be = best_branch.block_states[bid]['start_pos'], best_branch.block_states[bid]['end_pos']
                            mask_idx = (best_branch.x_t[0, bs:be] == self.mask_token_id).nonzero(as_tuple=True)[0]
                            if len(mask_idx) == 0: continue
                            mask_rel = mask_idx + bs - shared_prefix_end
                            valid_mask = mask_rel[(mask_rel >= 0) & (mask_rel < branch_len)]
                            if len(valid_mask) == 0: continue
                            logits_indices = branch_start_in_input + valid_mask
                            block_logits = best_logits[0, logits_indices, :]
                            confidence, x0, initial_confidence = spec_sample_tokens(block_logits, self.temperature, self.top_p, self.top_k, self.sampling_strategy)
                            high_conf_idx = (initial_confidence > self.skip_threshold).nonzero(as_tuple=True)[0]
                            fill_indices = []
                            if best_branch.block_states[bid]['is_complete'] and self.forcing_sample:
                                if len(high_conf_idx) > 0: fill_indices = high_conf_idx
                                else:
                                    run_stats["original_fallback_triggers"] += 1
                                    if len(confidence) > 0: _, most_conf = torch.topk(confidence, 1); fill_indices = most_conf
                            else:
                                fill_indices = high_conf_idx
                            if len(fill_indices) > 0:
                                for idx in fill_indices:
                                    pos = (valid_mask[idx.item()] + shared_prefix_end).item(); token = x0[idx.item()].item()
                                    tokens_to_update[pos] = token
                                    if token == self.pad_token_id: best_branch.eos_detected = True
                        run_stats["total_filled_original"] += len(tokens_to_update)
                        if tokens_to_update:
                            for pos, tok in tokens_to_update.items(): best_branch.x_t[0, pos] = tok
                best_branch.confidence = best_conf
                
                newly_spawned = []; base_completed = []
                for bid, st in best_branch.block_states.items():
                    if st['state'] == 'active':
                        st['mask_count'] = (best_branch.x_t[0, st['start_pos']:st['end_pos']] == self.mask_token_id).sum().item()
                        if st['mask_count'] == 0: base_completed.append(bid)
                
                if self.branch_verification_mode:
                    if self.base_branch_competition:
                        base = best_branch.copy(); base.branch_id = len(branches); base.is_base = True; base.confidence = 1.0; base.creation_token_confidence = 1.0
                        if self.branching_factor > 1:
                            self._generate_additional_branches(newly_spawned, best_branch, best_logits, active_block_ids, shared_prefix_end, set(tokens_to_update.keys()), run_stats, branches)
                        newly_spawned.append(base)
                    else: best_branch.is_base = True; best_branch.confidence = 1.0; best_branch.creation_token_confidence = 1.0; newly_spawned.append(best_branch)
                elif self.branching_factor > 1:
                    self._generate_additional_branches(newly_spawned, best_branch, best_logits, active_block_ids, shared_prefix_end, set(tokens_to_update.keys()), run_stats, branches)
                else: best_branch.is_base = True; best_branch.confidence = 1.0; best_branch.creation_token_confidence = 1.0; newly_spawned.append(best_branch)

                for nb in newly_spawned:
                    if update_kvcache_len > 0:
                        for bid in blocks_to_cache:
                            if bid in nb.block_states: nb.block_states[bid]['state'] = 'in_cache'
                    for bid in [b for b, s in nb.block_states.items() if s['state'] == 'active']:
                        nb.block_states[bid]['mask_count'] = (nb.x_t[0, nb.block_states[bid]['start_pos']:nb.block_states[bid]['end_pos']] == self.mask_token_id).sum().item()
                        if bid in base_completed and nb.block_states[bid]['mask_count'] == 0:
                            if all(p not in nb.block_states or nb.block_states[p]['state'] != 'active' for p in range(bid)):
                                nb.block_states[bid]['state'] = 'to_cache'

                branches = [b for b in branches if not b.is_active] + newly_spawned
                run_stats["branches_pruned"] += len(active_branches) - 1
                if forward_called: self._record_forward_pass()
                if len(active_branches) > 1: gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if not branches:
            logger.warning("No branches available, returning original prompt"); return [], 0, 0, run_stats
        
        completed = [b for b in branches if not b.is_active and b.steps_completed != -1]
        best_branch = min(completed, key=lambda x: x.steps_completed) if completed else max(branches, key=lambda x: x.generated_token_count)
        
        generated_ids = best_branch.x_t[0, prompt_length:].tolist()
        best_steps = best_branch.steps_completed if best_branch.steps_completed != -1 else run_stats["parallel_steps"]
        best_tokens = best_branch.generated_token_count
        _, actual_tokens_excluding_eos, generated_tokens_including_eos = self._compute_generation_token_stats(
            generated_ids,
            0,
        )
        self._record_generation(
            generated_tokens_including_eos,
            time.time() - start_time,
            actual_tokens_excluding_eos,
        )
        return generated_ids, best_steps, best_tokens, run_stats

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample: self._set_seed(1234)
        input_text = self._build_prompt(prompt)
        prompt_ids = self.tokenizer.encode(input_text)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        results = []
        for i in range(min(num_samples, self.batch_size)):
            try:
                generated_ids, _, _, _ = self._generate_enhanced_speculative(prompt_tensor)
                decoded_text = self.tokenizer.decode(generated_ids)
                pad_token = self.tokenizer.pad_token
                response = decoded_text.split(pad_token)[0] if pad_token else decoded_text
                print(f"Sample {i + 1}:\n{response}\n{'-' * 40}")
                results.append(response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error generating sample {i}: {str(e)}"); results.append("")
        return results

class VLlmDecoder(DecoderBase):
    def __init__(self, name: str, tensor_parallel_size = 1, **kwargs) -> None:
        super().__init__(name, **kwargs)
        kwargs = {"tensor_parallel_size": tensor_parallel_size, "dtype": self.dtype, "trust_remote_code": self.trust_remote_code, "enforce_eager": True, "gpu_memory_utilization": 0.98}
        print(kwargs)
        self.llm = LLM(model=name, max_model_len=1536, **kwargs)

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample: assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)
        vllm_outputs = self.llm.generate([prompt] * batch_size, SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens, top_p=0.95 if do_sample else 1.0, stop=self.eos), use_tqdm=False)
        return [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]

class VLlmAWQDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        kwargs = {"tensor_parallel_size": int(os.getenv("VLLM_N_GPUS", "1")), "dtype": torch.float16, "trust_remote_code": self.trust_remote_code, "quantization": "AWQ"}
        self.llm = LLM(model=name, max_model_len=2048, **kwargs)

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample: assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)
        vllm_outputs = self.llm.generate([prompt] * batch_size, SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens, top_p=0.95 if do_sample else 1.0, stop=self.eos), use_tqdm=False)
        return [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]

class AWQChatML(VLlmAWQDecoder):
    def __init__(self, name: str, tensor_parallel_size, **kwargs) -> None:
        kwargs["direct_completion"] = False
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample: assert self.temperature > 0, "Temperature must be greater than 0!"
        input = f"""<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
<|im_start|>user
Can you complete the following Python function?
```python
{prompt}
```
<|im_end|>
<|im_start|>assistant
```python
"""
        return VLlmDecoder.codegen(self, input, do_sample, num_samples)

class ChatML(VLlmDecoder):
    def __init__(self, name: str, tensor_parallel_size, **kwargs) -> None:
        kwargs["direct_completion"] = False
        super().__init__(name, tensor_parallel_size, **kwargs)
        self.eos += ["\n```"]

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200) -> List[str]:
        if do_sample: assert self.temperature > 0, "Temperature must be greater than 0!"
        input = f"""<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
<|im_start|>user
Can you complete the following Python function?
```python
{prompt}
```
<|im_end|>
<|im_start|>assistant
```python
"""
        return VLlmDecoder.codegen(self, input, do_sample, num_samples)

def make_model(
    model_type: str, model_size: str, model_path: str, batch_size: int = 1,
    temperature: float = 0.8, dataset: str = None, tensor_parallel_size: int = 1,
    device: str = "auto", max_new_tokens: int = 512, **kwargs,
):
    if model_type == "codeqwen" or model_type == "qwen2":
        if "chat" in model_size.lower():
            if "awq" in model_size.lower():
                return AWQChatML(batch_size=batch_size, name=model_path, temperature=temperature, max_new_tokens=2048, tensor_parallel_size=tensor_parallel_size)
            else:
                return ChatML(batch_size=batch_size, name=model_path, temperature=temperature, max_new_tokens=2048, tensor_parallel_size=tensor_parallel_size)
        else:
            return VLlmDecoder(batch_size=batch_size, name=model_path, temperature=temperature, dataset=dataset, tensor_parallel_size=tensor_parallel_size)
    elif model_type == "diffucoder":
        return DiffuCoder(batch_size=batch_size, name=model_path, temperature=temperature, max_new_tokens=max_new_tokens, dataset=dataset, device=device, **kwargs)
    elif model_type == "diffucoder_basic":
        # model_path should be the base model path only
        return DiffuCoderBasic(batch_size=batch_size, name=model_path, temperature=temperature, max_new_tokens=max_new_tokens, dataset=dataset, device=device, **kwargs)
    elif model_type == "diffucoder_parallel":
        return DiffuCoderParallel(
            batch_size=batch_size,
            name=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            dataset=dataset,
            device=device,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid model name: {model_type}@{model_size}")
