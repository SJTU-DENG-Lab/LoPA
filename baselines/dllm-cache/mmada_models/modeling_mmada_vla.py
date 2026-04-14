from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
from PIL import Image
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from .modeling_llada import LLaDAModelLM
from .sampling import cosine_schedule, mask_by_random_topk
from transformers import PretrainedConfig

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

class MMadaVLAConfig(PretrainedConfig):
    model_type = "mmada_vla"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            "codebook_size",
            "num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size",
        ]

        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])



class MMadaModelVLA(LLaDAModelLM):
    config_class = MMadaVLAConfig
    base_model_prefix = "model"
    def __init__(self, config: MMadaVLAConfig, *args, **kwargs):
        print(f"Initializing MMadaModelVLA with config: {config}")
        super().__init__(config, *args, **kwargs)

        # # resize token embeddings
        # print(f"Resizing token embeddings to {config.new_vocab_size}")
        # self.resize_token_embeddings(config.new_vocab_size)

    @torch.no_grad()
    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids
    
    def forward_process(
            self,
            input_ids, 
            labels,
            p_mask_its2ita=None,
            attention_mask=None, # for image generation
            answer_lengths_its2ita=None,
            prompt_lengths=None,
            # action_length=50,
            action_is_pad=None,
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        if attention_mask is not None:
            attention_bias_t2i = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            attention_bias[:] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias, prompt_len=prompt_lengths).logits
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]
        bsz = logits[:].shape[0]

        if attention_mask is not None:
            # 1. subgoal image genration loss, #TODO last length
            window_size = 258  # 256 img tokens + 2 special token

            # [L] → 0..L-1  再广播成  [B, L]
            idx = torch.arange(input_ids.shape[1], device=prompt_lengths.device).repeat(bsz, 1)

            # 三步布尔比较一次完成
            subgoal_gen_indices = (idx >= prompt_lengths) & (idx < prompt_lengths + window_size)
            loss_subgoal_gen = F.cross_entropy(
                logits[subgoal_gen_indices].contiguous().view(-1, self.output_size),
                labels[subgoal_gen_indices].contiguous().view(-1), ignore_index=-100,
            )
        else:
            loss_subgoal_gen = torch.tensor(0.0, device=input_ids.device)
            subgoal_gen_indices = None

        # TODO use is_pad for loss calculation
        loss_action_gen = torch.tensor(0.0, device=input_ids.device)
        # 2. action generation loss
        # loss_action_gen = F.cross_entropy(
        #     logits[:, -(action_length+2):].contiguous().view(-1, self.output_size),
        #     labels[:, -(action_length+2):].contiguous().view(-1), ignore_index=-100,
        #     )
        # NOTE we treat action generation as mmu task temporally, so this is the same as calculation of loss_mmu
        masked_indices = input_ids == self.config.mask_token_id
        if subgoal_gen_indices is not None:
            masked_indices = torch.where(subgoal_gen_indices, False, masked_indices)
        loss_all = F.cross_entropy(
            logits[masked_indices].contiguous().view(-1, self.output_size),
            labels[masked_indices].contiguous().view(-1), ignore_index=-100, reduction='none'
        ) / p_mask_its2ita[masked_indices]
        loss_all = torch.sum(loss_all / answer_lengths_its2ita[masked_indices]) / (logits.shape[0])

        return logits, loss_subgoal_gen, loss_all, loss_action_gen


    def forward_t2i(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            max_seq_length=128,
            t2i_masks=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        # print(f"logits shape: {logits.shape}") B, 359, vocab_size

        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
            labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )
        
        return loss_t2i

    @torch.no_grad()
    def mmu_generate_block_causal_mask(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128, block_length=128,
                                       temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0,
                                       remasking='low_confidence',
                                       mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        past_key_values = self(input_ids=idx, use_cache=True).past_key_values

        batch_size = idx.shape[0]
        #x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        #x[:, :idx.shape[1]] = idx.clone()
        x = torch.full((batch_size, max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        idx = torch.full((batch_size, 0), 0, dtype=torch.long).to(self.device)
        prompt_index = (x != mask_id)

        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        # print(f"num_blocks: {num_blocks}, steps: {steps}")
        # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (
                        num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
            # print(f"num_transfer_tokens: {num_transfer_tokens}, num_transfer_tokens.shape: {num_transfer_tokens.shape}")
            for i in range(steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.0:
                    assert NotImplementedError('cfg_scale > 0. is not supported.')
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias, past_key_values=past_key_values).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x

    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128, block_length=128,
                     temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence',
                     mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)

        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        # print(f"num_blocks: {num_blocks}, steps: {steps}")
        # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (
                        num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
            # print(f"num_transfer_tokens: {num_transfer_tokens}, num_transfer_tokens.shape: {num_transfer_tokens.shape}")
            for i in range(steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x
    

AutoConfig.register("mmada_vla", MMadaVLAConfig)
AutoModelForCausalLM.register(MMadaVLAConfig, MMadaModelVLA)
AutoModel.register(MMadaVLAConfig, MMadaModelVLA)
