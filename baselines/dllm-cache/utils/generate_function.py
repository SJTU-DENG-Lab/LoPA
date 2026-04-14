import torch
from dllm_cache.cache import dLLMCache
import torch.nn.functional as F
import numpy as np


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits.exp()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)


def generate(
    input_ids,
    attention_mask,
    model,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x[:, :prompt_length] = input_ids

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        feature_cache = dLLMCache()
        feature_cache.reset_cache(prompt_length)
        for num_block in range(num_blocks):
            start_idx = prompt_length + num_block * block_length
            end_idx = prompt_length + (num_block + 1) * block_length

            block_x = x[:, start_idx:end_idx]
            block_mask_index = block_x == mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    if hasattr(feature_cache, "cfg_interval_steps"):
                        feature_cache.update_step(layer_id=33)
                        if feature_cache.refresh_cfg(layer_id=33):
                            cfg_x = x.clone()
                            cfg_x[prompt_index] = mask_id
                            logits = model(x, attention_mask=attention_mask).logits[
                                :, prompt_length:
                            ]
                            feature_cache.cache_type = "cfg"
                            cfg_logits = model(
                                cfg_x, attention_mask=attention_mask
                            ).logits[:, prompt_length:]
                            cfg_residual = logits - cfg_logits
                            feature_cache.set_cache(
                                layer_id=33,
                                feature_name="cfg_residual",
                                features=cfg_residual,
                                cache_type="gen",
                            )
                            feature_cache.cache_type = "no_cfg"
                        else:
                            feature_cache.cache_type = "cfg"
                            cfg_residual = feature_cache.get_cache(
                                layer_id=33,
                                feature_name="cfg_residual",
                                cache_type="gen",
                            )
                            feature_cache.cache_type = "no_cfg"
                            logits = model(x, attention_mask=attention_mask).logits[
                                :, prompt_length:
                            ]
                    else:
                        cfg_x = x.clone()
                        cfg_x[prompt_index] = mask_id
                        logits = model(x, attention_mask=attention_mask).logits[
                            :, prompt_length:
                        ]
                        cfg_logits = model(cfg_x, attention_mask=attention_mask).logits[
                            :, prompt_length:
                        ]
                        cfg_residual = logits - cfg_logits
                    logits = (logits - cfg_residual) + (cfg_scale + 1) * cfg_residual
                else:
                    logits = model(x, attention_mask=attention_mask).logits[
                        :, prompt_length:
                    ]
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, (num_block + 1) * block_length :] = -np.inf

                x0 = torch.where(
                    mask_index[:, prompt_length:], x0, x[:, prompt_length:]
                )
                confidence = torch.where(mask_index[:, prompt_length:], x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    ).indices
                    transfer_index[j, select_index] = True
                x[:, prompt_length:][transfer_index] = x0[transfer_index]
        return x[:, prompt_length:]
