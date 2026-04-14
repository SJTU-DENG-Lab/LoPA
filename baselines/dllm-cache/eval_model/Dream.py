import logging
import os
import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union,Type,TypeVar

import jinja2
import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from datasets import Dataset
from accelerate.utils import get_max_memory
from huggingface_hub import HfApi
from packaging import version
from peft import PeftModel, PeftConfig
from peft import __version__ as PEFT_VERSION
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,

)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    handle_stop_sequences,
    pad_and_concat,
    stop_sequences_criteria,
)

eval_logger = logging.getLogger(__name__)
from dllm_cache.cache import  dLLMCacheConfig,dLLMCache
from dllm_cache.hooks import  register_cache_Dream
from dataclasses import asdict
T = TypeVar("T", bound="LM")
from lm_eval.api.model import LM
import time

@register_model("dream")
class Dream(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        lora_path: Optional[str] = None,
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
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        alg: Optional[str] = "entropy",
        alg_temp: Optional[float] = 0.0,
        escape_until: Optional[bool] = False,
        is_feature_cache: bool = False,
        is_cfg_cache: bool = False,
        prompt_interval_steps: int = 1,
        gen_interval_steps: int = 1,
        cfg_interval_steps: int = 1,
        transfer_ratio:float = 0.0,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.lora_path = lora_path
        self.prompt_interval_steps = prompt_interval_steps
        self.gen_interval_steps = gen_interval_steps
        self.cfg_interval_steps = cfg_interval_steps
        self.transfer_ratio = transfer_ratio
        self.add_bos_token = add_bos_token
        self.escape_until = escape_until
        self.save_dir = save_dir

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
            # TODO: include in warning that `load_in_8bit` etc. affect this too
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
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
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1


        if is_feature_cache:
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=transfer_ratio,
                    cfg_interval_steps=cfg_interval_steps if is_cfg_cache else 1,
                )))
            register_cache_Dream(self.model,"model.layers")
        else:
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                    prompt_interval_steps=1,
                    gen_interval_steps=1,
                    transfer_ratio=0,
                    cfg_interval_steps=cfg_interval_steps if is_cfg_cache else 1,
                )))

        if self.rank == 0:
                print(f"Feature Cache is {is_feature_cache}.CFG Cache is {is_cfg_cache},prompt_interval_steps={prompt_interval_steps}, gen_interval_steps={gen_interval_steps}, cfg_interval_steps={cfg_interval_steps}")

        self.max_length = max_length
        self.add_bos_token = add_bos_token
        # generation params
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp
        self.escape_until = escape_until

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
        # 获取正确的数据类型
        target_dtype = get_dtype(dtype)
        
        # 加载基础模型
        self.model = transformers.AutoModel.from_pretrained(
            pretrained,
            torch_dtype=target_dtype,
            trust_remote_code=trust_remote_code,
        ).eval()
        
        # 如果有LoRA路径，则加载LoRA模型
        if self.lora_path is not None:
            config = PeftConfig.from_pretrained(self.lora_path)
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            eval_logger.info(f"Loaded LoRA model from {self.lora_path}")
        
        # 移动到指定设备
        self.model = self.model.to(self.device)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
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
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError
    
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError

    def _count_non_eos_tokens_before_truncation(self, generated_sequence, prompt_length):
        """
        统一的token计算函数：计算生成序列中非EOS的token数量（截断前）
        """
        # 获取生成的部分（去掉prompt）
        generated_tokens = generated_sequence[prompt_length:]
        # 计算非EOS token数量
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            non_eos_count = (generated_tokens != eos_token_id).sum().item()
        else:
            non_eos_count = len(generated_tokens)
        return non_eos_count

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        if self.add_bos_token:
            prompts = [self.tokenizer.bos_token + p for p in prompts]
        # tokenize
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
        if len(prompt_ids) > self.max_length-self.max_new_tokens:
            eval_logger.warning(f"Prompt length {len(prompt_ids)} is larger than {self.max_length-self.max_new_tokens}, cutoff on the left side")
            prompt_ids = prompt_ids[-(self.max_length-self.max_new_tokens):]

        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)
        feature_cache = dLLMCache()
        feature_cache.reset_cache(prompt_ids.shape[1])
        # print("add_bos_token",self.add_bos_token)
        # print("prompt_ids",prompt_ids)
        # print("prompt_text",self.tokenizer.decode(prompt_ids[0],skip_special_tokens=False))
        # exit()
        generation_ids = self.model.diffusion_generate(
            prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.diffusion_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
        )

        # 使用统一的token计算函数
        if not hasattr(self, 'total_generated_tokens'):
            self.total_generated_tokens = 0
        non_eos_tokens = self._count_non_eos_tokens_before_truncation(
            generation_ids.sequences[0], prompt_ids.shape[1]
        )
        self.total_generated_tokens += non_eos_tokens

        # decode
        responses = [
            self.tokenizer.decode(g[len(p) :].tolist()).split(self.tokenizer.eos_token)[0]
            for p, g in zip(prompt_ids, generation_ids.sequences)
        ]

        return responses

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        
        # 初始化统计计数器
        if not hasattr(self, 'total_generated_tokens'):
            self.total_generated_tokens = 0
        num_tokens = 0
        num_nfe = 0  # Number of Forward Evaluations

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

        end_time = time.time()
        total_time = end_time - start_time
        
        # 累积统计数据
        num_tokens = self.total_generated_tokens
        num_nfe = self.diffusion_steps * len(requests)  # 估算NFE
        
        # 保存最终统计结果
        final_stats = {
            'processed_samples': len(requests),
            'total_samples': len(requests),
            'total_tokens': num_tokens,
            'total_nfe': num_nfe,
            'total_time': total_time,
            'tokens_per_second': num_tokens / total_time if total_time > 0 else 0,
            'nfe_per_token': num_nfe / num_tokens if num_tokens > 0 else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存统计结果到文件
        if self.save_dir is not None:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            
            # 保存回答结果
            save_path = os.path.join(self.save_dir, f'rank_{self.rank}_responses.jsonl')
            with open(save_path, 'w', encoding='utf-8') as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
            # 保存统计结果
            stats_path = os.path.join(self.save_dir, f'rank_{self.rank}_final_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)
        
        # 打印最终统计结果
        print("\n" + "="*60)
        print("=== 最终统计结果 ===")
        print("="*60)
        print(f"处理样本数: {final_stats['processed_samples']}")
        print(f"总样本数: {final_stats['total_samples']}")
        print(f"总token数: {final_stats['total_tokens']}")
        print(f"总NFE数: {final_stats['total_nfe']}")
        print(f"总时间: {final_stats['total_time']:.4f}秒")
        print(f"Token/秒: {final_stats['tokens_per_second']:.2f}")
        print(f"NFE/Token: {final_stats['nfe_per_token']:.4f}")
        print(f"完成时间: {final_stats['timestamp']}")
        print("="*60)

        return res

