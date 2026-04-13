import logging
import gc
import json
import time  # 添加时间模块
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

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate

import numpy as np  # 新增
import matplotlib.pyplot as plt  # 新增
import os  # 新增

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")

@register_model("dream")
class Dream(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
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
        save_dir: Optional[str] = "visualizations",  # 新增参数
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
            # TODO: include in warning that `load_in_8bit` etc. affect this too
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        
        # 添加统计指标追踪
        self.total_prompts = 0
        self.total_generation_time = 0.0
        self.total_generated_tokens = 0
        self.total_actual_tokens = 0  # 实际生成的token数（去掉EOS）
        self.all_generation_times = []
        self.all_generated_tokens = []
        self.all_actual_tokens = []
        self.save_dir = save_dir  # 保存 save_dir
        
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
        self.model = (
            transformers.AutoModel.from_pretrained(
                pretrained,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            )
            .eval()
        ).to(self.device)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

    def _compute_generation_token_stats(
        self,
        generated_ids: List[int],
        block_length: int,
        mask_token_id: int,
        eos_token_id: Optional[int],
    ) -> Tuple[List[int], int, int]:
        trimmed_ids = list(generated_ids)
        while trimmed_ids and trimmed_ids[-1] == mask_token_id:
            trimmed_ids.pop()

        if eos_token_id is not None and eos_token_id in trimmed_ids:
            eos_idx = trimmed_ids.index(eos_token_id)
            valid_sequence = trimmed_ids[:eos_idx]
            generated_tokens_including_eos = ((eos_idx // block_length) + 1) * block_length
        else:
            valid_sequence = trimmed_ids
            generated_tokens_including_eos = len(trimmed_ids)

        actual_tokens_excluding_eos = len(valid_sequence)
        return valid_sequence, actual_tokens_excluding_eos, generated_tokens_including_eos
    
    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

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

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        # Block-wise sequential generation with confidence threshold
        if self.add_bos_token:
            prompts = [self.tokenizer.bos_token + p for p in prompts]

        batch_start_time = time.time()

        # Tokenize and pad
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)

        # Block config
        max_length = self.max_length
        max_new_tokens = self.max_new_tokens
        block_length = 32  # 可调整
        threshold = 0.9    # 置信度阈值，可调整
        mask_token_id = self.tokenizer.mask_token_id if hasattr(self.tokenizer, "mask_token_id") else self.tokenizer.eos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        batch_size, prompt_len = prompt_ids.shape
        gen_length = max_new_tokens
        total_length = prompt_len + gen_length
        num_blocks = gen_length // block_length
        assert gen_length % block_length == 0, "gen_length必须能被block_length整除"

        # Pad to total_length with mask_token
        x = torch.full((batch_size, total_length), mask_token_id, dtype=torch.long, device=self.device)
        x[:, :prompt_len] = prompt_ids
        attn_mask_pad = torch.ones((batch_size, total_length), dtype=attn_mask.dtype, device=self.device)
        attn_mask_pad[:, :prompt_len] = attn_mask

        # --- 可视化初始化 ---
        vis_data = {
            'prompt_length': prompt_len,
            'steps': [], 
            'max_seq_length': total_length # 初始设为最大，后面会更新
        }
        step_global = 0
        
        # 增加完成标志位
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # [新增] 追踪实际处理到的最大位置，用于裁剪可视化图片的空白
        actual_process_end = prompt_len 

        # 逐block生成
        for block_idx in range(num_blocks):
            # 1. 只有在开启新 Block 前，检查是否所有样本都已结束
            if is_finished.all():
                break

            block_start = prompt_len + block_idx * block_length
            block_end = block_start + block_length
            
            # [新增] 更新实际处理位置到当前 block 的结束位置
            actual_process_end = block_end

            mask_region = x[:, block_start:block_end]
            mask_index = (mask_region == mask_token_id)
            
            if mask_index.sum() == 0:
                continue

            # 2. 内层循环：必须跑完，直到当前 Block 填满
            while mask_index.sum() > 0:
                step_global += 1
                prev_x_sample0 = x[0].clone()
                
                with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    model_output = self.model(x, attention_mask=attn_mask_pad)
                logits = model_output.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                logits_region = logits[:, block_start:block_end]
                mask_logits = logits_region[mask_index]
                probs = torch.softmax(mask_logits / (self.temperature if self.temperature > 0 else 1.0), dim=-1)
                confidence, x0 = probs.max(dim=-1)

                # --- 采集 Step 数据 ---
                step_data = {'step_idx': step_global - 1, 'eos_probs': [], 'decoded_positions': []}
                if eos_token_id is not None:
                    all_probs_region = torch.softmax(logits_region / (self.temperature if self.temperature > 0 else 1.0), dim=-1)
                    mask_indices_sample0 = (x[0, block_start:block_end] == mask_token_id).nonzero(as_tuple=True)[0]
                    if len(mask_indices_sample0) > 0:
                        eos_probs = all_probs_region[0, mask_indices_sample0, eos_token_id]
                        for idx, p_val in zip(mask_indices_sample0, eos_probs):
                            abs_pos = block_start + idx.item()
                            step_data['eos_probs'].append((abs_pos, p_val.item()))

                # 置信度阈值机制
                x_ = torch.full_like(mask_region, mask_token_id)
                x_[mask_index] = x0.clone()
                full_confidence = torch.full_like(mask_region, -float('inf'), dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                
                transfer_index = (full_confidence > threshold)
                
                if transfer_index.sum() > 0:
                    x[:, block_start:block_end][transfer_index] = x_[transfer_index]
                else:
                    max_idx = torch.argmax(full_confidence, dim=1)
                    for b in range(batch_size):
                        x[b, block_start + max_idx[b]] = x_[b, max_idx[b]]
                
                # 3. 更新完成状态
                if eos_token_id is not None:
                    for b in range(batch_size):
                        if not is_finished[b]:
                            current_gen = x[b, prompt_len:block_end]
                            if (current_gen == eos_token_id).any():
                                is_finished[b] = True

                mask_index = (x[:, block_start:block_end] == mask_token_id)

                # --- 计算本步解码的位置 ---
                was_mask = (prev_x_sample0 == mask_token_id)
                is_mask_now = (x[0] == mask_token_id)
                decoded_indices = (was_mask & ~is_mask_now).nonzero(as_tuple=True)[0]
                step_data['decoded_positions'] = decoded_indices.tolist()
                vis_data['steps'].append(step_data)

        batch_end_time = time.time()
        batch_generation_time = batch_end_time - batch_start_time

        # decode & 统计
        responses = []
        for i in range(batch_size):
            full_sequence = x[i, prompt_len:].tolist()
            (
                valid_sequence,
                actual_tokens_excluding_eos,
                generated_tokens_including_eos,
            ) = self._compute_generation_token_stats(
                full_sequence, block_length, mask_token_id, eos_token_id
            )
            response = self.tokenizer.decode(valid_sequence, skip_special_tokens=True)
            responses.append(response)
            
            prompt_generation_time = batch_generation_time / batch_size
            self.total_prompts += 1
            self.total_generation_time += prompt_generation_time
            self.total_generated_tokens += generated_tokens_including_eos
            self.total_actual_tokens += actual_tokens_excluding_eos
            self.all_generation_times.append(prompt_generation_time)
            self.all_generated_tokens.append(generated_tokens_including_eos)
            self.all_actual_tokens.append(actual_tokens_excluding_eos)
            
            throughput_generated = generated_tokens_including_eos / prompt_generation_time if prompt_generation_time > 0 else 0
            throughput_actual = actual_tokens_excluding_eos / prompt_generation_time if prompt_generation_time > 0 else 0
            print(f"\n=== Prompt {self.total_prompts} 指标 ===")
            print(f"生成时间: {prompt_generation_time:.4f}秒")
            print(f"generated_tokens_including_eos: {generated_tokens_including_eos}")
            print(f"actual_tokens_excluding_eos: {actual_tokens_excluding_eos}")
            print(f"生成token吞吐量: {throughput_generated:.2f} tokens/s")
            print(f"实际token吞吐量: {throughput_actual:.2f} tokens/s")
        
        # --- 生成可视化 ---
        if self.save_dir is not None:
            # [关键修改] 将最大序列长度更新为实际处理的结束位置
            # 这样绘图函数初始化矩阵和设置坐标轴时，就会自动裁剪掉右边的空白
            vis_data['max_seq_length'] = actual_process_end
            # self._generate_visualization(vis_data, prompt_len, self.total_prompts)

        return responses
    


    def _generate_visualization(self, vis_data, prompt_length, sample_id):
        """
        可视化：纵轴为 Step (从下往上)，横轴为 Token Position
        """
        if not vis_data['steps']: return
        
        max_seq_len = vis_data['max_seq_length']
        num_steps = len(vis_data['steps'])
        
        # 1. 矩阵初始化 (0.0 代表 Reds 的白背景)
        eos_prob_matrix = np.zeros((num_steps, max_seq_len))
        
        for i, step_data in enumerate(vis_data['steps']):
            for pos, prob in step_data['eos_probs']:
                if pos < max_seq_len:
                    eos_prob_matrix[i, pos] = prob
        
        # 2. 动态调整視野：聚焦生成区域
        display_start = max(0, prompt_length - 5)
        display_end = max_seq_len
        
        # 根据活跃区域宽度动态计算画布大小
        fig_width = max(10, (display_end - display_start) * 0.2)
        fig_height = max(8, num_steps * 0.3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # 3. 绘制热力图 (origin='lower' 确保纵轴从下往上增长)
        im = ax.imshow(eos_prob_matrix, aspect='auto', cmap='Reds', origin='lower',
                       interpolation='nearest', vmin=0, vmax=1)
        
        # 4. 绘制灰色圆圈 (表示该 step 解码了该 token)
        for i, step_data in enumerate(vis_data['steps']):
            for pos in step_data['decoded_positions']:
                if pos >= display_start:
                    circle = plt.Circle((pos, i), 0.38, color='gray', fill=False, linewidth=1.5, alpha=0.8)
                    ax.add_patch(circle)
        
        # 5. 设置视野范围和标注
        ax.set_xlim(display_start - 0.5, display_end - 0.5)
        
        # 黑色虚线：Prompt 与生成的界限
        ax.axvline(x=prompt_length - 0.5, color='black', linestyle='--', linewidth=1.5, label='Prompt End')
        ax.legend(loc='upper left', framealpha=0.8)

        ax.set_xlabel('Token Position (ID)', fontsize=10)
        ax.set_ylabel('Decoding Forward Step (1 -> Max)', fontsize=10)
        ax.set_title(f'EOS Probability Heatmap - Sample {sample_id}', fontsize=12)
        
        # 坐标刻度优化
        if (display_end - display_start) > 20:
            step_size = max(1, (display_end - display_start) // 15)
            ax.set_xticks(np.arange(display_start, display_end, step=step_size))
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Predicted EOS Probability', rotation=270, labelpad=15)
        
        # 保存
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f'rank_{self.rank}_vis_sample_{sample_id}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f">>> [Rank {self.rank}] Sample {sample_id} visualization saved to {save_path}")

        

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        start_time = time.time()

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(contexts)
            if not self.escape_until:
                for i, r in enumerate(responses):
                    for s in gen_args[0]['until']:
                        r = r.split(s)[0]
                    responses[i] = r

            # if self.rank == 0:
            #     print(f"Context:\n{contexts[0]}\nResponse:\n{responses[0]}\n")
            print(f"Context:\n{contexts[0]}\nResponse:\n{responses[0]}\n")
            res.extend(responses)
            pbar.update(len(contexts))

        # 打印所有提示词的平均指标
        if self.total_prompts > 0:
            avg_generation_time = self.total_generation_time / self.total_prompts
            avg_generated_tokens = self.total_generated_tokens / self.total_prompts
            avg_actual_tokens = self.total_actual_tokens / self.total_prompts
            
            # 计算平均吞吐量
            avg_throughput_generated = self.total_generated_tokens / self.total_generation_time if self.total_generation_time > 0 else 0
            avg_throughput_actual = self.total_actual_tokens / self.total_generation_time if self.total_generation_time > 0 else 0
            
            print("\n" + "="*60)
            print("=== 所有提示词的统计汇总 ===")
            print("="*60)
            print(f"总提示词数量: {self.total_prompts}")
            print(f"总生成时间: {self.total_generation_time:.4f}秒")
            print(f"总生成token数: {self.total_generated_tokens}")
            print(f"总实际token数: {self.total_actual_tokens}")
            print()
            print("=== 平均指标 ===")
            print(f"平均生成时间: {avg_generation_time:.4f}秒")
            print(f"平均生成token数: {avg_generated_tokens:.2f}")
            print(f"平均实际token数: {avg_actual_tokens:.2f}")
            print()
            print("=== 平均吞吐量 ===")
            print(f"平均生成token吞吐量: {avg_throughput_generated:.2f} tokens/s")
            print(f"平均实际token吞吐量: {avg_throughput_actual:.2f} tokens/s")
            print("="*60)

            if self.save_dir is not None:
                os.makedirs(self.save_dir, exist_ok=True)
                total_time = time.time() - start_time
                final_stats = {
                    "processed_samples": int(self.total_prompts),
                    "total_samples": len(requests),
                    "total_generated_tokens_including_eos": int(self.total_generated_tokens),
                    "total_actual_tokens_excluding_eos": int(self.total_actual_tokens),
                    "total_time": total_time,
                    "generated_tokens_including_eos_per_second": float(self.total_generated_tokens) / total_time if total_time > 0 else 0.0,
                    "actual_tokens_excluding_eos_per_second": float(self.total_actual_tokens) / total_time if total_time > 0 else 0.0,
                    "timestamp": time.time(),
                    "rank": self.rank,
                    "world_size": self.world_size,
                }
                with open(os.path.join(self.save_dir, f'rank_{self.rank}_stats.json'), 'w', encoding='utf-8') as f:
                    json.dump(final_stats, f, ensure_ascii=False, indent=2)

        return res

    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps

        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        # always unmask bos and eos
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.tokenizer.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        '''
        prompt_index : 1D bool tensor, length=batch.shape[1]
        '''
        if self.classifier_free_guidance > 1.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.tokenizer.mask_token_id
            batch = torch.cat([batch, un_batch])

        input = batch

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input).logits
            # since bos always unmask, the first logits will not be used
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

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
            # eval_logger.info("before noising")
            perturbed_seq_, p_mask = self._forward_process(seq)
            # eval_logger.info("end noising")
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.tokenizer.mask_token_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous() # l1*l1

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.tokenizer.mask_token_id
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

        # by default truncate on the left
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

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        print(ds[0])
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                # likelihood calculations are modified from https://github.com/ML-GSAI/SMDM/blob/main/evaluate_diff.py
                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type == 'ar_ftb' or self.nll_type == 'ar_btf':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                # TODO: greedy decoding
                is_target_greedy_dec = False

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError


if __name__ == "__main__":
    cli_evaluate()
