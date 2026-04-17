import logging
import gc
import time  # 添加时间模块
import warnings
import copy
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import torch
import torch.distributions as dists
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
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ModelOutput, is_torchdynamo_compiling, logging as hf_logging

eval_logger = logging.getLogger(__name__)
hf_generation_logger = hf_logging.get_logger(__name__)
T = TypeVar("T", bound="LM")


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


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
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
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", "origin")
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        if not self._from_model_config:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    hf_generation_logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

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
        decode_strategy: Optional[str] = "confidence_max",
        escape_until: Optional[bool] = False,
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
        self._rank = 0
        self._world_size = 1
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
        self.decode_strategy = decode_strategy
        self.escape_until = escape_until
        self.save_dir = save_dir
        self.analysis_case_counter = 0
        self.analysis_rank_path = None
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            self.analysis_rank_path = os.path.join(self.save_dir, f"rank_{self.rank}.jsonl")
            with open(self.analysis_rank_path, "w", encoding="utf-8"):
                pass

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

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        if is_torchdynamo_compiling():
            return

        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(self, generation_config, has_default_max_length, input_ids_length):
        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                hf_generation_logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.model.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.model.config)
            using_model_generation_config = True

        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            generation_config.update(**kwargs)
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.model.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.model.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.model.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.model.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            hf_generation_logger.warning(
                f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation."
            )

        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length

        sequences = []
        trace_histories = []
        sequence_histories = [] if (return_dict_in_generate and output_history) else None

        for sample_idx in range(input_ids.shape[0]):
            sample_input_ids = input_ids[sample_idx]
            sample_attention_mask = None if attention_mask is None else attention_mask[sample_idx]
            sample_sequence, sample_trace, sample_history = self._decode_single_sample(
                sample_input_ids=sample_input_ids,
                sample_attention_mask=sample_attention_mask,
                max_length=max_length,
                generation_config=generation_config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func,
                case_id=self.analysis_case_counter + sample_idx,
                collect_trace=self.analysis_rank_path is not None,
                collect_sequence_history=output_history,
            )
            sequences.append(sample_sequence)
            trace_histories.append(sample_trace)
            if sequence_histories is not None:
                sequence_histories.append(sample_history)

        sequences = torch.stack(sequences, dim=0)
        if return_dict_in_generate:
            history_payload = sequence_histories if sequence_histories is not None else trace_histories
            return DreamModelOutput(sequences=sequences, history=history_payload)
        return sequences

    def _build_single_attention_inputs(
        self,
        sample_attention_mask: Optional[torch.LongTensor],
        total_length: int,
    ) -> Tuple[Union[str, torch.Tensor], Optional[torch.LongTensor]]:
        if sample_attention_mask is None or not torch.any(sample_attention_mask == 0):
            return "full", None

        attn = F.pad(sample_attention_mask, (0, total_length - sample_attention_mask.shape[0]), value=1.0)
        tok_idx = attn.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attn == 0, 1)
        attn2d = torch.logical_and(attn.unsqueeze(0), attn.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        return attn2d, tok_idx.unsqueeze(0)

    def _compute_selection_confidence_and_tokens(
        self,
        logits: torch.Tensor,
        generation_config: DreamGenerationConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg

        probs = torch.softmax(logits, dim=-1)
        max_probs, top_token_ids = probs.max(dim=-1)

        if alg == "maskgit_plus":
            selection_confidence, sampled_tokens = sample_tokens(
                logits, temperature=temperature, top_p=top_p, top_k=top_k
            )
        elif alg == "topk_margin":
            selection_confidence, sampled_tokens = sample_tokens(
                logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True
            )
        elif alg == "entropy":
            selection_confidence, sampled_tokens = sample_tokens(
                logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True
            )
        elif alg == "origin":
            selection_confidence, sampled_tokens = sample_tokens(
                logits, temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            raise RuntimeError(f"Unknown alg: {alg}")

        return selection_confidence, sampled_tokens, max_probs, top_token_ids

    def _forward_single_sample(
        self,
        x: torch.LongTensor,
        sample_attention: Union[str, torch.Tensor],
        sample_tok_idx: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        logits = self.model(x.unsqueeze(0), sample_attention, sample_tok_idx).logits
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)[0]

    def _oracle_select_position(
        self,
        x: torch.LongTensor,
        remaining_positions: List[int],
        sampled_tokens: torch.Tensor,
        sample_attention: Union[str, torch.Tensor],
        sample_tok_idx: Optional[torch.LongTensor],
    ) -> int:
        best_idx = 0
        best_score = None

        for candidate_idx, candidate_pos in enumerate(remaining_positions):
            candidate_x = x.clone()
            candidate_x[candidate_pos] = sampled_tokens[candidate_idx]
            next_positions = remaining_positions[:candidate_idx] + remaining_positions[candidate_idx + 1 :]
            if not next_positions:
                candidate_score = 0.0
            else:
                next_logits = self._forward_single_sample(candidate_x, sample_attention, sample_tok_idx)
                next_position_tensor = torch.tensor(next_positions, device=self.device, dtype=torch.long)
                next_mask_logits = next_logits[next_position_tensor, :]
                next_max_probs = torch.softmax(next_mask_logits, dim=-1).max(dim=-1).values
                candidate_score = float(next_max_probs.mean().item())
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_idx = candidate_idx

        return best_idx

    def _decode_single_sample(
        self,
        sample_input_ids: torch.LongTensor,
        sample_attention_mask: Optional[torch.LongTensor],
        max_length: int,
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        case_id: int,
        collect_trace: bool,
        collect_sequence_history: bool,
    ) -> Tuple[torch.LongTensor, List[List[Dict[str, Union[int, float]]]], Optional[List[torch.LongTensor]]]:
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps

        x = F.pad(sample_input_ids, (0, max_length - sample_input_ids.shape[0]), value=mask_token_id)
        sample_attention, sample_tok_idx = self._build_single_attention_inputs(sample_attention_mask, max_length)

        generator = None
        if self.decode_strategy == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(20260416 + self.rank * 1000003 + case_id * 17)

        step_trace = []
        sample_history = [] if collect_sequence_history else None

        x = generation_tokens_hook_func(None, x.unsqueeze(0), None).squeeze(0)
        for step_idx in range(steps):
            mask_positions = torch.nonzero(x == mask_token_id, as_tuple=False).squeeze(-1)
            if mask_positions.numel() == 0:
                break

            logits = self._forward_single_sample(x, sample_attention, sample_tok_idx).unsqueeze(0)
            logits = generation_logits_hook_func(step_idx, x.unsqueeze(0), logits).squeeze(0)
            remaining_logits = logits[mask_positions, :]
            selection_confidence, sampled_tokens, max_probs, top_token_ids = self._compute_selection_confidence_and_tokens(
                remaining_logits,
                generation_config,
            )

            if collect_trace:
                step_trace.append(
                    [
                        {
                            "position_index": int(position),
                            "confidence": float(confidence),
                            "top_token_id": int(top_token_id),
                        }
                        for position, confidence, top_token_id in zip(
                            mask_positions.tolist(),
                            max_probs.tolist(),
                            top_token_ids.tolist(),
                        )
                    ]
                )

            remaining_positions = mask_positions.tolist()
            if self.decode_strategy == "confidence_max":
                if generation_config.alg_temp is None or generation_config.alg_temp == 0:
                    selected_idx = int(torch.argmax(selection_confidence).item())
                else:
                    position_probs = torch.softmax(selection_confidence / generation_config.alg_temp, dim=-1)
                    selected_idx = int(torch.multinomial(position_probs, num_samples=1).item())
            elif self.decode_strategy == "left_to_right":
                selected_idx = 0
            elif self.decode_strategy == "right_to_left":
                selected_idx = len(remaining_positions) - 1
            elif self.decode_strategy == "random":
                selected_idx = int(torch.randint(len(remaining_positions), (1,), generator=generator).item())
            elif self.decode_strategy == "oracle":
                selected_idx = self._oracle_select_position(
                    x=x,
                    remaining_positions=remaining_positions,
                    sampled_tokens=sampled_tokens,
                    sample_attention=sample_attention,
                    sample_tok_idx=sample_tok_idx,
                )
            else:
                raise RuntimeError(f"Unknown decode strategy: {self.decode_strategy}")

            selected_position = remaining_positions[selected_idx]
            x[selected_position] = sampled_tokens[selected_idx]
            x = generation_tokens_hook_func(step_idx, x.unsqueeze(0), logits.unsqueeze(0)).squeeze(0)

            if sample_history is not None:
                sample_history.append(x.clone())

        return x, step_trace, sample_history

    def _save_strategy_analysis_batch(
        self,
        prompts: List[str],
        attn_mask: torch.LongTensor,
        step_traces: Optional[List[List[List[Dict[str, Union[int, float]]]]]],
    ) -> None:
        if not self.analysis_rank_path:
            return

        if step_traces is None:
            raise RuntimeError("step_traces must be provided when save_dir is enabled.")

        prompt_lengths = attn_mask.sum(dim=-1).tolist()
        with open(self.analysis_rank_path, "a", encoding="utf-8") as fout:
            for prompt, prompt_length, strategy_trace in zip(prompts, prompt_lengths, step_traces):
                case_id = self.analysis_case_counter
                record = {
                    "case_id": case_id,
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "decode_strategy": self.decode_strategy,
                    "prompt_length": int(prompt_length),
                    "max_new_tokens": int(self.max_new_tokens),
                    "prompt": prompt,
                    "step_remaining_mask_details": strategy_trace,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.analysis_case_counter += 1

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id")
            and torch.any(input_ids == generation_config.pad_token_id)
            and attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
        )

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        if self.add_bos_token:
            prompts = [self.tokenizer.bos_token + p for p in prompts]
        
        # 记录批次开始时间
        batch_start_time = time.time()
        
        # tokenize
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
        if prompt_ids.shape[1] > self.max_length - self.max_new_tokens:
            eval_logger.warning(
                f"Prompt length {prompt_ids.shape[1]} is larger than {self.max_length-self.max_new_tokens}, cutoff on the left side"
            )
            prompt_ids = prompt_ids[:, -(self.max_length - self.max_new_tokens):]

        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)

        generation_ids = self.diffusion_generate(
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
        self._save_strategy_analysis_batch(prompts, attn_mask.cpu(), generation_ids.history)

        # 记录批次结束时间
        batch_end_time = time.time()
        batch_generation_time = batch_end_time - batch_start_time

        # decode
        responses = []
        for i, (p, g) in enumerate(zip(prompt_ids, generation_ids.sequences)):
            # 计算每个prompt的统计信息
            prompt_length = len(p)
            generated_sequence = g[prompt_length:].tolist()
            
            # 计算生成的token数
            generated_tokens = len(generated_sequence)
            
            # 计算实际token数（去掉EOS）
            actual_tokens = generated_tokens
            if self.tokenizer.eos_token_id is not None and generated_sequence and generated_sequence[-1] == self.tokenizer.eos_token_id:
                actual_tokens -= 1
            
            # 解码响应
            response = self.tokenizer.decode(generated_sequence).split(self.tokenizer.eos_token)[0]
            responses.append(response)
            
            # 为每个prompt分配时间（简化处理，假设时间平均分配）
            prompt_generation_time = batch_generation_time / len(prompts)
            
            # 更新统计数据
            self.total_prompts += 1
            self.total_generation_time += prompt_generation_time
            self.total_generated_tokens += generated_tokens
            self.total_actual_tokens += actual_tokens
            
            # 保存单个prompt的统计
            self.all_generation_times.append(prompt_generation_time)
            self.all_generated_tokens.append(generated_tokens)
            self.all_actual_tokens.append(actual_tokens)
            
            # 计算吞吐量
            throughput_generated = generated_tokens / prompt_generation_time if prompt_generation_time > 0 else 0
            throughput_actual = actual_tokens / prompt_generation_time if prompt_generation_time > 0 else 0
            
            # 打印当前prompt的指标
            print(f"\n=== Prompt {self.total_prompts} 指标 ===")
            print(f"生成时间: {prompt_generation_time:.4f}秒")
            print(f"生成token数: {generated_tokens}")
            print(f"实际生成token数 (去掉EOS): {actual_tokens}")
            print(f"生成token吞吐量: {throughput_generated:.2f} tokens/s")
            print(f"实际token吞吐量: {throughput_actual:.2f} tokens/s")

        return responses

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []

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
