from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": None,  # resolved by from_pretrained
}


class ModelRuntime:
    """Wraps a Hugging Face causal LM and tokenizer for ExactKV.

    This is the only place that touches the HF API. Everything else in the
    codebase calls through here.

    Phase 1 constraints:
    - One model, one device.
    - Greedy decoding only (callers are responsible for argmax; this class
      does not sample).
    - dtype defaults to float32 for determinism in tests.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "float32",
    ) -> None:
        self.model_name = model_name
        self.dtype_str = dtype

        torch_dtype = _DTYPE_MAP.get(dtype)
        if dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Choose from {list(_DTYPE_MAP)}"
            )

        # Resolve device.  "auto" lets accelerate decide; otherwise honour it.
        device_map: str | None = "auto" if device == "auto" else None
        explicit_device: str | None = None if device == "auto" else device

        load_kwargs: dict[str, Any] = {"device_map": device_map}
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # Ensure a pad token exists; reuse eos if absent.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, **load_kwargs
        )

        # Move to explicit device when device != "auto".
        if explicit_device is not None:
            self.model = self.model.to(explicit_device)

        self.model.eval()

        # Record the resolved device and dtype for callers.
        self.device: torch.device = next(self.model.parameters()).device
        self.dtype: torch.dtype = next(self.model.parameters()).dtype

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, prompt: str) -> torch.Tensor:
        """Tokenise prompt and return a 1-D LongTensor on self.device."""
        ids = self.tokenizer.encode(prompt, return_tensors="pt")
        return ids.to(self.device)  # shape: [1, seq]

    def decode(self, ids: torch.Tensor) -> str:
        """Decode a 1-D or 2-D token-ID tensor to a string."""
        flat = ids.squeeze().tolist()
        if isinstance(flat, int):
            flat = [flat]
        return self.tokenizer.decode(flat, skip_special_tokens=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Single forward pass.

        Returns the raw HF ModelOutput so callers can read logits and
        past_key_values without this class deciding their semantics.
        """
        inputs: dict[str, Any] = {
            "input_ids": input_ids.to(self.device),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask.to(self.device)
        inputs.update(kwargs)
        return self.model(**inputs)

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size
