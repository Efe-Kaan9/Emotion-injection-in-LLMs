"""
projector_agnostic.py — Model-Agnostic KV Projector Factory
============================================================

Extracts architecture dimensions from any HuggingFace config and
instantiates the correct BasicKVProjector / ModulatedKVProjector.

Verified architectures:
  - microsoft/Phi-4-mini-instruct  (GQA: num_kv_heads < num_heads)
  - Qwen/Qwen2.5-1.5B-Instruct    (GQA: num_kv_heads < num_heads)

Usage:
    from projector_agnostic import ModelConfig, create_projector

    mc = ModelConfig.from_pretrained("microsoft/Phi-4-mini-instruct")
    projector = create_projector(mc, variant=1, prefix_len=4, device="cuda")
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Tuple, Optional
from transformers import AutoConfig


# ──────────────────────────────────────────────────────────────
# 1.  ModelConfig — architecture-agnostic dimension holder
# ──────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    Normalised container for the four dimensions every KV-projector needs.

    Fields
    ------
    model_name       : HuggingFace model identifier (for bookkeeping / reloading)
    hidden_size      : transformer hidden dimension  (e.g. 3072 for Phi-4-mini)
    num_attention_heads : total query heads           (e.g. 32)
    num_key_value_heads : KV heads (GQA)              (e.g. 8)
    num_hidden_layers   : number of decoder layers    (e.g. 32)
    head_dim            : hidden_size // num_attention_heads
    """
    model_name: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int = field(init=False)

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads

    # ---- factory from HuggingFace config object ----
    @classmethod
    def from_hf_config(cls, hf_config, model_name: str = "unknown") -> "ModelConfig":
        """
        Build ModelConfig from an already-loaded HuggingFace config object.

        Handles naming differences across architectures:
          - Phi-4  : config.num_key_value_heads
          - Qwen2.5: config.num_key_value_heads
          - LLaMA  : config.num_key_value_heads
          - Fallback: num_key_value_heads = num_attention_heads (MHA)
        """
        num_kv = getattr(
            hf_config,
            "num_key_value_heads",
            getattr(hf_config, "num_attention_heads", None),
        )
        if num_kv is None:
            raise ValueError(
                f"Cannot determine num_key_value_heads from config: {type(hf_config)}"
            )

        return cls(
            model_name=model_name,
            hidden_size=hf_config.hidden_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=num_kv,
            num_hidden_layers=hf_config.num_hidden_layers,
        )

    # ---- factory from model name (downloads config only — no weights) ----
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        cache_dir: str = "./.hf_cache",
    ) -> "ModelConfig":
        """
        Download only the config JSON (a few KB) and build ModelConfig.
        This does NOT download model weights.
        """
        hf_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        return cls.from_hf_config(hf_config, model_name=model_name)

    def summary(self) -> str:
        return (
            f"ModelConfig({self.model_name})\n"
            f"  hidden_size        = {self.hidden_size}\n"
            f"  num_attention_heads = {self.num_attention_heads}\n"
            f"  num_key_value_heads = {self.num_key_value_heads}\n"
            f"  num_hidden_layers   = {self.num_hidden_layers}\n"
            f"  head_dim            = {self.head_dim}"
        )


# ──────────────────────────────────────────────────────────────
# 2.  Projector classes  (identical logic to notebook Var1/Var2,
#     but accept ModelConfig instead of raw ints)
# ──────────────────────────────────────────────────────────────

class AgnosticBasicKVProjector(nn.Module):
    """
    VARIANT 1 — Baseline KV Injection (model-agnostic version).

    Emotion vector → single linear projection → uniform KV prefix.
    Produces (k_prefix, v_prefix) shaped (B, num_kv_heads, prefix_len, head_dim).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        num_emotions: int = 28,
        prefix_len: int = 4,
    ):
        super().__init__()
        self.model_config = model_config
        self.num_emotions = num_emotions
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim
        self.prefix_len = prefix_len

        self.proj_k = nn.Linear(num_emotions, self.num_kv_heads * self.head_dim)
        self.proj_v = nn.Linear(num_emotions, self.num_kv_heads * self.head_dim)

    def forward(
        self, emotion_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            emotion_vec: (B, num_emotions)
        Returns:
            k_prefix: (B, num_kv_heads, prefix_len, head_dim)
            v_prefix: (B, num_kv_heads, prefix_len, head_dim)
        """
        B = emotion_vec.size(0)
        k = self.proj_k(emotion_vec).view(B, self.num_kv_heads, 1, self.head_dim)
        v = self.proj_v(emotion_vec).view(B, self.num_kv_heads, 1, self.head_dim)
        return k.repeat(1, 1, self.prefix_len, 1), v.repeat(1, 1, self.prefix_len, 1)


class AgnosticModulatedKVProjector(nn.Module):
    """
    VARIANT 2 — Head-Wise Modulated KV Injection (model-agnostic version).

    Emotion vector → base KV projection + per-head sigmoid gating.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        num_emotions: int = 28,
        prefix_len: int = 4,
        gating_hidden_dim: int = 64,
    ):
        super().__init__()
        self.model_config = model_config
        self.num_emotions = num_emotions
        self.num_heads = model_config.num_attention_heads
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim
        self.prefix_len = prefix_len

        # Base KV projection (uses num_kv_heads for cache compatibility)
        self.proj_k = nn.Linear(num_emotions, self.num_kv_heads * self.head_dim)
        self.proj_v = nn.Linear(num_emotions, self.num_kv_heads * self.head_dim)

        # Head-gating network (produces per-head scaling α_i)
        self.head_gate = nn.Sequential(
            nn.Linear(num_emotions, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, self.num_heads),
        )

    def forward(
        self, emotion_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            emotion_vec: (B, num_emotions)
        Returns:
            k_prefix:   (B, num_kv_heads, prefix_len, head_dim)
            v_prefix:   (B, num_kv_heads, prefix_len, head_dim)
            head_gates:  (B, num_heads)
        """
        B = emotion_vec.size(0)

        k = self.proj_k(emotion_vec).view(B, self.num_kv_heads, 1, self.head_dim)
        v = self.proj_v(emotion_vec).view(B, self.num_kv_heads, 1, self.head_dim)

        head_gates = torch.sigmoid(self.head_gate(emotion_vec))  # (B, num_heads)
        head_gates_kv = head_gates[:, : self.num_kv_heads]        # (B, num_kv_heads)

        k = k * head_gates_kv.unsqueeze(-1).unsqueeze(-1)
        v = v * head_gates_kv.unsqueeze(-1).unsqueeze(-1)

        return (
            k.repeat(1, 1, self.prefix_len, 1),
            v.repeat(1, 1, self.prefix_len, 1),
            head_gates,
        )


# ──────────────────────────────────────────────────────────────
# 3.  Factory + Checkpoint Loading
# ──────────────────────────────────────────────────────────────

def create_projector(
    model_config: ModelConfig,
    variant: int = 1,
    num_emotions: int = 28,
    prefix_len: int = 4,
    gating_hidden_dim: int = 64,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    One-liner to instantiate, load weights, and move to device.

    Parameters
    ----------
    model_config : ModelConfig
        Architecture specification (from any HF model).
    variant : {1, 2}
        1 = BasicKVProjector, 2 = ModulatedKVProjector.
    checkpoint_path : str, optional
        Path to a .pt file saved with {'model_state_dict': ...}.
        If the checkpoint was trained on a different architecture,
        loading will fail with a shape mismatch (by design).
    device : str
        Target device ("cuda", "cpu").

    Returns
    -------
    nn.Module
        Projector in eval mode on the target device.
    """
    if variant == 1:
        proj = AgnosticBasicKVProjector(
            model_config, num_emotions=num_emotions, prefix_len=prefix_len
        )
    elif variant == 2:
        proj = AgnosticModulatedKVProjector(
            model_config,
            num_emotions=num_emotions,
            prefix_len=prefix_len,
            gating_hidden_dim=gating_hidden_dim,
        )
    else:
        raise ValueError(f"variant must be 1 or 2, got {variant}")

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        proj.load_state_dict(state, strict=True)
        print(f"  [OK] Loaded checkpoint: {checkpoint_path}")
    elif checkpoint_path:
        print(f"  [WARN] Checkpoint not found: {checkpoint_path} -- using random weights")

    proj = proj.to(device).eval()
    n_params = sum(p.numel() for p in proj.parameters())
    print(f"  [OK] Variant {variant} projector ({n_params:,} params) on {device}")
    return proj


# ──────────────────────────────────────────────────────────────
# 4.  Quick self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Self-test: projector_agnostic.py")
    print("=" * 60)

    # ---- Test with Phi-4-mini config (no download needed for test) ----
    # Simulate a config for offline testing
    class _FakeConfig:
        hidden_size = 3072
        num_attention_heads = 32
        num_key_value_heads = 8
        num_hidden_layers = 32

    mc_phi = ModelConfig.from_hf_config(_FakeConfig(), model_name="phi-4-mini-sim")
    print(mc_phi.summary())

    p1 = create_projector(mc_phi, variant=1, device="cpu")
    dummy = torch.randn(1, 28)
    k, v = p1(dummy)
    print(f"  Var1 K shape: {k.shape}  V shape: {v.shape}")
    assert k.shape == (1, 8, 4, 96), f"Unexpected shape: {k.shape}"

    p2 = create_projector(mc_phi, variant=2, device="cpu")
    k2, v2, g = p2(dummy)
    print(f"  Var2 K shape: {k2.shape}  V shape: {v2.shape}  Gates: {g.shape}")
    assert g.shape == (1, 32), f"Unexpected gate shape: {g.shape}"

    # ---- Test with Qwen2.5-1.5B config ----
    class _FakeQwen:
        hidden_size = 1536
        num_attention_heads = 12
        num_key_value_heads = 2
        num_hidden_layers = 28

    mc_qwen = ModelConfig.from_hf_config(_FakeQwen(), model_name="qwen2.5-1.5b-sim")
    print("\n" + mc_qwen.summary())

    p1q = create_projector(mc_qwen, variant=1, device="cpu")
    kq, vq = p1q(dummy)
    print(f"  Var1 K shape: {kq.shape}  V shape: {vq.shape}")
    assert kq.shape == (1, 2, 4, 128), f"Unexpected shape: {kq.shape}"

    p2q = create_projector(mc_qwen, variant=2, device="cpu")
    kq2, vq2, gq = p2q(dummy)
    print(f"  Var2 K shape: {kq2.shape}  V shape: {vq2.shape}  Gates: {gq.shape}")
    assert gq.shape == (1, 12), f"Unexpected gate shape: {gq.shape}"

    print("\n[PASS] All assertions passed -- projector_agnostic.py is ready.")
