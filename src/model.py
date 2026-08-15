"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
from typing import Optional, cast

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.config.job_config import Model

# Try to use flash-attn package (faster on H100)
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) as described in RoFormer paper."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Precompute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Precompute cos and sin for max_seq_len positions
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        # Duplicate freqs for pairing: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given sequence length."""
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (B, n_head, T, head_dim)
        k: Key tensor of shape (B, n_head, T, head_dim)
        cos: Cosine tensor of shape (T, head_dim)
        sin: Sine tensor of shape (T, head_dim)

    Returns:
        Rotated q and k tensors
    """
    # Reshape cos and sin to broadcast: (1, 1, T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Rotate half: split into two halves and rotate
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        if FLASH_ATTN_AVAILABLE:
            # flash_attn expects (B, T, nh, hs) format
            q = q.view(B, T, self.n_head, C // self.n_head)
            k = k.view(B, T, self.n_head, C // self.n_head)
            v = v.view(B, T, self.n_head, C // self.n_head)
            # Apply RoPE if provided (need to transpose for apply_rotary_pos_emb)
            if rope_cos is not None and rope_sin is not None:
                q = q.transpose(1, 2)  # (B, nh, T, hs)
                k = k.transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
                q = q.transpose(1, 2)  # back to (B, T, nh, hs)
                k = k.transpose(1, 2)
            y = flash_attn_func(q, k, v, causal=True,
                               dropout_p=self.dropout if self.training else 0.0)
            y = y.view(B, T, C)
        else:
            # PyTorch SDPA expects (B, nh, T, hs) format
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            # Apply RoPE if provided
            if rope_cos is not None and rope_sin is not None:
                q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
            y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.ln_1(x), rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: Model):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # advanced options derived
        self.shared_token_embeddings: bool = bool(
            getattr(config, "shared_token_embeddings", False)
        )
        self.use_compartment_embeddings: bool = bool(
            getattr(config, "use_compartment_embeddings", False)
        )
        self.embedding_vocab_size: int = int(
            getattr(config, "embedding_vocab_size", config.vocab_size)
        )
        self.base_vocab_size: Optional[int] = getattr(config, "base_vocab_size", None)
        self.max_compartments: Optional[int] = getattr(config, "max_compartments", None)
        self.translation_token_id: Optional[int] = getattr(
            config, "translation_token_id", None
        )
        self.copy_compartment_embeddings: bool = bool(
            getattr(config, "copy_compartment_embeddings", False)
        )
        self.copy_compartment_lm_head: bool = bool(
            getattr(config, "copy_compartment_lm_head", False)
        )
        self.copy_compartment_id_embeddings: bool = bool(
            getattr(config, "copy_compartment_id_embeddings", False)
        )
        # RoPE configuration
        self.use_rope: bool = bool(getattr(config, "use_rope", False))
        rope_base: float = float(getattr(config, "rope_base", 10000.0))

        print(f"embedding_vocab_size: {self.embedding_vocab_size}")
        print(f"use_rope: {self.use_rope}")

        # Build transformer modules - conditionally include wpe based on use_rope
        transformer_modules = dict(
            wte=nn.Embedding(self.embedding_vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        # Only create learned positional embeddings if not using RoPE
        if not self.use_rope:
            transformer_modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)

        # Create RoPE module if enabled
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            self.rotary_emb = RotaryPositionEmbedding(
                dim=head_dim,
                max_seq_len=config.block_size,
                base=rope_base,
            )
        else:
            self.rotary_emb = None
        # Optional compartment embeddings (max_compartments x n_embd)
        if self.use_compartment_embeddings:
            assert self.max_compartments is not None, (
                "max_compartments required when using compartment embeddings"
            )
            self.comp_emb = nn.Embedding(self.max_compartments, config.n_embd)
        else:
            self.comp_emb = None

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ICL dual-stream head + embedding. When enabled, a parallel
        # per-example hashed view of the input is summed with tok_emb at
        # the input stage, and a second lm head produces logits over the
        # icl_vocab_size at the output. Both are no-ops when icl_enabled=False.
        self.icl_enabled: bool = bool(getattr(config, "icl_enabled", False))
        self.icl_vocab_size: int = int(getattr(config, "icl_vocab_size", 0))
        if self.icl_enabled:
            assert self.icl_vocab_size > 0, "icl_vocab_size must be set when icl_enabled"
            self.wte_icl = nn.Embedding(self.icl_vocab_size, config.n_embd)
            self.lm_head_icl = nn.Linear(
                config.n_embd, self.icl_vocab_size, bias=False
            )
        else:
            self.wte_icl = None
            self.lm_head_icl = None

        if config.weight_tying:
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            # incompatible with shared_token_embeddings; enforced in config validation
            cast(nn.Module, self.transformer.wte).weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # If configured: copy the base compartment token embeddings across all compartments
        if (
            not self.shared_token_embeddings
            and self.copy_compartment_embeddings
            and self.base_vocab_size is not None
            and self.max_compartments is not None
        ):
            with torch.no_grad():
                wte_weight: torch.Tensor = cast(
                    nn.Embedding, self.transformer.wte
                ).weight
                total_expected = self.base_vocab_size * self.max_compartments + 1
                if wte_weight.size(0) == total_expected:
                    base = wte_weight[: self.base_vocab_size].clone()
                    for c in range(self.max_compartments):
                        start = c * self.base_vocab_size
                        end = start + self.base_vocab_size
                        wte_weight[start:end].copy_(base)
                else:
                    raise ValueError(
                        f"wte_weight.size(0) != total_expected: {wte_weight.size(0)} != {total_expected}"
                    )

        # Optionally, copy the lm_head rows for base vocab across compartments (no validation/tile needed)
        if (
            not self.shared_token_embeddings
            and self.copy_compartment_lm_head
            and self.base_vocab_size is not None
            and self.max_compartments is not None
        ):
            with torch.no_grad():
                head_w: torch.Tensor = cast(nn.Linear, self.lm_head).weight
                # lm_head shape: [vocab_size, n_embd]
                total_expected = self.base_vocab_size * self.max_compartments + 1
                if head_w.size(0) == total_expected:
                    base = head_w[: self.base_vocab_size].clone()
                    for c in range(self.max_compartments):
                        start = c * self.base_vocab_size
                        end = start + self.base_vocab_size
                        head_w[start:end].copy_(base)
                # else: skip silently per request

        # Optionally, copy compartment 0's ID embedding to all other compartments
        if (
            self.copy_compartment_id_embeddings
            and self.use_compartment_embeddings
            and self.comp_emb is not None
            and self.max_compartments is not None
        ):
            with torch.no_grad():
                base_vec = self.comp_emb.weight[0].clone()
                for c in range(1, self.max_compartments):
                    self.comp_emb.weight[c].copy_(base_vec)

        # DANN: set of layer indices to collect outputs from (set from train.py before compile)
        self._dann_collect_layers: frozenset[int] | None = None

        # report number of parameters
        n_params = self.get_num_params()
        if n_params < 1e6:
            print("number of parameters: %.2fK" % (n_params / 1e3,))
        else:
            print("number of parameters: %.2fM" % (n_params / 1e6,))

    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.use_rope:
            n_params -= cast(nn.Embedding, self.transformer.wpe).weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx,
        targets=None,
        compartment_ids: Optional[torch.Tensor] = None,
        full_sequence_logits: bool = False,
        capture_layer: Optional[int] = None,
        icl_idx: Optional[torch.Tensor] = None,
        icl_targets: Optional[torch.Tensor] = None,
        icl_mask: Optional[torch.Tensor] = None,
        return_last_hidden: bool = False,
    ):
        """Forward.

        Standard return: (logits, loss).
        With DANN: (logits, loss, dann_layer_outputs).
        With capture_layer set: returns dict-form for the alignment hook —
            (None, None, captured_hidden_state)
        where captured_hidden_state is the post-block hidden state at layer
        `capture_layer` (0-indexed), shape (B, T, n_embd). The caller is
        responsible for pooling / loss. We short-circuit the LM head when
        capture_layer is set, so only the trunk through that layer runs.

        With ICL dual-stream enabled (icl_enabled=True at init) and icl_idx
        provided: an additional E_icl[icl_idx] is summed into the input
        embedding. If icl_targets is also provided, returns
        (logits, loss, icl_logits, icl_loss) where loss is the canonical
        next-token loss and icl_loss is the next-ICL-token loss.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        # forward the GPT model itself
        # if using shared token embeddings, map tokens into embedding vocab via modulo for regular tokens
        if self.shared_token_embeddings:
            assert (
                self.base_vocab_size is not None
                and self.translation_token_id is not None
            )
            emb_vocab_last = self.embedding_vocab_size - 1
            # modulo map into base vocab range, then put translation token to last index
            idx_mod = torch.remainder(idx, self.base_vocab_size)
            idx_mod = torch.where(
                idx == self.translation_token_id,
                torch.full_like(idx, emb_vocab_last),
                idx_mod,
            )
            tok_emb = cast(nn.Embedding, self.transformer.wte)(idx_mod)
        else:
            tok_emb = cast(nn.Embedding, self.transformer.wte)(idx)

        # ICL dual-stream with per-position masking: replace canonical
        # embedding with ICL embedding at masked positions (before pos/comp
        # encodings get added). Mask=True → use icl_emb; Mask=False → keep
        # canonical. This is the "stream dropout" variant that forces both
        # tasks to share residual-stream pathways. When icl_mask is None,
        # we fall through to the default summed dual-stream behavior below.
        if (
            self.icl_enabled
            and icl_idx is not None
            and icl_mask is not None
        ):
            icl_emb_for_mask = cast(nn.Embedding, self.wte_icl)(icl_idx)
            m = icl_mask.unsqueeze(-1).to(tok_emb.dtype)
            tok_emb = m * icl_emb_for_mask + (1.0 - m) * tok_emb

        # Position encoding: either learned embeddings or RoPE
        if self.use_rope:
            # RoPE: no positional embedding added to input, applied in attention
            x_input = tok_emb
            rope_cos, rope_sin = cast(RotaryPositionEmbedding, self.rotary_emb)(tok_emb, t)
        else:
            # Learned positional embeddings
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
            pos_emb = cast(nn.Embedding, self.transformer.wpe)(
                pos
            )  # position embeddings of shape (t, n_embd)
            x_input = tok_emb + pos_emb
            rope_cos, rope_sin = None, None

        if self.use_compartment_embeddings and compartment_ids is not None:
            # compartment_ids shape (b, t)
            comp_emb = cast(nn.Embedding, self.comp_emb)(compartment_ids)
            x_input = x_input + comp_emb
        # ICL dual-stream: add the per-example hashed-view embedding.
        # Skip when masking was already applied above — masked variant
        # REPLACES rather than sums.
        if self.icl_enabled and icl_idx is not None and icl_mask is None:
            icl_emb = cast(nn.Embedding, self.wte_icl)(icl_idx)
            x_input = x_input + icl_emb
        x = cast(nn.Dropout, self.transformer.drop)(x_input)
        layer_outputs: dict[int, torch.Tensor] = {}
        for i, block in enumerate(cast(nn.ModuleList, self.transformer.h)):
            x = block(x, rope_cos=rope_cos, rope_sin=rope_sin)
            if self._dann_collect_layers is not None and i in self._dann_collect_layers:
                layer_outputs[i] = x
            if capture_layer is not None and i == capture_layer:
                # short-circuit: return the post-block hidden state for alignment use
                return None, None, x
        x = cast(nn.LayerNorm, self.transformer.ln_f)(x)

        # decide how to compute logits
        if targets is not None or full_sequence_logits:
            logits = self.lm_head(x)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
        # compute loss only if targets are provided
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            loss = None

        # ICL head: produce parallel logits over the ICL vocab and compute the
        # ICL next-token loss when targets are provided.
        icl_logits = None
        icl_loss = None
        if self.icl_enabled:
            if targets is not None or full_sequence_logits:
                icl_logits = cast(nn.Linear, self.lm_head_icl)(x)
            else:
                icl_logits = cast(nn.Linear, self.lm_head_icl)(x[:, [-1], :])
            if icl_targets is not None:
                icl_loss = F.cross_entropy(
                    icl_logits.view(-1, icl_logits.size(-1)),
                    icl_targets.view(-1),
                    ignore_index=-1,
                )

        if self._dann_collect_layers is not None:
            return logits, loss, layer_outputs
        if self.icl_enabled:
            return logits, loss, icl_logits, icl_loss
        if return_last_hidden:
            return logits, loss, x
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size  # pyright: ignore[reportAttributeAccessIssue]
        if not self.use_rope:
            cast(nn.Embedding, self.transformer.wpe).weight = nn.Parameter(
                cast(nn.Embedding, self.transformer.wpe).weight[:block_size]
            )
        for block in cast(nn.ModuleList, self.transformer.h):
            if hasattr(block.attn, "bias"):
                cast(CausalSelfAttention, block.attn).bias = cast(
                    nn.Parameter, cast(CausalSelfAttention, block.attn).bias
                )[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel  # pyright: ignore[reportMissingImports]

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = Model(**config_args)  # pyright: ignore[reportArgumentType]
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # Dense bf16 peak FLOPS, no sparsity, matched against torch's device name.
    # Order matters: "A100" must be tested before the bare fallback, and the
    # more specific SKU strings before their families.
    _BF16_PEAK_FLOPS = (
        ("B200", 2.25e15),
        ("B100", 1.8e15),
        ("H200", 989e12),
        ("H100", 989e12),
        ("A100", 312e12),
        ("L40", 181e12),
        ("A6000", 155e12),
        ("V100", 125e12),   # fp16 tensor cores; V100 has no bf16 path at all
    )

    @classmethod
    def _peak_flops(cls, device_name):
        for key, peak in cls._BF16_PEAK_FLOPS:
            if key in device_name:
                return peak, key
        return None, None

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization against the *current* device's peak.

        This used to divide by a hardcoded 312e12 (A100) regardless of what it
        ran on, so every MFU figure read off a B200 was inflated ~7x and every
        H100 figure ~3x. Since MFU is the number we tune throughput against,
        that made cross-hardware comparisons actively misleading.
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second

        name = ""
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name()
            except Exception:
                name = ""
        flops_promised, matched = self._peak_flops(name)
        if flops_promised is None:
            # An unrecognised device would silently get A100 numbers, which is
            # exactly the bug this replaced. Report NaN so a bad MFU reading is
            # visible in the logs instead of plausible.
            if not getattr(type(self), "_warned_unknown_gpu", False):
                type(self)._warned_unknown_gpu = True
                print(
                    f"[mfu] unknown device {name!r}: no bf16 peak on record, "
                    f"reporting MFU as nan. Add it to GPT._BF16_PEAK_FLOPS."
                )
            return float("nan")
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
