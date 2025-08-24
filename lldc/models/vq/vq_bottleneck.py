# lldc/models/vq/vq_bottleneck.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Sequence, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class VQOutput:
    z_q: torch.Tensor
    indices: torch.LongTensor
    commit_loss: torch.Tensor
    codebook_loss: torch.Tensor

class VectorQuantiser(nn.Module):
    def __init__(self, codebook_size: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim) * 0.02)
    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> VQOutput:
        B, T, D = h.shape
        flat = h.reshape(-1, D)
        cb = self.codebook
        h2 = (flat**2).sum(-1, keepdim=True)
        c2 = (cb**2).sum(-1)
        dist = h2 + c2 - 2.0 * flat @ cb.t()
        indices = torch.argmin(dist, dim=-1)
        z = cb[indices].view(B, T, D)
        if mask is not None:
            m = mask.to(h.dtype).view(B, T, 1)
            denom = m.sum() * D + 1e-8
            commit_loss = (((h - z.detach()) ** 2) * m).sum() / denom
            codebook_loss = (((h.detach() - z) ** 2) * m).sum() / denom
        else:
            commit_loss = F.mse_loss(h, z.detach())
            codebook_loss = F.mse_loss(h.detach(), z)
        z_q = h + (z - h).detach()
        return VQOutput(z_q=z_q, indices=indices.view(B, T), commit_loss=commit_loss, codebook_loss=codebook_loss)

class VQBottleneckWrapper(nn.Module):
    def __init__(self, lm: nn.Module, layer_after: Union[int, Sequence[int]], codebook_size: int, beta: float = 0.25):
        super().__init__()
        self.lm = lm
        if isinstance(layer_after, (list, tuple)):
            la = [int(x) for x in layer_after]
        else:
            la = [int(layer_after)]
        self.layers_after: List[int] = sorted(list({max(0, int(x)) for x in la}))
        try:
            D = lm.config.n_embd
        except Exception:
            D = lm.base_model.model.config.n_embd
        self.vq = VectorQuantiser(codebook_size=codebook_size, dim=D, beta=beta)
        with torch.no_grad():
            dtype = next(lm.parameters()).dtype
            self.vq.codebook.data = self.vq.codebook.data.to(dtype)
        self.use_checkpoint = True
    def _get_transformer(self):
        return self.lm.transformer if hasattr(self.lm, "transformer") else self.lm.base_model.transformer
    def _apply_block(self, block, x_in, attn):
        out = block(x_in, attention_mask=attn)
        return out[0] if isinstance(out, tuple) else out
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None, labels: torch.LongTensor | None = None):
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        model = self.lm
        transformer = self._get_transformer()
        extended_attention_mask: torch.Tensor | None = None
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=next(model.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(extended_attention_mask.dtype).min
        wte, wpe, drop, h = transformer.wte, transformer.wpe, transformer.drop, transformer.h
        device = input_ids.device
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = wte(input_ids) + wpe(pos)
        x = drop(x)
        commit_total = 0.0
        codebook_total = 0.0
        indices_all: List[torch.LongTensor] = []
        cur_layer = 0
        la_list = self.layers_after
        def maybe_ck(block, xin):
            if self.training and self.use_checkpoint:
                return checkpoint(lambda t: self._apply_block(block, t, extended_attention_mask), xin)
            return self._apply_block(block, xin, extended_attention_mask)
        for target in la_list:
            while cur_layer < min(target, len(h)):
                x = maybe_ck(h[cur_layer], x)
                cur_layer += 1
            vqo = self.vq(x, mask=attention_mask if attention_mask is not None else None)
            x = vqo.z_q
            indices_all.append(vqo.indices)
            commit_total = commit_total + vqo.commit_loss
            codebook_total = codebook_total + vqo.codebook_loss
        while cur_layer < len(h):
            x = maybe_ck(h[cur_layer], x)
            cur_layer += 1
        ln_f = transformer.ln_f
        logits = model.lm_head(ln_f(x))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = ce + codebook_total + self.vq.beta * commit_total
        main_indices = indices_all[-1] if indices_all else torch.empty((B, T), dtype=torch.long, device=input_ids.device)
        return {"logits": logits, "loss": loss, "vq_commit": torch.as_tensor(commit_total), "vq_codebook": torch.as_tensor(codebook_total), "indices": main_indices, "indices_all": indices_all}
    @torch.no_grad()
    def decode_from_indices(self, indices: torch.LongTensor, attention_mask: torch.Tensor | None = None, depth_index: int = -1) -> torch.LongTensor:
        model = self.lm
        transformer = self._get_transformer()
        h = transformer.h
        extended_attention_mask: torch.Tensor | None = None
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=next(model.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(extended_attention_mask.dtype).min
        start_layer = self.layers_after[depth_index] if self.layers_after else 0
        xq = self.vq.codebook[indices.view(-1)].view(indices.size(0), indices.size(1), -1)
        def _apply(block, xin):
            out = block(xin, attention_mask=extended_attention_mask)
            return out[0] if isinstance(out, tuple) else out
        for i in range(start_layer, len(h)):
            xq = _apply(h[i], xq)
        ln_f = transformer.ln_f
        logits = model.lm_head(ln_f(xq))
        pred = logits.argmax(dim=-1)
        return pred

