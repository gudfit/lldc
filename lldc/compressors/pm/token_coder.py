# lldc/compressors/pm/token_coder.py

from __future__ import annotations
from typing import List, Tuple
import torch


@torch.no_grad()
def encode_kept_stream_with_oracle(
    kept_text: str,
    oracle_tok,
    oracle_model,
) -> tuple[List[int], List[List[float]], int]:
    """
    Walk kept tokens left→right.
    At step t:
      - record P(sym_t | sym_<t>) using KV cache (teacher-forced on kept-only stream),
      - then feed true sym_t to advance cache.
    """
    device = next(oracle_model.parameters()).device
    vocab_size = int(getattr(oracle_model.config, "vocab_size", len(oracle_tok)))
    if not kept_text:
        return [], [], vocab_size

    enc = oracle_tok(kept_text, return_tensors="pt", add_special_tokens=False)
    kept_ids = enc["input_ids"][0].to(device)
    start_id = oracle_tok.bos_token_id or oracle_tok.eos_token_id or 0

    out = oracle_model(input_ids=torch.tensor([[start_id]], device=device))
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)

    symbols, probs_list = [], []
    for t in range(kept_ids.numel()):
        probs_list.append(probs.squeeze(0).detach().cpu().tolist())
        sym = int(kept_ids[t].item())
        symbols.append(sym)
        step_out = oracle_model(
            input_ids=kept_ids[t : t + 1].view(1, 1),
            past_key_values=past,
            use_cache=True,
        )
        past = step_out.past_key_values
        logits = step_out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

    return symbols, probs_list, vocab_size
