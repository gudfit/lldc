# lldc/metrics/crumpled_paper.py

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    PreTrainedTokenizerBase,
)


@dataclass
class Oracle:
    name: str = "gpt2-large"
    device: str = "cuda"

    def __post_init__(self):
        self.tok = AutoTokenizer.from_pretrained(self.name, use_fast=True)
        self.m = AutoModelForCausalLM.from_pretrained(self.name).to(self.device).eval()
        self.vocab_size = int(getattr(self.m.config, "vocab_size", len(self.tok)))

    @torch.no_grad()
    def surprisal_bits(self, text: str) -> Tuple[List[int], List[float]]:
        model_max_length = self.m.config.max_position_embeddings
        enc = self.tok(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=model_max_length,
        ).to(self.device)
        ids = enc["input_ids"][0]
        logits = self.m(input_ids=ids.unsqueeze(0)).logits[0]
        if ids.size(0) < 2:
            return ids.tolist(), []
        logp = torch.log_softmax(logits[:-1], dim=-1)
        true_next = ids[1:].unsqueeze(-1)
        nats = -logp.gather(-1, true_next).squeeze(-1)
        bits = (nats / math.log(2.0)).tolist()
        return ids.tolist(), [float(x) for x in bits]


class OracleEnsemble:
    def __init__(
        self,
        model_names: Iterable[str] = ("gpt2-large",),
        device: str = "cuda",
        measuring_tokenizer: str | None = "oracle",
    ):
        names = list(model_names) if model_names else ["gpt2-large"]
        self.device = device
        self._members: List[tuple[PreTrainedTokenizerBase, AutoModelForCausalLM]] = []
        for nm in names:
            tok = AutoTokenizer.from_pretrained(nm, use_fast=True)
            m = AutoModelForCausalLM.from_pretrained(nm).to(device).eval()
            self._members.append((tok, m))
        if measuring_tokenizer and measuring_tokenizer != "oracle":
            self.measure_tok = AutoTokenizer.from_pretrained(
                measuring_tokenizer, use_fast=True
            )
        else:
            self.measure_tok = self._members[0][0]
        self.vocab_size = int(
            getattr(
                getattr(self._members[0][1], "config", None),
                "vocab_size",
                len(self.measure_tok),
            )
        )

    @torch.no_grad()
    def surprisal_bits(self, text: str) -> Tuple[List[int], List[float]]:
        if not text:
            return [], []
        L_chars = len(text)
        if L_chars == 0:
            return [], []
        acc = torch.zeros(L_chars, dtype=torch.float32, device="cpu")
        cnt = torch.zeros(L_chars, dtype=torch.float32, device="cpu")
        for tok, model in self._members:
            model_max_length = model.config.max_position_embeddings
            enc = tok(
                text,
                add_special_tokens=False,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=model_max_length,
            )
            if "offset_mapping" not in enc:
                raise RuntimeError(
                    f"Fast tokenizer with offset mapping required for {getattr(tok, 'name_or_path', 'oracle')}."
                )
            ids = enc["input_ids"][0].to(model.device)
            offs = enc["offset_mapping"][0].tolist()
            if ids.numel() == 0:
                continue
            logits = model(input_ids=ids.unsqueeze(0)).logits[0]
            if ids.size(0) < 2:
                continue
            logp = F.log_softmax(logits[:-1], dim=-1)
            true_next = ids[1:].unsqueeze(-1)
            nats = -logp.gather(-1, true_next).squeeze(-1)
            bits = (nats / math.log(2.0)).detach().cpu().tolist()
            for j in range(len(bits)):
                if j + 1 < len(offs):
                    s, e = offs[j + 1]
                    s = int(max(0, s))
                    e = int(min(L_chars, e))
                    if e > s:
                        acc[s:e] += float(bits[j])
                        cnt[s:e] += 1.0
        avg_chars = torch.zeros_like(acc)
        mask = cnt > 0
        avg_chars[mask] = acc[mask] / cnt[mask]
        measure_tok_max_length = self._members[0][1].config.max_position_embeddings
        m_enc = self.measure_tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=measure_tok_max_length,
        )
        m_ids = m_enc["input_ids"]
        m_offs = m_enc["offset_mapping"]
        m_bits: List[float] = []
        for j in range(1, len(m_ids)):
            s, e = m_offs[j]
            s = int(max(0, s))
            e = int(min(L_chars, e))
            if e > s:
                seg = avg_chars[s:e]
                m_bits.append(float(seg.mean().item()))
            else:
                m_bits.append(0.0)
        return list(m_ids), m_bits


def _needleman_wunsch_cost(
    a_s: List[float], b_s: List[float], gap_cost: float
) -> Tuple[float, List[Tuple[int, int]]]:
    n, m = len(a_s), len(b_s)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        bt[i][0] = (i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j * gap_cost
        bt[0][j] = (0, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c_sub = dp[i - 1][j - 1] + abs(a_s[i - 1] - b_s[j - 1])
            c_del = dp[i - 1][j] + gap_cost
            c_ins = dp[i][j - 1] + gap_cost
            if c_sub <= c_del and c_sub <= c_ins:
                dp[i][j] = c_sub
                bt[i][j] = (i - 1, j - 1)
            elif c_del <= c_ins:
                dp[i][j] = c_del
                bt[i][j] = (i - 1, j)
            else:
                dp[i][j] = c_ins
                bt[i][j] = (i, j - 1)
    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj = bt[i][j]
        if pi == i - 1 and pj == j - 1:
            path.append((i - 1, j - 1))
            i, j = pi, pj
        elif pi == i - 1 and pj == j:
            path.append((i - 1, -1))
            i, j = pi, pj
        else:
            path.append((-1, j - 1))
            i, j = pi, pj
    path.reverse()
    return dp[n][m], path


def _needleman_wunsch_tokens(
    ids_a: List[int],
    ids_b: List[int],
    gap_cost: float = 1.0,
    mismatch_cost: float = 1.0,
) -> List[Tuple[int, int]]:
    n, m = len(ids_a), len(ids_b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        bt[i][0] = (i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j * gap_cost
        bt[0][j] = (0, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0.0 if ids_a[i - 1] == ids_b[j - 1] else mismatch_cost
            c_sub = dp[i - 1][j - 1] + sub
            c_del = dp[i - 1][j] + gap_cost
            c_ins = dp[i][j - 1] + gap_cost
            if c_sub <= c_del and c_sub <= c_ins:
                dp[i][j] = c_sub
                bt[i][j] = (i - 1, j - 1)
            elif c_del <= c_ins:
                dp[i][j] = c_del
                bt[i][j] = (i - 1, j)
            else:
                dp[i][j] = c_ins
                bt[i][j] = (i, j - 1)
    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj = bt[i][j]
        if pi == i - 1 and pj == j - 1:
            path.append((i - 1, j - 1))
            i, j = pi, pj
        elif pi == i - 1 and pj == j:
            path.append((i - 1, -1))
            i, j = pi, pj
        else:
            path.append((-1, j - 1))
            i, j = pi, pj
    path.reverse()
    return path


def tcm_pcm_from_texts(
    orig: str, recon: str, oracle: Oracle | OracleEnsemble
) -> Dict[str, float]:
    _, s0 = oracle.surprisal_bits(orig)
    _, s1 = oracle.surprisal_bits(recon)
    return tcm_pcm_from_surprisal(s0, s1, vocab_size=oracle.vocab_size)


def tcm_pcm_from_surprisal(
    s0: List[float], s1: List[float], vocab_size: int
) -> Dict[str, float]:
    gap = math.log2(max(2, vocab_size))
    _, path = _needleman_wunsch_cost(s0, s1, gap_cost=gap)
    tcm = 0.0
    pcm = 0.0
    for i, j in path:
        if i == -1 or j == -1:
            c = gap
        else:
            c = abs(s0[i] - s1[j])
        tcm += c
        pcm = max(pcm, c)
    return {"tcm_bits": float(tcm), "pcm_bits": float(pcm)}


@torch.no_grad()
def _pll_log2_at_pos(
    model: AutoModelForMaskedLM,
    tok: PreTrainedTokenizerBase,
    seq_ids: torch.LongTensor,
    pos: int,
    target_id: int,
) -> float:
    device = next(model.parameters()).device
    ids = seq_ids.clone().to(device)
    if pos < 0 or pos >= ids.numel():
        return 0.0
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise RuntimeError(
            "Bi-directional oracle tokenizer must define a [MASK] token."
        )
    ids[pos] = int(mask_id)
    out = model(input_ids=ids.unsqueeze(0)).logits[0, pos]
    logp = F.log_softmax(out, dim=-1)[int(target_id)]
    return float(logp.item() / math.log(2.0))


def delta_log_likelihood_bits(
    original_text: str,
    reconstructed_text: str,
    bi_model: AutoModelForMaskedLM,
    bi_tokenizer: PreTrainedTokenizerBase,
) -> float:
    device = next(bi_model.parameters()).device
    encA = bi_tokenizer(original_text, add_special_tokens=False, return_tensors="pt")
    encB = bi_tokenizer(
        reconstructed_text, add_special_tokens=False, return_tensors="pt"
    )
    idsA = encA["input_ids"][0].to(device)
    idsB = encB["input_ids"][0].to(device)
    a_ids_list: List[int] = [int(x) for x in idsA.tolist()]
    b_ids_list: List[int] = [int(x) for x in idsB.tolist()]
    path = _needleman_wunsch_tokens(
        a_ids_list, b_ids_list, gap_cost=1.0, mismatch_cost=1.0
    )
    diffs: List[float] = []
    for i, j in path:
        if i < 0 or j < 0:
            continue
        tok_id = a_ids_list[i]
        pll_orig = _pll_log2_at_pos(bi_model, bi_tokenizer, idsA, i, tok_id)
        pll_recon_ctx = _pll_log2_at_pos(bi_model, bi_tokenizer, idsB, j, tok_id)
        diffs.append(pll_orig - pll_recon_ctx)
    if not diffs:
        return 0.0
    return float(sum(diffs) / len(diffs))
