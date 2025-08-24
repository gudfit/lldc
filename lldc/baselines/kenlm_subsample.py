# lldc/baselines/kenlm_subsample.py

from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
from collections import Counter
import os, subprocess, shutil, time
import kenlm, math
from tqdm import tqdm


def _find_kenlm_binary(name: str) -> str | None:
    binary_path = shutil.which(name)
    if binary_path:
        return binary_path
    kenlm_bin_dir = os.environ.get("KENLM_BIN")
    if kenlm_bin_dir:
        path = Path(kenlm_bin_dir) / name
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    default_path = Path("/usr/local/bin") / name
    if default_path.is_file() and os.access(default_path, os.X_OK):
        return str(default_path)
    return None


def train_kenlm(
    sentences: List[str],
    order: int,
    workdir: str,
    pruning: str = "0 1 2",
    memory: str | None = "30%",
    show_progress: bool = True,
) -> Tuple[str, str]:
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    corpus = wd / "train.txt"
    with corpus.open("w", encoding="utf-8") as f:
        itr = (
            tqdm(sentences, desc="Writing corpus", total=len(sentences))
            if show_progress
            else sentences
        )
        for s in itr:
            f.write(s + "\n")

    lmplz_path = _find_kenlm_binary("lmplz")
    build_binary_path = _find_kenlm_binary("build_binary")
    if not lmplz_path or not build_binary_path:
        raise RuntimeError(
            "KenLM binaries ('lmplz', 'build_binary') not found. Please run setup.sh, ensure they are in PATH, or set KENLM_BIN."
        )

    arpa = wd / f"lm{order}.arpa"
    binary = wd / f"lm_{order}.binary"
    mem = memory or os.environ.get("KENLM_MEMORY", "30%")
    tmp = wd / "tmp"
    tmp.mkdir(exist_ok=True)
    prune_parts = [p for p in str(pruning).split() if p]

    cmd_lmplz = [lmplz_path, "-o", str(order), "--discount_fallback"]
    if prune_parts:
        cmd_lmplz += ["--prune", *prune_parts]
    cmd_lmplz += ["--temp_prefix", str(tmp), "--memory", str(mem)]

    with corpus.open("rb") as fin, arpa.open("wb") as fout:
        try:
            subprocess.run(cmd_lmplz, stdin=fin, stdout=fout, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"KenLM lmplz failed (order={order}). {e}") from e

    cmd_bin = [build_binary_path, "trie", "-T", str(tmp), str(arpa), str(binary)]
    try:
        subprocess.run(cmd_bin, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"KenLM build_binary failed. {e}") from e

    return str(arpa), str(binary)


def _load_kenlm(arpa_path: str, binary_path: str | None = None):
    if binary_path and Path(binary_path).exists():
        return kenlm.Model(binary_path)
    return kenlm.Model(arpa_path)


def _build_vocab(docs: List[str], max_vocab: int = 50000) -> List[str]:
    cnt = Counter()
    for t in docs:
        cnt.update(t.split())
    return [w for w, _ in cnt.most_common(max_vocab)]


def _uniform_subsample_words(words: List[str], N: int) -> Tuple[List[str], List[int]]:
    kept = words[::N]
    idxs = list(range(0, len(words), N))
    return kept, idxs


def _state_from_context(model, ctx: List[str]):
    st = kenlm.State()
    model.BeginSentenceWrite(st)
    for w in ctx[-4:]:
        nxt = kenlm.State()
        model.BaseScore(st, w, nxt)
        st = nxt
    return st


def _viterbi_fill(
    model,
    ctx: List[str],
    end_word: str,
    gap_len: int,
    vocab: List[str],
    beam_size: int = 16,
    cand_per_step: int = 200,
) -> List[str]:
    if gap_len <= 0:
        return []
    cand = vocab[:cand_per_step]
    beams = [(_state_from_context(model, ctx), [], 0.0)]
    for _ in range(gap_len):
        nxt = []
        for st, seq, sc in beams:
            for w in cand:
                st2 = kenlm.State()
                inc = model.BaseScore(st, w, st2)
                nxt.append((st2, seq + [w], sc + inc))
        nxt.sort(key=lambda x: x[2], reverse=True)
        beams = nxt[:beam_size]
    best = None
    for st, seq, sc in beams:
        st2 = kenlm.State()
        inc = model.BaseScore(st, end_word, st2)
        if best is None or (sc + inc) > best[2]:
            best = (st2, seq, sc + inc)
    return best[1] if best else []


def subsample_and_reconstruct_kenlm5(
    test_texts: List[str],
    train_texts: List[str],
    rates: List[int],
    workdir: str,
    beam_size: int = 16,
    cand_per_step: int = 200,
    max_vocab: int = 50000,
) -> List[Dict]:
    arpa, binary = train_kenlm(train_texts, order=5, workdir=workdir)
    lm = _load_kenlm(arpa, binary)
    vocab = _build_vocab(train_texts, max_vocab=max_vocab)
    outputs = []
    for N in rates:
        recons, payloads = [], []
        for t in test_texts:
            words = t.split()
            kept, idxs = _uniform_subsample_words(words, N)
            payloads.append(" ".join(kept))
            out: List[str] = []
            for j, i in enumerate(idxs):
                out.append(words[i])
                if j + 1 < len(idxs):
                    gap = max(0, idxs[j + 1] - i - 1)
                    if gap > 0:
                        fill = _viterbi_fill(
                            lm,
                            out[:],
                            words[idxs[j + 1]],
                            gap,
                            vocab,
                            beam_size=beam_size,
                            cand_per_step=cand_per_step,
                        )
                        out.extend(fill)
            recons.append(" ".join(out))
        outputs.append(
            {"rate_N": N, "reconstructions": recons, "subsamples_payload": payloads}
        )
    return outputs


def kenlm_ngram_bpc(
    train_texts: List[str],
    test_texts: List[str],
    workdir: str,
    order: int = 8,
    pruning: str = "0 1 2",
) -> Tuple[float, float]:
    arpa, binary = train_kenlm(
        train_texts, order=order, workdir=workdir, pruning=pruning
    )
    lm = _load_kenlm(arpa, binary)
    total_log10 = 0.0
    total_chars = 0
    t0 = time.perf_counter()
    for t in test_texts:
        total_log10 += lm.score(t, bos=True, eos=True)
        total_chars += len(t)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    bits = -total_log10 / math.log10(2.0)
    bpc = bits / max(1, total_chars)
    avg_decode_ms = dt_ms / max(1, len(test_texts))
    return bpc, avg_decode_ms
