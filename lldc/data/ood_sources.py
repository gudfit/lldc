# lldc/data/ood_sources.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import random


def _nCr(n: int, r: int) -> int:
    from math import comb

    r = max(0, min(r, n))
    return comb(n, r)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


@dataclass
class QAPair:
    q: str
    a: str
    meta: Dict[str, str]


def _kid_arith(r: random.Random) -> QAPair:
    a, b = r.randint(10, 999), r.randint(10, 999)
    q = f"Solve: {a} + {b} = ? Explain each step in words. Finish with the final number."
    return QAPair(q=q, a=str(a + b), meta={"topic": "arithmetic", "level": "kid"})


def _mid_mult_div(r: random.Random) -> QAPair:
    a, b = r.randint(12, 99), r.randint(3, 19)
    if r.random() < 0.5:
        q = f"Solve: {a} × {b} = ? Show your steps."
        ans = a * b
    else:
        prod = a * b
        q = f"Solve: {prod} ÷ {a} = ? Show your steps."
        ans = b
    return QAPair(q=q, a=str(ans), meta={"topic": "arithmetic", "level": "middle"})


def _algebra_linear(r: random.Random) -> QAPair:
    a = r.randint(2, 9)
    b = r.randint(-20, 20)
    x = r.randint(-10, 10)
    c = a * x + b
    q = f"Solve for x: {a}x + {b} = {c}. Show your steps and give the final value of x."
    return QAPair(q=q, a=str(x), meta={"topic": "algebra", "level": "hs"})


def _algebra_system(r: random.Random) -> QAPair:
    x, y = r.randint(-5, 5), r.randint(-5, 5)
    a, b = r.randint(1, 6), r.randint(1, 6)
    c, d = r.randint(1, 6), r.randint(1, 6)
    e = a * x + b * y
    f = c * x + d * y
    q = (
        f"Solve the system:\n"
        f"{a}x + {b}y = {e}\n"
        f"{c}x + {d}y = {f}\n"
        f"Explain steps; give the ordered pair (x, y)."
    )
    return QAPair(
        q=q, a=f"({x}, {y})", meta={"topic": "algebra_systems", "level": "hs"}
    )


def _probability_combo(r: random.Random) -> QAPair:
    n = r.randint(4, 10)
    k = r.randint(2, min(4, n))
    q = f"How many ways to choose {k} objects out of {n} distinct objects? Give a number."
    return QAPair(
        q=q, a=str(_nCr(n, k)), meta={"topic": "combinatorics", "level": "hs"}
    )


def _number_theory_gcd(r: random.Random) -> QAPair:
    a, b = r.randint(50, 300), r.randint(50, 300)
    q = f"Compute gcd({a}, {b}). Provide the integer result."
    return QAPair(
        q=q, a=str(_gcd(a, b)), meta={"topic": "number_theory", "level": "hs"}
    )


def _calculus_derivative_eval(r: random.Random) -> QAPair:
    a, b, c = [r.randint(-5, 6) or 1 for _ in range(3)]
    t = r.randint(-5, 5)
    ans = 3 * a * t * t + 2 * b * t + c
    q = (
        f"Let f(x) = {a}x^3 + {b}x^2 + {c}x + 7. "
        f"Compute f'({t}). Show the differentiation and the final value."
    )
    return QAPair(
        q=q, a=str(ans), meta={"topic": "calculus_derivative", "level": "univ"}
    )


def _calculus_def_integral_poly(r: random.Random) -> QAPair:
    a, b = r.randint(-4, 6) or 1, r.randint(-8, 8)
    m, n = sorted([r.randint(-5, 5), r.randint(-5, 5)])
    val = 0.5 * a * (n * n - m * m) + b * (n - m)
    q = (
        f"Compute the definite integral ∫_{m}^{n} ({a}x + {b}) dx. "
        f"Show antiderivative and evaluation; give a numeric answer."
    )
    if abs(val - round(val)) < 1e-9:
        ans = str(int(round(val)))
    else:
        ans = str(val)
    return QAPair(q=q, a=ans, meta={"topic": "calculus_integral", "level": "univ"})


def _geometry_area(r: random.Random) -> QAPair:
    w, h = r.randint(3, 40), r.randint(3, 40)
    q = f"A rectangle has width {w} and height {h}. What is its area?"
    return QAPair(q=q, a=str(w * h), meta={"topic": "geometry_area", "level": "ms"})


def generate_math_qa(n: int, seed: int) -> List[QAPair]:
    r = random.Random(seed)
    makers = [
        _kid_arith,
        _mid_mult_div,
        _algebra_linear,
        _algebra_system,
        _probability_combo,
        _number_theory_gcd,
        _geometry_area,
        _calculus_derivative_eval,
        _calculus_def_integral_poly,
    ]
    out: List[QAPair] = []
    for i in range(n):
        q = makers[i % len(makers)](r)
        out.append(q)
    r.shuffle(out)
    return out


_BUGGY_TEMPLATES = [
    (
        "square_then_add_three",
        'def f_{i}(x):\n    """Return x^2 + 3."""\n    return xx + 3\n',
        lambda x: x * x + 3,
    ),
    (
        "cube_minus_one",
        'def f_{i}(x):\n    """Return x^3 - 1."""\n    return x*x - 1  # BUG: not cube\n',
        lambda x: x * x * x - 1,
    ),
    (
        "abs_plus_seven",
        'def f_{i}(x):\n    """Return |x| + 7."""\n    return x + 7  # BUG: missing abs\n',
        lambda x: abs(x) + 7,
    ),
    (
        "sum_first_n",
        'def f_{i}(x):\n    """Return the sum 1+2+...+x for integer x>=1."""\n    s = 0\n    for k in range(x):\n        s += k  # BUG: off by one; misses x\n    return s\n',
        lambda x: x * (x + 1) // 2,
    ),
    (
        "factorial",
        'def f_{i}(x):\n    """Return x! for integer x>=0."""\n    if x == 0:\n        return 0  # BUG: 0! should be 1\n    out = 1\n    for k in range(1, x):  # BUG: stops early\n        out *= k\n    return out\n',
        lambda x: 1 if x == 0 else __import__("math").prod(range(1, x + 1)),
    ),
]


def generate_code_qa(n: int, seed: int) -> List[QAPair]:
    r = random.Random(seed)
    items: List[QAPair] = []
    for i in range(n):
        name, tmpl, correct = _BUGGY_TEMPLATES[i % len(_BUGGY_TEMPLATES)]
        v = r.randint(-8, 8)
        code = tmpl.replace("{i}", str(i))
        q = (
            "You are given a buggy Python function. "
            "First, fix the bug according to the docstring; then compute its return value.\n"
            f"{code}\n"
            f"Question: After fixing, what is f_{i}({v})? "
            "Explain briefly and give the final integer."
        )
        a = str(int(correct(v)))
        items.append(QAPair(q=q, a=a, meta={"topic": f"code_{name}", "level": "mixed"}))
    r.shuffle(items)
    return items


def to_text_only(items: List[QAPair]) -> List[str]:
    return [it.q for it in items]


def to_jsonl_dicts(items: List[QAPair]) -> List[Dict]:
    return [{"q": it.q, "a": it.a, "meta": it.meta} for it in items]
