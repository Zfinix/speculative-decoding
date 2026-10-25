import random
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from spec import BIG


def pass_time(model, cache, tokens, runs=30):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        mx.eval(model(mx.array([tokens]), cache=cache))
        times.append(time.perf_counter() - start)
        trim_prompt_cache(cache, len(tokens))
    return statistics.median(times) * 1000


def one_vs_many():
    big, tok = load(BIG)
    cache = make_prompt_cache(big)
    context = tok.encode("The quick brown fox jumps over the lazy dog. " * 40)
    mx.eval(big(mx.array([context]), cache=cache))
    for n in (1, 2, 5, 8, 16, 32):
        print(f"{n:3d} tokens in one pass  {pass_time(big, cache, context[:n]):6.1f} ms")


def sampling_rule(draws=1_000_000):
    tokens = ["Paris", "the", "a", "France", "it"]
    p = [0.60, 0.25, 0.08, 0.05, 0.02]
    q = [0.30, 0.45, 0.15, 0.02, 0.08]
    leftover = [max(0, a - b) for a, b in zip(p, q)]
    leftover = [x / sum(leftover) for x in leftover]
    counts = dict.fromkeys(tokens, 0)
    for _ in range(draws):
        i = random.choices(range(len(tokens)), q)[0]
        if random.random() >= min(1, p[i] / q[i]):
            i = random.choices(range(len(tokens)), leftover)[0]
        counts[tokens[i]] += 1
    print(f"{'token':8s} {'big model':>10s} {'with draft':>11s}")
    for t, want in zip(tokens, p):
        print(f"{t:8s} {want:10.3f} {counts[t] / draws:11.3f}")


if __name__ == "__main__":
    sampling_rule()
    one_vs_many()
