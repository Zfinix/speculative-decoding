import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

BIG = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DRAFT = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
GREEN, CORAL, RESET = "\033[38;2;113;208;131m", "\033[38;2;255;92;92m", "\033[0m"


def picks(model, tokens, cache):
    logits = model(mx.array([tokens]), cache=cache)
    return mx.argmax(logits[0], axis=-1).tolist()


def plain(big, prompt, n):
    cache = make_prompt_cache(big)
    out = [picks(big, prompt, cache)[-1]]
    while len(out) < n:
        out.append(picks(big, out[-1:], cache)[-1])
    return out


def speculative(big, draft, prompt, n, k=4, trace=None):
    big_cache, draft_cache = make_prompt_cache(big), make_prompt_cache(draft)
    out = [picks(big, prompt, big_cache)[-1]]
    pending = prompt + out
    rounds = kept = 0
    while len(out) < n:
        guesses = []
        for _ in range(k):
            guesses.append(picks(draft, pending, draft_cache)[-1])
            pending = guesses[-1:]

        checked = picks(big, out[-1:] + guesses, big_cache)
        accepted = 0
        while accepted < k and guesses[accepted] == checked[accepted]:
            accepted += 1

        out += guesses[:accepted] + [checked[accepted]]
        trim_prompt_cache(big_cache, k - accepted)
        if accepted == k:
            pending = guesses[-1:] + out[-1:]
        else:
            trim_prompt_cache(draft_cache, k - 1 - accepted)
            pending = out[-1:]

        rounds += 1
        kept += accepted
        if trace:
            trace(guesses[:accepted], checked[accepted])
    return out[:n], kept / (rounds * k), rounds


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "Write a Python function that checks if a string is a palindrome."
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    big, tok = load(BIG)
    draft, _ = load(DRAFT)
    prompt = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True)

    plain(big, prompt, 8)
    speculative(big, draft, prompt, 8)

    def show(kept, fixed):
        print(GREEN + tok.decode(kept) + CORAL + tok.decode([fixed]) + RESET, end="", flush=True)

    base, t_plain = timed(lambda: plain(big, prompt, n))
    (fast, rate, rounds), t_spec = timed(lambda: speculative(big, draft, prompt, n, k, trace=show))
    print()
    print(f"plain        {n / t_plain:6.1f} tokens/s")
    print(f"speculative  {n / t_spec:6.1f} tokens/s   {t_plain / t_spec:.2f}x")
    print(f"acceptance   {rate:.2f}   tokens per big pass {n / rounds:.2f}")
    print("identical" if base == fast else f"differs at token {next(i for i, (a, b) in enumerate(zip(base, fast)) if a != b)}")


if __name__ == "__main__":
    main()
