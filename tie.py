import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from spec import BIG, CORAL, DRAFT, RESET, plain, speculative

QUESTION = "Explain in simple words why the sky is blue."


def scores(big, prompt, out):
    cache = make_prompt_cache(big)
    logits = big(mx.array([prompt]), cache=cache)
    for t in out:
        logits = big(mx.array([[t]]), cache=cache)
    return logits[0, -1]


def main(n=200, k=4):
    big, tok = load(BIG)
    draft, _ = load(DRAFT)
    prompt = tok.apply_chat_template([{"role": "user", "content": QUESTION}], add_generation_prompt=True)
    base = plain(big, prompt, n)
    fast = speculative(big, draft, prompt, n, k)[0]
    if base == fast:
        print("identical")
        return
    i = next(i for i, (a, b) in enumerate(zip(base, fast)) if a != b)
    print(tok.decode(base[:i]))
    print()
    print(f"first difference at token {i}")
    print(f"plain        {CORAL}{tok.decode(base[i:i + 6])!r}{RESET}")
    print(f"speculative  {CORAL}{tok.decode(fast[i:i + 6])!r}{RESET}")
    print()
    logits = scores(big, prompt, base[:i])
    print(f"big model's top three at token {i}")
    for t in mx.argsort(-logits)[:3].tolist():
        print(f"  {tok.decode([t])!r:10s} {logits[t].item():7.3f}")


if __name__ == "__main__":
    main()
