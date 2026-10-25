import time

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from spec import BIG, picks, plain

FUNCTION = '''def summarize(orders):
    t = 0
    count = 0
    for order in orders:
        if order["status"] == "paid":
            t += order["amount"]
            count += 1
    average = t / count if count else 0
    return {"total": t, "count": count, "average": average}'''

QUESTION = f"Rename the variable t to total in this function. Reply with only the full function.\n\n{FUNCTION}"


def look_back(tokens, k, longest=3):
    for n in range(longest, 0, -1):
        tail = tokens[-n:]
        for start in range(len(tokens) - n - 1, -1, -1):
            if tokens[start:start + n] == tail:
                return tokens[start + n:start + n + k]
    return []


def lookup_speculative(big, prompt, n, k=8):
    cache = make_prompt_cache(big)
    out = [picks(big, prompt, cache)[-1]]
    rounds = kept = guessed = 0
    while len(out) < n:
        guesses = look_back(prompt + out, k)
        checked = picks(big, out[-1:] + guesses, cache)
        accepted = 0
        while accepted < len(guesses) and guesses[accepted] == checked[accepted]:
            accepted += 1
        out += guesses[:accepted] + [checked[accepted]]
        trim_prompt_cache(cache, len(guesses) - accepted)
        rounds += 1
        kept += accepted
        guessed += len(guesses)
    return out[:n], kept / max(guessed, 1), rounds


def main(n=76):
    big, tok = load(BIG)
    prompt = tok.apply_chat_template([{"role": "user", "content": QUESTION}], add_generation_prompt=True)
    plain(big, prompt, 8)
    start = time.perf_counter()
    base = plain(big, prompt, n)
    t_plain = time.perf_counter() - start
    print(tok.decode(base))
    print(f"plain        {n / t_plain:6.1f} tokens/s")
    for k in (2, 4, 8):
        start = time.perf_counter()
        fast, rate, rounds = lookup_speculative(big, prompt, n, k)
        t_fast = time.perf_counter() - start
        same = "identical" if base == fast else "differs"
        print(f"look back {k}  {n / t_fast:6.1f} tokens/s   {t_plain / t_fast:.2f}x   kept {rate:.2f}   "
              f"tokens per pass {n / rounds:.2f}   {same}")


if __name__ == "__main__":
    main()
