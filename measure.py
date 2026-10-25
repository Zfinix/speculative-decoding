import json
import statistics
import time

from mlx_lm import load

from bench import one_vs_many
from spec import BIG, DRAFT, plain, speculative

PROMPTS = {
    "code": "Write a Python function that checks if a string is a palindrome.",
    "poem": "Write a short poem about New York at night.",
    "english": "Explain in simple words why the sky is blue.",
}


def median_time(fn, runs=3):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, statistics.median(times)


def main(n=200):
    big, tok = load(BIG)
    draft, _ = load(DRAFT)
    results = {}
    for name, question in PROMPTS.items():
        prompt = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True)
        plain(big, prompt, 8)
        base, t_plain = median_time(lambda: plain(big, prompt, n))
        row = {"plain_tps": n / t_plain}
        for k in (2, 4, 8):
            (fast, rate, rounds), t_spec = median_time(lambda: speculative(big, draft, prompt, n, k=k))
            row[f"k{k}"] = {"tps": n / t_spec, "speedup": t_plain / t_spec, "acceptance": rate,
                            "tokens_per_pass": n / rounds, "identical": base == fast}
        results[name] = row
        print(name, json.dumps(row, indent=1))
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    one_vs_many()
    main()
