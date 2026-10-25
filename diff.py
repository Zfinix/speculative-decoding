import sys

from mlx_lm import load

from spec import BIG, DRAFT, plain, speculative

QUESTION = "Write a Python function that checks if a string is a palindrome."


def main(n=200):
    big, tok = load(BIG)
    prompt = tok.apply_chat_template([{"role": "user", "content": QUESTION}], add_generation_prompt=True)
    if sys.argv[1] == "plain":
        out = plain(big, prompt, n)
    else:
        draft, _ = load(DRAFT)
        out = speculative(big, draft, prompt, n)[0]
    print(tok.decode(out))


if __name__ == "__main__":
    main()
