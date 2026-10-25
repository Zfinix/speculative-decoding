# Speculative decoding in about sixty lines

The code from the Inference Junkie video on speculative decoding. A small draft model guesses the next few tokens, the big model checks all of them in one pass, and the output is the same as the big model on its own.

Everything runs on Apple silicon with [MLX](https://github.com/ml-explore/mlx). The models are Qwen2.5-7B-Instruct (big) and Qwen2.5-0.5B-Instruct (draft), both 4-bit from `mlx-community`. They share a tokenizer, which speculative decoding needs.

## Run it

```sh
uv run python spec.py                                   # code prompt, 200 tokens, 4 guesses a round
uv run python spec.py "Write a short poem about New York at night."
uv run python spec.py "Write a Python function that checks if a string is a palindrome." 200 2
uv run python lookup.py                                 # no draft model: guesses copied from the context
uv run python bench.py                                  # the sampling rule over 1M draws, and pass time for 1 to 32 tokens
uv run python measure.py                                # the full sweep, written to results.json
```

The first run downloads about 4.6 GB of weights.

| file | what |
| --- | --- |
| `spec.py` | greedy speculative decoding with a draft model, plus a plain decoder to compare against |
| `lookup.py` | the same loop with the draft model swapped for a look-back search of the conversation |
| `bench.py` | the accept/reject rule for sampling, simulated; one pass of 1, 2, 5, 8, 16 and 32 tokens, timed |
| `measure.py` | speedup, share of guesses kept and tokens per big pass for 2, 4 and 8 guesses on three prompts |
| `diff.py` | prints one decoder's 200 tokens, to diff plain against speculative |
| `tie.py` | finds where plain and speculative split on "why is the sky blue" and prints the big model's top scores there |
| `results.json` | the numbers from the video |

## Results on an M4 Pro, 24 GB

| prompt | 2 guesses | 4 guesses | 8 guesses |
| --- | --- | --- | --- |
| Python palindrome function | 1.72x | 1.41x | 1.03x |
| Why is the sky blue | 1.16x | 0.84x | 0.56x |
| Poem about New York at night | 1.12x | 0.80x | 0.46x |

Every run printed identical output to plain decoding except the sky prompt at 4 and 8 guesses. At 4 guesses it split at token 116, where the big model's top two scores were tied to the third decimal place (25.344 against 25.328). Checking four tokens in one batch adds the numbers in a different order than checking one, which was enough to flip the tie. That is the "within hardware numerics" caveat from the papers.

Your numbers will move with the machine, what else is running, and the prompt.

## Papers

- Leviathan, Kalman, Matias (2023), [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- Chen et al. (2023), [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
