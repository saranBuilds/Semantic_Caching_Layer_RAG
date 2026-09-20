"""
Parses logs/app.log and prints cache performance: hit rate, latency
comparison, and estimated tokens/time saved by caching.
Run standalone: python scripts/analyze_logs.py
"""
import re

LOG_PATH = "logs/app.log"

# Rough $ equivalent if these tokens had gone to a hosted API instead —
# purely illustrative, using GPT-4o-mini-ish pricing per 1M tokens.
PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_OUTPUT = 0.60

hit_pattern = re.compile(
    r"CACHE_HIT \| matched=.*? \| similarity=([\d.]+) \| total_latency=([\d.]+)s"
)
miss_pattern = re.compile(
    r"CACHE_MISS \| best_similarity=([\d.]+) \| prompt_tokens=(\d+) \| "
    r"completion_tokens=(\d+) \| llm_latency=([\d.]+)s \| total_latency=([\d.]+)s"
)


def analyze(log_path: str = LOG_PATH):
    hits, misses = [], []

    with open(log_path) as f:
        for line in f:
            hm = hit_pattern.search(line)
            if hm:
                hits.append({"similarity": float(hm.group(1)), "latency": float(hm.group(2))})
                continue
            mm = miss_pattern.search(line)
            if mm:
                misses.append({
                    "best_similarity": float(mm.group(1)),
                    "prompt_tokens": int(mm.group(2)),
                    "completion_tokens": int(mm.group(3)),
                    "llm_latency": float(mm.group(4)),
                    "total_latency": float(mm.group(5)),
                })

    total = len(hits) + len(misses)
    if total == 0:
        print("No queries found in log yet — ask some questions first.")
        return

    hit_rate = len(hits) / total * 100
    avg_hit_latency = sum(h["latency"] for h in hits) / len(hits) if hits else 0
    avg_miss_latency = sum(m["total_latency"] for m in misses) / len(misses) if misses else 0
    avg_prompt_tokens = sum(m["prompt_tokens"] for m in misses) / len(misses) if misses else 0
    avg_completion_tokens = sum(m["completion_tokens"] for m in misses) / len(misses) if misses else 0

    tokens_saved = (avg_prompt_tokens + avg_completion_tokens) * len(hits)
    time_saved = max(0, (avg_miss_latency - avg_hit_latency)) * len(hits)
    cost_saved = (
        (avg_prompt_tokens * len(hits) / 1_000_000) * PRICE_PER_1M_INPUT
        + (avg_completion_tokens * len(hits) / 1_000_000) * PRICE_PER_1M_OUTPUT
    )

    print("=== Cache Performance Summary ===")
    print(f"Total queries:        {total}")
    print(f"Cache hits:           {len(hits)}")
    print(f"Cache misses:         {len(misses)}")
    print(f"Hit rate:             {hit_rate:.1f}%")
    print()
    print(f"Avg latency (hit):    {avg_hit_latency:.3f}s")
    print(f"Avg latency (miss):   {avg_miss_latency:.3f}s")
    print(f"Speedup on a hit:     {avg_miss_latency / avg_hit_latency:.1f}x" if avg_hit_latency else "")
    print()
    print(f"Est. tokens saved:    {tokens_saved:.0f}")
    print(f"Est. time saved:      {time_saved:.1f}s")
    print(f"Est. $ saved (if on a hosted API): ${cost_saved:.4f}")

    if misses:
        print("\n=== Near-miss similarity scores (misses, highest first) ===")
        for m in sorted(misses, key=lambda x: -x["best_similarity"])[:5]:
            print(f"  best_similarity={m['best_similarity']:.3f}")


if __name__ == "__main__":
    analyze()