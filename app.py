    p10: float
    p50: float
    p90: float
    mean: float
    confidence: str


def mean_from_signals(prior: float, news_rec: float, reddit_rec: float, wiki_trend: float, horizon_days: float) -> float:
    """
    Convert signal features into a mean probability.
    Conservative, bounded adjustments.
    """
    # Signals: recency gives directional "activity"; wiki_trend gives attention change.
    activity = 0.55 * news_rec + 0.45 * reddit_rec  # [0..1]
    # Convert to centered [-0.5..+0.5]
    activity_centered = activity - 


def monte_carlo_beta(mean_prob: float, strength: float, n: int = 25000, seed: Optional[int] = None) -> Tuple[float, float, float, float]:
    """
    Monte Carlo sampling from Beta distribution parameterized by mean and strength.
    Returns (p10, p50, p90, mean).
    """
    import random

    if seed is not None:
        random.seed(seed)

    m = clamp(mean_prob, 0.001, 0.999)
    s = max(2.0, strength)

    alpha = m * s
    beta = (1.0 - m) * s

    samples = []
    for _ in range(int(n)):
        # random.betavariate exists in Python stdlib
        samples.append(random.betavariate(alpha, beta))

    samples.sort()
    def pct(p: float) -> float:
        idx = int(round(p * (len(samples) - 1)))
        return samples[clamp(idx, 0, len(samples) - 1)]

    p10 = pct(0.10)
    p50 = pct(0.50)
    p90 = pct(0.90)
    mean_s = sum(samples) / len(samples)
    return p10, p50, p90, mean_s


# =========================
# Narrative + Spine
# =========================

def get_lean(p: float) -> str:
   def get_lean(p: float) -> str:
    if p < 0.15:
        return "Very Unlikely"
    elif p < 0.30:
        return "Unlikely"
    elif p < 0.45:
        return "Lean Unlikely"
    elif p < 0.55:
        return "Even"
    elif p < 0.70:
        return "Lean Likely"
    elif p < 0.85:
        return "Likely"
    else:
        return "Highly Likely"
