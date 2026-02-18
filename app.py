import streamlit as st
import random
from datetime import datetime

import requests
import feedparser
from dateutil import parser as dateparser
import math

st.set_page_config(
    page_title="Carnac (MVP)",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Carnac (MVP)")
st.caption("Carnac is rebuilding the signal cortex…")
def fetch_rss_items(url: str, limit: int = 8):
    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:limit]:
            items.append({
                "title": getattr(e, "title", ""),
                "link": getattr(e, "link", ""),
                "published": getattr(e, "published", "") or getattr(e, "updated", "")
            })
        return items
    except Exception:
        return []

def recency_score(items, horizon_days: float = 14.0) -> float:
    if not items:
        return 0.0

    now = datetime.now()
    half_life_days = max(1.0, min(14.0, horizon_days / 2.0))

    scores = []
    for it in items:
        pub = it.get("published", "") or ""
        try:
            dt = dateparser.parse(pub, fuzzy=True, default=now)
        except Exception:
            dt = now

        age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
        w = math.exp(-math.log(2) * (age_days / half_life_days))
        scores.append(w)

    return min(1.0, sum(scores) / max(1, len(scores)))

def density_label(n_news: int, n_reddit: int) -> str:
    total = n_news + n_reddit
    if total >= 12:
        return "High"
    if total >= 4:
        return "Medium"
    return "Low"


q = st.text_input("Ask Carnac:", placeholder="e.g., Will I need a raincoat in Chicago next week?")

show_sources = st.checkbox("Show sources", value=True)
use_live_signals = st.checkbox("Use live signals (RSS)", value=True)
# ===============================
# SIMPLE STABLE CORE (NO TRY BLOCK)
# ===============================

import random

def monte_carlo_beta(mean_prob: float, strength: float, n: int = 20000):
    alpha = mean_prob * strength
    beta = (1.0 - mean_prob) * strength
    samples = [random.betavariate(alpha, beta) for _ in range(n)]
    samples.sort()
    return (
        samples[int(0.10 * n)],
        samples[int(0.50 * n)],
        samples[int(0.90 * n)],
        sum(samples) / n
    )

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

def carnac_reveal(p50: float) -> str:
    if p50 < 0.30:
        return "**Carnac senses opposing currents.**"
    elif p50 < 0.55:
        return "**Carnac reads a balanced atmosphere.**"
    elif p50 < 0.75:
        return "**Carnac observes gathering momentum.**"
    else:
        return "**Carnac detects strong convergence.**"

# ===============================
# UI
# ===============================

q = st.text_input("Ask Carnac:", placeholder="Will it rain next week?")

# ⬇️ NEW BLOCK STARTS HERE
from datetime import datetime

def is_bulletin(p: float) -> bool:
    ...
if q:
    st.write("✅ Query received:", q)

    base_prob = 0.40
    p10, p50, p90, mean_val = monte_carlo_beta(base_prob, strength=40)

    bulletin = (p50 >= 0.70) or (p50 <= 0.20)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if bulletin:
        st.subheader("Carnac Bulletin")
        st.caption(f"Issued: {timestamp}")
    else:
        st.subheader("Carnac Reading")

    st.markdown(carnac_reveal(p50))
    st.markdown(f"**Lean:** {get_lean(p50)}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P10", f"{p10:.0%}")
    c2.metric("P50", f"{p50:.0%}")
    c3.metric("P90", f"{p90:.0%}")
    c4.metric("Mean", f"{mean_val:.0%}")
