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
