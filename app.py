import streamlit as st

st.set_page_config(
    page_title="Carnac (MVP)",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Carnac (MVP)")
st.caption("Safe-mode boot. Carnac rebuilding…")

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
