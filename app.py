# app.py
# Streamlit MVP: "Carnac" - Magic 8-ball meets real intel
# Adds: follow-up questions + signal density meter
#
# Run:
#   pip install streamlit requests python-dateutil
#   streamlit run app.py

import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
import streamlit as st
from dateutil import parser as dateparser
import xml.etree.ElementTree as ET

# ----------------------------
# Utilities
# ----------------------------

UA = "CarnacMVP/0.2 (+streamlit)"

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def safe_get(url: str, timeout: int = 12) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    return r

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def days_until(target: datetime) -> float:
    return (target - now_utc()).total_seconds() / 86400.0

# ----------------------------
# RSS Parsing
# ----------------------------

def parse_rss_items(xml_text: str, max_items: int = 25) -> List[Dict]:
    """
    Best-effort RSS/Atom parser: returns list of dicts: title, link, published(datetime|None).
    """
    items: List[Dict] = []
    root = ET.fromstring(xml_text)

    rss_items = root.findall(".//item")
    atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    def parse_date(text: Optional[str]) -> Optional[datetime]:
        if not text:
            return None
        try:
            dt = dateparser.parse(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    if rss_items:
        for it in rss_items[:max_items]:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            pub = parse_date((it.findtext("pubDate") or "").strip())
            items.append({"title": title, "link": link, "published": pub})
    elif atom_entries:
        ns = "{http://www.w3.org/2005/Atom}"
        for en in atom_entries[:max_items]:
            title = (en.findtext(f"{ns}title") or "").strip()
            link_el = en.find(f"{ns}link")
            link = (link_el.attrib.get("href", "") if link_el is not None else "").strip()
            pub = parse_date((en.findtext(f"{ns}updated") or en.findtext(f"{ns}published") or "").strip())
            items.append({"title": title, "link": link, "published": pub})
    return items

def recency_score(items: List[Dict], half_life_hours: float = 24.0) -> float:
    """
    Returns 0..1 based on how recent the items are (higher means more recent).
    """
    if not items:
        return 0.0
    now = now_utc()
    weights = []
    for it in items:
        dt = it.get("published")
        if not isinstance(dt, datetime):
            continue
        age_h = max(0.0, (now - dt).total_seconds() / 3600.0)
        w = math.exp(-age_h / half_life_hours)
        weights.append(w)
    if not weights:
        return 0.0
    return clamp(sum(weights) / len(weights), 0.0, 1.0)

# ----------------------------
# Sources
# ----------------------------

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_google_news_rss(query: str, max_items: int = 25) -> List[Dict]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    xml_text = safe_get(url).text
    return parse_rss_items(xml_text, max_items=max_items)

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_reddit_search_rss(query: str, max_items: int = 25) -> List[Dict]:
    url = f"https://www.reddit.com/search.rss?q={quote_plus(query)}&t=day"
    xml_text = safe_get(url).text
    return parse_rss_items(xml_text, max_items=max_items)

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_wikipedia_pageviews(project: str, article: str, days: int = 30) -> List[int]:
    article = article.replace(" ", "_")
    end = (now_utc() - timedelta(days=1)).strftime("%Y%m%d")
    start = (now_utc() - timedelta(days=days)).strftime("%Y%m%d")
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{project}/all-access/user/{quote_plus(article)}/daily/{start}/{end}"
    )
    data = safe_get(url).json()
    items = data.get("items", [])
    return [int(it.get("views", 0)) for it in items]

def wiki_trend_score(views: List[int]) -> float:
    if len(views) < 10:
        return 0.0
    last = views[-7:]
    prev = views[-14:-7] if len(views) >= 14 else views[:-7]
    if not prev:
        return 0.0
    a = sum(last) / len(last)
    b = sum(prev) / len(prev)
    if b <= 0:
        return 0.0
    ratio = a / b
    score = (ratio - 0.7) / (1.5 - 0.7)
    return clamp(score, 0.0, 1.0)

# ----------------------------
# Query Router (heuristic MVP)
# Later: swap with LLM for structured parsing.
# ----------------------------

@dataclass
class ParsedQuery:
    raw: str
    entity_guess: str
    target_date_utc: Optional[datetime]
    domain_hint: str
    needs_entity: bool
    needs_time: bool
    timeframe_label: str

def guess_domain(q: str) -> str:
    ql = q.lower()
    if any(w in ql for w in ["flight", "delayed", "gate", "united", "delta", "southwest", "aa ", "ord", "sfo", "lax"]):
        return "flights"
    if any(w in ql for w in ["rain", "snow", "weather", "coat", "umbrella", "forecast", "temperature", "wind"]):
        return "weather"
    if any(w in ql for w in ["win", "lose", "score", "giants", "niners", "warriors", "lakers", "nfl", "mlb", "nba"]):
        return "sports"
    if any(w in ql for w in ["emmy", "oscar", "grammy", "award", "nomination"]):
        return "awards"
    if any(w in ql for w in ["tariff", "sanction", "bill", "congress", "election", "white house"]):
        return "policy"
    return "general"

def extract_entity_guess(q: str) -> str:
    m = re.search(r'"([^"]+)"', q)
    if m:
        return m.group(1).strip()

    cleaned = q.strip()
    cleaned = re.sub(r"\?$", "", cleaned).strip()
    cleaned = re.sub(r"^(will|what|chances|chance|is|are|do|does|did)\s+", "", cleaned, flags=re.I)

    # Remove common filler phrases
    cleaned = re.sub(r"\b(next week|this week|tomorrow|today|on saturday|on sunday|this weekend)\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned[:80].strip()

def contains_time_hint(q: str) -> bool:
    ql = q.lower()
    if any(w in ql for w in ["tomorrow", "today", "next week", "this week", "this weekend", "tonight"]):
        return True
    weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    if any(w in ql for w in weekdays):
        return True
    # crude date tokens
    if re.search(r"\b(\d{1,2}/\d{1,2}(/\d{2,4})?)\b", ql):
        return True
    if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", ql):
        return True
    return False

def estimate_target_date(q: str) -> Optional[datetime]:
    ql = q.lower()
    base = now_utc()

    if "tomorrow" in ql:
        return base + timedelta(days=1)
    if "next week" in ql:
        return base + timedelta(days=7)
    if "this weekend" in ql:
        return base + timedelta(days=2)

    weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for i, wd in enumerate(weekdays):
        if wd in ql:
            today_i = base.weekday()  # monday=0
            target_i = i
            delta = (target_i - today_i) % 7
            if delta == 0:
                delta = 7
            return base + timedelta(days=delta)

    # Try date parsing if user included a date
    try:
        dt = dateparser.parse(q, default=base)
        if dt:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass

    return None

def is_vague_entity(entity_guess: str, raw: str) -> bool:
    # Too short / too generic / basically the whole question
    eg = entity_guess.strip().lower()
    rawl = raw.strip().lower()

    if len(eg) < 4:
        return True

    # If entity guess still contains typical question scaffolding, it’s vague
    if any(w in eg for w in ["will ", "chances", "chance", "need", "win", "lose", "delayed", "raincoat", "tariffs"]):
        # Not always wrong, but often indicates the entity guess is the whole query
        pass

    # If entity guess is nearly the same length as raw, it's not really an entity
    if len(entity_guess) >= int(0.85 * len(raw)):
        return True

    # Common vague stand-ins
    if eg in {"it", "that", "this", "giants", "weather", "stocks", "politics"}:
        return True

    return False

def timeframe_label_from_target(target: datetime) -> str:
    d = days_until(target)
    if d <= 2:
        return "short"
    if d <= 14:
        return "medium"
    return "long"

def parse_query(q: str) -> ParsedQuery:
    domain = guess_domain(q)
    entity_guess = extract_entity_guess(q)

    target = estimate_target_date(q)
    needs_time = not contains_time_hint(q) or target is None

    needs_entity = is_vague_entity(entity_guess, q)

    tf = timeframe_label_from_target(target) if target else "unknown"
    return ParsedQuery(
        raw=q,
        entity_guess=entity_guess,
        target_date_utc=target,
        domain_hint=domain,
        needs_entity=needs_entity,
        needs_time=needs_time,
        timeframe_label=tf
    )

# ----------------------------
# Monte Carlo engine (Beta distribution over probability)
# ----------------------------

@dataclass
class MCResult:
    mean: float
    p10: float
    p50: float
    p90: float
    confidence: str

def beta_from_mean_strength(mean: float, strength: float) -> Tuple[float, float]:
    mean = clamp(mean, 1e-6, 1 - 1e-6)
    strength = max(2.0, strength)
    return mean * strength, (1 - mean) * strength

def percentile(sorted_vals: List[float], pct: float) -> float:
    k = (len(sorted_vals) - 1) * pct
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

def confidence_label(p10: float, p90: float) -> str:
    width = p90 - p10
    if width < 0.15:
        return "High"
    if width < 0.30:
        return "Medium"
    return "Low"

def monte_carlo_beta(mean_prob: float, strength: float, n: int = 25000, seed: Optional[int] = None) -> MCResult:
    if seed is not None:
        random.seed(seed)
    a, b = beta_from_mean_strength(mean_prob, strength)
    samples = [random.betavariate(a, b) for _ in range(n)]
    samples.sort()
    p10 = percentile(samples, 0.10)
    p50 = percentile(samples, 0.50)
    p90 = percentile(samples, 0.90)
    return MCResult(
        mean=sum(samples) / len(samples),
        p10=p10,
        p50=p50,
        p90=p90,
        confidence=confidence_label(p10, p90),
    )

# ----------------------------
# Feature building / scoring
# ----------------------------

def compute_base_rate(domain: str) -> float:
    priors = {
        "flights": 0.35,
        "weather": 0.30,
        "sports": 0.50,
        "awards": 0.15,
        "policy": 0.20,
        "general": 0.30,
    }
    return priors.get(domain, 0.30)

def strength_from_signal_density(n_news: int, n_reddit: int, wiki_has: bool, density_label: str) -> float:
    # Higher density => higher pseudo-evidence strength => narrower bands
    base = 8.0
    base += min(n_news, 25) * 0.6
    base += min(n_reddit, 25) * 0.4
    base += 10.0 if wiki_has else 0.0

    # density adjustment
    if density_label == "High":
        base += 8.0
    elif density_label == "Medium":
        base += 3.0
    else:
        base -= 2.0

    return clamp(base, 6.0, 55.0)

def mean_prob_from_signals(
    base: float,
    news_recency: float,
    reddit_recency: float,
    wiki_trend: float,
    time_to_event_days: float
) -> float:
    momentum = 0.55 * (news_recency - 0.5) + 0.45 * (reddit_recency - 0.5)
    momentum += 0.35 * (wiki_trend - 0.5)

    t = clamp(1.2 - (time_to_event_days / 14.0), 0.4, 1.2)

    x = (base - 0.5) * 1.2 + momentum * 1.4 * t
    p = sigmoid(x)
    return clamp(p, 0.02, 0.98)

def signal_density_label(n_news: int, n_reddit: int, news_rec: float, reddit_rec: float, wiki_has: bool) -> str:
    # A single blended score 0..1
    count_score = clamp((n_news / 25.0) * 0.55 + (n_reddit / 25.0) * 0.45, 0.0, 1.0)
    rec_score = clamp(news_rec * 0.55 + reddit_rec * 0.45, 0.0, 1.0)
    wiki_bonus = 0.10 if wiki_has else 0.0

    score = clamp(0.55 * count_score + 0.35 * rec_score + wiki_bonus, 0.0, 1.0)

    if score >= 0.62:
        return "High"
    if score >= 0.32:
        return "Medium"
    return "Sparse"

# ----------------------------
# Carnac Narration (no external LLM required yet)
# ----------------------------

def carnac_reveal(domain: str, p50: float, conf: str, density: str) -> str:
    # Elemental / signal metaphors chosen deterministically from probability tier

    if p50 < 0.15:
        line1 = "Carnac registers strong opposing currents."
        line2 = "The outcome appears very unlikely."
    elif p50 < 0.30:
        line1 = "Carnac senses opposing currents."
        line2 = "The outcome appears unlikely."
    elif p50 < 0.45:
        line1 = "Carnac detects mild headwinds."
        line2 = "The balance tilts against the outcome."
    elif p50 < 0.55:
        line1 = "Carnac reads a balanced atmosphere."
        line2 = "The outcome remains evenly poised."
    elif p50 < 0.70:
        line1 = "Carnac observes gathering momentum."
        line2 = "The balance tilts toward the outcome."
    elif p50 < 0.85:
        line1 = "Carnac registers strengthening convergence."
        line2 = "The outcome appears likely."
    else:
        line1 = "Carnac detects strong convergence."
        line2 = "The outcome appears highly likely."

    return f"**{line1}**\n\n{line2}"

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Carnac (MVP)", page_icon="🔮", layout="centered")

st.title("🔮 Carnac (MVP)")
st.caption("Useful novelty — Magic 8-ball meets real intel (news + Reddit + Wikipedia attention) + Monte Carlo uncertainty bands.")

q = st.text_input("Ask Carnac:", placeholder='e.g., "Will I need a raincoat in Chicago next week?"')

colA, colB, colC = st.columns(3)
with colA:
    show_sources = st.checkbox("Show sources", value=True)
with colB:
    mc_runs = st.selectbox("Monte Carlo runs", [5000, 10000, 25000, 50000], index=2)
with colC:
    seed = st.number_input("Seed (optional)", value=0, step=1)

if q:
    parsed = parse_query(q)

    st.write(f"**Domain hint:** `{parsed.domain_hint}`")

    # ----------------------------
    # Follow-up questions (only if needed)
    # ----------------------------
    entity = parsed.entity_guess
    target_dt = parsed.target_date_utc

    if parsed.needs_entity:
        entity = st.text_input(
            "Quick follow-up: what’s the main entity/topic?",
            value=parsed.entity_guess if len(parsed.entity_guess) < 50 else "",
            placeholder='e.g., "San Francisco Giants", "Sabrina Carpenter", "US tariffs on China", "United UA123 SFO→ORD"'
        )

    if parsed.needs_time:
        default_day = (datetime.now().date() + timedelta(days=7))
        picked = st.date_input(
            "Quick follow-up: when is the target date?",
            value=default_day,
            min_value=datetime.now().date(),
            max_value=(datetime.now().date() + timedelta(days=365))
        )
        # Convert to a UTC datetime (noon UTC is fine for a day-level MVP)
        target_dt = datetime(picked.year, picked.month, picked.day, 12, 0, 0, tzinfo=timezone.utc)

    # If still missing essentials, stop gracefully
    if not entity or not target_dt:
        st.warning("Give me a clearer entity/topic and a target date, and I’ll make the call.")
        st.stop()

    st.write(f"**Entity/topic:** `{entity}`  •  **Target:** `{target_dt.strftime('%Y-%m-%d')}`  •  **Horizon:** `{timeframe_label_from_target(target_dt)}`")

    # Build a general-purpose search term
    query_for_feeds = f"{entity} {q}".strip()

    with st.spinner("Consulting the crowd…"):
        news_items: List[Dict] = []
        reddit_items: List[Dict] = []
        wiki_views: List[int] = []
        wiki_project = "en.wikipedia.org"

        # Fetch signals (best-effort)
        try:
            news_items = fetch_google_news_rss(query_for_feeds, max_items=25)
        except Exception as e:
            st.warning(f"News feed unavailable right now: {e}")

        try:
            reddit_items = fetch_reddit_search_rss(query_for_feeds, max_items=25)
        except Exception as e:
            st.warning(f"Reddit feed unavailable right now: {e}")

        # Wikipedia: guess article title from entity (with one sports nicety)
        article_guess = entity.strip()
        if parsed.domain_hint == "sports" and re.fullmatch(r"giants", article_guess, flags=re.I):
            article_guess = "San Francisco Giants"

        try:
            wiki_views = fetch_wikipedia_pageviews(wiki_project, article_guess, days=30)
        except Exception:
            wiki_views = []

        news_rec = recency_score(news_items, half_life_hours=24)
        reddit_rec = recency_score(reddit_items, half_life_hours=18)
        wt = wiki_trend_score(wiki_views) if wiki_views else 0.0

        density = signal_density_label(len(news_items), len(reddit_items), news_rec, reddit_rec, bool(wiki_views))

        base = compute_base_rate(parsed.domain_hint)
        tdays = max(0.0, days_until(target_dt))

        mean_p = mean_prob_from_signals(
            base=base,
            news_recency=news_rec,
            reddit_recency=reddit_rec,
            wiki_trend=wt,
            time_to_event_days=tdays
        )

        strength = strength_from_signal_density(len(news_items), len(reddit_items), bool(wiki_views), density)

        # Monte Carlo
        res = monte_carlo_beta(
            mean_prob=mean_p,
            strength=strength,
            n=int(mc_runs),
            seed=(int(seed) if seed != 0 else None)
        )
    lean = get_lean(res.p50)
bulletin = is_bulletin(res.p50, res.confidence, density)

from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

if bulletin:
    st.subheader("Carnac Bulletin")
    st.caption(f"Issued: {timestamp}")
else:
    st.subheader("Carnac Reading")

st.markdown(carnac_reveal(parsed.domain_hint, res.p50, res.confidence, density))
st.markdown(f"**Lean:** {lean}")
    
        lean = get_lean(res.p50)
    bulletin = is_bulletin(res.p50, res.confidence, density)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if bulletin:
        st.subheader("Carnac Bulletin")
        st.caption(f"Issued: {timestamp}")
    else:
        st.subheader("Carnac Reading")

    st.markdown(carnac_reveal(parsed.domain_hint, res.p50, res.confidence, density))
    st.markdown(f"**Lean:** {lean}")
    st.markdown(carnac_reveal(parsed.domain_hint, res.p50, res.confidence, density))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P10", f"{res.p10:.0%}")
    c2.metric("P50", f"{res.p50:.0%}")
    c3.metric("P90", f"{res.p90:.0%}")
    c4.metric("Mean", f"{res.mean:.0%}")
    c5.metric("Signal Density", density)

    st.write("**Why (signals):**")
    for b in bullets_from_signals(news_items, reddit_items, news_rec, reddit_rec, wiki_views, wt, target_dt):
        st.write(f"• {b}")

    st.write("**Carnac’s suggestion:**")
    if parsed.domain_hint == "weather":
        st.write("• If P50 ≥ 50%, a light rain layer is a cheap hedge.")
    elif parsed.domain_hint == "flights":
        st.write("• If P50 ≥ 50%, plan for friction: arrive earlier and keep alerts on.")
    elif parsed.domain_hint == "sports":
        st.write("• Check late-breaking lineup/injury news; it can swing the band fast.")
    elif parsed.domain_hint == "awards":
        st.write("• Watch nomination/shortlist and credible pundit picks—those are high-leverage signals.")
    elif parsed.domain_hint == "policy":
        st.write("• Track official statements + legislative/actionable steps; headlines alone are noisy.")
    else:
        st.write("• If the signal is sparse, treat this as entertainment with receipts—not certainty.")

    if show_sources:
        st.divider()
        st.subheader("Sources (best-effort)")

        st.write("**News (Google News RSS)**")
        for it in news_items[:8]:
            title = it["title"] or "(untitled)"
            link = it["link"] or ""
            dt = it.get("published")
            when = dt.strftime("%Y-%m-%d %H:%MZ") if isinstance(dt, datetime) else ""
            st.write(f"- [{title}]({link}) {('— ' + when) if when else ''}")

        st.write("**Reddit (search RSS)**")
        for it in reddit_items[:8]:
            title = it["title"] or "(untitled)"
            link = it["link"] or ""
            dt = it.get("published")
            when = dt.strftime("%Y-%m-%d %H:%MZ") if isinstance(dt, datetime) else ""
            st.write(f"- [{title}]({link}) {('— ' + when) if when else ''}")

        if wiki_views:
            st.write("**Wikipedia pageviews (30 days)**")
            st.write(f"- Article guess: `{article_guess}` • Trend score: **{wt:.2f}**")
        else:
            st.write("**Wikipedia pageviews:** not available for the guessed article title.")
else:
    st.info("Type a question to begin. If your query is vague, Carnac will ask one quick follow-up.")
