import re
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import streamlit as st

# External deps typically in requirements:
# - requests
# - feedparser
# - python-dateutil
import requests
import feedparser
from dateutil import parser as dateparser


# =========================
# Utilities
# =========================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def safe_get(d: dict, key: str, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default


def days_until(target_dt: datetime) -> float:
    # If target_dt is naive, treat as local; if aware, use that.
    now = datetime.now(target_dt.tzinfo) if target_dt.tzinfo else datetime.now()
    delta = target_dt - now
    return delta.total_seconds() / 86400.0


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# =========================
# Query parsing / routing
# =========================

@dataclass
class ParsedQuery:
    raw: str
    domain_hint: str
    topic: str
    target_dt: datetime
    time_window_days: float


def infer_target_datetime(q: str) -> Tuple[datetime, float]:
    """
    Attempts to extract a time horizon from the query.
    Returns (target_dt, window_days).
    """
    q_l = q.lower()

    # Default: 14 days
    default_days = 14.0

    # Quick patterns
    if "tomorrow" in q_l:
        return datetime.now() + timedelta(days=1), 1.0
    if "today" in q_l:
        return datetime.now() + timedelta(hours=12), 0.5
    if "next week" in q_l:
        return datetime.now() + timedelta(days=7), 7.0
    if "this week" in q_l:
        return datetime.now() + timedelta(days=5), 5.0
    if "next month" in q_l:
        return datetime.now() + timedelta(days=30), 30.0

    # "in X days/weeks"
    m = re.search(r"\bin\s+(\d+)\s*(day|days|week|weeks|month|months)\b", q_l)
    if m:
        n = float(m.group(1))
        unit = m.group(2)
        if "week" in unit:
            days = n * 7.0
        elif "month" in unit:
            days = n * 30.0
        else:
            days = n
        return datetime.now() + timedelta(days=days), days

    # "by <date>" or any parseable date present
    # We'll try a forgiving date parse on the whole query; if it fails, fallback.
    try:
        dt = dateparser.parse(q, fuzzy=True, default=datetime.now())
        # If parse returned "now-ish" from fuzzy parse without a real date,
        # it can be misleading. We'll only accept if it differs meaningfully.
        if abs((dt - datetime.now()).total_seconds()) > 3600 * 6:
            # estimate window as days until that dt (bounded)
            w = clamp(abs((dt - datetime.now()).total_seconds()) / 86400.0, 1.0, 365.0)
            return dt, w
    except Exception:
        pass

    return datetime.now() + timedelta(days=default_days), default_days


def classify_domain(q: str) -> str:
    q_l = q.lower()

    # Weather
    if any(k in q_l for k in ["rain", "snow", "weather", "forecast", "temperature", "wind", "storm", "umbrella", "raincoat"]):
        return "weather"

    # Flights / travel disruption
    if any(k in q_l for k in ["flight", "delayed", "delay", "gate", "boarding", "tsa", "airport", "united", "delta", "aa ", "southwest"]):
        return "flights"

    # Sports
    if any(k in q_l for k in ["giants", "dodgers", "yankees", "nba", "nfl", "mlb", "nhl", "win", "beat", "score", "spring training"]):
        return "sports"

    # Awards / culture
    if any(k in q_l for k in ["oscar", "emmy", "grammy", "award", "best song", "won", "win an emmy"]):
        return "awards"

    # Policy / politics
    if any(k in q_l for k in ["congress", "tariff", "sanction", "bill", "senate", "house", "election", "fed", "rate cut", "government shutdown"]):
        return "policy"

    return "general"


def extract_topic(q: str) -> str:
    # Strip filler words; keep nouns-ish.
    fillers = r"\b(will|would|could|should|the|a|an|to|in|on|for|of|my|i|we|you|need|chances|odds|predict|probability|chance|likely|unlikely)\b"
    s = re.sub(fillers, " ", q, flags=re.IGNORECASE)
    s = re.sub(r"[^A-Za-z0-9\s\-\&\.\,]", " ", s)
    s = normalize_whitespace(s)
    # Keep it short
    return s[:120] if s else q.strip()[:120]


def parse_query(q: str) -> ParsedQuery:
    target_dt, window_days = infer_target_datetime(q)
    domain = classify_domain(q)
    topic = extract_topic(q)
    return ParsedQuery(raw=q, domain_hint=domain, topic=topic, target_dt=target_dt, time_window_days=window_days)


# =========================
# Signal sources (RSS + Wikipedia attention)
# =========================

USER_AGENT = "CarnacMVP/1.1 (streamlit demo; contact: example@example.com)"


def fetch_rss_items(url: str, limit: int = 8) -> List[dict]:
    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:limit]:
            title = safe_get(e, "title", "") or ""
            link = safe_get(e, "link", "") or ""
            published = safe_get(e, "published", "") or safe_get(e, "updated", "") or ""
            items.append({"title": title, "link": link, "published": published})
        return items
    except Exception:
        return []


def guess_wikipedia_title(topic: str) -> Optional[str]:
    # Very lightweight heuristic; for better results you can add a Wikipedia search API call later.
    # For now, use the topic as-is if it looks plausible.
    t = topic.strip()
    if not t:
        return None
    # Avoid long question-like strings
    if len(t.split()) > 10:
        return None
    return t


def wikipedia_pageviews_30d(title: str) -> Optional[List[int]]:
    """
    Uses Wikimedia Pageviews API to fetch last ~30 days of daily views for an article.
    """
    if not title:
        return None
    try:
        # Basic normalization
        title_enc = requests.utils.quote(title.replace(" ", "_"), safe="")
        end = (utc_now() - timedelta(days=1)).strftime("%Y%m%d")
        start = (utc_now() - timedelta(days=31)).strftime("%Y%m%d")

        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia/all-access/user/{title_enc}/daily/{start}/{end}"
        )
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        items = data.get("items", [])
        if not items:
            return None
        views = [int(it.get("views", 0)) for it in items][-30:]
        return views if views else None
    except Exception:
        return None


def trend_score(views: List[int]) -> float:
    """
    Simple 30-day trend: compare last 7 days average to prior 21 days average.
    Returns a bounded score in [-1, +1].
    """
    if not views or len(views) < 14:
        return 0.0
    last7 = views[-7:]
    prev = views[:-7]
    if not prev:
        return 0.0
    a = sum(last7) / max(1, len(last7))
    b = sum(prev) / max(1, len(prev))
    if b <= 0:
        return 0.0
    ratio = (a - b) / b  # e.g., +0.25 means +25%
    # squash
    return clamp(ratio / 1.0, -1.0, 1.0)


def recency_score(items: List[dict], horizon_days: float) -> float:
    """
    Compute a simple recency score: more recent items within the horizon contribute more.
    Returns [0, 1].
    """
    if not items:
        return 0.0

    now = datetime.now()
    scores = []
    # Half-life grows with horizon (but bounded)
    half_life_days = clamp(horizon_days / 2.0, 1.0, 14.0)

    for it in items:
        pub = it.get("published", "") or ""
        try:
            dt = dateparser.parse(pub, fuzzy=True, default=now)
        except Exception:
            dt = now - timedelta(days=30)

        age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
        # exponential decay
        w = math.exp(-math.log(2) * (age_days / half_life_days))
        scores.append(w)

    # normalize
    return clamp(sum(scores) / max(1, len(scores)), 0.0, 1.0)


def gather_signals(parsed: ParsedQuery, limit_each: int = 8):
    """
    Returns (news_items, reddit_items, wiki_views, derived scores)
    """
    # Lightweight RSS sources (no API keys)
    # Google News RSS: topic query
    q = requests.utils.quote(parsed.topic)
    news_rss = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    reddit_rss = f"https://www.reddit.com/search.rss?q={q}&sort=new"

    news_items = fetch_rss_items(news_rss, limit=limit_each)
    reddit_items = fetch_rss_items(reddit_rss, limit=limit_each)

    news_rec = recency_score(news_items, parsed.time_window_days)
    reddit_rec = recency_score(reddit_items, parsed.time_window_days)

    wiki_title = guess_wikipedia_title(parsed.topic)
    wiki_views = wikipedia_pageviews_30d(wiki_title) if wiki_title else None
    wiki_trend = trend_score(wiki_views) if wiki_views else 0.0

    return news_items, reddit_items, wiki_views, wiki_trend, news_rec, reddit_rec, wiki_title


def signal_density_label(n_news: int, n_reddit: int, has_wiki: bool) -> str:
    score = 0
    if n_news >= 6:
        score += 2
    elif n_news >= 2:
        score += 1
    if n_reddit >= 6:
        score += 2
    elif n_reddit >= 2:
        score += 1
    if has_wiki:
        score += 1

    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def strength_from_signal_density(n_news: int, n_reddit: int, has_wiki: bool, density: str) -> float:
    """
    Beta strength: higher means tighter distribution (more confidence/less variance).
    """
    base = {"Low": 20.0, "Medium": 35.0, "High": 55.0}.get(density, 30.0)
    # small additive effect
    base += min(20.0, 2.0 * n_news + 1.5 * n_reddit)
    if has_wiki:
        base += 10.0
    return clamp(base, 15.0, 120.0)


def confidence_label(density: str, horizon_days: float) -> str:
    # Longer horizon reduces confidence
    if horizon_days >= 60:
        return "Low"
    if horizon_days >= 21 and density != "High":
        return "Low"
    if density == "High" and horizon_days <= 14:
        return "High"
    if density == "Low":
        return "Low"
    return "Medium"


# =========================
# Probability model + Monte Carlo
# =========================

def base_rate(domain: str) -> float:
    # Conservative priors; tune later.
    return {
        "sports": 0.45,
        "weather": 0.40,
        "flights": 0.35,
        "awards": 0.30,
        "policy": 0.25,
        "general": 0.30
    }.get(domain, 0.30)


@dataclass
class ForecastResult:
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
    activity_centered = activity - 0.5

    # wiki_trend is [-1..1]
    # Weight decays with horizon (far horizon => dampening)
    horizon_damp = clamp(14.0 / max(7.0, horizon_days), 0.2, 1.0)

    delta = (
        0.18 * activity_centered * horizon_damp +
        0.10 * wiki_trend * horizon_damp
    )

    # Pull toward prior strongly to avoid overconfidence
    mean_p = prior + delta
    return clamp(mean_p, 0.02, 0.98)


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
   
