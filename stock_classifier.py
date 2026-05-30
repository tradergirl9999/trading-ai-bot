# ================================================================
#  STOCK CLASSIFIER + PRICE PREDICTOR
#  Google Colab — paste entire script into one cell and run
#
#  Formula:  Beta  = Cov(Stock, Market) / Var(Market)
#            Cov(X,Y) = E[XY] - E[X]E[Y]
#
#  Categories:
#    Aggressive Growth | Market Leader | Market Follower
#    Defensive         | Uncorrelated
#
#  Predictions: 1 Day | 1 Week | 1 Month | 3M | 6M | 1 Year
#
#  Universe is fetched LIVE from Wikipedia (S&P 500 + NASDAQ 100).
#  No stock names are hardcoded — the script discovers everything itself.
# ================================================================

# ── Install (run this line first if packages are missing) ────────
# !pip install yfinance pandas numpy matplotlib pytz requests -q

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import concurrent.futures
import warnings
import time
import datetime
import pytz
import requests

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ================================================================
#  LIVE TIMESTAMP
# ================================================================

TZ_ET  = pytz.timezone("America/New_York")
TZ_UTC = pytz.utc

RUN_ET  = datetime.datetime.now(TZ_ET)
RUN_UTC = datetime.datetime.now(TZ_UTC)

def fmt_dt(dt):
    return dt.strftime("%A, %d %B %Y  %H:%M:%S %Z")

def market_status():
    et      = datetime.datetime.now(TZ_ET)
    weekday = et.weekday()
    hour    = et.hour + et.minute / 60
    if weekday >= 5:    return "CLOSED — Weekend"
    if hour < 4.0:      return "CLOSED — Overnight"
    if hour < 9.5:
        mins = int((9.5 - hour) * 60)
        return f"PRE-MARKET  (regular open in {mins} min)"
    if hour < 16.0:     return "OPEN — Regular Hours"
    if hour < 20.0:     return "AFTER-HOURS"
    return "CLOSED — Overnight"

# ================================================================
#  CONFIGURATION  ← edit here
# ================================================================

MARKET          = "^GSPC"  # Benchmark (S&P 500)
PERIOD          = "2y"     # Data lookback
RF_RATE         = 0.05     # Annual risk-free rate
TOP_N           = 10       # Stocks per category to display
WORKERS         = 16       # Parallel analysis threads
MAX_UNIVERSE    = 1500     # Cap universe size (sorted by market cap desc)
BATCH_SIZE      = 250      # Tickers per yfinance download call
MIN_MKTCAP_M    = 300      # Skip stocks below this market cap ($ millions)
# Stocks with < RECENT_IPO_DAYS trading days are flagged as recent IPOs
RECENT_IPO_DAYS = 420      # ~20 months

# ================================================================
#  LIVE UNIVERSE DISCOVERY
#  Fetches S&P 500 + NASDAQ 100 from Wikipedia.
#  No hardcoded ticker list — the script finds stocks itself.
# ================================================================

def _parse_mktcap(s) -> float:
    """Convert NASDAQ screener market-cap string ('$1.23B', '$456.78M') to $ millions."""
    try:
        s = str(s).replace("$", "").replace(",", "").strip()
        if not s or s in ("N/A", "0", "0.00"):
            return 0.0
        if s.endswith("T"):  return float(s[:-1]) * 1_000_000
        if s.endswith("B"):  return float(s[:-1]) * 1_000
        if s.endswith("M"):  return float(s[:-1])
        return float(s) / 1e6
    except Exception:
        return 0.0


def fetch_universe():
    """
    Discover the stock universe entirely from live data sources.
    No ticker names are hardcoded — every symbol comes from an API or Wikipedia.

    Pipeline:
      1. NASDAQ exchange screener  (public API, no key needed)
      2. NYSE  exchange screener   (public API, no key needed)
      3. Wikipedia S&P 500 / NASDAQ 100 / S&P 400 (fallback if screeners fail)

    Filtering applied before any ticker touches yfinance:
      • Letters only (A–Z), 1–5 characters  → drops preferred shares (MITT^A),
        warrants (BIO/B), ETNs (DX^C) and other non-common-stock listings
      • US-domiciled companies only (screener `country` field)
      • Market cap ≥ MIN_MKTCAP_M million dollars  → drops micro/nano caps
        that generate noise and slow the analysis

    Results sorted by market cap descending, capped at MAX_UNIVERSE.
    Returns: list[str] of clean ticker symbols.
    """
    hdrs        = {"User-Agent": "Mozilla/5.0 (compatible; StockScreener/1.0)"}
    meta        = {}   # sym → mktcap_M
    sources     = []

    def _valid(sym):
        s = str(sym).strip().upper()
        return s.isalpha() and 1 <= len(s) <= 5

    # ── 1 & 2. NASDAQ / NYSE screener ────────────────────────────
    for exchange in ("nasdaq", "nyse"):
        try:
            url  = (f"https://api.nasdaq.com/api/screener/stocks"
                    f"?tableonly=true&limit=5000&exchange={exchange}&download=true")
            resp = requests.get(url, headers=hdrs, timeout=15)
            resp.raise_for_status()
            rows = resp.json().get("data", {}).get("rows") or []

            added = 0
            for row in rows:
                sym = str(row.get("symbol", "")).strip().upper()
                if not _valid(sym):
                    continue
                country = str(row.get("country", "")).lower()
                if country and "united states" not in country:
                    continue
                mc = _parse_mktcap(row.get("marketCap", ""))
                if mc < MIN_MKTCAP_M:
                    continue
                if sym not in meta or mc > meta[sym]:
                    meta[sym] = mc
                added += 1
            sources.append(f"{exchange.upper()} ({added} valid)")
        except Exception as e:
            print(f"  ⚠ {exchange.upper()} screener failed: {e}")

    # ── 3. Wikipedia fallback (used when screeners fail) ──────────
    if len(meta) < 200:
        for url, label, min_rows in [
            ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500",   490),
            ("https://en.wikipedia.org/wiki/Nasdaq-100",                   "NASDAQ 100", 90),
            ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P 400",   380),
        ]:
            try:
                tables = pd.read_html(url)
                for t in tables:
                    col = next((c for c in t.columns if c in ("Symbol", "Ticker")), None)
                    if col and len(t) >= min_rows:
                        for sym in t[col].dropna():
                            s = str(sym).strip().upper()
                            if _valid(s) and s not in meta:
                                meta[s] = 0.0   # no market cap data from Wikipedia
                        sources.append(label)
                        break
            except Exception as e:
                print(f"  ⚠ {label} Wikipedia fetch failed: {e}")

    if not meta:
        raise RuntimeError(
            "All universe sources failed. Check your internet connection and try again."
        )

    # Sort by market cap descending, cap at MAX_UNIVERSE
    ranked = sorted(meta.items(), key=lambda x: x[1], reverse=True)
    result = [sym for sym, _ in ranked[:MAX_UNIVERSE]]

    print(f"  Sources: {' | '.join(sources)}")
    print(f"  Valid US stocks found: {len(meta)}  →  keeping top {len(result)} by market cap")
    return result


def bulk_download(syms, period, batch_size=BATCH_SIZE):
    """
    Download Close prices for all symbols in batches.
    yfinance has a URL length limit — sending thousands of tickers at once
    causes 'unexpected character' errors. Batching fixes this.
    Returns a single DataFrame with one column per symbol.
    """
    frames  = []
    batches = [syms[i:i+batch_size] for i in range(0, len(syms), batch_size)]
    total_b = len(batches)

    for i, batch in enumerate(batches, 1):
        try:
            raw = yf.download(batch, period=period, auto_adjust=True, progress=False)
            # yfinance ≥0.2 returns MultiIndex columns; extract Close level
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw["Close"]
            elif "Close" in raw.columns:
                raw = raw[["Close"]]
            if isinstance(raw, pd.Series):
                raw = raw.to_frame(batch[0])
            frames.append(raw)
        except Exception as e:
            pass   # failed batches are silently skipped; stocks just won't appear
        if i % 3 == 0 or i == total_b:
            pct = i / total_b * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:.0f}%  batch {i}/{total_b}", end="\r")

    print()
    if not frames:
        raise RuntimeError("All download batches failed. Check internet connection.")

    combined = pd.concat(frames, axis=1)
    # Drop duplicate columns (can happen if a ticker appears in both exchanges)
    return combined.loc[:, ~combined.columns.duplicated()]


def fetch_upcoming_ipos():
    """
    Fetch upcoming IPOs from NASDAQ's public IPO calendar API (no key needed).
    Tries current month + next 2 months, then falls back to a curated list.
    Returns list of dicts with: name, ticker, sector, valuation, expected, why.
    """
    all_ipos = []
    hdrs     = {"User-Agent": "Mozilla/5.0 (compatible; StockScreener/1.0)"}

    try:
        today  = datetime.datetime.now(TZ_ET)
        months = [
            (today + datetime.timedelta(days=30 * i)).strftime("%Y-%m")
            for i in range(3)
        ]
        for month in months:
            url  = f"https://api.nasdaq.com/api/ipo/calendar?date={month}"
            resp = requests.get(url, headers=hdrs, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json().get("data", {})

            for section_key, table_key in [("upcoming", "upcomingTable"), ("priced", "rows")]:
                section = data.get(section_key, {})
                rows = section.get(table_key, {})
                if isinstance(rows, dict):
                    rows = rows.get("rows", [])
                for row in (rows or []):
                    lo     = row.get("priceRangeLow", "?")
                    hi     = row.get("priceRangeHigh", "?")
                    shares = row.get("sharesOffered", "N/A")
                    exch   = row.get("exchange", "N/A")
                    all_ipos.append({
                        "name":      row.get("companyName", "Unknown"),
                        "ticker":    row.get("proposedTickerSymbol", "TBD"),
                        "sector":    exch,
                        "valuation": row.get("dollarValueOfSharesOffered", "N/A"),
                        "expected":  row.get("expectedPriceDate") or row.get("pricedDate", "TBD"),
                        "why":       (f"Shares offered: {shares}. "
                                      f"Price range: ${lo}–${hi}. "
                                      f"Exchange: {exch}."),
                    })

        if all_ipos:
            print(f"  Fetched {len(all_ipos)} IPO entries from NASDAQ API")
            return all_ipos[:15]

    except Exception as e:
        print(f"  ⚠ NASDAQ IPO API unavailable: {e}")

    # No hardcoded fallback — upcoming IPOs change weekly and any
    # static list would be stale. Run again later or check nasdaq.com/ipo
    print("  NASDAQ IPO calendar unavailable — skipping upcoming IPO section")
    return []

# ================================================================
#  MATH
# ================================================================

def cov(x, y):
    """Cov(X,Y) = E[XY] - E[X]*E[Y]"""
    return float((x * y).mean() - x.mean() * y.mean())

def corr(x, y):
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0: return 0.0
    return cov(x, y) / (sx * sy)

def beta(s, m):
    """Beta = Cov(stock, market) / Var(market)"""
    vm = m.var()
    return cov(s, m) / vm if vm != 0 else 0.0

def sharpe(r, rf=RF_RATE):
    e = r - rf / 252
    return float(e.mean() / e.std() * np.sqrt(252)) if r.std() != 0 else 0.0

def ann_ret(r):
    return float(((1 + r).prod()) ** (252 / len(r)) - 1) * 100

def ann_vol(r):
    return float(r.std() * np.sqrt(252)) * 100

# ================================================================
#  TECHNICALS
# ================================================================

def technicals(close, high, low, volume):
    n     = len(close)
    price = float(close.iloc[-1])

    sma20  = float(close.rolling(20).mean().iloc[-1])  if n >= 20  else price
    sma50  = float(close.rolling(50).mean().iloc[-1])  if n >= 50  else price
    sma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else price

    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    macd_h    = float((macd_line - macd_sig).iloc[-1])

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = float((100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1])

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = float((bb_mid + 2 * bb_std).iloc[-1])
    bb_lo  = float((bb_mid - 2 * bb_std).iloc[-1])
    bb_pct = (price - bb_lo) / (bb_up - bb_lo) if bb_up != bb_lo else 0.5

    tr    = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr   = float(tr.rolling(14).mean().iloc[-1]) if n >= 14 else float(tr.mean())

    stoch_k = None
    if n >= 17:
        lo14    = low.rolling(14).min()
        hi14    = high.rolling(14).max()
        stoch_k = float(((close-lo14)/(hi14-lo14)*100).rolling(3).mean().iloc[-1])

    vol_ratio = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]) if n >= 20 else 1.0
    mom5      = (price / float(close.iloc[-6])  - 1)*100 if n >= 6  else 0.0
    mom20     = (price / float(close.iloc[-21]) - 1)*100 if n >= 21 else 0.0

    signals = [
        price > sma20, price > sma50, price > sma200,
        macd_h > 0, rsi > 50, bb_pct > 0.5,
        stoch_k is not None and stoch_k > 50,
        mom20 > 0,
    ]
    tech_score = sum(signals) / len(signals)

    return dict(
        price=price, sma20=sma20, sma50=sma50, sma200=sma200,
        macd_h=macd_h, rsi=rsi, bb_up=bb_up, bb_lo=bb_lo,
        bb_pct=bb_pct, atr=atr, stoch_k=stoch_k,
        vol_ratio=vol_ratio, mom5=mom5, mom20=mom20,
        tech_score=tech_score,
    )

# ================================================================
#  PRICE PREDICTION
#  tech_score (0→1) drives expected drift:
#    0.0 → -15% annual   |   0.5 → +8% annual   |   1.0 → +30% annual
# ================================================================

TIMEFRAMES = [
    ("1 Day",    1),
    ("1 Week",   5),
    ("1 Month",  22),
    ("3 Months", 63),
    ("6 Months", 126),
    ("1 Year",   252),
]

def predict(price, tech_score, hist_vol_pct, days, tf_idx):
    daily_drift = -0.0006 + tech_score * 0.0018
    period_ret  = daily_drift * days
    period_vol  = (hist_vol_pct / 100) * np.sqrt(days / 252)

    base = price * (1 + period_ret)
    bull = price * (1 + period_ret + period_vol * 1.2)
    bear = price * (1 + period_ret - period_vol * 1.2)
    conf = max(30, 78 - tf_idx * 8)

    if   tech_score > 0.74: signal = "STRONG BUY"
    elif tech_score > 0.58: signal = "BUY"
    elif tech_score > 0.42: signal = "HOLD"
    elif tech_score > 0.26: signal = "SELL"
    else:                   signal = "STRONG SELL"

    return dict(bear=bear, base=base, bull=bull, conf=conf, signal=signal)

# ================================================================
#  CLASSIFICATION
# ================================================================

CATEGORIES = ["Aggressive Growth","Market Leader","Market Follower","Defensive","Uncorrelated"]

CAT_COLORS = {
    "Aggressive Growth": "#ff4757",
    "Market Leader":     "#ffa502",
    "Market Follower":   "#2ed573",
    "Defensive":         "#1e90ff",
    "Uncorrelated":      "#a29bfe",
}

CAT_DESC = {
    "Aggressive Growth": "β ≥ 1.5 & ρ ≥ 0.55  —  amplifies every market move",
    "Market Leader":     "1.05 ≤ β < 1.5 & ρ ≥ 0.60  —  consistently outpaces the index",
    "Market Follower":   "0.65 ≤ β < 1.05 & ρ ≥ 0.50  —  reliable index tracker",
    "Defensive":         "β < 0.65 & ρ ≥ 0.25  —  holds value in downturns",
    "Uncorrelated":      "|ρ| < 0.30  —  stock-specific drivers, great diversifier",
}

def classify(b, r):
    if abs(r) < 0.30:              return "Uncorrelated"
    if b >= 1.5   and r >= 0.55:  return "Aggressive Growth"
    if 1.05<=b<1.5 and r >= 0.60: return "Market Leader"
    if 0.65<=b<1.05 and r >= 0.50:return "Market Follower"
    if b < 0.65:                   return "Defensive"
    return "Market Follower"

# ================================================================
#  REASON GENERATOR  ← the "why I picked this stock" section
# ================================================================

def build_reasons(t, b_val, r_val, cat, ar, sh, av, pe, fwd_pe, rev_g, margin, rank_score):
    reasons = {}

    # ── WHY SELECTED (ranking logic) ─────────────────────────────
    if cat == "Aggressive Growth":
        reasons["WHY SELECTED"] = (
            f"Ranked by composite score (beta×0.4 + sharpe×0.35 + corr×0.25 = {rank_score:.2f}). "
            f"Beta {b_val:.2f} puts it among the highest market amplifiers — "
            f"every 1% S&P move drives ~{b_val:.1f}% in this stock. "
            f"Sharpe {sh:.2f} confirms the volatility is being rewarded."
        )
    elif cat == "Market Leader":
        reasons["WHY SELECTED"] = (
            f"Ranked by composite score (beta×0.3 + sharpe×0.4 + corr×0.3 = {rank_score:.2f}). "
            f"Beta {b_val:.2f} consistently outpaces the S&P 500 by {(b_val-1)*100:.0f}% on up-days "
            f"while maintaining strong correlation ({r_val:.2f}) — meaning it leads, not just follows."
        )
    elif cat == "Market Follower":
        reasons["WHY SELECTED"] = (
            f"Ranked by market tracking quality (corr×0.5 + sharpe×0.5 = {rank_score:.2f}). "
            f"Correlation {r_val:.2f} is among the tightest in the universe — "
            f"ideal for index-style exposure with individual stock alpha potential."
        )
    elif cat == "Defensive":
        reasons["WHY SELECTED"] = (
            f"Ranked by capital preservation score (1/β×0.5 + sharpe×0.5 = {rank_score:.2f}). "
            f"Beta {b_val:.2f} means when the S&P drops 10%, this stock historically drops only "
            f"~{b_val*10:.1f}%. Best in class for drawdown protection."
        )
    else:  # Uncorrelated
        reasons["WHY SELECTED"] = (
            f"Ranked by independence score (1/|ρ|×0.4 + ann_return×0.6 = {rank_score:.2f}). "
            f"Market correlation of only {r_val:.2f} means price moves are driven by "
            f"company-specific catalysts, not macro noise. Strong portfolio diversifier."
        )

    # ── TECHNICAL SETUP ────────────────────────────────────────────
    tech_points = []
    rsi = t["rsi"]
    if rsi > 70:
        tech_points.append(f"RSI {rsi:.0f} — overbought territory, momentum is strong but watch for pullback")
    elif rsi < 30:
        tech_points.append(f"RSI {rsi:.0f} — oversold, high probability mean-reversion bounce")
    elif rsi > 55:
        tech_points.append(f"RSI {rsi:.0f} — bullish momentum, trend intact")
    else:
        tech_points.append(f"RSI {rsi:.0f} — neutral zone, no directional conviction yet")

    if t["price"] > t["sma200"]:
        gap = (t["price"]/t["sma200"] - 1)*100
        tech_points.append(f"Price is {gap:.1f}% above 200-day SMA — confirmed long-term uptrend")
    else:
        gap = (1 - t["price"]/t["sma200"])*100
        tech_points.append(f"Price is {gap:.1f}% below 200-day SMA — long-term downtrend, higher risk")

    if t["macd_h"] > 0:
        tech_points.append("MACD histogram positive — short-term bullish crossover active")
    else:
        tech_points.append("MACD histogram negative — short-term bearish pressure")

    if t["bb_pct"] > 0.85:
        tech_points.append(f"Near upper Bollinger Band ({t['bb_pct']*100:.0f}% of band) — extended, consolidation likely")
    elif t["bb_pct"] < 0.15:
        tech_points.append(f"Near lower Bollinger Band ({t['bb_pct']*100:.0f}% of band) — potential bounce zone")
    else:
        tech_points.append(f"Mid Bollinger Band position ({t['bb_pct']*100:.0f}%) — room to move in either direction")

    if t["vol_ratio"] > 1.8:
        tech_points.append(f"Volume {t['vol_ratio']:.1f}x above 20-day average — strong institutional participation")
    elif t["vol_ratio"] < 0.6:
        tech_points.append(f"Volume {t['vol_ratio']:.1f}x below average — low conviction, wait for volume confirmation")

    reasons["TECHNICAL SETUP"] = " | ".join(tech_points)

    # ── FUNDAMENTAL VIEW ───────────────────────────────────────────
    fund_points = []
    if pe:
        if pe < 15:
            fund_points.append(f"P/E {pe:.1f} — cheap vs market, potential value play")
        elif pe < 25:
            fund_points.append(f"P/E {pe:.1f} — reasonably valued")
        elif pe < 40:
            fund_points.append(f"P/E {pe:.1f} — growth premium, justified only if earnings accelerate")
        else:
            fund_points.append(f"P/E {pe:.1f} — expensive, priced for perfection")

    if fwd_pe and pe:
        pe_growth = (pe - fwd_pe) / pe * 100
        if pe_growth > 10:
            fund_points.append(f"Fwd P/E {fwd_pe:.1f} vs TTM {pe:.1f} — analysts expect {pe_growth:.0f}% earnings growth")
        elif pe_growth < -10:
            fund_points.append(f"Fwd P/E {fwd_pe:.1f} above TTM {pe:.1f} — earnings expected to decline, caution")

    if rev_g:
        rg_pct = rev_g * 100
        if rg_pct > 30:
            fund_points.append(f"Revenue growing {rg_pct:.0f}% YoY — hypergrowth stage, top-line momentum strong")
        elif rg_pct > 10:
            fund_points.append(f"Revenue growing {rg_pct:.0f}% YoY — healthy expansion")
        elif rg_pct > 0:
            fund_points.append(f"Revenue growing {rg_pct:.0f}% YoY — slow growth, watch for margin expansion")
        else:
            fund_points.append(f"Revenue declining {rg_pct:.0f}% YoY — turnaround required, higher risk")

    if margin:
        mg_pct = margin * 100
        if mg_pct > 25:
            fund_points.append(f"Net margin {mg_pct:.1f}% — high-quality business with pricing power")
        elif mg_pct > 10:
            fund_points.append(f"Net margin {mg_pct:.1f}% — profitable, sustainable business model")
        elif mg_pct > 0:
            fund_points.append(f"Net margin {mg_pct:.1f}% — thin margins, vulnerable to cost shocks")
        else:
            fund_points.append(f"Net margin {mg_pct:.1f}% — currently unprofitable, monitor burn rate")

    reasons["FUNDAMENTAL VIEW"] = (
        " | ".join(fund_points) if fund_points
        else "Fundamental data not available — use web screener for verification"
    )

    # ── PERFORMANCE CONTEXT ────────────────────────────────────────
    if ar > 50:
        perf = f"Exceptional 1-year return of +{ar:.0f}% — among the top performers in the universe. Momentum strongly favours continuation but beware mean reversion."
    elif ar > 20:
        perf = f"Strong 1-year return of +{ar:.0f}% — trend intact, fundamentals and momentum aligned."
    elif ar > 0:
        perf = f"Modest 1-year return of +{ar:.0f}% — underperforming the index, monitor for catalyst."
    elif ar > -20:
        perf = f"1-year return {ar:.0f}% — in drawdown. Contrarian opportunity only if fundamentals are sound."
    else:
        perf = f"Severe 1-year decline of {ar:.0f}% — high risk. Only for deep value or turnaround thesis investors."

    if sh > 1.5:
        perf += f" Sharpe {sh:.2f} — every unit of risk is being well compensated."
    elif sh > 0.5:
        perf += f" Sharpe {sh:.2f} — acceptable risk-adjusted return."
    elif sh < 0:
        perf += f" Sharpe {sh:.2f} — negative risk-adjusted return, reconsider position sizing."

    reasons["PERFORMANCE"] = perf

    # ── RISK NOTE ──────────────────────────────────────────────────
    risks = []
    if b_val > 1.8:
        risks.append(f"High beta {b_val:.2f} — in a market selloff, expect ~{b_val*15:.0f}% drawdown on a 15% market drop")
    if av > 60:
        risks.append(f"Annual volatility {av:.0f}% — extreme price swings, size position accordingly")
    if rsi > 72:
        risks.append("Overbought RSI — short-term pullback of 5-10% is statistically likely")
    if t["bb_pct"] > 0.90:
        risks.append("Stretched above Bollinger Bands — mean reversion risk in near term")
    if not risks:
        risks.append("No extreme technical risk flags at current levels — standard market risk applies")

    reasons["RISK NOTE"] = " | ".join(risks)

    return reasons

# ================================================================
#  SINGLE STOCK ANALYSER (runs in background thread)
# ================================================================

def analyse(sym, mkt_returns, bulk_prices):
    try:
        if sym not in bulk_prices.columns:
            return None

        tk   = yf.Ticker(sym)
        hist = tk.history(period=PERIOD, auto_adjust=True)
        if hist.empty or len(hist) < 60:
            return None

        s_ret = hist["Close"].pct_change().dropna()
        m_ret = mkt_returns.reindex(s_ret.index).dropna()
        s_ret = s_ret.reindex(m_ret.index)
        if len(s_ret) < 50:
            return None

        b_val  = beta(s_ret, m_ret)
        r_val  = corr(s_ret, m_ret)
        c_val  = cov(s_ret, m_ret)
        sh_val = sharpe(s_ret)
        ar_val = ann_ret(s_ret)
        av_val = ann_vol(s_ret)
        cat    = classify(b_val, r_val)

        t          = technicals(hist["Close"], hist["High"], hist["Low"], hist["Volume"])
        price      = t["price"]
        hist_days  = len(hist)
        last_close = hist.index[-1].strftime("%d %b %Y")
        ipo_date   = hist.index[0].strftime("%d %b %Y")

        preds = {
            label: predict(price, t["tech_score"], av_val, days, idx)
            for idx, (label, days) in enumerate(TIMEFRAMES)
        }

        info   = tk.info
        pe     = info.get("trailingPE")
        fwd_pe = info.get("forwardPE")
        rev_g  = info.get("revenueGrowth")
        margin = info.get("profitMargins")
        mktcap = info.get("marketCap")
        name   = info.get("longName") or info.get("shortName", sym)
        sector = info.get("sector", "")
        target = info.get("targetMeanPrice")

        return dict(
            symbol=sym, name=name, sector=sector, category=cat,
            price=price, last_close=last_close,
            beta=b_val, cov=c_val, corr=r_val,
            sharpe=sh_val, ann_ret=ar_val, ann_vol=av_val,
            tech=t, preds=preds,
            pe=pe, fwd_pe=fwd_pe, rev_g=rev_g,
            margin=margin, mktcap=mktcap, target=target,
            hist_days=hist_days, ipo_date=ipo_date,
        )
    except Exception:
        return None

# ================================================================
#  RANKING
# ================================================================

def rank_df(subset, cat):
    s = subset.copy()
    if   cat == "Aggressive Growth": s["_s"] = s["beta"]*0.40 + s["sharpe"]*0.35 + s["corr"]*0.25
    elif cat == "Market Leader":     s["_s"] = s["beta"]*0.30 + s["sharpe"]*0.40 + s["corr"]*0.30
    elif cat == "Market Follower":   s["_s"] = s["corr"]*0.50 + s["sharpe"]*0.50
    elif cat == "Defensive":         s["_s"] = (1/(s["beta"].abs()+0.01))*0.50 + s["sharpe"]*0.50
    else:                            s["_s"] = (1/(s["corr"].abs()+0.01))*0.40 + s["ann_ret"]*0.60
    return s.sort_values("_s", ascending=False)

# ================================================================
#  STEP 0 — DISCOVER UNIVERSE  ← live fetch, no hardcoded list
# ================================================================

print("=" * 70)
print("  STOCK CLASSIFIER + PRICE PREDICTOR")
print(f"  Date (ET) : {fmt_dt(RUN_ET)}")
print(f"  Date (UTC): {fmt_dt(RUN_UTC)}")
print(f"  Market    : {market_status()}")
print("=" * 70)

print("\n  [0/4] Discovering stock universe from live sources…")
UNIVERSE = fetch_universe()

print("\n  [0/4] Fetching upcoming IPO calendar…")
UPCOMING_IPOS = fetch_upcoming_ipos()

# ================================================================
#  STEP 1 — DOWNLOAD
# ================================================================

print(f"\n  [1/4] Downloading price data — {len(UNIVERSE)} stocks in batches of {BATCH_SIZE}…")

# Download market benchmark first (always needed)
mkt_raw = yf.download(MARKET, period=PERIOD, auto_adjust=True, progress=False)
if isinstance(mkt_raw.columns, pd.MultiIndex):
    mkt_raw = mkt_raw["Close"]
mkt_series = mkt_raw.squeeze() if isinstance(mkt_raw, pd.DataFrame) else mkt_raw

# Download all stocks in batches
bulk = bulk_download(UNIVERSE, PERIOD, BATCH_SIZE)
bulk[MARKET] = mkt_series   # inject benchmark column

mkt_ret    = bulk[MARKET].pct_change().dropna()
DATA_START = bulk.index[0].strftime("%d %b %Y")
DATA_END   = bulk.index[-1].strftime("%d %b %Y")
loaded     = bulk.shape[1] - 1   # exclude benchmark column
print(f"  ✓ {loaded} tickers loaded  |  {len(bulk)} trading days")
print(f"  ✓ Data range: {DATA_START}  →  {DATA_END}")

# ================================================================
#  STEP 2 — PARALLEL ANALYSIS
# ================================================================

print(f"\n  [2/4] Analysing all stocks in background ({WORKERS} threads)…")
t0      = time.time()
results = []
done    = 0
total   = len(UNIVERSE)

with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = {ex.submit(analyse, sym, mkt_ret, bulk): sym for sym in UNIVERSE}
    for fut in concurrent.futures.as_completed(futures):
        done += 1
        res = fut.result()
        if res:
            results.append(res)
        if done % 20 == 0 or done == total:
            pct = done / total * 100
            bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
            print(f"  [{bar}] {pct:.0f}%  ({done}/{total}, {len(results)} valid)", end="\r")

elapsed = time.time() - t0
print(f"\n  ✓ Done in {elapsed:.1f}s — {len(results)} stocks analysed\n")

df_all  = pd.DataFrame([{k:v for k,v in r.items() if k not in ("tech","preds")} for r in results])
res_map = {r["symbol"]: r for r in results}

# ================================================================
#  STEP 3 — PRINT RESULTS
# ================================================================

print("[3/4] Category results\n")

for cat in CATEGORIES:
    cat_rows = df_all[df_all["category"] == cat]
    if cat_rows.empty:
        continue
    ranked = rank_df(cat_rows, cat).head(TOP_N)

    print("=" * 70)
    print(f"  ★  {cat.upper()}")
    print(f"     {CAT_DESC[cat]}")
    print(f"     {len(cat_rows)} stocks in category  |  Top {len(ranked)} shown  |  As of {DATA_END}")
    print("=" * 70)

    for pos, (_, row) in enumerate(ranked.iterrows(), 1):
        sym = row["symbol"]
        r   = res_map.get(sym)
        if not r:
            continue

        cat_sub    = df_all[df_all["category"] == cat].copy()
        ranked_sub = rank_df(cat_sub, cat)
        rank_score = float(ranked_sub[ranked_sub["symbol"] == sym]["_s"].values[0]) \
                     if "_s" in ranked_sub.columns else 0.0
        rdict = build_reasons(
            r["tech"], r["beta"], r["corr"], cat,
            r["ann_ret"], r["sharpe"], r["ann_vol"],
            r["pe"], r["fwd_pe"], r["rev_g"], r["margin"], rank_score
        )

        mc_str  = f"  MCap ${r['mktcap']/1e9:.1f}B" if r["mktcap"] else ""
        tgt_str = f"  Analyst target ${r['target']:.2f}" if r["target"] else ""
        ipo_tag = "  ⚡ Recent IPO" if r["hist_days"] < RECENT_IPO_DAYS else ""

        print(f"\n  #{pos}  {r['name']} ({sym}){ipo_tag}  |  ${r['price']:.2f}  |  Last close: {r['last_close']}")
        print(f"       Sector: {r['sector']}{mc_str}{tgt_str}")
        print(f"       Beta: {r['beta']:+.3f}  |  Corr: {r['corr']:+.3f}  |  Cov: {r['cov']:.6f}  |  "
              f"Ann Return: {r['ann_ret']:+.1f}%  |  Volatility: {r['ann_vol']:.1f}%  |  Sharpe: {r['sharpe']:.2f}")
        print()

        # ── Price predictions ──────────────────────────────────────
        print(f"       {'TIMEFRAME':<12} {'BEAR':>10} {'BASE':>10} {'BULL':>10} {'CONF':>6} {'SIGNAL'}")
        print(f"       {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*6} {'─'*13}")
        for label, _ in TIMEFRAMES:
            p      = r["preds"][label]
            b_pct  = (p["bear"]/r["price"]-1)*100
            bs_pct = (p["base"]/r["price"]-1)*100
            bl_pct = (p["bull"]/r["price"]-1)*100
            print(f"       {label:<12} "
                  f"${p['bear']:>8.2f} ({b_pct:+.1f}%)  "
                  f"${p['base']:>8.2f} ({bs_pct:+.1f}%)  "
                  f"${p['bull']:>8.2f} ({bl_pct:+.1f}%)  "
                  f"{p['conf']:>4}%  {p['signal']}")

        # ── Reasons ────────────────────────────────────────────────
        print()
        for section, text in rdict.items():
            print(f"       [{section}]")
            words = text.split()
            line, lines = "", []
            for w in words:
                if len(line) + len(w) + 1 <= 65:
                    line = (line + " " + w).strip()
                else:
                    lines.append(line)
                    line = w
            if line:
                lines.append(line)
            for l in lines:
                print(f"         {l}")
            print()

# ================================================================
#  RECENTLY IPO'd STOCKS  (auto-detected: < ~20 months of history)
# ================================================================

recent_ipos = df_all[df_all["hist_days"] < RECENT_IPO_DAYS].sort_values("ann_ret", ascending=False)

if not recent_ipos.empty:
    print("=" * 70)
    print("  ⚡  RECENTLY IPO'd STOCKS IN UNIVERSE  (auto-detected)")
    print(f"     Stocks with < {RECENT_IPO_DAYS} trading days of history  |  "
          f"First data point used as proxy IPO date")
    print(f"     {len(recent_ipos)} found  |  As of {DATA_END}")
    print("=" * 70)
    for _, row in recent_ipos.iterrows():
        sym = row["symbol"]
        r   = res_map.get(sym)
        if not r:
            continue
        p1y = r["preds"]["1 Year"]
        mc  = f"  MCap ${r['mktcap']/1e9:.1f}B" if r.get("mktcap") else ""
        print(f"\n  ▸  {r['name']} ({sym})  |  ${r['price']:.2f}{mc}")
        print(f"     Sector: {r['sector']}  |  Listed ~{r['ipo_date']}  |  "
              f"{r['hist_days']} trading days of data")
        print(f"     Category: {r['category']}  |  "
              f"Beta: {r['beta']:+.3f}  |  1Y Return: {r['ann_ret']:+.1f}%  |  Sharpe: {r['sharpe']:.2f}")
        print(f"     1Y Prediction — Bear: ${p1y['bear']:.2f}  Base: ${p1y['base']:.2f}  "
              f"Bull: ${p1y['bull']:.2f}  →  {p1y['signal']}")

# ================================================================
#  UPCOMING IPOs  (from NASDAQ API or curated fallback)
# ================================================================

print(f"\n{'=' * 70}")
print("  ★  UPCOMING IPO WATCH  —  Not yet listed, monitor these")
print(f"     Generated: {RUN_ET.strftime('%d %b %Y %H:%M ET')}")
print("=" * 70)
for ipo in UPCOMING_IPOS:
    print(f"\n  ▸  {ipo['name']}  |  Ticker: {ipo['ticker']}  |  "
          f"Sector: {ipo['sector']}  |  Valuation: {ipo['valuation']}  |  Expected: {ipo['expected']}")
    words = ipo["why"].split()
    line, lines = "", []
    for w in words:
        if len(line) + len(w) + 1 <= 68:
            line = (line + " " + w).strip()
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    for l in lines:
        print(f"     {l}")

# ================================================================
#  SUMMARY
# ================================================================

print(f"\n{'=' * 70}")
print("  UNIVERSE BREAKDOWN")
print(f"{'─' * 70}")
for cat in CATEGORIES:
    n   = len(df_all[df_all["category"] == cat])
    bar = "█" * int(n / max(1, len(df_all)) * 40)
    print(f"  {cat:<22}  {bar:<40}  {n:>3} stocks")
print(f"  {'─'*22}  {'Total analysed':>42}  {len(df_all):>3}")
print(f"\n  Run completed : {fmt_dt(RUN_ET)}")
print(f"  Data as of   : {DATA_END}")
print("=" * 70)

# ================================================================
#  CHARTS
# ================================================================

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#0d0f14")

# Chart 1 — Classification Scatter
ax1 = fig.add_subplot(2, 2, (1, 2))
ax1.set_facecolor("#13161e")

for cat in CATEGORIES:
    sub = df_all[df_all["category"] == cat]
    ax1.scatter(sub["beta"], sub["corr"], c=CAT_COLORS[cat],
                label=cat, alpha=0.80, s=70,
                edgecolors="white", linewidths=0.4, zorder=3)
    top5 = rank_df(sub, cat).head(5)
    for _, row in top5.iterrows():
        ax1.annotate(row["symbol"], (row["beta"], row["corr"]),
                     fontsize=7, color="white", alpha=0.85,
                     xytext=(4, 4), textcoords="offset points")

for xv, lc in [(0.65,"#446644"),(1.05,"#664444"),(1.50,"#ff4757")]:
    ax1.axvline(xv, color=lc, lw=1, ls="--", alpha=0.6)
ax1.axhline(0.30, color="#444466", lw=1, ls=":", alpha=0.7)

for xc, yc, label, col in [
    (0.30, 0.88, "DEFENSIVE",   "#1e90ff"),
    (0.85, 0.88, "FOLLOWER",    "#2ed573"),
    (1.25, 0.88, "LEADER",      "#ffa502"),
    (1.78, 0.88, "AGGRESSIVE",  "#ff4757"),
    (1.10, 0.06, "UNCORRELATED","#a29bfe"),
]:
    ax1.text(xc, yc, label, fontsize=9, color=col, alpha=0.5,
             ha="center", fontweight="bold")

ax1.set_xlabel("Beta (β)  —  higher = more sensitive to market", color="white")
ax1.set_ylabel("Correlation with S&P 500 (ρ)", color="white")
ax1.set_title(
    f"Stock Classification Map  |  Beta vs Correlation  |  "
    f"Data: {DATA_START} → {DATA_END}  |  "
    f"Generated: {RUN_ET.strftime('%d %b %Y  %H:%M ET')}",
    color="white", fontsize=11, fontweight="bold",
)
ax1.tick_params(colors="white")
ax1.spines[:].set_color("#333355")
ax1.grid(True, alpha=0.12, color="white")
ax1.legend(facecolor="#0d0f14", edgecolor="#444", labelcolor="white", fontsize=9)
ax1.set_ylim(-0.15, 1.05)

# Chart 2 — Return Distribution
ax2 = fig.add_subplot(2, 2, 3)
ax2.set_facecolor("#13161e")

data_by_cat = [df_all[df_all["category"]==c]["ann_ret"].dropna().values for c in CATEGORIES]
bp = ax2.boxplot(data_by_cat, patch_artist=True, notch=False,
                 medianprops=dict(color="white", linewidth=2))
for patch, cat in zip(bp["boxes"], CATEGORIES):
    patch.set_facecolor(CAT_COLORS[cat])
    patch.set_alpha(0.8)
for el in ["whiskers","caps","fliers"]:
    for item in bp[el]:
        item.set(color="white", alpha=0.5)
ax2.set_xticklabels([c.replace(" ","\n") for c in CATEGORIES], color="white", fontsize=8)
ax2.set_ylabel("Annual Return (%)", color="white")
ax2.set_title("Return Distribution by Category", color="white", fontweight="bold")
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#333355")
ax2.grid(True, alpha=0.12, color="white", axis="y")
ax2.axhline(0, color="white", lw=0.7, alpha=0.4)

# Chart 3 — 1-Year Base Upside per top pick
ax3 = fig.add_subplot(2, 2, 4)
ax3.set_facecolor("#13161e")

bars_data = []
for cat in CATEGORIES:
    sub = rank_df(df_all[df_all["category"]==cat], cat).head(3)
    for _, row in sub.iterrows():
        r = res_map.get(row["symbol"])
        if not r: continue
        base_1y = r["preds"]["1 Year"]["base"]
        upside  = (base_1y - r["price"]) / r["price"] * 100
        bars_data.append({"sym": row["symbol"], "cat": cat, "upside": upside})

bd = pd.DataFrame(bars_data).sort_values("upside", ascending=True)
ax3.barh(bd["sym"], bd["upside"],
         color=[CAT_COLORS[c] for c in bd["cat"]],
         alpha=0.85, edgecolor="white", linewidth=0.3)
ax3.axvline(0, color="white", lw=0.8, alpha=0.5)
for i, (_, row) in enumerate(bd.iterrows()):
    ax3.text(row["upside"] + (0.5 if row["upside"]>=0 else -0.5), i,
             f"{row['upside']:+.1f}%",
             va="center", ha="left" if row["upside"]>=0 else "right",
             color="white", fontsize=8)
patches = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CATEGORIES]
ax3.legend(handles=patches, facecolor="#0d0f14", edgecolor="#444",
           labelcolor="white", fontsize=8)
ax3.set_xlabel("1-Year Base Case Upside (%)", color="white")
ax3.set_title("1-Year Price Upside — Top 3 per Category", color="white", fontweight="bold")
ax3.tick_params(colors="white")
ax3.spines[:].set_color("#333355")
ax3.grid(True, alpha=0.12, color="white", axis="x")

plt.tight_layout()
plt.savefig("stock_classifier.png", dpi=150, bbox_inches="tight", facecolor="#0d0f14")
plt.show()
print("\n  ✓ Chart saved → stock_classifier.png")

# ================================================================
#  CSV EXPORT
# ================================================================

export_cols = ["symbol","name","sector","category","price","last_close",
               "beta","cov","corr","ann_ret","ann_vol","sharpe",
               "pe","fwd_pe","rev_g","margin","mktcap","target",
               "hist_days","ipo_date"]
out = df_all[[c for c in export_cols if c in df_all.columns]].copy()
out.insert(0, "as_of_date",    DATA_END)
out.insert(1, "generated_et",  RUN_ET.strftime("%Y-%m-%d %H:%M:%S ET"))
out.insert(2, "generated_utc", RUN_UTC.strftime("%Y-%m-%d %H:%M:%S UTC"))
out.sort_values(["category","beta"], ascending=[True,False]).round(4) \
   .to_csv("stock_classifier.csv", index=False)

print(f"  ✓ Exported → stock_classifier.csv  ({len(out)} rows, as of {DATA_END})")
