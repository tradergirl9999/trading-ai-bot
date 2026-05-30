# ================================================================
#  STOCK CLASSIFIER + MULTI-TIMEFRAME PRICE PREDICTOR
#  Google Colab — run all cells top to bottom
#
#  Formula:  Beta  = Cov(Stock, Market) / Var(Market)
#            Cov(X,Y) = E[XY] - E[X]E[Y]
#
#  Categories:
#    Aggressive Growth | Market Leader | Market Follower
#    Defensive Stock   | Uncorrelated Opportunity
#
#  Price Predictions:  1 Day | 1 Week | 1 Month | 3M | 6M | 1 Year
# ================================================================

# Cell 1 — Install
# !pip install yfinance pandas numpy matplotlib seaborn -q

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import concurrent.futures
import time
import datetime
import pytz
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ── Live timestamp helpers ────────────────────────────────────────
def now_utc() -> datetime.datetime:
    return datetime.datetime.now(pytz.utc)

def now_et() -> datetime.datetime:
    """Eastern Time (US market time zone)."""
    return datetime.datetime.now(pytz.timezone("America/New_York"))

def fmt_dt(dt: datetime.datetime) -> str:
    return dt.strftime("%A, %d %B %Y  %H:%M:%S %Z")

def fmt_date(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def market_status() -> str:
    et = now_et()
    weekday = et.weekday()          # 0=Mon … 6=Sun
    hour    = et.hour + et.minute / 60
    if weekday >= 5:
        return "CLOSED (Weekend)"
    if hour < 9.5:
        return f"PRE-MARKET  (opens in {int((9.5 - hour) * 60)} min)"
    if hour < 16.0:
        return "OPEN"
    return "AFTER-HOURS"

RUN_UTC = now_utc()
RUN_ET  = now_et()

# ================================================================
#  CONFIGURATION
# ================================================================

MARKET   = "^GSPC"      # S&P 500
PERIOD   = "2y"         # lookback for all calculations
RF_RATE  = 0.05         # annual risk-free rate
TOP_N    = 10           # stocks shown per category
WORKERS  = 12           # parallel threads (increase if Colab allows)

# ── Full stock universe (150 stocks + 10 recent IPOs) ───────────
STOCKS = [
    # ── Technology ─────────────────────────────────────────────
    "AAPL","MSFT","NVDA","GOOGL","META","AMD","INTC","CRM","ORCL","ADBE",
    "QCOM","TXN","NOW","SNOW","PLTR","PANW","AMAT","MU","LRCX","KLAC",
    "SMCI","DELL","HPE","CSCO","AVGO","MRVL","NXPI","ON","ENPH","SEDG",
    # ── Healthcare ──────────────────────────────────────────────
    "JNJ","UNH","LLY","ABBV","PFE","MRK","TMO","ABT","DHR","AMGN",
    "GILD","ISRG","CVS","BMY","VRTX","REGN","ZTS","SYK","BSX","ELV",
    "HCA","CI","CNC","MOH","DXCM","IDXX","IQV","ALGN","HOLX","PODD",
    # ── Financials ──────────────────────────────────────────────
    "JPM","BAC","WFC","GS","MS","BLK","C","AXP","SPGI","CB",
    "AON","MMC","PGR","TRV","USB","COF","SCHW","ICE","CME","MCO",
    "V","MA","PYPL","SQ","AFRM","UPST","LC","SOFI","NU","HOOD",
    # ── Consumer Discretionary ──────────────────────────────────
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","COST","LOW","BKNG",
    "MAR","HLT","GM","F","ABNB","EBAY","ETSY","RCL","CCL","LVS",
    # ── Consumer Staples ────────────────────────────────────────
    "PG","KO","PEP","WMT","MDLZ","CL","KMB","GIS","K","HSY",
    "MO","PM","STZ","CLX","EL","COTY","HRL","SJM","CAG","CPB",
    # ── Energy ──────────────────────────────────────────────────
    "XOM","CVX","COP","EOG","SLB","PSX","VLO","MPC","OXY","HAL",
    "BKR","DVN","FANG","APA","MRO","HES","CNQ","TTE","BP","SHEL",
    # ── Industrials ─────────────────────────────────────────────
    "CAT","BA","HON","UNP","LMT","RTX","GE","DE","MMM","UPS",
    "FDX","EMR","ETN","PH","ITW","CSX","NSC","WM","ROK","AME",
    # ── Materials ───────────────────────────────────────────────
    "LIN","APD","ECL","SHW","FCX","NEM","DOW","DD","NUE","CF",
    # ── Utilities ───────────────────────────────────────────────
    "NEE","DUK","SO","D","AEP","EXC","XEL","WEC","ES","ETR",
    # ── Real Estate ─────────────────────────────────────────────
    "AMT","PLD","CCI","EQIX","SPG","PSA","EQR","AVB","O","WY",
    # ── Communication ───────────────────────────────────────────
    "NFLX","DIS","CMCSA","T","VZ","CHTR","TMUS","WBD","PARA","FOX",
    # ── High Beta / Speculative ─────────────────────────────────
    "COIN","MSTR","MARA","RIOT","CLSK","IREN","BTBT",
    # ── Recent IPOs (2023-2025) ─────────────────────────────────
    "RDDT",   # Reddit       — IPO Mar 2024
    "ARM",    # ARM Holdings — IPO Sep 2023
    "CART",   # Instacart    — IPO Sep 2023
    "KVYO",   # Klaviyo      — IPO Sep 2023
    "CRWV",   # CoreWeave    — IPO Mar 2025
    "ASTS",   # AST SpaceMobile (recent listing)
    "RKLB",   # Rocket Lab
    "ACHR",   # Archer Aviation
    "JOBY",   # Joby Aviation
    "MNTN",   # Montauk Renewables
]

UPCOMING_IPOS = [
    {"name": "Klarna",        "ticker": "KLAR",  "sector": "Fintech",    "valuation": "~$15B",
     "note": "BNPL leader, profitable, strong EU market share. High growth potential."},
    {"name": "Chime",         "ticker": "CHYM",  "sector": "Neobank",    "valuation": "~$8B",
     "note": "US digital bank, 22M+ users. Delayed multiple times, IPO expected 2025."},
    {"name": "StubHub",       "ticker": "STUB",  "sector": "Ticketing",  "valuation": "~$16B",
     "note": "Live events marketplace. Revenue tied to concert/sports cycle."},
    {"name": "Cerebras",      "ticker": "CBRS",  "sector": "AI Chips",   "valuation": "~$4B",
     "note": "Wafer-scale AI chip. Competes with NVIDIA. High risk, massive upside if adopted."},
    {"name": "eToro",         "ticker": "ETOR",  "sector": "Fintech",    "valuation": "~$3.5B",
     "note": "Social trading platform. Profitable in 2023. Crypto-exposed revenue."},
    {"name": "Medline",       "ticker": "MDL",   "sector": "Healthcare", "valuation": "~$30B",
     "note": "Largest US private medical supply co. Defensive, steady revenue."},
    {"name": "Shein",         "ticker": "SHEIN", "sector": "Fast Fashion","valuation": "~$45B",
     "note": "Ultra-fast fashion. Massive scale but regulatory/ESG risk."},
    {"name": "Panera Brands", "ticker": "PNRA",  "sector": "Restaurants","valuation": "~$10B",
     "note": "Second re-IPO attempt. Stable but debt-heavy."},
]

# ================================================================
#  MATH FUNCTIONS
# ================================================================

def cov(x: pd.Series, y: pd.Series) -> float:
    """Cov(X,Y) = E[XY] - E[X]*E[Y]"""
    return float((x * y).mean() - x.mean() * y.mean())

def corr(x: pd.Series, y: pd.Series) -> float:
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0: return 0.0
    return cov(x, y) / (sx * sy)

def beta(s: pd.Series, m: pd.Series) -> float:
    """Beta = Cov(stock, market) / Var(market)"""
    var_m = m.var()
    return cov(s, m) / var_m if var_m != 0 else 0.0

def sharpe(r: pd.Series, rf=RF_RATE) -> float:
    excess = r - rf / 252
    return float(excess.mean() / excess.std() * np.sqrt(252)) if r.std() != 0 else 0.0

def ann_ret(r: pd.Series) -> float:
    return float(((1 + r).prod()) ** (252 / len(r)) - 1) * 100

def ann_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(252)) * 100

# ================================================================
#  TECHNICAL ANALYSIS
# ================================================================

def technicals(close: pd.Series, high: pd.Series, low: pd.Series, vol: pd.Series) -> dict:
    n = len(close)
    price = float(close.iloc[-1])

    sma20  = float(close.rolling(20).mean().iloc[-1])  if n >= 20  else price
    sma50  = float(close.rolling(50).mean().iloc[-1])  if n >= 50  else price
    sma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else price

    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd_h = float((ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()).iloc[-1])

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = float((100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1])

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = float((bb_mid + 2 * bb_std).iloc[-1])
    bb_lo  = float((bb_mid - 2 * bb_std).iloc[-1])
    bb_pct = (price - bb_lo) / (bb_up - bb_lo) if bb_up != bb_lo else 0.5

    tr    = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr   = float(tr.rolling(14).mean().iloc[-1]) if n >= 14 else float(tr.mean())

    stoch_k = None
    if n >= 17:
        lo14 = low.rolling(14).min()
        hi14 = high.rolling(14).max()
        stoch_k = float(((close - lo14) / (hi14 - lo14) * 100).rolling(3).mean().iloc[-1])

    vol_ratio = float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]) if n >= 20 else 1.0
    mom20     = (price / float(close.iloc[-21]) - 1) * 100 if n >= 21 else 0.0

    # Bull score (0.0–1.0)
    signals = [
        price > sma20,
        price > sma50,
        price > sma200,
        macd_h > 0,
        rsi > 50,
        bb_pct > 0.5,
        stoch_k is not None and stoch_k > 50,
        mom20 > 0,
    ]
    tech_score = sum(signals) / len(signals)

    return {
        "price": price, "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "macd_h": macd_h, "rsi": rsi,
        "bb_up": bb_up, "bb_lo": bb_lo, "bb_pct": bb_pct,
        "atr": atr, "stoch_k": stoch_k,
        "vol_ratio": vol_ratio, "mom20": mom20,
        "tech_score": tech_score,
    }

# ================================================================
#  PRICE PREDICTION MODEL
#  Uses historical volatility + technical score to build
#  Bear / Base / Bull targets for each timeframe
# ================================================================

TIMEFRAMES = [
    ("1 Day",    1),
    ("1 Week",   5),
    ("1 Month",  22),
    ("3 Months", 63),
    ("6 Months", 126),
    ("1 Year",   252),
]

def predict(price: float, tech_score: float, hist_vol_annual: float, days: int, tf_idx: int) -> dict:
    """
    tech_score 0→1 maps to daily_drift:
      0.0 → -0.06% / day  (≈ -15% annual)
      0.5 → +0.032% / day (≈ +8%  annual, market avg)
      1.0 → +0.12% / day  (≈ +30% annual)
    """
    daily_drift  = -0.0006 + tech_score * 0.0018
    period_ret   = daily_drift * days
    period_vol   = (hist_vol_annual / 100) * np.sqrt(days / 252)

    base = price * (1 + period_ret)
    bull = price * (1 + period_ret + period_vol * 1.15)
    bear = price * (1 + period_ret - period_vol * 1.15)

    # Confidence decreases with horizon
    confidence = max(30, 78 - tf_idx * 8)

    # Signal from tech_score
    if tech_score > 0.74:   signal = "STRONG BUY"
    elif tech_score > 0.58: signal = "BUY"
    elif tech_score > 0.42: signal = "HOLD"
    elif tech_score > 0.26: signal = "SELL"
    else:                   signal = "STRONG SELL"

    return {"bear": bear, "base": base, "bull": bull,
            "confidence": confidence, "signal": signal}

# ================================================================
#  CLASSIFICATION
# ================================================================

CAT_COLORS = {
    "Aggressive Growth": "#ff4757",
    "Market Leader":     "#ffa502",
    "Market Follower":   "#2ed573",
    "Defensive":         "#1e90ff",
    "Uncorrelated":      "#a29bfe",
}
CATEGORIES = list(CAT_COLORS.keys())

CAT_DESC = {
    "Aggressive Growth": "β ≥ 1.5 & ρ ≥ 0.55 — amplifies every market move",
    "Market Leader":     "1.05 ≤ β < 1.5 & ρ ≥ 0.60 — outpaces index on up-days",
    "Market Follower":   "0.65 ≤ β < 1.05 & ρ ≥ 0.50 — reliable index tracker",
    "Defensive":         "β < 0.65 & ρ ≥ 0.25 — holds value in downturns",
    "Uncorrelated":      "|ρ| < 0.30 — moves independently, great diversifier",
}

def classify(b: float, r: float) -> str:
    if abs(r) < 0.30:                        return "Uncorrelated"
    if b >= 1.5   and r >= 0.55:             return "Aggressive Growth"
    if 1.05 <= b < 1.5 and r >= 0.60:       return "Market Leader"
    if 0.65 <= b < 1.05 and r >= 0.50:      return "Market Follower"
    if b < 0.65:                             return "Defensive"
    return "Market Follower"

# ================================================================
#  REASON GENERATOR
# ================================================================

def reasons(t: dict, b_val: float, r_val: float, cat: str, ar: float, sh: float) -> list[str]:
    out = []

    # RSI
    if t["rsi"] > 72:
        out.append(f"RSI {t['rsi']:.0f} — overbought, momentum strong but pullback risk")
    elif t["rsi"] < 30:
        out.append(f"RSI {t['rsi']:.0f} — oversold, mean-reversion bounce likely")
    elif t["rsi"] > 55:
        out.append(f"RSI {t['rsi']:.0f} — healthy bullish momentum")
    else:
        out.append(f"RSI {t['rsi']:.0f} — neutral, no clear momentum signal")

    # Trend
    if t["price"] > t["sma200"]:
        out.append("Price above 200-day SMA — confirmed long-term uptrend")
    else:
        out.append("Price below 200-day SMA — long-term downtrend, caution warranted")

    # MACD
    if t["macd_h"] > 0:
        out.append("MACD histogram positive — bullish crossover in effect")
    else:
        out.append("MACD histogram negative — bearish pressure on short-term momentum")

    # Category-specific
    if cat == "Aggressive Growth":
        out.append(f"Beta {b_val:.2f} — stock amplifies S&P 500 by {b_val:.1f}x, ideal in bull markets")
    elif cat == "Market Leader":
        out.append(f"Beta {b_val:.2f} — consistently outpaces index, sector leadership evident")
    elif cat == "Defensive":
        out.append(f"Low beta {b_val:.2f} — capital preservation in volatility, low drawdowns")
    elif cat == "Uncorrelated":
        out.append(f"Market correlation only {r_val:.2f} — stock-specific drivers dominate")
    elif cat == "Market Follower":
        out.append(f"Beta {b_val:.2f}, correlation {r_val:.2f} — reliable market proxy")

    # Performance
    if ar > 40:
        out.append(f"Exceptional 1-year return (+{ar:.0f}%) — momentum favours continuation")
    elif ar > 15:
        out.append(f"Solid 1-year return (+{ar:.0f}%) — trend intact")
    elif ar < -20:
        out.append(f"Weak 1-year return ({ar:.0f}%) — contrarian opportunity or value trap, verify fundamentals")
    else:
        out.append(f"1-year return {ar:+.0f}% — moderate performance")

    # Sharpe
    if sh > 1.5:
        out.append(f"Sharpe {sh:.2f} — excellent risk-adjusted return, reward justifies risk")
    elif sh > 0.8:
        out.append(f"Sharpe {sh:.2f} — good risk-adjusted performance")
    elif sh < 0:
        out.append(f"Negative Sharpe {sh:.2f} — risk is not being compensated, reassess position sizing")

    # Volume
    if t["vol_ratio"] > 1.8:
        out.append(f"Volume {t['vol_ratio']:.1f}x above average — strong institutional activity")
    elif t["vol_ratio"] < 0.5:
        out.append(f"Volume only {t['vol_ratio']:.1f}x average — low conviction, wait for breakout")

    # BB
    if t["bb_pct"] > 0.90:
        out.append("Near upper Bollinger Band — extended, short-term consolidation expected")
    elif t["bb_pct"] < 0.10:
        out.append("Near lower Bollinger Band — potential bounce zone")

    return out[:6]   # max 6 reasons

# ================================================================
#  SINGLE STOCK ANALYSIS (runs in thread)
# ================================================================

def analyse(sym: str, mkt_returns: pd.Series, all_prices: pd.DataFrame) -> dict | None:
    try:
        if sym not in all_prices.columns:
            return None

        prices = all_prices[sym].dropna()
        if len(prices) < 60:
            return None

        # Get high/low from individual ticker (bulk download only gives Close)
        tk = yf.Ticker(sym)
        hist = tk.history(period=PERIOD, auto_adjust=True)
        if hist.empty or len(hist) < 60:
            return None

        s_ret = hist["Close"].pct_change().dropna()
        m_ret = mkt_returns.reindex(s_ret.index).dropna()
        s_ret = s_ret.reindex(m_ret.index)

        b_val  = beta(s_ret, m_ret)
        r_val  = corr(s_ret, m_ret)
        c_val  = cov(s_ret, m_ret)
        sh_val = sharpe(s_ret)
        ar_val = ann_ret(s_ret)
        av_val = ann_vol(s_ret)
        cat    = classify(b_val, r_val)

        t = technicals(hist["Close"], hist["High"], hist["Low"], hist["Volume"])
        price = t["price"]

        # Price predictions for all 6 timeframes
        preds = {}
        for idx, (label, days) in enumerate(TIMEFRAMES):
            preds[label] = predict(price, t["tech_score"], av_val, days, idx)

        # Fundamentals (quick fetch)
        info   = tk.info
        pe     = info.get("trailingPE")
        fwd_pe = info.get("forwardPE")
        rev_g  = info.get("revenueGrowth")
        margin = info.get("profitMargins")
        mktcap = info.get("marketCap")
        name   = info.get("longName") or info.get("shortName", sym)
        sector = info.get("sector", "")

        rs = reasons(t, b_val, r_val, cat, ar_val, sh_val)

        last_date = hist.index[-1]
        last_close_str = (last_date.strftime("%d %b %Y")
                          if hasattr(last_date, "strftime")
                          else str(last_date)[:10])

        return {
            "symbol":     sym,
            "name":       name,
            "sector":     sector,
            "category":   cat,
            "price":      price,
            "last_close": last_close_str,
            "beta":       b_val,
            "cov":        c_val,
            "corr":       r_val,
            "sharpe":     sh_val,
            "ann_ret":    ar_val,
            "ann_vol":    av_val,
            "tech":       t,
            "preds":      preds,
            "pe":         pe,
            "fwd_pe":     fwd_pe,
            "rev_g":      rev_g,
            "margin":     margin,
            "mktcap":     mktcap,
            "reasons":    rs,
        }
    except Exception:
        return None

# ================================================================
#  PARALLEL EXECUTION
# ================================================================

print("=" * 70)
print("  STOCK CLASSIFIER + PRICE PREDICTOR")
print(f"  Run date : {fmt_dt(RUN_ET)}")
print(f"  Run date : {fmt_dt(RUN_UTC)}")
print(f"  Market   : {market_status()}")
print(f"  Universe : {len(STOCKS)} stocks  |  Period: {PERIOD}  |  Benchmark: S&P 500")
print("=" * 70)

# Bulk download prices for fast parallel access
print("\n  [1/3] Downloading market + universe prices…")
all_tickers = STOCKS + [MARKET]
bulk = yf.download(all_tickers, period=PERIOD, auto_adjust=True, progress=False)["Close"]
if isinstance(bulk, pd.Series):
    bulk = bulk.to_frame()
mkt_ret = bulk[MARKET].pct_change().dropna()

DATA_START = bulk.index[0].strftime("%d %b %Y")
DATA_END   = bulk.index[-1].strftime("%d %b %Y")
print(f"  ✓ {bulk.shape[1]-1} instruments  |  {len(bulk)} trading days")
print(f"  ✓ Data range: {DATA_START} → {DATA_END}  (latest close)")

print(f"\n  [2/3] Analysing stocks in parallel ({WORKERS} threads)…")
t0      = time.time()
results = []
done    = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = {ex.submit(analyse, sym, mkt_ret, bulk): sym for sym in STOCKS}
    for fut in concurrent.futures.as_completed(futures):
        done += 1
        res = fut.result()
        if res:
            results.append(res)
        if done % 20 == 0 or done == len(STOCKS):
            pct = done / len(STOCKS) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{bar}] {pct:.0f}%  ({done}/{len(STOCKS)} done, {len(results)} valid)", end="\r")

print(f"\n  ✓ Completed in {time.time()-t0:.1f}s — {len(results)} stocks analysed\n")

df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("tech","preds","reasons")}
                   for r in results])

# ================================================================
#  RANKING WITHIN CATEGORY
# ================================================================

def rank(subset, cat):
    s = subset.copy()
    if cat == "Aggressive Growth":
        s["_s"] = s["beta"] * 0.4 + s["sharpe"] * 0.35 + s["corr"] * 0.25
    elif cat == "Market Leader":
        s["_s"] = s["beta"] * 0.3 + s["sharpe"] * 0.4  + s["corr"] * 0.3
    elif cat == "Market Follower":
        s["_s"] = s["corr"] * 0.5 + s["sharpe"] * 0.5
    elif cat == "Defensive":
        s["_s"] = (1 / (s["beta"].abs() + 0.01)) * 0.5 + s["sharpe"] * 0.5
    else:  # Uncorrelated
        s["_s"] = (1 / (s["corr"].abs() + 0.01)) * 0.4 + s["ann_ret"] * 0.6
    return s.sort_values("_s", ascending=False)

# Map symbol → full result dict
res_map = {r["symbol"]: r for r in results}

# ================================================================
#  OUTPUT — FULL RESULTS WITH PRICE PREDICTIONS + REASONS
# ================================================================

print("[3/3] Results\n")

for cat in CATEGORIES:
    cat_rows = df[df["category"] == cat]
    if cat_rows.empty:
        continue
    ranked   = rank(cat_rows, cat).head(TOP_N)

    print("=" * 70)
    print(f"  ★  {cat.upper()}")
    print(f"     {CAT_DESC[cat]}")
    print(f"     {len(cat_rows)} stocks in category  |  Showing top {len(ranked)}  |  As of {DATA_END}")
    print("=" * 70)

    for _, row in ranked.iterrows():
        sym  = row["symbol"]
        r    = res_map.get(sym)
        if not r:
            continue

        pe_str  = f"P/E {r['pe']:.1f}" if r["pe"]  else ""
        fpe_str = f"Fwd P/E {r['fwd_pe']:.1f}" if r["fwd_pe"] else ""
        rg_str  = f"Rev Growth {r['rev_g']*100:.1f}%" if r["rev_g"] else ""
        mg_str  = f"Margin {r['margin']*100:.1f}%" if r["margin"] else ""
        mc_str  = f"MCap ${r['mktcap']/1e9:.1f}B" if r["mktcap"] else ""
        fundamentals_line = "  ".join(filter(None, [pe_str, fpe_str, rg_str, mg_str, mc_str]))

        print(f"\n  ▸  {r['name']} ({sym})  |  ${r['price']:.2f}  |  Last close: {r.get('last_close', DATA_END)}  |  {r['sector']}")
        print(f"     Beta: {r['beta']:+.3f}  |  Corr: {r['corr']:+.3f}  |  "
              f"Ann Return: {r['ann_ret']:+.1f}%  |  Sharpe: {r['sharpe']:.2f}  |  "
              f"Vol: {r['ann_vol']:.1f}%")
        if fundamentals_line:
            print(f"     {fundamentals_line}")

        # Price prediction table
        print()
        print(f"     {'TIMEFRAME':<12} {'BEAR':>10} {'BASE':>10} {'BULL':>10} {'CONF':>6} {'SIGNAL'}")
        print(f"     {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*6} {'─'*12}")
        for label, _ in TIMEFRAMES:
            p = r["preds"][label]
            print(f"     {label:<12} "
                  f"${p['bear']:>9.2f} "
                  f"${p['base']:>9.2f} "
                  f"${p['bull']:>9.2f} "
                  f"{p['confidence']:>5}% "
                  f"{p['signal']}")

        # Reasons
        print()
        print("     REASONS:")
        for i, reason in enumerate(r["reasons"], 1):
            print(f"     {i}. {reason}")
        print()

# ================================================================
#  UPCOMING IPOS SECTION
# ================================================================

print("=" * 70)
print("  ★  UPCOMING IPO WATCH")
print("     Stocks not yet listed — monitor for listing date")
print("=" * 70)
print()
for ipo in UPCOMING_IPOS:
    print(f"  ▸  {ipo['name']} (Expected: {ipo['ticker']})  |  {ipo['sector']}  |  Valuation: {ipo['valuation']}")
    print(f"     {ipo['note']}")
    print()

# ================================================================
#  SUMMARY TABLE
# ================================================================

print("=" * 70)
print("  UNIVERSE BREAKDOWN")
print("=" * 70)
for cat in CATEGORIES:
    n   = len(df[df["category"] == cat])
    bar = "█" * int(n / max(1, len(df)) * 40)
    print(f"  {cat:<22} {bar:<40} {n:>3}")
print(f"  {'─'*22}  {'Total screened':>42} {len(df):>3}")

# ================================================================
#  VISUALISATION
# ================================================================

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#0d0f14")

# ── Chart 1: Classification Scatter ──────────────────────────────
ax1 = fig.add_subplot(2, 2, (1, 2))
ax1.set_facecolor("#13161e")

for cat in CATEGORIES:
    sub = df[df["category"] == cat]
    ax1.scatter(sub["beta"], sub["corr"],
                c=CAT_COLORS[cat], label=cat,
                alpha=0.80, s=70,
                edgecolors="white", linewidths=0.4, zorder=3)
    # Label top picks in each category
    top5 = rank(sub, cat).head(5)
    for _, row in top5.iterrows():
        ax1.annotate(row["symbol"],
                     (row["beta"], row["corr"]),
                     fontsize=7, color="white", alpha=0.85,
                     xytext=(4, 4), textcoords="offset points")

for xv, lc in [(0.65,"#446644"),(1.05,"#664444"),(1.50,"#ff4757")]:
    ax1.axvline(xv, color=lc, lw=1, ls="--", alpha=0.6)
ax1.axhline(0.30, color="#444466", lw=1, ls=":", alpha=0.7)

ax1.set_xlabel("Beta (β)  —  higher = more sensitive to market", color="white")
ax1.set_ylabel("Correlation with S&P 500 (ρ)", color="white")
ax1.set_title("Stock Classification Map  |  Beta vs Market Correlation",
              color="white", fontsize=12, fontweight="bold")
ax1.tick_params(colors="white")
ax1.spines[:].set_color("#333355")
ax1.grid(True, alpha=0.12, color="white")
ax1.legend(facecolor="#0d0f14", edgecolor="#444", labelcolor="white", fontsize=9)

for xc, yc, label, col in [
    (0.30, 0.82, "DEFENSIVE",  "#1e90ff"),
    (0.85, 0.82, "FOLLOWER",   "#2ed573"),
    (1.25, 0.82, "LEADER",     "#ffa502"),
    (1.75, 0.82, "AGGRESSIVE", "#ff4757"),
    (1.10, 0.08, "UNCORRELATED","#a29bfe"),
]:
    ax1.text(xc, yc, label, fontsize=9, color=col, alpha=0.5,
             ha="center", fontweight="bold")

# ── Chart 2: Annual Return by Category (box) ─────────────────────
ax2 = fig.add_subplot(2, 2, 3)
ax2.set_facecolor("#13161e")

data_by_cat = [df[df["category"] == c]["ann_ret"].values for c in CATEGORIES]
bp = ax2.boxplot(data_by_cat, patch_artist=True, notch=False,
                 medianprops=dict(color="white", linewidth=2))
for patch, cat in zip(bp["boxes"], CATEGORIES):
    patch.set_facecolor(CAT_COLORS[cat])
    patch.set_alpha(0.8)
for element in ["whiskers","caps","fliers"]:
    for item in bp[element]:
        item.set(color="white", alpha=0.5)

ax2.set_xticklabels([c.replace(" ", "\n") for c in CATEGORIES],
                    color="white", fontsize=8)
ax2.set_ylabel("Annual Return (%)", color="white")
ax2.set_title("Return Distribution by Category", color="white", fontweight="bold")
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#333355")
ax2.grid(True, alpha=0.12, color="white", axis="y")
ax2.axhline(0, color="white", lw=0.7, alpha=0.4)

# ── Chart 3: Top 3 per category — base price predictions bar ──────
ax3 = fig.add_subplot(2, 2, 4)
ax3.set_facecolor("#13161e")

bars_data = []
for cat in CATEGORIES:
    sub = rank(df[df["category"] == cat], cat).head(3)
    for _, row in sub.iterrows():
        r = res_map.get(row["symbol"])
        if not r: continue
        base_1y  = r["preds"]["1 Year"]["base"]
        upside   = (base_1y - r["price"]) / r["price"] * 100
        bars_data.append({"sym": row["symbol"], "cat": cat, "upside": upside})

bd = pd.DataFrame(bars_data).sort_values("upside", ascending=True)
colors = [CAT_COLORS[c] for c in bd["cat"]]
ax3.barh(bd["sym"], bd["upside"], color=colors, alpha=0.85,
         edgecolor="white", linewidth=0.3)
ax3.axvline(0, color="white", lw=0.8, alpha=0.5)
for i, (_, row) in enumerate(bd.iterrows()):
    ax3.text(row["upside"] + (0.5 if row["upside"] >= 0 else -0.5),
             i, f"{row['upside']:+.1f}%",
             va="center", ha="left" if row["upside"] >= 0 else "right",
             color="white", fontsize=8)

patches = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CATEGORIES]
ax3.legend(handles=patches, facecolor="#0d0f14", edgecolor="#444",
           labelcolor="white", fontsize=8)
ax3.set_xlabel("1-Year Base Upside (%)", color="white")
ax3.set_title("1-Year Price Upside — Top 3 per Category", color="white", fontweight="bold")
ax3.tick_params(colors="white")
ax3.spines[:].set_color("#333355")
ax3.grid(True, alpha=0.12, color="white", axis="x")

plt.suptitle(
    f"Stock Classifier + Price Predictor  |  S&P 500 Benchmark  |  {PERIOD}  |  "
    f"Data: {DATA_START} → {DATA_END}  |  Generated: {RUN_ET.strftime('%d %b %Y %H:%M ET')}  |  N={len(df)}",
    color="white", fontsize=11, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("stock_classifier.png", dpi=150, bbox_inches="tight", facecolor="#0d0f14")
plt.show()
print("\n  ✓ Chart saved → stock_classifier.png")

# ================================================================
#  EXPORT CSV
# ================================================================

export_cols = ["symbol","name","sector","category","price","beta","cov",
               "corr","ann_ret","ann_vol","sharpe","pe","fwd_pe","rev_g","margin"]
out = df[export_cols].sort_values(["category","beta"], ascending=[True, False]).round(4).copy()
out.insert(0, "as_of_date",    DATA_END)
out.insert(1, "generated_utc", RUN_UTC.strftime("%Y-%m-%d %H:%M:%S UTC"))
out.insert(2, "generated_et",  RUN_ET.strftime("%Y-%m-%d %H:%M:%S ET"))
out.to_csv("stock_classifier.csv", index=False)
print(f"  ✓ Data exported → stock_classifier.csv  (as of {DATA_END})")
print(f"  ✓ Generated: {fmt_dt(RUN_ET)}")
print("=" * 70)
print("=" * 70)
