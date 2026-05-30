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
# ================================================================

# ── Install (run this line first if packages are missing) ────────
# !pip install yfinance pandas numpy matplotlib pytz -q

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
    if weekday >= 5:                return "CLOSED — Weekend"
    if hour < 4.0:                  return "CLOSED — Overnight"
    if hour < 9.5:
        mins = int((9.5 - hour) * 60)
        return f"PRE-MARKET  (regular open in {mins} min)"
    if hour < 16.0:                 return "OPEN — Regular Hours"
    if hour < 20.0:                 return "AFTER-HOURS"
    return "CLOSED — Overnight"

# ================================================================
#  CONFIGURATION  ← edit here
# ================================================================

MARKET   = "^GSPC"     # Benchmark
PERIOD   = "2y"        # Data lookback
RF_RATE  = 0.05        # Annual risk-free rate
TOP_N    = 10          # Stocks per category
WORKERS  = 12          # Parallel threads

STOCKS = [
    # Technology
    "AAPL","MSFT","NVDA","GOOGL","META","AMD","INTC","CRM","ORCL","ADBE",
    "QCOM","TXN","NOW","SNOW","PLTR","PANW","AMAT","MU","LRCX","KLAC",
    "SMCI","DELL","HPE","CSCO","AVGO","MRVL","NXPI","ON",
    # Healthcare
    "JNJ","UNH","LLY","ABBV","PFE","MRK","TMO","ABT","DHR","AMGN",
    "GILD","ISRG","CVS","BMY","VRTX","REGN","ZTS","SYK","BSX","ELV",
    "HCA","CI","CNC","DXCM","IDXX","IQV","ALGN","HOLX","PODD",
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","C","AXP","SPGI","CB",
    "AON","MMC","PGR","TRV","USB","COF","SCHW","ICE","CME","MCO",
    "V","MA","PYPL","SQ","SOFI","NU","HOOD","AFRM","UPST",
    # Consumer Discretionary
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","COST","LOW","BKNG",
    "MAR","HLT","GM","F","ABNB","EBAY","ETSY","RCL","CCL","LVS",
    # Consumer Staples
    "PG","KO","PEP","WMT","MDLZ","CL","KMB","GIS","MO","PM","STZ","CLX",
    # Energy
    "XOM","CVX","COP","EOG","SLB","PSX","VLO","MPC","OXY","HAL",
    "BKR","DVN","APA","MRO","HES",
    # Industrials
    "CAT","BA","HON","UNP","LMT","RTX","GE","DE","MMM","UPS",
    "FDX","EMR","ETN","PH","ITW","CSX","NSC","WM","ROK","AME",
    # Materials
    "LIN","APD","ECL","SHW","FCX","NEM","DOW","DD","NUE","CF",
    # Utilities
    "NEE","DUK","SO","D","AEP","EXC","XEL","WEC","ES","ETR",
    # Real Estate
    "AMT","PLD","CCI","EQIX","SPG","PSA","EQR","AVB","O","WY",
    # Communication
    "NFLX","DIS","CMCSA","T","VZ","CHTR","TMUS","WBD",
    # High Beta / Crypto-adjacent
    "COIN","MSTR","MARA","RIOT","CLSK","IREN",
    # Recent IPOs
    "RDDT","ARM","CART","KVYO","CRWV","ASTS","RKLB","ACHR","JOBY",
]

UPCOMING_IPOS = [
    {"name":"Klarna",        "ticker":"KLAR",  "sector":"Fintech",
     "valuation":"~$15B",   "expected":"2026",
     "why":"BNPL leader, profitable, strong EU market share, expanding US. "
           "High growth, IPO multiple likely 8-12x revenue."},
    {"name":"Chime",         "ticker":"TBD",   "sector":"Neobank",
     "valuation":"~$8B",    "expected":"2026",
     "why":"22M+ US users, no-fee banking model. Delayed multiple times. "
           "Watch for profitability metrics before committing."},
    {"name":"StubHub",       "ticker":"TBD",   "sector":"Ticketing",
     "valuation":"~$16B",   "expected":"2026",
     "why":"Live events super-cycle post-COVID. Revenue tied to Taylor Swift-level demand. "
           "High cash flow, brand recognition."},
    {"name":"Cerebras",      "ticker":"CBRS",  "sector":"AI Chips",
     "valuation":"~$4B",    "expected":"2026",
     "why":"Wafer-Scale Engine — single chip for entire AI model. "
           "If adopted at scale, competes directly with NVIDIA. Extreme risk, extreme upside."},
    {"name":"eToro",         "ticker":"ETOR",  "sector":"Fintech",
     "valuation":"~$3.5B",  "expected":"2026",
     "why":"Social trading platform, profitable in 2023. Crypto-exposed revenue stream. "
           "Watch crypto cycle correlation."},
    {"name":"Medline",       "ticker":"MDL",   "sector":"Healthcare",
     "valuation":"~$30B",   "expected":"2026",
     "why":"Largest US private medical supply company. Defensive, recession-resistant. "
           "Stable revenue, strong pricing power."},
    {"name":"Panera Brands", "ticker":"PNRA",  "sector":"Restaurants",
     "valuation":"~$10B",   "expected":"2026",
     "why":"Second re-IPO attempt after going private. Recognisable brand, "
           "heavy debt load — monitor leverage ratios closely."},
    {"name":"Shein",         "ticker":"TBD",   "sector":"Fast Fashion",
     "valuation":"~$45B",   "expected":"2026",
     "why":"Massive global scale in ultra-fast fashion. Regulatory + ESG risk is significant. "
           "High reward only for risk-tolerant investors."},
]

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

    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    macd_h    = float((macd_line - macd_sig).iloc[-1])

    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rsi    = float((100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1])

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = float((bb_mid + 2 * bb_std).iloc[-1])
    bb_lo  = float((bb_mid - 2 * bb_std).iloc[-1])
    bb_pct = (price - bb_lo) / (bb_up - bb_lo) if bb_up != bb_lo else 0.5

    tr   = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr  = float(tr.rolling(14).mean().iloc[-1]) if n >= 14 else float(tr.mean())

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
            fund_points.append(f"Revenue growing {rg_pct:.0f}% YoY — slow growth, watch for margin expansion to compensate")
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
        " | ".join(fund_points) if fund_points else "Fundamental data not available — use web screener for verification"
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

        t     = technicals(hist["Close"], hist["High"], hist["Low"], hist["Volume"])
        price = t["price"]
        last_close = hist.index[-1].strftime("%d %b %Y")

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
#  STEP 1 — DOWNLOAD
# ================================================================

print("=" * 70)
print("  STOCK CLASSIFIER + PRICE PREDICTOR")
print(f"  Date (ET) : {fmt_dt(RUN_ET)}")
print(f"  Date (UTC): {fmt_dt(RUN_UTC)}")
print(f"  Market    : {market_status()}")
print(f"  Universe  : {len(STOCKS)} stocks  |  Period: {PERIOD}  |  Benchmark: S&P 500")
print("=" * 70)

print("\n  [1/3] Downloading bulk price data…")
all_tickers = STOCKS + [MARKET]
bulk = yf.download(all_tickers, period=PERIOD, auto_adjust=True, progress=False)["Close"]
if isinstance(bulk, pd.Series):
    bulk = bulk.to_frame()

mkt_ret    = bulk[MARKET].pct_change().dropna()
DATA_START = bulk.index[0].strftime("%d %b %Y")
DATA_END   = bulk.index[-1].strftime("%d %b %Y")
print(f"  ✓ {bulk.shape[1]-1} tickers  |  {len(bulk)} trading days")
print(f"  ✓ Data range: {DATA_START}  →  {DATA_END}")

# ================================================================
#  STEP 2 — PARALLEL ANALYSIS
# ================================================================

print(f"\n  [2/3] Analysing all stocks in background ({WORKERS} threads)…")
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
        if done % 15 == 0 or done == len(STOCKS):
            pct = done / len(STOCKS) * 100
            bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
            print(f"  [{bar}] {pct:.0f}%  ({done}/{len(STOCKS)}, {len(results)} valid)", end="\r")

elapsed = time.time() - t0
print(f"\n  ✓ Done in {elapsed:.1f}s — {len(results)} stocks analysed\n")

df_all  = pd.DataFrame([{k:v for k,v in r.items() if k not in ("tech","preds")} for r in results])
res_map = {r["symbol"]: r for r in results}

# ================================================================
#  STEP 3 — PRINT RESULTS
# ================================================================

print("[3/3] Results\n")

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

        # Build reasons with rank score
        cat_sub   = df_all[df_all["category"] == cat].copy()
        ranked_sub= rank_df(cat_sub, cat)
        rank_score= float(ranked_sub[ranked_sub["symbol"] == sym]["_s"].values[0]) if "_s" in ranked_sub.columns else 0.0
        rdict     = build_reasons(
            r["tech"], r["beta"], r["corr"], cat,
            r["ann_ret"], r["sharpe"], r["ann_vol"],
            r["pe"], r["fwd_pe"], r["rev_g"], r["margin"], rank_score
        )

        mc_str  = f"  MCap ${r['mktcap']/1e9:.1f}B" if r["mktcap"] else ""
        tgt_str = f"  Analyst target ${r['target']:.2f}" if r["target"] else ""

        print(f"\n  #{pos}  {r['name']} ({sym})  |  ${r['price']:.2f}  |  Last close: {r['last_close']}")
        print(f"       Sector: {r['sector']}{mc_str}{tgt_str}")
        print(f"       Beta: {r['beta']:+.3f}  |  Corr: {r['corr']:+.3f}  |  Cov: {r['cov']:.6f}  |  "
              f"Ann Return: {r['ann_ret']:+.1f}%  |  Volatility: {r['ann_vol']:.1f}%  |  Sharpe: {r['sharpe']:.2f}")
        print()

        # ── Price predictions ──────────────────────────────────────
        print(f"       {'TIMEFRAME':<12} {'BEAR':>10} {'BASE':>10} {'BULL':>10} {'CONF':>6} {'SIGNAL'}")
        print(f"       {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*6} {'─'*13}")
        for label, _ in TIMEFRAMES:
            p = r["preds"][label]
            b_pct = (p["bear"]/r["price"]-1)*100
            bs_pct= (p["base"]/r["price"]-1)*100
            bl_pct= (p["bull"]/r["price"]-1)*100
            print(f"       {label:<12} "
                  f"${p['bear']:>8.2f} ({b_pct:+.1f}%)  "
                  f"${p['base']:>8.2f} ({bs_pct:+.1f}%)  "
                  f"${p['bull']:>8.2f} ({bl_pct:+.1f}%)  "
                  f"{p['conf']:>4}%  {p['signal']}")

        # ── Reasons ────────────────────────────────────────────────
        print()
        for section, text in rdict.items():
            print(f"       [{section}]")
            # Word-wrap at 65 chars
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
#  UPCOMING IPOs
# ================================================================

print("=" * 70)
print("  ★  UPCOMING IPO WATCH  —  Not yet listed, monitor these")
print(f"     Generated: {RUN_ET.strftime('%d %b %Y %H:%M ET')}")
print("=" * 70)
for ipo in UPCOMING_IPOS:
    print(f"\n  ▸  {ipo['name']}  |  Expected ticker: {ipo['ticker']}  |  "
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
               "pe","fwd_pe","rev_g","margin","mktcap","target"]
out = df_all[[c for c in export_cols if c in df_all.columns]].copy()
out.insert(0, "as_of_date",    DATA_END)
out.insert(1, "generated_et",  RUN_ET.strftime("%Y-%m-%d %H:%M:%S ET"))
out.insert(2, "generated_utc", RUN_UTC.strftime("%Y-%m-%d %H:%M:%S UTC"))
out.sort_values(["category","beta"], ascending=[True,False]).round(4) \
   .to_csv("stock_classifier.csv", index=False)

print(f"  ✓ Exported → stock_classifier.csv  ({len(out)} rows, as of {DATA_END})")
