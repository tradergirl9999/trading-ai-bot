"""
Stock classifier using Beta + Covariance.

Beta = Cov(Stock Return, Market Return) / Var(Market Return)

Categories:
  Aggressive Growth Stock  — β ≥ 1.5,  ρ ≥ 0.55
  Market Leader            — 1.05 ≤ β < 1.5, ρ ≥ 0.60
  Market Follower          — 0.65 ≤ β < 1.05, ρ ≥ 0.50
  Defensive Stock          — β < 0.65, ρ ≥ 0.25
  Uncorrelated Opportunity — |ρ| < 0.30 (any beta)
"""

import yfinance as yf
import pandas as pd

MARKET = "^GSPC"

# ── Stock universe ─────────────────────────────────────────────────────────────
UNIVERSE: dict[str, list[str]] = {
    "Technology":          ["AAPL","MSFT","NVDA","GOOGL","META","AMD","INTC","CRM","ORCL","ADBE",
                            "QCOM","TXN","NOW","SNOW","PLTR","PANW","AMAT","MU","LRCX","KLAC"],
    "Healthcare":          ["JNJ","UNH","LLY","ABBV","PFE","MRK","TMO","ABT","DHR","AMGN",
                            "GILD","ISRG","CVS","BMY","VRTX","REGN","ZTS","SYK","BSX","ELV"],
    "Financials":          ["JPM","BAC","WFC","GS","MS","BLK","C","AXP","SPGI","CB",
                            "AON","MMC","PGR","TRV","USB","COF","SCHW","ICE","CME","MCO"],
    "Consumer Disc.":      ["AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","COST","LOW","BKNG",
                            "MAR","HLT","GM","F","ABNB","EBAY","ETSY","RCL","CCL","LVS"],
    "Consumer Staples":    ["PG","KO","PEP","WMT","MDLZ","CL","KMB","GIS","K","HSY",
                            "MO","PM","STZ","TAP","CLX"],
    "Energy":              ["XOM","CVX","COP","EOG","SLB","PSX","VLO","MPC","OXY","HAL",
                            "BKR","DVN","FANG","APA","MRO"],
    "Industrials":         ["CAT","BA","HON","UNP","LMT","RTX","GE","DE","MMM","UPS",
                            "FDX","EMR","ETN","PH","ITW","CSX","NSC","WM","ROK","AME"],
    "Materials":           ["LIN","APD","ECL","SHW","FCX","NEM","DOW","DD","NUE","CF",
                            "ALB","MOS","PKG","IP","SEE"],
    "Utilities":           ["NEE","DUK","SO","D","AEP","EXC","XEL","WEC","ES","ETR",
                            "FE","PPL","EIX","PEG","AES"],
    "Real Estate":         ["AMT","PLD","CCI","EQIX","SPG","PSA","EQR","AVB","O","WY",
                            "DLR","SBAC","WELL","VTR","HST"],
    "Communication":       ["NFLX","DIS","CMCSA","T","VZ","CHTR","TMUS","WBD","PARA","FOX"],
    "Crypto / High Beta":  ["COIN","MSTR","MARA","RIOT","CLSK","SOFI","HOOD","UPST","AFRM","SQ"],
}

ALL_SYMBOLS = [s for symbols in UNIVERSE.values() for s in symbols]

# Reverse lookup: symbol → sector
SYMBOL_SECTOR = {s: sec for sec, syms in UNIVERSE.items() for s in syms}


# ── Math ───────────────────────────────────────────────────────────────────────

def _cov(x: pd.Series, y: pd.Series) -> float:
    """Cov(X,Y) = E[XY] - E[X]·E[Y]"""
    xy = x * y
    return float(xy.mean() - x.mean() * y.mean())


def _corr(x: pd.Series, y: pd.Series) -> float:
    sx, sy = float(x.std()), float(y.std())
    if sx == 0 or sy == 0:
        return 0.0
    return _cov(x, y) / (sx * sy)


def _beta(stock: pd.Series, market: pd.Series) -> float:
    """β = Cov(stock, market) / Var(market)"""
    var_m = float(market.var())
    return _cov(stock, market) / var_m if var_m != 0 else 0.0


def _sharpe(ret: pd.Series, rf_daily: float = 0.00019) -> float:
    """Annualised Sharpe (assumes 252 trading days)."""
    excess = ret - rf_daily
    if float(ret.std()) == 0:
        return 0.0
    return float(excess.mean() / excess.std() * (252 ** 0.5))


def _annual_return(ret: pd.Series) -> float:
    return float(((1 + ret).prod()) ** (252 / len(ret)) - 1) * 100


# ── Classification ─────────────────────────────────────────────────────────────

CATEGORIES = [
    "Aggressive Growth Stock",
    "Market Leader",
    "Market Follower",
    "Defensive Stock",
    "Uncorrelated Opportunity",
]

CATEGORY_DESC = {
    "Aggressive Growth Stock":  "β ≥ 1.5 & ρ ≥ 0.55 — amplifies market moves, high risk/reward",
    "Market Leader":            "1.05 ≤ β < 1.5 & ρ ≥ 0.60 — outpaces market on up-days",
    "Market Follower":          "0.65 ≤ β < 1.05 & ρ ≥ 0.50 — tracks market reliably",
    "Defensive Stock":          "β < 0.65 & ρ ≥ 0.25 — holds up in downturns",
    "Uncorrelated Opportunity": "|ρ| < 0.30 — moves independently, diversification gem",
}

def _classify(beta_val: float, corr_val: float) -> str:
    if abs(corr_val) < 0.30:
        return "Uncorrelated Opportunity"
    if beta_val >= 1.5 and corr_val >= 0.55:
        return "Aggressive Growth Stock"
    if 1.05 <= beta_val < 1.5 and corr_val >= 0.60:
        return "Market Leader"
    if 0.65 <= beta_val < 1.05 and corr_val >= 0.50:
        return "Market Follower"
    if beta_val < 0.65:
        return "Defensive Stock"
    return "Market Follower"


# ── Explain each category ─────────────────────────────────────────────────────

def _category_insight(cat: str, beta_val: float, corr_val: float, annual_ret: float, sharpe: float) -> str:
    if cat == "Aggressive Growth Stock":
        return (f"High beta ({beta_val:.2f}) and strong market correlation ({corr_val:.2f}). "
                f"Moves ~{beta_val:.1f}x the market. Great for bull markets, painful in crashes. "
                f"Annual return: {annual_ret:+.1f}%.")
    if cat == "Market Leader":
        return (f"Beta {beta_val:.2f} — consistently outpaces S&P 500 on up-days by {(beta_val-1)*100:.0f}%. "
                f"Strong correlation ({corr_val:.2f}). Usually sector leaders or large-cap growth. "
                f"Sharpe: {sharpe:.2f}.")
    if cat == "Market Follower":
        return (f"Beta {beta_val:.2f} closely mirrors the market (ρ={corr_val:.2f}). "
                f"Good for riding the index with individual stock alpha potential. "
                f"Annual return: {annual_ret:+.1f}%.")
    if cat == "Defensive Stock":
        return (f"Low beta ({beta_val:.2f}) means muted reaction to market swings. "
                f"Holds value during corrections. Ideal for capital preservation. "
                f"Sharpe: {sharpe:.2f}.")
    if cat == "Uncorrelated Opportunity":
        return (f"Very low market correlation (ρ={corr_val:.2f}). Driven by company-specific factors. "
                f"Excellent diversifier — reduces portfolio volatility when added. "
                f"Annual return: {annual_ret:+.1f}%.")
    return ""


# ── Main functions ─────────────────────────────────────────────────────────────

def classify_single_stock(symbol: str) -> str:
    """Classify one stock and explain why."""
    sym = symbol.upper()
    try:
        raw = yf.download([sym, MARKET], period="1y", auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            return f"Could not fetch data for {sym}."
        returns = raw.pct_change().dropna()
        if sym not in returns.columns or MARKET not in returns.columns:
            return f"Insufficient data for {sym}."

        s_ret = returns[sym]
        m_ret = returns[MARKET]

        cov_val  = _cov(s_ret, m_ret)
        corr_val = _corr(s_ret, m_ret)
        beta_val = _beta(s_ret, m_ret)
        sharpe_v = _sharpe(s_ret)
        ann_ret  = _annual_return(s_ret)
        category = _classify(beta_val, corr_val)
        insight  = _category_insight(category, beta_val, corr_val, ann_ret, sharpe_v)

        info = yf.Ticker(sym).info
        name   = info.get("longName") or info.get("shortName", sym)
        sector = info.get("sector", SYMBOL_SECTOR.get(sym, "Unknown"))
        price  = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")

        lines = [
            "═" * 56,
            f" STOCK CLASSIFICATION: {name} ({sym})",
            f" Sector: {sector}  |  Price: ${price}",
            "═" * 56,
            f"",
            f"  ★  CATEGORY:    {category.upper()}",
            f"",
            f"  Beta (1Y):      {beta_val:+.4f}",
            f"  Covariance:     {cov_val:.6f}",
            f"  Correlation ρ:  {corr_val:+.4f}",
            f"  Annual Return:  {ann_ret:+.1f}%",
            f"  Sharpe Ratio:   {sharpe_v:.2f}",
            f"",
            f"  What this means:",
            f"  {insight}",
            f"",
            f"  Category rule:  {CATEGORY_DESC[category]}",
            "═" * 56,
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Classification failed for {symbol}: {e}"


def screen_stocks(
    category: str = "all",
    sector: str = "all",
    top_n: int = 10,
    period: str = "1y",
) -> str:
    """
    Screen the full universe and return top stocks matching a category.
    category: 'Aggressive Growth Stock' | 'Market Leader' | 'Market Follower' |
              'Defensive Stock' | 'Uncorrelated Opportunity' | 'all'
    sector:   sector name or 'all'
    top_n:    how many stocks to return per category
    """
    # Filter universe by sector
    if sector.lower() != "all":
        matched_sector = next((k for k in UNIVERSE if sector.lower() in k.lower()), None)
        symbols = UNIVERSE.get(matched_sector, ALL_SYMBOLS) if matched_sector else ALL_SYMBOLS
    else:
        symbols = ALL_SYMBOLS

    symbols = list(set(symbols))

    print(f"  [classifier] Downloading {len(symbols)} stocks + market data ({period})…")

    try:
        tickers = symbols + [MARKET]
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
        if MARKET not in raw.columns:
            return "Could not download market data."
        returns = raw.pct_change().dropna()
        m_ret = returns[MARKET]
    except Exception as e:
        return f"Data download failed: {e}"

    results = []
    for sym in symbols:
        if sym not in returns.columns:
            continue
        s_ret = returns[sym].dropna()
        if len(s_ret) < 50:
            continue
        m_aligned = m_ret.reindex(s_ret.index).dropna()
        s_aligned  = s_ret.reindex(m_aligned.index).dropna()
        if len(s_aligned) < 50:
            continue

        try:
            cov_val  = _cov(s_aligned, m_aligned)
            corr_val = _corr(s_aligned, m_aligned)
            beta_val = _beta(s_aligned, m_aligned)
            sharpe_v = _sharpe(s_aligned)
            ann_ret  = _annual_return(s_aligned)
            cat      = _classify(beta_val, corr_val)
        except Exception:
            continue

        results.append({
            "symbol":   sym,
            "sector":   SYMBOL_SECTOR.get(sym, "Unknown"),
            "category": cat,
            "beta":     beta_val,
            "cov":      cov_val,
            "corr":     corr_val,
            "sharpe":   sharpe_v,
            "ann_ret":  ann_ret,
        })

    if not results:
        return "No results — check your sector/period parameters."

    df = pd.DataFrame(results)

    # Filter by requested category
    target_cats = CATEGORIES if category.lower() == "all" else [
        c for c in CATEGORIES if category.lower() in c.lower()
    ]
    if not target_cats:
        target_cats = CATEGORIES

    lines = []
    lines.append("═" * 70)
    lines.append(f" STOCK PICKER — Beta + Covariance Classification ({period})")
    lines.append(f" Universe: {len(df)} stocks screened  |  Market: S&P 500")
    lines.append("═" * 70)

    for cat in target_cats:
        subset = df[df["category"] == cat].copy()
        if subset.empty:
            continue

        # Rank within category
        if cat == "Aggressive Growth Stock":
            subset = subset.sort_values("beta", ascending=False)
        elif cat == "Market Leader":
            subset["score"] = subset["beta"] * 0.5 + subset["sharpe"] * 0.3 + subset["corr"] * 0.2
            subset = subset.sort_values("score", ascending=False)
        elif cat == "Market Follower":
            subset = subset.sort_values("corr", ascending=False)
        elif cat == "Defensive Stock":
            subset["score"] = (1 / (subset["beta"].abs() + 0.01)) * 0.6 + subset["sharpe"] * 0.4
            subset = subset.sort_values("score", ascending=False)
        elif cat == "Uncorrelated Opportunity":
            subset["score"] = (1 / (subset["corr"].abs() + 0.01)) * 0.5 + subset["ann_ret"] * 0.5
            subset = subset.sort_values("score", ascending=False)

        top = subset.head(top_n)

        lines.append(f"\n{'─'*70}")
        lines.append(f" ★  {cat.upper()}")
        lines.append(f"    {CATEGORY_DESC[cat]}")
        lines.append(f"{'─'*70}")
        lines.append(f"  {'Symbol':<8} {'Sector':<22} {'Beta':>6} {'Corr':>6} {'Ann Ret':>8} {'Sharpe':>7}")
        lines.append(f"  {'─'*8} {'─'*22} {'─'*6} {'─'*6} {'─'*8} {'─'*7}")

        for _, row in top.iterrows():
            ret_str = f"{row['ann_ret']:+.1f}%"
            lines.append(
                f"  {row['symbol']:<8} {row['sector']:<22} "
                f"{row['beta']:>+6.3f} {row['corr']:>+6.3f} "
                f"{ret_str:>8} {row['sharpe']:>7.2f}"
            )

    lines.append(f"\n{'═'*70}")
    lines.append(" HOW TO USE THESE PICKS:")
    lines.append("  Aggressive Growth  → High conviction bull thesis, tight stop loss")
    lines.append("  Market Leader      → Core portfolio holdings, ride the trend")
    lines.append("  Market Follower    → Index-plus strategy, low research overhead")
    lines.append("  Defensive Stock    → Allocate during uncertainty or bear markets")
    lines.append("  Uncorrelated       → Add to reduce portfolio correlation, diversify")
    lines.append("═" * 70)

    # Summary count
    lines.append(f"\n BREAKDOWN: " +
                 " | ".join(f"{cat.split()[0]}: {len(df[df['category']==cat])}"
                            for cat in CATEGORIES))

    return "\n".join(lines)


def smart_pick(goal: str = "growth", sector: str = "all", top_n: int = 5) -> str:
    """
    Pick stocks based on investor goal:
      'growth'      → Aggressive Growth + Market Leaders
      'safe'        → Defensive Stocks
      'diversify'   → Uncorrelated Opportunities
      'income'      → Defensive + low-beta (stable dividend candidates)
      'momentum'    → Market Leaders + high Sharpe Aggressive Growth
    """
    goal_map = {
        "growth":    ["Aggressive Growth Stock", "Market Leader"],
        "safe":      ["Defensive Stock"],
        "defensive": ["Defensive Stock"],
        "diversify": ["Uncorrelated Opportunity"],
        "income":    ["Defensive Stock", "Market Follower"],
        "momentum":  ["Market Leader", "Aggressive Growth Stock"],
        "hedge":     ["Uncorrelated Opportunity", "Defensive Stock"],
    }
    matched = next((v for k, v in goal_map.items() if goal.lower() in k), None)
    cat_filter = matched[0] if matched else "all"
    return screen_stocks(category=cat_filter, sector=sector, top_n=top_n)
