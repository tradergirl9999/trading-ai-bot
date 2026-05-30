"""
Covariance analysis: Cov(X,Y) = E[XY] - E[X]E[Y]
Stock returns vs S&P 500, Nasdaq 100, Sector ETF, Gold, Bitcoin.
"""
import yfinance as yf
import pandas as pd

# ── Benchmark tickers ─────────────────────────────────────────────────────────
BENCHMARKS = {
    "S&P 500":    "^GSPC",
    "Nasdaq 100": "^NDX",
    "Gold":       "GC=F",
    "Bitcoin":    "BTC-USD",
}

# ── Sector → ETF mapping ──────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Financial Services":     "XLF",
    "Financials":             "XLF",
    "Energy":                 "XLE",
    "Consumer Cyclical":      "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive":     "XLP",
    "Consumer Staples":       "XLP",
    "Industrials":            "XLI",
    "Basic Materials":        "XLB",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Communication Services": "XLC",
    "Telecommunication":      "XLC",
}

# ── Analysis windows (trading days) ──────────────────────────────────────────
WINDOWS = {
    "30 Days":  30,
    "90 Days":  90,
    "180 Days": 180,
    "1 Year":   252,
}


# ── Core math ─────────────────────────────────────────────────────────────────

def _daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def _cov(x: pd.Series, y: pd.Series) -> float:
    """Cov(X,Y) = E[XY] - E[X]·E[Y]  — exact formula as specified."""
    aligned_x, aligned_y = x.align(y, join="inner")
    if len(aligned_x) < 5:
        return float("nan")
    return float((aligned_x * aligned_y).mean() - aligned_x.mean() * aligned_y.mean())


def _corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation = Cov(X,Y) / (σX · σY)"""
    aligned_x, aligned_y = x.align(y, join="inner")
    if len(aligned_x) < 5:
        return float("nan")
    sx, sy = float(aligned_x.std()), float(aligned_y.std())
    if sx == 0 or sy == 0:
        return 0.0
    return _cov(aligned_x, aligned_y) / (sx * sy)


def _beta(stock_ret: pd.Series, market_ret: pd.Series) -> float:
    """β = Cov(stock, market) / Var(market)"""
    aligned_s, aligned_m = stock_ret.align(market_ret, join="inner")
    if len(aligned_s) < 5:
        return float("nan")
    var_m = float(aligned_m.var())
    return _cov(aligned_s, aligned_m) / var_m if var_m != 0 else float("nan")


# ── Scoring & interpretation ───────────────────────────────────────────────────

def _dependency_score(corr_map: dict) -> float:
    """
    Weighted absolute correlation → 0–100 Market Dependency Score.
    S&P 500 carries highest weight as primary market proxy.
    """
    weights = {
        "S&P 500":    0.35,
        "Nasdaq 100": 0.25,
        "Sector ETF": 0.20,
        "Gold":       0.10,
        "Bitcoin":    0.10,
    }
    total_w, score = 0.0, 0.0
    for name, corr in corr_map.items():
        if corr != corr:   # nan
            continue
        w = weights.get(name, 0.1)
        score   += abs(corr) * w * 100
        total_w += w
    return score / total_w if total_w > 0 else 0.0


def _score_label(score: float) -> str:
    if score <= 20:  return "Independent"
    if score <= 40:  return "Weakly Dependent"
    if score <= 60:  return "Moderately Dependent"
    if score <= 80:  return "Strongly Dependent"
    return "Highly Dependent"


def _cov_label(cov: float, corr: float) -> str:
    if abs(corr) < 0.1 or cov != cov:
        return "Near-zero (weak relationship)"
    direction = "Positive" if cov > 0 else "Negative"
    if abs(corr) > 0.7:   strength = "Strong"
    elif abs(corr) > 0.4: strength = "Moderate"
    else:                  strength = "Weak"
    return f"{direction} ({strength})"


def _corr_plain(name: str, corr: float) -> str:
    if corr != corr:
        return f"  No data available for {name}."
    if abs(corr) < 0.1:
        return f"  {name}: Very weak link — stock moves independently."
    if corr > 0:
        adv = "rises and falls together with"
        qual = "closely" if corr > 0.7 else "moderately" if corr > 0.4 else "loosely"
    else:
        adv = "tends to move opposite to"
        qual = "strongly" if corr < -0.7 else "moderately" if corr < -0.4 else "loosely"
    return f"  {name}: Stock {qual} {adv} this benchmark (ρ={corr:+.3f})."


# ── Main function ─────────────────────────────────────────────────────────────

def get_covariance_analysis(symbol: str) -> str:
    sym = symbol.upper()

    # ── Fetch 1-year + of price data for all instruments ─────────────────────
    try:
        info = yf.Ticker(sym).info
        name = info.get("longName") or info.get("shortName", sym)
        sector = info.get("sector", "")
        sector_etf_ticker = SECTOR_ETFS.get(sector, "SPY")
        sector_etf_name   = f"Sector ETF ({sector_etf_ticker})"
    except Exception:
        name, sector, sector_etf_ticker, sector_etf_name = sym, "", "SPY", "Sector ETF (SPY)"

    all_tickers = [sym] + list(BENCHMARKS.values()) + [sector_etf_ticker]

    try:
        raw = yf.download(all_tickers, period="2y", auto_adjust=True, progress=False)["Close"]
    except Exception as e:
        return f"Data download failed: {e}"

    if isinstance(raw, pd.Series):
        raw = raw.to_frame(sym)

    # Map benchmark tickers → readable names
    col_map = {v: k for k, v in BENCHMARKS.items()}
    col_map[sector_etf_ticker] = sector_etf_name
    raw = raw.rename(columns=col_map)

    if sym not in raw.columns:
        return f"No price data found for {sym}."

    # ── Compute returns for each instrument ───────────────────────────────────
    returns = raw.pct_change().dropna()
    stock_ret_full = returns[sym]

    bench_names = list(BENCHMARKS.keys()) + [sector_etf_name]

    lines = []
    lines.append("=" * 62)
    lines.append(f" COVARIANCE ANALYSIS: {name} ({sym})")
    if sector:
        lines.append(f" Sector: {sector}  |  Sector ETF: {sector_etf_ticker}")
    lines.append("=" * 62)

    # ── Per-window table ──────────────────────────────────────────────────────
    # Store 1-year results for the summary section
    year_covs  = {}
    year_corrs = {}
    year_betas = {}

    for window_label, n_days in WINDOWS.items():
        stock_ret  = stock_ret_full.tail(n_days)
        bench_rets = {b: returns[b].tail(n_days) for b in bench_names if b in returns.columns}

        lines.append(f"\n{'─'*62}")
        lines.append(f" WINDOW: {window_label}  ({len(stock_ret)} trading days)")
        lines.append(f"{'─'*62}")
        lines.append(f"  {'Benchmark':<22} {'Covariance':>12} {'Correlation':>12} {'Label'}")
        lines.append(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*24}")

        for b_name, b_ret in bench_rets.items():
            cov  = _cov(stock_ret, b_ret)
            corr = _corr(stock_ret, b_ret)
            label = _cov_label(cov, corr)
            cov_str  = f"{cov:.6f}"  if cov == cov  else "  N/A"
            corr_str = f"{corr:+.4f}" if corr == corr else "  N/A"
            lines.append(f"  {b_name:<22} {cov_str:>12} {corr_str:>12} {label}")

            if window_label == "1 Year":
                year_covs[b_name]  = cov
                year_corrs[b_name] = corr
                year_betas[b_name] = _beta(stock_ret, b_ret)

    # ── Market Dependency Score (based on 1-year correlations) ───────────────
    dep_score = _dependency_score(year_corrs)
    dep_label = _score_label(dep_score)
    sp500_corr = year_corrs.get("S&P 500", float("nan"))
    sp500_beta = year_betas.get("S&P 500", float("nan"))

    lines.append(f"\n{'='*62}")
    lines.append(f" MARKET DEPENDENCY SCORE  (based on 1-year data)")
    lines.append(f"{'='*62}")
    lines.append(f"  Score:  {dep_score:.1f} / 100")
    lines.append(f"  Rating: {dep_label}")
    bar_filled = int(dep_score / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    lines.append(f"  [{bar}] {dep_score:.0f}%")

    # ── Beta ─────────────────────────────────────────────────────────────────
    lines.append(f"\n{'─'*62}")
    lines.append(" BETA vs BENCHMARKS (1 Year)")
    lines.append(f"{'─'*62}")
    for b_name in bench_names:
        beta = year_betas.get(b_name, float("nan"))
        if beta == beta:
            if   beta > 1.5:  interp = "Very high market sensitivity"
            elif beta > 1.0:  interp = "Amplifies market moves"
            elif beta > 0.5:  interp = "Below-average sensitivity"
            elif beta > 0:    interp = "Low sensitivity"
            elif beta > -0.5: interp = "Slight inverse to market"
            else:             interp = "Inverse / defensive"
            lines.append(f"  {b_name:<22} β = {beta:+.3f}  ({interp})")

    # ── Plain-English Interpretation ──────────────────────────────────────────
    lines.append(f"\n{'='*62}")
    lines.append(" PLAIN-ENGLISH INTERPRETATION")
    lines.append(f"{'='*62}")
    for b_name in bench_names:
        corr = year_corrs.get(b_name, float("nan"))
        lines.append(_corr_plain(b_name, corr))

    # ── Diversification & Hedging ─────────────────────────────────────────────
    lines.append(f"\n{'─'*62}")
    lines.append(" PORTFOLIO IMPLICATIONS")
    lines.append(f"{'─'*62}")

    # Diversification benefit
    if sp500_corr == sp500_corr:
        if sp500_corr < 0.2:
            div_benefit = "HIGH — Low/negative correlation with market. Adding this stock reduces portfolio volatility."
        elif sp500_corr < 0.5:
            div_benefit = "MODERATE — Some diversification benefit. Partially decorrelated from market."
        else:
            div_benefit = "LOW — Moves closely with the market. Adds limited diversification."
    else:
        div_benefit = "INSUFFICIENT DATA"

    # Hedging benefit
    gold_corr = year_corrs.get("Gold", float("nan"))
    if sp500_corr == sp500_corr and sp500_corr < -0.3:
        hedge = "STRONG — Negative market correlation makes this a natural hedge."
    elif gold_corr == gold_corr and gold_corr < -0.3:
        hedge = "MODERATE — Negative gold correlation. Useful in inflation/crisis hedging."
    elif sp500_corr == sp500_corr and sp500_corr < 0.2:
        hedge = "PARTIAL — Low market correlation provides some hedging utility."
    else:
        hedge = "WEAK — Positive market correlation limits hedging effectiveness."

    # Portfolio risk contribution
    if sp500_beta == sp500_beta:
        if sp500_beta > 1.2:
            risk_contrib = f"HIGH — β={sp500_beta:.2f}. Amplifies portfolio swings. Size position conservatively."
        elif sp500_beta > 0.8:
            risk_contrib = f"MARKET-LEVEL — β={sp500_beta:.2f}. Risk in line with S&P 500."
        elif sp500_beta > 0.3:
            risk_contrib = f"LOW-MODERATE — β={sp500_beta:.2f}. Below-average systematic risk."
        else:
            risk_contrib = f"LOW — β={sp500_beta:.2f}. Minimal systematic risk contribution."
    else:
        risk_contrib = "INSUFFICIENT DATA"

    # Systematic vs idiosyncratic risk
    if sp500_corr == sp500_corr:
        r_squared = sp500_corr ** 2 * 100
        systematic_pct = f"{r_squared:.1f}%"
        idio_pct       = f"{100 - r_squared:.1f}%"
        sys_exp = (f"R²={r_squared:.1f}% — {systematic_pct} of this stock's variance is explained "
                   f"by the market (systematic). {idio_pct} is stock-specific (idiosyncratic).")
    else:
        sys_exp = "INSUFFICIENT DATA"

    lines.append(f"  Diversification Benefit:  {div_benefit}")
    lines.append(f"  Hedging Benefit:          {hedge}")
    lines.append(f"  Portfolio Risk Contrib:   {risk_contrib}")
    lines.append(f"  Systematic Risk Exposure: {sys_exp}")

    # ── Summary verdict ───────────────────────────────────────────────────────
    lines.append(f"\n{'='*62}")
    lines.append(" SUMMARY")
    lines.append(f"{'='*62}")
    lines.append(f"  {sym} is rated {dep_label.upper()} on market dependency.")
    if sp500_corr == sp500_corr:
        if sp500_corr > 0.7:
            lines.append(f"  It is tightly coupled to the S&P 500 (ρ={sp500_corr:.3f}) — "
                         f"expect it to mirror broad market moves.")
        elif sp500_corr > 0.3:
            lines.append(f"  Moderate market coupling (ρ={sp500_corr:.3f}) — "
                         f"partially influenced by macro, but retains stock-specific drivers.")
        else:
            lines.append(f"  Low market coupling (ρ={sp500_corr:.3f}) — "
                         f"stock-specific factors dominate price action.")
    lines.append("=" * 62)

    return "\n".join(lines)
