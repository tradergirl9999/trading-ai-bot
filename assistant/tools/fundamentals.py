"""
Comprehensive financial fundamentals via yfinance.
Covers income statement, balance sheet, cash flow, valuation, analyst targets.
"""
import yfinance as yf


def _fmt(val, pct=False, billions=False, dollars=False):
    if val is None or val == "N/A":
        return "N/A"
    try:
        v = float(val)
        if pct:
            return f"{v*100:.1f}%"
        if billions or abs(v) >= 1e9:
            return f"${v/1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v/1e6:.1f}M"
        if dollars:
            return f"${v:.2f}"
        return f"{v:.2f}"
    except Exception:
        return str(val)


def get_fundamentals(symbol: str) -> str:
    """Full fundamental snapshot: valuation, profitability, growth, balance sheet, cash flow, analyst targets, ownership."""
    try:
        t = yf.Ticker(symbol.upper())
        i = t.info
        name = i.get("longName") or i.get("shortName", symbol)
        price = i.get("currentPrice") or i.get("regularMarketPrice")

        sections = []

        # ── Valuation ──────────────────────────────────────────────────────
        vals = []
        for label, key in [
            ("Market Cap",     "marketCap"),
            ("Enterprise Val", "enterpriseValue"),
            ("P/E (TTM)",      "trailingPE"),
            ("P/E (Fwd)",      "forwardPE"),
            ("P/S Ratio",      "priceToSalesTrailing12Months"),
            ("P/B Ratio",      "priceToBook"),
            ("EV/EBITDA",      "enterpriseToEbitda"),
            ("EV/Revenue",     "enterpriseToRevenue"),
            ("Beta",           "beta"),
        ]:
            v = i.get(key)
            if v and v not in (None, "N/A"):
                vals.append(f"  {label:<18} {_fmt(v)}")
        if vals:
            sections.append("VALUATION\n" + "\n".join(vals))

        # ── Revenue & Profitability ────────────────────────────────────────
        prof = []
        for label, key, is_pct in [
            ("Revenue (TTM)",   "totalRevenue",                  False),
            ("Gross Profit",    "grossProfits",                  False),
            ("EBITDA",          "ebitda",                        False),
            ("Net Income",      "netIncomeToCommon",             False),
            ("EPS (TTM)",       "trailingEps",                   False),
            ("EPS (Forward)",   "forwardEps",                    False),
            ("Gross Margin",    "grossMargins",                  True),
            ("Operating Margin","operatingMargins",              True),
            ("Net Margin",      "profitMargins",                 True),
            ("ROE",             "returnOnEquity",                True),
            ("ROA",             "returnOnAssets",                True),
            ("ROIC",            "returnOnCapital",               True),
        ]:
            v = i.get(key)
            if v and v not in (None, "N/A"):
                prof.append(f"  {label:<18} {_fmt(v, pct=is_pct)}")
        if prof:
            sections.append("REVENUE & PROFITABILITY\n" + "\n".join(prof))

        # ── Growth ────────────────────────────────────────────────────────
        grow = []
        for label, key in [
            ("Revenue Growth",  "revenueGrowth"),
            ("Earnings Growth", "earningsGrowth"),
            ("EPS Growth (Q)",  "earningsQuarterlyGrowth"),
        ]:
            v = i.get(key)
            if v and v not in (None, "N/A"):
                grow.append(f"  {label:<18} {_fmt(v, pct=True)}")
        if grow:
            sections.append("GROWTH RATES\n" + "\n".join(grow))

        # ── Balance Sheet ─────────────────────────────────────────────────
        bs = []
        for label, key, is_pct in [
            ("Total Cash",      "totalCash",       False),
            ("Total Debt",      "totalDebt",       False),
            ("Debt / Equity",   "debtToEquity",    False),
            ("Current Ratio",   "currentRatio",    False),
            ("Quick Ratio",     "quickRatio",      False),
            ("Book Value/Share","bookValue",        False),
        ]:
            v = i.get(key)
            if v and v not in (None, "N/A"):
                bs.append(f"  {label:<18} {_fmt(v, pct=is_pct)}")
        if bs:
            sections.append("BALANCE SHEET HEALTH\n" + "\n".join(bs))

        # ── Cash Flow ────────────────────────────────────────────────────
        cf = []
        for label, key in [
            ("Operating CF",    "operatingCashflow"),
            ("Free Cash Flow",  "freeCashflow"),
            ("CF per Share",    "operatingCashflowPerShare"),
        ]:
            v = i.get(key)
            if v and v not in (None, "N/A"):
                cf.append(f"  {label:<18} {_fmt(v)}")
        if cf:
            sections.append("CASH FLOW\n" + "\n".join(cf))

        # ── Analyst Consensus ────────────────────────────────────────────
        ana = []
        if price:
            ana.append(f"  {'Current Price':<18} ${price:.2f}")
        for label, key in [
            ("Target (Mean)",   "targetMeanPrice"),
            ("Target (High)",   "targetHighPrice"),
            ("Target (Low)",    "targetLowPrice"),
        ]:
            v = i.get(key)
            if v and v not in (None, "N/A") and price:
                upside = (float(v) - price) / price * 100
                sign = "+" if upside >= 0 else ""
                ana.append(f"  {label:<18} ${float(v):.2f}  ({sign}{upside:.1f}%)")
        rec = i.get("recommendationKey", "")
        n_analysts = i.get("numberOfAnalystOpinions")
        if rec:
            ana.append(f"  {'Consensus':<18} {rec.upper().replace('_',' ')}"
                       f"{f'  ({n_analysts} analysts)' if n_analysts else ''}")
        if ana:
            sections.append("ANALYST TARGETS\n" + "\n".join(ana))

        # ── Ownership ────────────────────────────────────────────────────
        own = []
        for label, key in [
            ("Institutional",   "heldPercentInstitutions"),
            ("Insider",         "heldPercentInsiders"),
            ("Short % Float",   "shortPercentOfFloat"),
            ("Short Ratio",     "shortRatio"),
        ]:
            v = i.get(key)
            if v and v not in (None, "N/A"):
                own.append(f"  {label:<18} {_fmt(v, pct=True)}")
        if own:
            sections.append("OWNERSHIP\n" + "\n".join(own))

        # ── Dividends ────────────────────────────────────────────────────
        div = []
        for label, key in [
            ("Dividend Yield",  "dividendYield"),
            ("Dividend Rate",   "dividendRate"),
            ("Payout Ratio",    "payoutRatio"),
        ]:
            v = i.get(key)
            if v and v not in (None, "N/A"):
                div.append(f"  {label:<18} {_fmt(v, pct=(key != 'dividendRate'))}")
        if div:
            sections.append("DIVIDENDS\n" + "\n".join(div))

        header = f"{'='*50}\n{name} ({symbol.upper()})  —  Full Fundamentals\n{'='*50}"
        return header + "\n\n" + "\n\n".join(sections)

    except Exception as e:
        return f"Fundamentals fetch failed for {symbol}: {e}"


def get_earnings_history(symbol: str) -> str:
    """Recent earnings: actual vs estimate, surprise %, revenue beats."""
    try:
        t = yf.Ticker(symbol.upper())
        hist = t.earnings_history
        if hist is None or hist.empty:
            # try quarterly earnings
            qe = t.quarterly_earnings
            if qe is not None and not qe.empty:
                lines = [f"{symbol.upper()} Quarterly Earnings:"]
                for idx, row in qe.head(8).iterrows():
                    rev = f"  Revenue: {_fmt(row.get('Revenue', 'N/A'))}"
                    earn = f"  Earnings: {_fmt(row.get('Earnings', 'N/A'))}"
                    lines.append(f"\n{idx}\n{rev}\n{earn}")
                return "\n".join(lines)
            return f"No earnings history available for {symbol}."

        lines = [f"{symbol.upper()} Earnings History (recent quarters):"]
        for _, row in hist.head(8).iterrows():
            q_date = str(row.get("period", row.get("quarter", "N/A")))
            eps_est = row.get("epsEstimate")
            eps_act = row.get("epsActual")
            surprise = row.get("epsDifference")
            pct = row.get("surprisePercent")

            parts = [f"\n  {q_date}"]
            if eps_est is not None:
                parts.append(f"    EPS Estimate:  ${float(eps_est):.2f}")
            if eps_act is not None:
                parts.append(f"    EPS Actual:    ${float(eps_act):.2f}")
            if surprise is not None:
                sign = "+" if float(surprise) >= 0 else ""
                parts.append(f"    Surprise:      {sign}{float(surprise):.2f}"
                             f"{f'  ({sign}{float(pct):.1f}%)' if pct is not None else ''}")
            lines.extend(parts)
        return "\n".join(lines)
    except Exception as e:
        return f"Earnings history unavailable for {symbol}: {e}"


def get_income_statement(symbol: str) -> str:
    """Annual income statement — revenue, gross profit, EBIT, net income."""
    try:
        t = yf.Ticker(symbol.upper())
        fin = t.financials  # columns = years, rows = line items
        if fin is None or fin.empty:
            return f"No income statement data for {symbol}."

        key_rows = [
            "Total Revenue", "Gross Profit", "Operating Income",
            "Ebit", "Net Income", "Ebitda",
        ]
        lines = [f"{symbol.upper()} Annual Income Statement:"]
        cols = list(fin.columns[:4])  # last 4 years
        lines.append("  " + "  ".join(str(c)[:10] for c in cols))
        for row in key_rows:
            matches = [r for r in fin.index if row.lower() in r.lower()]
            if matches:
                data = fin.loc[matches[0], cols]
                vals = "  ".join(
                    _fmt(v) if v is not None and v == v else "  N/A  "
                    for v in data
                )
                lines.append(f"  {matches[0]:<25} {vals}")
        return "\n".join(lines)
    except Exception as e:
        return f"Income statement unavailable for {symbol}: {e}"


def get_cashflow_statement(symbol: str) -> str:
    """Annual cash flow statement."""
    try:
        t = yf.Ticker(symbol.upper())
        cf = t.cashflow
        if cf is None or cf.empty:
            return f"No cash flow data for {symbol}."

        key_rows = [
            "Total Cash From Operating Activities",
            "Capital Expenditures",
            "Free Cash Flow",
            "Total Cash From Investing Activities",
            "Total Cash From Financing Activities",
            "Change In Cash",
        ]
        lines = [f"{symbol.upper()} Annual Cash Flow Statement:"]
        cols = list(cf.columns[:4])
        lines.append("  " + "  ".join(str(c)[:10] for c in cols))
        for row in key_rows:
            matches = [r for r in cf.index if any(
                word.lower() in r.lower() for word in row.split()[:3]
            )]
            if matches:
                data = cf.loc[matches[0], cols]
                vals = "  ".join(
                    _fmt(v) if v is not None and v == v else "  N/A  "
                    for v in data
                )
                lines.append(f"  {matches[0][:28]:<28} {vals}")
        return "\n".join(lines)
    except Exception as e:
        return f"Cash flow statement unavailable for {symbol}: {e}"


def get_ipo_data(company: str) -> str:
    """Search for IPO information: price, date, valuation, underwriter, lock-up."""
    from tools.search import web_search, news_search
    search_results = web_search(
        f"{company} IPO price date valuation underwriter 2024 2025", max_results=6
    )
    news_results = news_search(f"{company} IPO", max_results=4)
    return (
        f"IPO Search Results for '{company}':\n\n"
        f"{search_results}\n\n"
        f"Recent News:\n{news_results}"
    )
