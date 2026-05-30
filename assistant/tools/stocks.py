import yfinance as yf
import pandas as pd
import math


def get_stock_info(symbol: str) -> str:
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        name = info.get("longName") or info.get("shortName", symbol)
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        day_high = info.get("dayHigh")
        day_low = info.get("dayLow")
        volume = info.get("volume")
        market_cap = info.get("marketCap")
        pe = info.get("trailingPE")
        w52_high = info.get("fiftyTwoWeekHigh")
        w52_low = info.get("fiftyTwoWeekLow")
        sector = info.get("sector", "")
        summary = info.get("longBusinessSummary", "")[:200] if info.get("longBusinessSummary") else ""

        lines = [f"{name} ({symbol.upper()})"]
        if price:
            change = price - prev_close if prev_close else None
            pct = (change / prev_close * 100) if change and prev_close else None
            price_str = f"${price:.2f}"
            if change is not None:
                sign = "+" if change >= 0 else ""
                price_str += f"  {sign}{change:.2f} ({sign}{pct:.2f}%)"
            lines.append(f"Price: {price_str}")
        if day_low and day_high:
            lines.append(f"Day Range: ${day_low:.2f} – ${day_high:.2f}")
        if w52_low and w52_high:
            lines.append(f"52W Range: ${w52_low:.2f} – ${w52_high:.2f}")
        if volume:
            lines.append(f"Volume: {volume:,}")
        if market_cap:
            lines.append(f"Market Cap: ${market_cap/1e9:.2f}B")
        if pe:
            lines.append(f"P/E Ratio: {pe:.2f}")
        if sector:
            lines.append(f"Sector: {sector}")
        if summary:
            lines.append(f"About: {summary}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not fetch data for {symbol}: {e}"


def _compute_technicals(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> dict:
    """Compute all technical indicators from OHLCV series."""
    n = len(close)
    current = float(close.iloc[-1])

    # SMAs
    sma20  = float(close.rolling(20).mean().iloc[-1])  if n >= 20  else None
    sma50  = float(close.rolling(50).mean().iloc[-1])  if n >= 50  else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else None

    # EMA
    ema12 = float(close.ewm(span=12, adjust=False).mean().iloc[-1])
    ema26 = float(close.ewm(span=26, adjust=False).mean().iloc[-1])

    # MACD
    macd_line   = ema12 - ema26
    macd_signal_series = (close.ewm(span=12, adjust=False).mean() -
                          close.ewm(span=26, adjust=False).mean()).ewm(span=9, adjust=False).mean()
    macd_signal = float(macd_signal_series.iloc[-1])
    macd_hist   = macd_line - macd_signal

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = float((100 - 100 / (1 + rs)).iloc[-1])

    # Bollinger Bands (20,2)
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = float((bb_mid + 2 * bb_std).iloc[-1])
    bb_lower = float((bb_mid - 2 * bb_std).iloc[-1])
    bb_pct   = (current - bb_lower) / (bb_upper - bb_lower) * 100 if bb_upper != bb_lower else 50.0

    # ATR (14)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if n >= 14 else None

    # Volume analysis
    avg_vol_10  = float(volume.rolling(10).mean().iloc[-1])
    avg_vol_20  = float(volume.rolling(20).mean().iloc[-1]) if n >= 20 else avg_vol_10
    last_vol    = float(volume.iloc[-1])
    vol_ratio   = last_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

    # Support & Resistance (simple: recent swing highs/lows over last 20 bars)
    window = min(20, n)
    support    = float(low.tail(window).min())
    resistance = float(high.tail(window).max())

    # Stochastic %K (14,3)
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = float(((close - low14) / (high14 - low14) * 100).rolling(3).mean().iloc[-1]) if n >= 17 else None

    # Price momentum
    mom_5  = (current / float(close.iloc[-6])  - 1) * 100 if n >= 6  else None
    mom_10 = (current / float(close.iloc[-11]) - 1) * 100 if n >= 11 else None
    mom_20 = (current / float(close.iloc[-21]) - 1) * 100 if n >= 21 else None

    return {
        "current": current,
        "sma20": sma20, "sma50": sma50, "sma200": sma200,
        "ema12": ema12, "ema26": ema26,
        "macd": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist,
        "rsi": rsi,
        "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": float(bb_mid.iloc[-1]), "bb_pct": bb_pct,
        "atr": atr,
        "support": support, "resistance": resistance,
        "vol_ratio": vol_ratio,
        "stoch_k": stoch_k,
        "mom_5": mom_5, "mom_10": mom_10, "mom_20": mom_20,
    }


def analyze_stock(symbol: str, period: str = "1mo") -> str:
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)
        if hist.empty:
            return f"No historical data for {symbol}."

        t = _compute_technicals(hist["Close"], hist["High"], hist["Low"], hist["Volume"])
        current = t["current"]
        start   = float(hist["Close"].iloc[0])
        change_pct = (current - start) / start * 100

        lines = [f"{symbol.upper()} Technical Analysis ({period})",
                 f"Current:  ${current:.2f}",
                 f"Change:   {'+' if change_pct >= 0 else ''}{change_pct:.2f}% this period"]

        if t["sma20"]:
            lines.append(f"SMA20:    ${t['sma20']:.2f}  ({'▲ above' if current > t['sma20'] else '▼ below'})")
        if t["sma50"]:
            lines.append(f"SMA50:    ${t['sma50']:.2f}  ({'▲ above' if current > t['sma50'] else '▼ below'})")
        if t["sma200"]:
            lines.append(f"SMA200:   ${t['sma200']:.2f}  ({'▲ above' if current > t['sma200'] else '▼ below'})")

        lines.append(f"MACD:     {t['macd']:.3f}  Signal: {t['macd_signal']:.3f}  "
                     f"Hist: {t['macd_hist']:+.3f}  ({'bullish cross' if t['macd_hist'] > 0 else 'bearish cross'})")
        lines.append(f"RSI(14):  {t['rsi']:.1f}  "
                     f"({'overbought ⚠' if t['rsi'] > 70 else 'oversold ⚠' if t['rsi'] < 30 else 'neutral'})")
        lines.append(f"BB:       ${t['bb_lower']:.2f} – ${t['bb_upper']:.2f}  "
                     f"(price at {t['bb_pct']:.0f}% of band)")
        if t["stoch_k"] is not None:
            lines.append(f"Stoch %K: {t['stoch_k']:.1f}  "
                         f"({'overbought' if t['stoch_k'] > 80 else 'oversold' if t['stoch_k'] < 20 else 'neutral'})")
        if t["atr"]:
            lines.append(f"ATR(14):  ${t['atr']:.2f}  (daily volatility range)")
        lines.append(f"Support:  ${t['support']:.2f}  |  Resistance: ${t['resistance']:.2f}")
        lines.append(f"Volume:   {t['vol_ratio']:.1f}x avg  ({'high' if t['vol_ratio'] > 1.5 else 'low' if t['vol_ratio'] < 0.7 else 'normal'})")
        if t["mom_20"] is not None:
            lines.append(f"Momentum: 5d {t['mom_5']:+.1f}%  10d {t['mom_10']:+.1f}%  20d {t['mom_20']:+.1f}%")

        # Overall bias score
        bull = sum([
            t["sma20"] and current > t["sma20"],
            t["sma50"] and current > t["sma50"],
            t["sma200"] and current > t["sma200"],
            t["macd_hist"] > 0,
            t["rsi"] > 50,
            t["stoch_k"] is not None and t["stoch_k"] > 50,
            t["mom_20"] is not None and t["mom_20"] > 0,
        ])
        total_signals = sum([
            t["sma20"] is not None, t["sma50"] is not None, t["sma200"] is not None,
            True, True,
            t["stoch_k"] is not None,
            t["mom_20"] is not None,
        ])
        bias = ("Strong Bullish" if bull >= total_signals * 0.8 else
                "Bullish"        if bull >= total_signals * 0.6 else
                "Bearish"        if bull <= total_signals * 0.3 else
                "Strong Bearish" if bull <= total_signals * 0.15 else "Neutral")
        lines.append(f"Bias:     {bias}  ({bull}/{total_signals} signals bullish)")

        return "\n".join(lines)
    except Exception as e:
        return f"Analysis failed for {symbol}: {e}"


def get_multi_timeframe_analysis(symbol: str) -> str:
    """
    Run technical analysis across 6 timeframes simultaneously.
    Returns structured data Claude uses to build the multi-horizon prediction table.
    """
    sym = symbol.upper()
    # Map: label → (yfinance period, expected bars for context)
    frames = [
        ("1 Day",    "5d",   "Short-term intraday momentum"),
        ("1 Week",   "1mo",  "Short-term swing"),
        ("1 Month",  "3mo",  "Medium-term trend"),
        ("3 Months", "6mo",  "Medium-term structural"),
        ("6 Months", "1y",   "Long-term trend"),
        ("1 Year",   "2y",   "Long-term investment"),
    ]

    try:
        ticker = yf.Ticker(sym)
        # Fetch 2y once — slice for shorter periods
        hist_2y = ticker.history(period="2y")
        if hist_2y.empty:
            return f"No data for {sym}."

        info    = ticker.info
        current = info.get("currentPrice") or info.get("regularMarketPrice") or float(hist_2y["Close"].iloc[-1])
        name    = info.get("longName") or info.get("shortName", sym)

        sections = [f"MULTI-TIMEFRAME ANALYSIS: {name} ({sym})  |  Current: ${current:.2f}\n"]

        bar_counts = {"5d": 5, "1mo": 22, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504}

        for label, period, context in frames:
            bars = bar_counts.get(period, 252)
            hist = hist_2y.tail(bars)
            if len(hist) < 5:
                sections.append(f"── {label} ──\nInsufficient data\n")
                continue
            try:
                t = _compute_technicals(hist["Close"], hist["High"], hist["Low"], hist["Volume"])
                start = float(hist["Close"].iloc[0])
                period_chg = (current - start) / start * 100

                # Bull signal count for this timeframe
                bull = sum([
                    t["sma20"] and current > t["sma20"],
                    t["sma50"] and current > t["sma50"],
                    t["macd_hist"] > 0,
                    t["rsi"] > 50,
                    t["stoch_k"] is not None and t["stoch_k"] > 50,
                    t["mom_20"] is not None and t["mom_20"] > 0,
                ])
                total = sum([
                    t["sma20"] is not None, t["sma50"] is not None,
                    True, True,
                    t["stoch_k"] is not None,
                    t["mom_20"] is not None,
                ])
                score_pct = bull / total * 100 if total else 50

                atr_str = f"${t['atr']:.2f}" if t["atr"] else "N/A"

                section = [
                    f"── {label} ({context}) ──",
                    f"  Period Change: {'+' if period_chg >= 0 else ''}{period_chg:.2f}%",
                    f"  RSI:           {t['rsi']:.1f}",
                    f"  MACD Hist:     {t['macd_hist']:+.3f}",
                    f"  BB Position:   {t['bb_pct']:.0f}% of band",
                    f"  Support:       ${t['support']:.2f}  |  Resistance: ${t['resistance']:.2f}",
                    f"  ATR:           {atr_str}",
                    f"  Vol Ratio:     {t['vol_ratio']:.1f}x",
                    f"  Bull Score:    {score_pct:.0f}%  ({bull}/{total} signals)",
                ]
                if t["sma50"]:
                    section.append(f"  SMA50:         ${t['sma50']:.2f} ({'above' if current > t['sma50'] else 'below'})")
                if t["sma200"]:
                    section.append(f"  SMA200:        ${t['sma200']:.2f} ({'above' if current > t['sma200'] else 'below'})")
                sections.append("\n".join(section))
            except Exception as e:
                sections.append(f"── {label} ──\nError: {e}")

        sections.append(
            "\nINSTRUCTION FOR AI: Using all timeframe data above plus fundamentals and news, "
            "produce a prediction table with Bear/Base/Bull price targets, "
            "% upside/downside, confidence, and signal (STRONG BUY/BUY/HOLD/SELL/STRONG SELL) "
            "for EACH of the 6 timeframes: 1 Day, 1 Week, 1 Month, 3 Months, 6 Months, 1 Year."
        )
        return "\n\n".join(sections)

    except Exception as e:
        return f"Multi-timeframe analysis failed for {symbol}: {e}"


def get_market_overview() -> str:
    indices = {
        "S&P 500":  "^GSPC",
        "NASDAQ":   "^IXIC",
        "Dow Jones":"^DJI",
        "VIX":      "^VIX",
        "Gold":     "GC=F",
        "Oil (WTI)":"CL=F",
        "BTC/USD":  "BTC-USD",
    }
    lines = ["Market Overview:"]
    for name, sym in indices.items():
        try:
            t = yf.Ticker(sym)
            info = t.info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            prev  = info.get("previousClose")
            if price and prev:
                chg = price - prev
                pct = chg / prev * 100
                sign = "+" if chg >= 0 else ""
                lines.append(f"  {name}: ${price:,.2f} ({sign}{pct:.2f}%)")
        except Exception:
            pass
    return "\n".join(lines)
