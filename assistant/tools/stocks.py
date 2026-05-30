import yfinance as yf
import pandas as pd


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


def analyze_stock(symbol: str, period: str = "1mo") -> str:
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)
        if hist.empty:
            return f"No historical data for {symbol}."

        close = hist["Close"]
        current = close.iloc[-1]
        start = close.iloc[0]
        change_pct = (current - start) / start * 100

        sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
        sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = float((100 - 100 / (1 + rs)).iloc[-1])

        # Bollinger Bands
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper_band = float((mid + 2 * std).iloc[-1])
        lower_band = float((mid - 2 * std).iloc[-1])

        # Volume trend
        avg_vol = hist["Volume"].rolling(10).mean().iloc[-1]
        last_vol = hist["Volume"].iloc[-1]
        vol_signal = "above average" if last_vol > avg_vol else "below average"

        lines = [
            f"{symbol.upper()} Technical Analysis ({period})",
            f"Current Price: ${current:.2f}",
            f"Period Change: {'+' if change_pct >= 0 else ''}{change_pct:.2f}%",
        ]
        if sma20:
            lines.append(f"SMA20: ${sma20:.2f} ({'above' if current > sma20 else 'below'})")
        if sma50:
            lines.append(f"SMA50: ${sma50:.2f} ({'above' if current > sma50 else 'below'})")
        if sma200:
            lines.append(f"SMA200: ${sma200:.2f} ({'above' if current > sma200 else 'below'})")
        lines.append(f"RSI(14): {rsi:.1f} — {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'}")
        lines.append(f"Bollinger Bands: ${lower_band:.2f} – ${upper_band:.2f}")
        lines.append(f"Volume: {vol_signal}")

        # Simple bias
        bias_signals = 0
        if sma20 and current > sma20:
            bias_signals += 1
        if sma50 and current > sma50:
            bias_signals += 1
        if rsi > 50:
            bias_signals += 1
        if current > (upper_band + lower_band) / 2:
            bias_signals += 1
        total = sum([sma20 is not None, sma50 is not None, True, True])
        bias = "Bullish" if bias_signals >= total * 0.6 else "Bearish" if bias_signals <= total * 0.3 else "Neutral"
        lines.append(f"Overall Bias: {bias}")

        return "\n".join(lines)
    except Exception as e:
        return f"Analysis failed for {symbol}: {e}"


def get_market_overview() -> str:
    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "VIX": "^VIX",
        "Gold": "GC=F",
        "Oil (WTI)": "CL=F",
        "BTC/USD": "BTC-USD",
    }
    lines = ["Market Overview:"]
    for name, sym in indices.items():
        try:
            t = yf.Ticker(sym)
            info = t.info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            prev = info.get("previousClose")
            if price and prev:
                chg = price - prev
                pct = chg / prev * 100
                sign = "+" if chg >= 0 else ""
                lines.append(f"  {name}: ${price:,.2f} ({sign}{pct:.2f}%)")
        except Exception:
            pass
    return "\n".join(lines)
