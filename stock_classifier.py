# ============================================================
#  Stock Classifier — Beta + Covariance Analysis
#  Run in Google Colab: just execute all cells top to bottom
# ============================================================

# ── Install dependencies ─────────────────────────────────────
# !pip install yfinance pandas matplotlib seaborn -q

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  CONFIGURATION — edit this section
# ============================================================

MARKET_TICKER = "^GSPC"          # Benchmark (S&P 500)
PERIOD        = "1y"             # Lookback: 1y, 2y, 6mo
RISK_FREE_RATE= 0.05             # Annual risk-free rate (5%)
TOP_N         = 10               # Top stocks to show per category

STOCKS = [
    # Technology
    "AAPL","MSFT","NVDA","GOOGL","META","AMD","INTC","CRM","ORCL","ADBE",
    "QCOM","TXN","NOW","SNOW","PLTR","PANW","AMAT","MU","LRCX","KLAC",
    # Healthcare
    "JNJ","UNH","LLY","ABBV","PFE","MRK","TMO","ABT","DHR","AMGN",
    "GILD","ISRG","CVS","BMY","VRTX","REGN","ZTS","SYK","BSX","ELV",
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","C","AXP","SPGI","CB",
    "AON","MMC","PGR","TRV","USB","COF","SCHW","ICE","CME","MCO",
    # Consumer
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","COST","LOW","BKNG",
    "MAR","HLT","GM","F","ABNB","PG","KO","PEP","WMT","MDLZ",
    # Energy
    "XOM","CVX","COP","EOG","SLB","PSX","VLO","MPC","OXY","HAL",
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
    "NFLX","DIS","CMCSA","T","VZ","CHTR","TMUS",
    # High Beta / Crypto-adjacent
    "COIN","MSTR","MARA","RIOT","SOFI","HOOD","SQ","UPST","AFRM","PYPL",
]

# ============================================================
#  MATH FUNCTIONS
# ============================================================

def covariance(x: pd.Series, y: pd.Series) -> float:
    """Cov(X,Y) = E[XY] - E[X]*E[Y]"""
    return float((x * y).mean() - x.mean() * y.mean())

def correlation(x: pd.Series, y: pd.Series) -> float:
    """ρ = Cov(X,Y) / (σX * σY)"""
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return 0.0
    return covariance(x, y) / (sx * sy)

def beta(stock_ret: pd.Series, market_ret: pd.Series) -> float:
    """β = Cov(stock, market) / Var(market)"""
    var_m = market_ret.var()
    if var_m == 0:
        return 0.0
    return covariance(stock_ret, market_ret) / var_m

def sharpe(ret: pd.Series, rf_annual: float = RISK_FREE_RATE) -> float:
    """Annualised Sharpe Ratio"""
    rf_daily = rf_annual / 252
    excess   = ret - rf_daily
    return float(excess.mean() / excess.std() * np.sqrt(252)) if ret.std() != 0 else 0.0

def annual_return(ret: pd.Series) -> float:
    """Compound annual return (%)"""
    return float(((1 + ret).prod()) ** (252 / len(ret)) - 1) * 100

def annual_volatility(ret: pd.Series) -> float:
    """Annualised volatility (%)"""
    return float(ret.std() * np.sqrt(252)) * 100

# ============================================================
#  CLASSIFICATION LOGIC
# ============================================================

#  Beta vs Correlation matrix:
#
#            │  β < 0.65    │ 0.65–1.05  │ 1.05–1.50  │ β ≥ 1.50
#  ─────────────────────────────────────────────────────────────────
#  |ρ| < 0.30│  Uncorrelated│ Uncorrelated│ Uncorrelated│ Uncorrelated
#  ρ ≥ 0.25  │  Defensive   │   Follower  │   Leader   │  Aggressive
#

CATEGORIES = [
    "Aggressive Growth",
    "Market Leader",
    "Market Follower",
    "Defensive",
    "Uncorrelated",
]

CAT_COLORS = {
    "Aggressive Growth": "#ff4757",
    "Market Leader":     "#ffa502",
    "Market Follower":   "#2ed573",
    "Defensive":         "#1e90ff",
    "Uncorrelated":      "#a29bfe",
}

CAT_DESC = {
    "Aggressive Growth": "β ≥ 1.5 & ρ ≥ 0.55  |  Amplifies market moves — high risk/reward",
    "Market Leader":     "1.05 ≤ β < 1.5 & ρ ≥ 0.60  |  Outpaces market on up-days",
    "Market Follower":   "0.65 ≤ β < 1.05 & ρ ≥ 0.50  |  Tracks market reliably",
    "Defensive":         "β < 0.65 & ρ ≥ 0.25  |  Holds value in downturns",
    "Uncorrelated":      "|ρ| < 0.30  |  Moves independently — diversification gem",
}

def classify(b: float, r: float) -> str:
    if abs(r) < 0.30:
        return "Uncorrelated"
    if b >= 1.5  and r >= 0.55:
        return "Aggressive Growth"
    if 1.05 <= b < 1.5 and r >= 0.60:
        return "Market Leader"
    if 0.65 <= b < 1.05 and r >= 0.50:
        return "Market Follower"
    if b < 0.65:
        return "Defensive"
    return "Market Follower"

# ============================================================
#  DATA DOWNLOAD & ANALYSIS
# ============================================================

print("=" * 65)
print("  STOCK CLASSIFIER — Beta + Covariance Analysis")
print(f"  Benchmark: {MARKET_TICKER}  |  Period: {PERIOD}  |  Universe: {len(STOCKS)} stocks")
print("=" * 65)
print("\n  Downloading price data…")

all_tickers = STOCKS + [MARKET_TICKER]
raw = yf.download(all_tickers, period=PERIOD, auto_adjust=True, progress=False)["Close"]

if isinstance(raw, pd.Series):
    raw = raw.to_frame()

returns   = raw.pct_change().dropna()
mkt_ret   = returns[MARKET_TICKER]

print(f"  ✓ Downloaded {raw.shape[1]-1} instruments, {len(returns)} trading days\n")

# ── Compute metrics for every stock ──────────────────────────
rows = []
for sym in STOCKS:
    if sym not in returns.columns:
        continue
    s = returns[sym].dropna()
    m = mkt_ret.reindex(s.index).dropna()
    s = s.reindex(m.index)
    if len(s) < 50:
        continue

    b_val   = beta(s, m)
    r_val   = correlation(s, m)
    cov_val = covariance(s, m)
    sh_val  = sharpe(s)
    ar_val  = annual_return(s)
    vol_val = annual_volatility(s)
    cat     = classify(b_val, r_val)

    rows.append({
        "Symbol":     sym,
        "Category":   cat,
        "Beta":       b_val,
        "Covariance": cov_val,
        "Correlation":r_val,
        "Ann Return %": ar_val,
        "Volatility %": vol_val,
        "Sharpe":     sh_val,
    })

df = pd.DataFrame(rows)

# ============================================================
#  RANK WITHIN EACH CATEGORY
# ============================================================

def rank_category(subset: pd.DataFrame, cat: str) -> pd.DataFrame:
    if cat == "Aggressive Growth":
        subset = subset.copy()
        subset["score"] = subset["Beta"] * 0.4 + subset["Sharpe"] * 0.4 + subset["Correlation"] * 0.2
    elif cat == "Market Leader":
        subset = subset.copy()
        subset["score"] = subset["Beta"] * 0.35 + subset["Sharpe"] * 0.4 + subset["Correlation"] * 0.25
    elif cat == "Market Follower":
        subset = subset.copy()
        subset["score"] = subset["Correlation"] * 0.5 + subset["Sharpe"] * 0.5
    elif cat == "Defensive":
        subset = subset.copy()
        subset["score"] = (1 / (subset["Beta"].abs() + 0.01)) * 0.5 + subset["Sharpe"] * 0.5
    else:  # Uncorrelated
        subset = subset.copy()
        subset["score"] = (1 / (subset["Correlation"].abs() + 0.01)) * 0.4 + subset["Ann Return %"] * 0.6
    return subset.sort_values("score", ascending=False)

# ============================================================
#  PRINT RESULTS TABLE
# ============================================================

print("=" * 65)
print("  CLASSIFICATION RESULTS")
print("=" * 65)

summary = {}
for cat in CATEGORIES:
    subset = df[df["Category"] == cat]
    summary[cat] = len(subset)
    if subset.empty:
        continue

    ranked = rank_category(subset, cat).head(TOP_N)

    print(f"\n  ★  {cat.upper()}")
    print(f"     {CAT_DESC[cat]}")
    print(f"     Found: {len(subset)} stocks  |  Showing top {min(TOP_N, len(subset))}")
    print()
    print(f"  {'Symbol':<8} {'Beta':>7} {'Corr':>7} {'Cov':>10} {'Ann Ret':>9} {'Vol':>8} {'Sharpe':>8}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*10} {'─'*9} {'─'*8} {'─'*8}")
    for _, row in ranked.iterrows():
        print(f"  {row['Symbol']:<8} "
              f"{row['Beta']:>+7.3f} "
              f"{row['Correlation']:>+7.3f} "
              f"{row['Covariance']:>10.6f} "
              f"{row['Ann Return %']:>+8.1f}% "
              f"{row['Volatility %']:>7.1f}% "
              f"{row['Sharpe']:>8.2f}")

print(f"\n{'=' * 65}")
print("  UNIVERSE BREAKDOWN")
print(f"{'─' * 65}")
for cat, count in summary.items():
    bar = "█" * int(count / max(summary.values()) * 30)
    print(f"  {cat:<22} {bar:<30} {count:>3} stocks")
print(f"  {'Total':<22} {'':<30} {sum(summary.values()):>3} stocks")
print("=" * 65)

# ============================================================
#  VISUALISATION 1 — Scatter: Beta vs Correlation (coloured)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#0d0f14")

ax1 = axes[0]
ax1.set_facecolor("#13161e")

for cat in CATEGORIES:
    subset = df[df["Category"] == cat]
    ax1.scatter(
        subset["Beta"],
        subset["Correlation"],
        c=CAT_COLORS[cat],
        label=cat,
        alpha=0.85,
        s=80,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    for _, row in subset.iterrows():
        ax1.annotate(
            row["Symbol"],
            (row["Beta"], row["Correlation"]),
            fontsize=6.5, color="white", alpha=0.75,
            xytext=(3, 3), textcoords="offset points",
        )

# Classification boundary lines
ax1.axhline(0.30, color="#444466", lw=1, ls="--", alpha=0.7, label="|ρ|=0.30 boundary")
ax1.axvline(0.65, color="#446644", lw=1, ls=":",  alpha=0.7)
ax1.axvline(1.05, color="#664444", lw=1, ls=":",  alpha=0.7)
ax1.axvline(1.50, color="#ff4757", lw=1, ls=":",  alpha=0.7)

ax1.set_xlabel("Beta  (β)", color="white", fontsize=11)
ax1.set_ylabel("Correlation with S&P 500  (ρ)", color="white", fontsize=11)
ax1.set_title("Stock Classification Map\nBeta vs Market Correlation", color="white", fontsize=12, fontweight="bold")
ax1.tick_params(colors="white")
ax1.spines[:].set_color("#333355")
ax1.grid(True, alpha=0.15, color="white")
legend = ax1.legend(facecolor="#0d0f14", edgecolor="#333355", labelcolor="white", fontsize=9)
ax1.set_xlim(df["Beta"].min() - 0.2, df["Beta"].max() + 0.2)
ax1.set_ylim(-0.1, 1.05)

# ── Annotations for zones ─────────────────────────────────────
zone_labels = [
    (0.30, 0.85, "DEFENSIVE", "#1e90ff"),
    (0.80, 0.85, "FOLLOWER",  "#2ed573"),
    (1.20, 0.85, "LEADER",    "#ffa502"),
    (1.70, 0.85, "AGGRESSIVE","#ff4757"),
    (1.00, 0.10, "UNCORRELATED","#a29bfe"),
]
for bx, ry, label, color in zone_labels:
    ax1.text(bx, ry, label, fontsize=8, color=color, alpha=0.6,
             ha="center", fontweight="bold")

# ── Bar chart: top picks per category ─────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#13161e")

top_picks = []
for cat in CATEGORIES:
    subset = df[df["Category"] == cat]
    if subset.empty:
        continue
    ranked = rank_category(subset, cat).head(5)
    for _, row in ranked.iterrows():
        top_picks.append({"symbol": row["Symbol"], "cat": cat,
                          "return": row["Ann Return %"], "sharpe": row["Sharpe"]})

top_df   = pd.DataFrame(top_picks)
colors   = [CAT_COLORS[c] for c in top_df["cat"]]
y_pos    = range(len(top_df))
bars     = ax2.barh(list(y_pos), top_df["return"], color=colors, alpha=0.85,
                    edgecolor="white", linewidth=0.3)

ax2.set_yticks(list(y_pos))
ax2.set_yticklabels(
    [f"{r['symbol']}  (β {df[df['Symbol']==r['symbol']]['Beta'].values[0]:+.2f})"
     for _, r in top_df.iterrows()],
    color="white", fontsize=9,
)
ax2.set_xlabel("Annual Return (%)", color="white", fontsize=11)
ax2.set_title("Top Picks per Category\n(Annual Return, 1Y)", color="white", fontsize=12, fontweight="bold")
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#333355")
ax2.grid(True, alpha=0.15, color="white", axis="x")
ax2.axvline(0, color="white", lw=0.8, alpha=0.5)
ax2.set_facecolor("#13161e")

# Return value labels on bars
for bar_obj, val in zip(bars, top_df["return"]):
    ax2.text(
        val + (1 if val >= 0 else -1),
        bar_obj.get_y() + bar_obj.get_height() / 2,
        f"{val:+.1f}%",
        va="center", ha="left" if val >= 0 else "right",
        color="white", fontsize=8,
    )

# Legend
patches = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CATEGORIES]
ax2.legend(handles=patches, facecolor="#0d0f14", edgecolor="#333355",
           labelcolor="white", fontsize=8, loc="lower right")

plt.suptitle(
    f"Stock Classifier — Beta + Covariance  |  Benchmark: S&P 500  |  Period: {PERIOD}",
    color="white", fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("stock_classification.png", dpi=150, bbox_inches="tight",
            facecolor="#0d0f14")
plt.show()
print("\n  Chart saved → stock_classification.png")

# ============================================================
#  PRINT FULL DATAFRAME (sortable in Colab)
# ============================================================

display_cols = ["Symbol","Category","Beta","Correlation","Covariance","Ann Return %","Volatility %","Sharpe"]
df_display   = df[display_cols].sort_values("Beta", ascending=False).reset_index(drop=True)

df_display["Beta"]        = df_display["Beta"].round(3)
df_display["Correlation"] = df_display["Correlation"].round(4)
df_display["Covariance"]  = df_display["Covariance"].round(6)
df_display["Ann Return %"]= df_display["Ann Return %"].round(1)
df_display["Volatility %"]= df_display["Volatility %"].round(1)
df_display["Sharpe"]      = df_display["Sharpe"].round(2)

print("\n  FULL RESULTS TABLE")
print(df_display.to_string(index=False))

# ============================================================
#  EXPORT TO CSV
# ============================================================

df_display.to_csv("stock_classification.csv", index=False)
print("\n  Data exported → stock_classification.csv")
print("=" * 65)
