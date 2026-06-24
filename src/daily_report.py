"""
Daily Stock Market Report Agent
================================
Pulls real market data for the S&P 500 (via Yahoo Finance / yfinance),
analyzes movers, volume spikes, and basic technical indicators, pulls
recent headlines for the most notable names, and emails a detailed
HTML report via Gmail SMTP.

IMPORTANT / HONESTY NOTE
-------------------------
This script surfaces real, publicly available market data and flags
patterns (price moves, volume spikes, RSI, 52-week proximity). It does
NOT provide licensed financial advice, and nothing it outputs should be
treated as a guaranteed "buy" signal. It is a research/screening aid.

Data source: Yahoo Finance via the `yfinance` library. Data is typically
delayed ~15 minutes and is "best effort" — Yahoo can throttle or change
its undocumented endpoints at any time. This is free-tier data, not a
paid real-time feed.
"""

import os
import sys
import time
import smtplib
import traceback
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import numpy as np
import yfinance as yf
import requests

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

# How many tickers to pull per yfinance batch download (avoids rate limits)
BATCH_SIZE = 50
# Seconds to sleep between batches
BATCH_SLEEP = 2
# How many top gainers / losers / volume spikes to show in detail
TOP_N = 8
# RSI thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
# Volume spike threshold (today's volume vs 20-day average)
VOLUME_SPIKE_MULTIPLE = 2.0
# How close to a 52-week high/low counts as "near" (percent)
NEAR_52W_PCT = 3.0

EMAIL_FROM = os.environ.get("GMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_FROM)

# --------------------------------------------------------------------------
# TICKER UNIVERSE
# --------------------------------------------------------------------------


def get_sp500_tickers():
    """
    Pulls the current S&P 500 constituent list from Wikipedia.
    Falls back to a hardcoded slice of large caps if that fails,
    so the job never hard-fails just because Wikipedia's table changed.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 400:
            return tickers
    except Exception as e:
        print(f"[WARN] Could not fetch S&P 500 list from Wikipedia: {e}")

    # Fallback: a reasonably diverse large-cap list so the script still runs
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM",
        "V", "UNH", "XOM", "JNJ", "PG", "MA", "HD", "MRK", "COST", "ABBV", "AVGO",
        "PEP", "KO", "WMT", "BAC", "CVX", "ADBE", "CRM", "AMD", "NFLX", "TMO",
        "DIS", "PFE", "ORCL", "INTC", "VZ", "CSCO", "ABT", "NKE", "QCOM", "TXN",
    ]


# --------------------------------------------------------------------------
# DATA FETCHING
# --------------------------------------------------------------------------


def fetch_batch_history(tickers, period="3mo", interval="1d"):
    """
    Downloads OHLCV history for a batch of tickers in one call.
    Returns a dict: ticker -> DataFrame (or None if unavailable).
    """
    try:
        data = yf.download(
            tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        print(f"[WARN] Batch download failed for {tickers[:3]}...: {e}")
        return {t: None for t in tickers}

    result = {}
    for t in tickers:
        try:
            if len(tickers) == 1:
                df = data
            else:
                df = data[t]
            df = df.dropna(how="all")
            result[t] = df if not df.empty else None
        except Exception:
            result[t] = None
    return result


def fetch_all_data(tickers):
    """Fetches history for all tickers in batches, returns combined dict."""
    all_data = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        print(f"[INFO] Fetching batch {i // BATCH_SIZE + 1} ({len(batch)} tickers)...")
        batch_data = fetch_batch_history(batch)
        all_data.update(batch_data)
        time.sleep(BATCH_SLEEP)
    return all_data


# --------------------------------------------------------------------------
# ANALYSIS
# --------------------------------------------------------------------------


def compute_rsi(close_prices, period=14):
    """Standard 14-day RSI."""
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def analyze_ticker(ticker, df):
    """Computes a row of stats for one ticker. Returns dict or None."""
    if df is None or len(df) < 20:
        return None
    try:
        df = df.copy()
        df["RSI"] = compute_rsi(df["Close"])

        last = df.iloc[-1]
        prev = df.iloc[-2]

        last_close = float(last["Close"])
        prev_close = float(prev["Close"])
        pct_change = (last_close - prev_close) / prev_close * 100

        avg_vol_20 = float(df["Volume"].tail(20).mean())
        last_vol = float(last["Volume"])
        vol_ratio = last_vol / avg_vol_20 if avg_vol_20 > 0 else np.nan

        high_52w = float(df["High"].max())
        low_52w = float(df["Low"].min())
        pct_from_high = (last_close - high_52w) / high_52w * 100
        pct_from_low = (last_close - low_52w) / low_52w * 100

        rsi_val = float(last["RSI"]) if not pd.isna(last["RSI"]) else None

        return {
            "ticker": ticker,
            "last_close": last_close,
            "pct_change": pct_change,
            "volume": last_vol,
            "avg_vol_20": avg_vol_20,
            "vol_ratio": vol_ratio,
            "rsi": rsi_val,
            "pct_from_52w_high": pct_from_high,
            "pct_from_52w_low": pct_from_low,
        }
    except Exception as e:
        print(f"[WARN] Analysis failed for {ticker}: {e}")
        return None


def build_analysis_table(all_data):
    rows = []
    for ticker, df in all_data.items():
        row = analyze_ticker(ticker, df)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def get_recent_news(ticker, max_items=3):
    """Pulls recent headlines for a ticker via yfinance's news endpoint."""
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        items = []
        for n in news[:max_items]:
            title = n.get("title")
            publisher = n.get("publisher", "")
            link = n.get("link", "")
            if title:
                items.append({"title": title, "publisher": publisher, "link": link})
        return items
    except Exception as e:
        print(f"[WARN] News fetch failed for {ticker}: {e}")
        return []


def get_market_index_summary():
    """Pulls today's move for major indices: S&P 500, Nasdaq, Dow, VIX."""
    indices = {
        "^GSPC": "S&P 500",
        "^IXIC": "Nasdaq Composite",
        "^DJI": "Dow Jones",
        "^VIX": "VIX (Volatility Index)",
    }
    summary = []
    try:
        data = yf.download(
            list(indices.keys()), period="5d", interval="1d",
            group_by="ticker", progress=False, auto_adjust=True,
        )
        for symbol, name in indices.items():
            try:
                df = data[symbol].dropna()
                last = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                pct = (last - prev) / prev * 100
                summary.append({"name": name, "last": last, "pct_change": pct})
            except Exception:
                continue
    except Exception as e:
        print(f"[WARN] Index summary failed: {e}")
    return summary


# --------------------------------------------------------------------------
# EMAIL FORMATTING
# --------------------------------------------------------------------------


def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}%"


def fmt_num(x, decimals=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:,.{decimals}f}"


def color_for_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "#666666"
    return "#1a7f37" if x >= 0 else "#c0152f"


def row_html(label, value, color=None):
    color_style = f"color:{color};" if color else ""
    return f"""
    <tr>
      <td style="padding:6px 10px;border-bottom:1px solid #eee;font-weight:600;">{label}</td>
      <td style="padding:6px 10px;border-bottom:1px solid #eee;{color_style}">{value}</td>
    </tr>"""


def build_index_section(index_summary):
    if not index_summary:
        return "<p>Index data unavailable today.</p>"
    rows = ""
    for idx in index_summary:
        color = color_for_pct(idx["pct_change"])
        rows += f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #eee;">{idx['name']}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #eee;">{fmt_num(idx['last'])}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #eee;color:{color};font-weight:600;">{fmt_pct(idx['pct_change'])}</td>
        </tr>"""
    return f"""
    <table style="width:100%;border-collapse:collapse;font-family:Arial,sans-serif;font-size:14px;">
      <tr style="background:#f5f5f5;">
        <th style="text-align:left;padding:8px 12px;">Index</th>
        <th style="text-align:left;padding:8px 12px;">Level</th>
        <th style="text-align:left;padding:8px 12px;">Change</th>
      </tr>
      {rows}
    </table>"""


def build_movers_table(df, ascending=False, title="Top Gainers"):
    sub = df.sort_values("pct_change", ascending=ascending).head(TOP_N)
    rows = ""
    for _, r in sub.iterrows():
        color = color_for_pct(r["pct_change"])
        rsi_display = fmt_num(r["rsi"], 1) if r["rsi"] is not None else "N/A"
        rows += f"""
        <tr>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;font-weight:600;">{r['ticker']}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;">${fmt_num(r['last_close'])}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;color:{color};font-weight:600;">{fmt_pct(r['pct_change'])}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;">{fmt_num(r['vol_ratio'], 1)}x avg</td>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;">{rsi_display}</td>
        </tr>"""
    return f"""
    <h3 style="font-family:Arial,sans-serif;margin-bottom:4px;">{title}</h3>
    <table style="width:100%;border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px;">
      <tr style="background:#f5f5f5;">
        <th style="text-align:left;padding:6px 10px;">Ticker</th>
        <th style="text-align:left;padding:6px 10px;">Price</th>
        <th style="text-align:left;padding:6px 10px;">Change</th>
        <th style="text-align:left;padding:6px 10px;">Volume</th>
        <th style="text-align:left;padding:6px 10px;">RSI(14)</th>
      </tr>
      {rows}
    </table>"""


def build_volume_spike_table(df):
    sub = df[df["vol_ratio"] >= VOLUME_SPIKE_MULTIPLE].sort_values("vol_ratio", ascending=False).head(TOP_N)
    if sub.empty:
        return "<p style='font-family:Arial,sans-serif;font-size:13px;'>No unusual volume spikes detected today.</p>"
    rows = ""
    for _, r in sub.iterrows():
        color = color_for_pct(r["pct_change"])
        rows += f"""
        <tr>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;font-weight:600;">{r['ticker']}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;">${fmt_num(r['last_close'])}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;color:{color};">{fmt_pct(r['pct_change'])}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;font-weight:600;">{fmt_num(r['vol_ratio'], 1)}x avg</td>
        </tr>"""
    return f"""
    <h3 style="font-family:Arial,sans-serif;margin-bottom:4px;">Unusual Volume (&ge;{VOLUME_SPIKE_MULTIPLE}x 20-day avg)</h3>
    <table style="width:100%;border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px;">
      <tr style="background:#f5f5f5;">
        <th style="text-align:left;padding:6px 10px;">Ticker</th>
        <th style="text-align:left;padding:6px 10px;">Price</th>
        <th style="text-align:left;padding:6px 10px;">Change</th>
        <th style="text-align:left;padding:6px 10px;">Volume vs Avg</th>
      </tr>
      {rows}
    </table>"""


def build_technical_setups(df):
    """Flags oversold/overbought RSI and proximity to 52-week highs/lows."""
    oversold = df[(df["rsi"].notna()) & (df["rsi"] <= RSI_OVERSOLD)].sort_values("rsi").head(TOP_N)
    overbought = df[(df["rsi"].notna()) & (df["rsi"] >= RSI_OVERBOUGHT)].sort_values("rsi", ascending=False).head(TOP_N)
    near_high = df[df["pct_from_52w_high"] >= -NEAR_52W_PCT].sort_values("pct_from_52w_high", ascending=False).head(TOP_N)
    near_low = df[df["pct_from_52w_low"] <= NEAR_52W_PCT].sort_values("pct_from_52w_low").head(TOP_N)

    def mini_table(sub, value_col, label):
        if sub.empty:
            return f"<p style='font-family:Arial,sans-serif;font-size:13px;'>None today.</p>"
        rows = ""
        for _, r in sub.iterrows():
            rows += f"""
            <tr>
              <td style="padding:6px 10px;border-bottom:1px solid #eee;font-weight:600;">{r['ticker']}</td>
              <td style="padding:6px 10px;border-bottom:1px solid #eee;">${fmt_num(r['last_close'])}</td>
              <td style="padding:6px 10px;border-bottom:1px solid #eee;">{fmt_num(r[value_col], 2)}</td>
            </tr>"""
        return f"""
        <table style="width:100%;border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px;margin-bottom:14px;">
          <tr style="background:#f5f5f5;">
            <th style="text-align:left;padding:6px 10px;">Ticker</th>
            <th style="text-align:left;padding:6px 10px;">Price</th>
            <th style="text-align:left;padding:6px 10px;">{label}</th>
          </tr>
          {rows}
        </table>"""

    html = "<h3 style='font-family:Arial,sans-serif;margin-bottom:4px;'>Technical Screens</h3>"
    html += "<p style='font-family:Arial,sans-serif;font-size:13px;'><b>RSI Oversold (≤30)</b> — historically associated with short-term bounce potential, not a guarantee:</p>"
    html += mini_table(oversold, "rsi", "RSI")
    html += "<p style='font-family:Arial,sans-serif;font-size:13px;'><b>RSI Overbought (≥70)</b> — may be due for a pullback or simply in a strong trend:</p>"
    html += mini_table(overbought, "rsi", "RSI")
    html += "<p style='font-family:Arial,sans-serif;font-size:13px;'><b>Near 52-Week High</b> (within 3%):</p>"
    html += mini_table(near_high, "pct_from_52w_high", "% From High")
    html += "<p style='font-family:Arial,sans-serif;font-size:13px;'><b>Near 52-Week Low</b> (within 3%):</p>"
    html += mini_table(near_low, "pct_from_52w_low", "% From Low")
    return html


def build_news_section(tickers_of_interest):
    """Pulls news for a short list of the day's most notable tickers."""
    html = "<h3 style='font-family:Arial,sans-serif;margin-bottom:4px;'>Headlines on Today's Notable Movers</h3>"
    any_news = False
    for ticker in tickers_of_interest:
        items = get_recent_news(ticker)
        if not items:
            continue
        any_news = True
        html += f"<p style='font-family:Arial,sans-serif;font-size:13px;margin-bottom:2px;'><b>{ticker}</b></p><ul style='font-family:Arial,sans-serif;font-size:13px;margin-top:0;'>"
        for item in items:
            html += f"<li><a href='{item['link']}' style='color:#0a58ca;'>{item['title']}</a> <span style='color:#888;'>({item['publisher']})</span></li>"
        html += "</ul>"
    if not any_news:
        html += "<p style='font-family:Arial,sans-serif;font-size:13px;'>No headlines available today.</p>"
    return html


def build_watchlist_ideas(df):
    """
    Builds a 'things to look into' section: stocks with a confluence of
    signals (e.g. strong move + high volume + RSI not yet overbought).
    Framed explicitly as a screen, not advice.
    """
    candidates = df[
        (df["pct_change"] > 0)
        & (df["vol_ratio"] >= 1.5)
        & (df["rsi"].notna())
        & (df["rsi"] < RSI_OVERBOUGHT)
    ].copy()
    candidates["score"] = candidates["pct_change"] * candidates["vol_ratio"]
    candidates = candidates.sort_values("score", ascending=False).head(5)

    if candidates.empty:
        return "<p style='font-family:Arial,sans-serif;font-size:13px;'>No names cleared the screen criteria today.</p>", []

    html = """
    <p style="font-family:Arial,sans-serif;font-size:13px;">
    These names show a <b>positive price move + above-average volume + RSI not yet overbought</b> —
    a combination some traders watch for confirmed momentum. This is a mechanical screen on
    public data, <b>not personalized financial advice</b>. Always check the news/fundamentals
    behind a move before acting.
    </p>"""
    rows = ""
    for _, r in candidates.iterrows():
        rows += f"""
        <tr>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;font-weight:600;">{r['ticker']}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;">${fmt_num(r['last_close'])}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;color:{color_for_pct(r['pct_change'])};">{fmt_pct(r['pct_change'])}</td>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;">{fmt_num(r['vol_ratio'],1)}x avg</td>
          <td style="padding:6px 10px;border-bottom:1px solid #eee;">{fmt_num(r['rsi'],1)}</td>
        </tr>"""
    html += f"""
    <table style="width:100%;border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px;">
      <tr style="background:#f5f5f5;">
        <th style="text-align:left;padding:6px 10px;">Ticker</th>
        <th style="text-align:left;padding:6px 10px;">Price</th>
        <th style="text-align:left;padding:6px 10px;">Change</th>
        <th style="text-align:left;padding:6px 10px;">Volume</th>
        <th style="text-align:left;padding:6px 10px;">RSI</th>
      </tr>
      {rows}
    </table>"""
    return html, candidates["ticker"].tolist()


def build_email_html(index_summary, df, report_date):
    gainers_table = build_movers_table(df, ascending=False, title=f"Top {TOP_N} Gainers")
    losers_table = build_movers_table(df, ascending=True, title=f"Top {TOP_N} Losers")
    volume_table = build_volume_spike_table(df)
    technicals = build_technical_setups(df)
    ideas_html, idea_tickers = build_watchlist_ideas(df)

    # Pull news for: top 3 gainers, top 2 volume spikes, and screen ideas (deduped)
    top_gainers = df.sort_values("pct_change", ascending=False).head(3)["ticker"].tolist()
    top_volume = df.sort_values("vol_ratio", ascending=False).head(2)["ticker"].tolist()
    news_tickers = list(dict.fromkeys(top_gainers + top_volume + idea_tickers))[:6]
    news_html = build_news_section(news_tickers)

    index_html = build_index_section(index_summary)

    html = f"""
    <html>
    <body style="margin:0;padding:0;background:#f9f9f9;">
    <div style="max-width:760px;margin:0 auto;padding:20px;background:#ffffff;">
      <h1 style="font-family:Arial,sans-serif;font-size:22px;border-bottom:3px solid #1a1a1a;padding-bottom:10px;">
        📈 Daily Market Report — {report_date}
      </h1>
      <p style="font-family:Arial,sans-serif;font-size:12px;color:#888;">
        Data source: Yahoo Finance (yfinance), delayed ~15 min. This report is automated,
        screen-based analysis of public market data — not personalized financial advice.
      </p>

      <h2 style="font-family:Arial,sans-serif;font-size:18px;margin-top:24px;">Market Snapshot</h2>
      {index_html}

      <h2 style="font-family:Arial,sans-serif;font-size:18px;margin-top:24px;">Movers</h2>
      {gainers_table}
      <div style="height:14px;"></div>
      {losers_table}

      <h2 style="font-family:Arial,sans-serif;font-size:18px;margin-top:24px;">Volume Activity</h2>
      {volume_table}

      <h2 style="font-family:Arial,sans-serif;font-size:18px;margin-top:24px;">{technicals}</h2>

      <h2 style="font-family:Arial,sans-serif;font-size:18px;margin-top:24px;">Today's Screen: Momentum + Volume Confirmation</h2>
      {ideas_html}

      <h2 style="font-family:Arial,sans-serif;font-size:18px;margin-top:24px;">{news_html}</h2>

      <p style="font-family:Arial,sans-serif;font-size:11px;color:#aaa;margin-top:30px;border-top:1px solid #eee;padding-top:10px;">
        Generated automatically. Not financial advice. Past performance and technical
        signals do not guarantee future results. Always do your own research.
      </p>
    </div>
    </body>
    </html>
    """
    return html


# --------------------------------------------------------------------------
# EMAIL SENDING
# --------------------------------------------------------------------------


def send_email(subject, html_body):
    if not EMAIL_FROM or not EMAIL_APP_PASSWORD:
        print("[ERROR] Missing GMAIL_ADDRESS or GMAIL_APP_PASSWORD environment variables.")
        sys.exit(1)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    print(f"[INFO] Email sent to {EMAIL_TO}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------


def main():
    report_date = datetime.now().strftime("%A, %B %d, %Y")
    print(f"[INFO] Starting daily report run for {report_date}")

    print("[INFO] Fetching index summary...")
    index_summary = get_market_index_summary()

    print("[INFO] Fetching S&P 500 ticker list...")
    tickers = get_sp500_tickers()
    print(f"[INFO] {len(tickers)} tickers to analyze.")

    print("[INFO] Fetching historical data...")
    all_data = fetch_all_data(tickers)

    print("[INFO] Running analysis...")
    df = build_analysis_table(all_data)
    print(f"[INFO] Successfully analyzed {len(df)} / {len(tickers)} tickers.")

    if df.empty:
        send_email(
            f"⚠️ Daily Market Report ({report_date}) — Data Error",
            "<p>The report could not be generated today because no ticker data was retrieved. "
            "Check the GitHub Actions log for details.</p>",
        )
        sys.exit(1)

    print("[INFO] Building email...")
    html = build_email_html(index_summary, df, report_date)

    subject = f"📈 Daily Market Report — {report_date}"
    send_email(subject, html)
    print("[INFO] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)
