"""
Lighter Exchange Alert Bot
Scans all active markets every 2 hours for:
  - SFP (Swing Failure Pattern) on 4h and Daily timeframes
  - Filtered by 200 EMA bias on Daily timeframe
Sends alerts to Telegram when setups align.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

import aiohttp

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

BASE_URL = "https://mainnet.zklighter.elliot.ai"
SCAN_INTERVAL_HOURS = 2

# SFP detection: how many candles to look back for the prior swing high/low
SFP_LOOKBACK = 20

# 200 EMA period
EMA_PERIOD = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── LIGHTER API ──────────────────────────────────────────────────────────────

async def fetch_active_markets(session: aiohttp.ClientSession) -> list[dict]:
    """Fetch all active markets from Lighter and return list of {symbol, market_id}."""
    url = f"{BASE_URL}/api/v1/orderBooks"
    try:
        async with session.get(url, params={"filter": "all"}, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json()
            markets = []
            for ob in data.get("order_books", []):
                if ob.get("status") == "active":
                    markets.append({
                        "symbol": ob["symbol"],
                        "market_id": ob["market_id"],
                        "market_type": ob.get("market_type", "perp"),
                    })
            log.info(f"Found {len(markets)} active markets on Lighter")
            return markets
    except Exception as e:
        log.error(f"Failed to fetch markets: {e}")
        return []


async def fetch_candles(
    session: aiohttp.ClientSession,
    market_id: int,
    resolution: str,
    count: int,
) -> list[dict]:
    """
    Fetch `count` candles for a market at a given resolution.
    Resolution options: 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w
    Returns list of candle dicts with keys: t, o, h, l, c
    """
    url = f"{BASE_URL}/api/v1/candles"
    now_ms = int(time.time() * 1000)

    # Resolution to milliseconds map for start_timestamp calculation
    res_ms = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "4h": 14_400_000, "12h": 43_200_000,
        "1d": 86_400_000, "1w": 604_800_000,
    }
    ms_per_candle = res_ms.get(resolution, 86_400_000)
    start_ms = now_ms - (count + 5) * ms_per_candle  # small buffer

    params = {
        "market_id": market_id,
        "resolution": resolution,
        "start_timestamp": start_ms,
        "end_timestamp": now_ms,
        "count_back": count,
    }
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json()
            candles = data.get("c", [])
            # Sort ascending by timestamp
            candles.sort(key=lambda x: x["t"])
            return candles
    except Exception as e:
        log.debug(f"Candle fetch error market={market_id} res={resolution}: {e}")
        return []


# ─── INDICATORS ───────────────────────────────────────────────────────────────

def calculate_ema(prices: list[float], period: int) -> float | None:
    """Calculate EMA for a list of closing prices. Returns the last EMA value."""
    if len(prices) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period  # seed with SMA
    for price in prices[period:]:
        ema = price * k + ema * (1 - k)
    return ema


def detect_sfp(candles: list[dict]) -> dict | None:
    """
    Detect a Swing Failure Pattern on the most recent completed candle.

    Bullish SFP: price wicks BELOW the prior swing low (within lookback),
                 but CLOSES BACK ABOVE it. Signals long.

    Bearish SFP: price wicks ABOVE the prior swing high (within lookback),
                 but CLOSES BACK BELOW it. Signals short.

    Returns dict with keys: direction ('long'/'short'), swing_level, wick_extreme
    or None if no SFP detected.
    """
    if len(candles) < SFP_LOOKBACK + 1:
        return None

    # Use the last completed candle (index -2; -1 may be forming)
    current = candles[-2]
    lookback_candles = candles[-(SFP_LOOKBACK + 2):-2]

    if not lookback_candles:
        return None

    prior_swing_low = min(c["l"] for c in lookback_candles)
    prior_swing_high = max(c["h"] for c in lookback_candles)

    c_low = current["l"]
    c_high = current["h"]
    c_close = current["c"]
    c_open = current["o"]

    # Bullish SFP: wick below prior swing low, close back above it
    if c_low < prior_swing_low and c_close > prior_swing_low:
        return {
            "direction": "long",
            "swing_level": prior_swing_low,
            "wick_extreme": c_low,
            "close": c_close,
        }

    # Bearish SFP: wick above prior swing high, close back below it
    if c_high > prior_swing_high and c_close < prior_swing_high:
        return {
            "direction": "short",
            "swing_level": prior_swing_high,
            "wick_extreme": c_high,
            "close": c_close,
        }

    return None


# ─── TELEGRAM ─────────────────────────────────────────────────────────────────

async def send_telegram(session: aiohttp.ClientSession, message: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            result = await resp.json()
            if not result.get("ok"):
                log.error(f"Telegram error: {result}")
    except Exception as e:
        log.error(f"Telegram send failed: {e}")


def format_alert(
    symbol: str,
    market_type: str,
    timeframe: str,
    sfp: dict,
    ema_value: float,
    current_price: float,
) -> str:
    direction = sfp["direction"].upper()
    emoji = "🟢" if sfp["direction"] == "long" else "🔴"
    bias = "ABOVE" if sfp["direction"] == "long" else "BELOW"

    return (
        f"{emoji} <b>SFP SETUP — {symbol} ({market_type.upper()})</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Timeframe:</b> {timeframe}\n"
        f"📍 <b>Direction:</b> {direction}\n"
        f"💰 <b>Current Price:</b> {current_price:.4f}\n"
        f"📏 <b>Swing Level:</b> {sfp['swing_level']:.4f}\n"
        f"🔽 <b>Wick Extreme:</b> {sfp['wick_extreme']:.4f}\n"
        f"📈 <b>Daily 200 EMA:</b> {ema_value:.4f}\n"
        f"✅ <b>Bias:</b> Price {bias} 200 EMA → {direction} bias confirmed\n"
        f"🕒 <b>Scan time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


# ─── SCAN LOGIC ───────────────────────────────────────────────────────────────

async def scan_market(
    session: aiohttp.ClientSession,
    market: dict,
) -> list[str]:
    """
    Scan a single market. Returns list of alert messages (empty if no setups).
    """
    symbol = market["symbol"]
    market_id = market["market_id"]
    market_type = market["market_type"]
    alerts = []

    # Fetch daily candles for 200 EMA (need at least 205 for reliable EMA)
    daily_candles = await fetch_candles(session, market_id, "1d", 210)
    if len(daily_candles) < EMA_PERIOD + 5:
        log.debug(f"{symbol}: insufficient daily candles ({len(daily_candles)}), skipping")
        return []

    closes = [c["c"] for c in daily_candles]
    ema_200 = calculate_ema(closes, EMA_PERIOD)
    if ema_200 is None:
        return []

    current_price = daily_candles[-1]["c"]
    price_above_ema = current_price > ema_200

    # ── Daily SFP scan ──────────────────────────────────────────────────
    daily_sfp = detect_sfp(daily_candles)
    if daily_sfp:
        sfp_is_long = daily_sfp["direction"] == "long"
        # SFP direction must match EMA bias
        if sfp_is_long == price_above_ema:
            msg = format_alert(symbol, market_type, "Daily", daily_sfp, ema_200, current_price)
            alerts.append(msg)
            log.info(f"ALERT: {symbol} Daily SFP {daily_sfp['direction'].upper()}")

    # ── 4H SFP scan ─────────────────────────────────────────────────────
    h4_candles = await fetch_candles(session, market_id, "4h", 50)
    if len(h4_candles) >= SFP_LOOKBACK + 2:
        h4_sfp = detect_sfp(h4_candles)
        if h4_sfp:
            sfp_is_long = h4_sfp["direction"] == "long"
            if sfp_is_long == price_above_ema:
                msg = format_alert(symbol, market_type, "4H", h4_sfp, ema_200, current_price)
                alerts.append(msg)
                log.info(f"ALERT: {symbol} 4H SFP {h4_sfp['direction'].upper()}")

    return alerts


async def run_scan(session: aiohttp.ClientSession) -> None:
    """Run a full scan across all active markets."""
    log.info("━━━ Starting market scan ━━━")
    markets = await fetch_active_markets(session)
    if not markets:
        log.warning("No markets returned, skipping scan")
        return

    all_alerts = []

    # Scan markets with small delay between each to be API-friendly
    for market in markets:
        try:
            alerts = await scan_market(session, market)
            all_alerts.extend(alerts)
        except Exception as e:
            log.error(f"Error scanning {market['symbol']}: {e}")
        await asyncio.sleep(0.3)  # gentle rate limiting

    if all_alerts:
        # Send a header first
        header = (
            f"🔔 <b>Lighter Scan — {len(all_alerts)} Setup(s) Found</b>\n"
            f"Scanned {len(markets)} markets at "
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        await send_telegram(session, header)
        await asyncio.sleep(0.5)
        for alert in all_alerts:
            await send_telegram(session, alert)
            await asyncio.sleep(0.5)
    else:
        log.info(f"Scan complete — no setups found across {len(markets)} markets")
        # Send a quiet heartbeat every scan so you know it's alive
        heartbeat = (
            f"💤 <b>Lighter Scan Complete</b> — No setups\n"
            f"Scanned {len(markets)} markets at "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        await send_telegram(session, heartbeat)


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("Lighter Alert Bot starting...")

    # Validate config
    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        log.error("Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables before running.")
        return

    async with aiohttp.ClientSession() as session:
        # Send startup message
        await send_telegram(
            session,
            f"🤖 <b>Lighter Alert Bot Online</b>\n"
            f"Scanning all active markets every {SCAN_INTERVAL_HOURS}h\n"
            f"Filters: Daily 200 EMA bias + SFP (4H & Daily)\n"
            f"Started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        while True:
            try:
                await run_scan(session)
            except Exception as e:
                log.error(f"Scan loop error: {e}")
                await send_telegram(session, f"⚠️ Bot error: {e}")

            log.info(f"Next scan in {SCAN_INTERVAL_HOURS} hours")
            await asyncio.sleep(SCAN_INTERVAL_HOURS * 3600)


if __name__ == "__main__":
    asyncio.run(main())
