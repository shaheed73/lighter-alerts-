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

# Minimum wick size as a percentage of price (filters weak SFPs)
MIN_WICK_PCT = 0.0075  # 0.75%

# Minimum R:R ratio required to send alert
MIN_RR = 2.0

# Deduplication: tracks already-alerted SFPs {symbol_timeframe: candle_timestamp}
alerted_sfps: dict = {}

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
    start_ms = now_ms - (count + 5) * ms_per_candle  # small buffer

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
    ema = sum(prices[:period]) / period  # seed with SMA
    for price in prices[period:]:
        ema = price * k + ema * (1 - k)
    return ema


def find_nearest_target(candles: list[dict], direction: str, entry: float, stop: float) -> float | None:
    """
    Find the nearest swing high (for longs) or swing low (for shorts)
    in the lookback window that gives at least MIN_RR.
    Returns the target price or None if no valid target exists.
    """
    risk = abs(entry - stop)
    if risk == 0:
        return None

    min_target_distance = risk * MIN_RR
    lookback_candles = candles[-(SFP_LOOKBACK + 2):-2]

    if direction == "long":
        # Find swing highs above entry that give at least 2:1
        candidates = [
            c["h"] for c in lookback_candles
            if c["h"] > entry + min_target_distance
        ]
        if not candidates:
            return None
        return min(candidates)  # nearest (lowest) qualifying high
    else:
        # Find swing lows below entry that give at least 2:1
        candidates = [
            c["l"] for c in lookback_candles
            if c["l"] < entry - min_target_distance
        ]
        if not candidates:
            return None
        return max(candidates)  # nearest (highest) qualifying low


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

    # Minimum close distance back inside as % of swing level
    MIN_CLOSE_PCT = 0.005  # 0.5%

    # Bullish SFP: wick below prior swing low, close back above it
    if c_low < prior_swing_low and c_close > prior_swing_low:
        wick_size = (prior_swing_low - c_low) / prior_swing_low
        if wick_size < MIN_WICK_PCT:
            return None
        close_distance = (c_close - prior_swing_low) / prior_swing_low
        if close_distance < MIN_CLOSE_PCT:
            return None
        return {
            "direction": "long",
            "swing_level": prior_swing_low,
            "wick_extreme": c_low,
            "close": c_close,
            "wick_pct": wick_size,
        }

    # Bearish SFP: wick above prior swing high, close back below it
    if c_high > prior_swing_high and c_close < prior_swing_high:
        wick_size = (c_high - prior_swing_high) / prior_swing_high
        if wick_size < MIN_WICK_PCT:
            return None
        close_distance = (prior_swing_high - c_close) / prior_swing_high
        if close_distance < MIN_CLOSE_PCT:
            return None
        return {
            "direction": "short",
            "swing_level": prior_swing_high,
            "wick_extreme": c_high,
            "close": c_close,
            "wick_pct": wick_size,
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


def bst_now() -> str:
    """Return current time as BST (UTC+1) and UTC string."""
    from datetime import timedelta
    utc_now = datetime.now(timezone.utc)
    bst = utc_now + timedelta(hours=1)
    return f"{bst.strftime('%Y-%m-%d %H:%M')} BST ({utc_now.strftime('%H:%M')} UTC)"


def format_alert(
    symbol: str,
    market_type: str,
    timeframe: str,
    sfp: dict,
    ema_daily: float,
    ema_4h: float,
    current_price: float,
    target: float,
    rr: float,
) -> str:
    direction = sfp["direction"].upper()
    emoji = "🟢" if sfp["direction"] == "long" else "🔴"
    bias = "ABOVE" if sfp["direction"] == "long" else "BELOW"
    wick_pct = sfp.get("wick_pct", 0) * 100
    entry = sfp["swing_level"]
    stop = sfp["wick_extreme"]

    return (
        f"{emoji} <b>SFP SETUP — {symbol} ({market_type.upper()})</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Timeframe:</b> {timeframe}\n"
        f"📍 <b>Direction:</b> {direction}\n"
        f"💰 <b>Current Price:</b> {current_price:.4f}\n"
        f"📏 <b>Swing Level:</b> {sfp['swing_level']:.4f}\n"
        f"🔽 <b>Wick Extreme:</b> {sfp['wick_extreme']:.4f} ({wick_pct:.2f}% breach)\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Entry Zone:</b> {entry:.4f}\n"
        f"🛑 <b>Stop:</b> {stop:.4f}\n"
        f"🏁 <b>Target:</b> {target:.4f}\n"
        f"📐 <b>R:R:</b> {rr:.1f}:1\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Daily 200 EMA:</b> {ema_daily:.4f}\n"
        f"📈 <b>4H 200 EMA:</b> {ema_4h:.4f}\n"
        f"✅ <b>Bias:</b> Price {bias} both EMAs → {direction} bias confirmed\n"
        f"🕒 <b>Scan time:</b> {bst_now()}"
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
    ema_200_daily = calculate_ema(closes, EMA_PERIOD)
    if ema_200_daily is None:
        return []

    current_price = daily_candles[-1]["c"]
    price_above_daily_ema = current_price > ema_200_daily

    # Fetch 4H candles for 4H 200 EMA confirmation (need 210 candles)
    h4_ema_candles = await fetch_candles(session, market_id, "4h", 210)
    if len(h4_ema_candles) < EMA_PERIOD + 5:
        log.debug(f"{symbol}: insufficient 4H candles for EMA ({len(h4_ema_candles)}), skipping")
        return []

    h4_closes = [c["c"] for c in h4_ema_candles]
    ema_200_4h = calculate_ema(h4_closes, EMA_PERIOD)
    if ema_200_4h is None:
        return []

    price_above_4h_ema = current_price > ema_200_4h

    # Both EMAs must agree on direction
    if price_above_daily_ema != price_above_4h_ema:
        log.debug(f"{symbol}: Daily and 4H EMA bias conflict, skipping")
        return []

    price_above_ema = price_above_daily_ema

    # ── Daily SFP scan ──────────────────────────────────────────────────
    daily_sfp = detect_sfp(daily_candles)
    if daily_sfp:
        sfp_is_long = daily_sfp["direction"] == "long"
        if sfp_is_long == price_above_ema:
            # Check price is still on the correct side of swing level
            swing = daily_sfp["swing_level"]
            if sfp_is_long and current_price < swing:
                log.debug(f"{symbol} Daily bullish SFP stale — price back below swing level")
            elif not sfp_is_long and current_price > swing:
                log.debug(f"{symbol} Daily bearish SFP stale — price back above swing level")
            else:
                dedup_key = f"{symbol}_Daily"
                candle_ts = daily_candles[-2]["t"]
                if alerted_sfps.get(dedup_key) != candle_ts:
                    # R:R check
                    entry = daily_sfp["swing_level"]
                    stop = daily_sfp["wick_extreme"]
                    risk = abs(entry - stop)
                    target = find_nearest_target(daily_candles, daily_sfp["direction"], entry, stop)
                    if target is None:
                        log.debug(f"{symbol} Daily SFP no valid 2:1 target found, skipping")
                    else:
                        rr = abs(target - entry) / risk
                        alerted_sfps[dedup_key] = candle_ts
                        msg = format_alert(symbol, market_type, "Daily", daily_sfp, ema_200_daily, ema_200_4h, current_price, target, rr)
                        alerts.append(msg)
                        log.info(f"ALERT: {symbol} Daily SFP {daily_sfp['direction'].upper()} R:R={rr:.1f}")
                else:
                    log.debug(f"{symbol} Daily SFP already alerted for this candle, skipping")

    # ── 4H SFP scan ─────────────────────────────────────────────────────
    h4_candles = await fetch_candles(session, market_id, "4h", 50)
    if len(h4_candles) >= SFP_LOOKBACK + 2:
        h4_sfp = detect_sfp(h4_candles)
        if h4_sfp:
            sfp_is_long = h4_sfp["direction"] == "long"
            if sfp_is_long == price_above_ema:
                # Check price is still on the correct side of swing level
                swing = h4_sfp["swing_level"]
                if sfp_is_long and current_price < swing:
                    log.debug(f"{symbol} 4H bullish SFP stale — price back below swing level")
                elif not sfp_is_long and current_price > swing:
                    log.debug(f"{symbol} 4H bearish SFP stale — price back above swing level")
                else:
                    dedup_key = f"{symbol}_4H"
                    candle_ts = h4_candles[-2]["t"]
                    if alerted_sfps.get(dedup_key) != candle_ts:
                        # R:R check
                        entry = h4_sfp["swing_level"]
                        stop = h4_sfp["wick_extreme"]
                        risk = abs(entry - stop)
                        target = find_nearest_target(h4_candles, h4_sfp["direction"], entry, stop)
                        if target is None:
                            log.debug(f"{symbol} 4H SFP no valid 2:1 target found, skipping")
                        else:
                            rr = abs(target - entry) / risk
                            alerted_sfps[dedup_key] = candle_ts
                            msg = format_alert(symbol, market_type, "4H", h4_sfp, ema_200_daily, ema_200_4h, current_price, target, rr)
                            alerts.append(msg)
                            log.info(f"ALERT: {symbol} 4H SFP {h4_sfp['direction'].upper()} R:R={rr:.1f}")
                    else:
                        log.debug(f"{symbol} 4H SFP already alerted for this candle, skipping")

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
        await asyncio.sleep(0.3)  # gentle rate limiting

    if all_alerts:
        # Send a header first
        header = (
            f"🔔 <b>Lighter Scan — {len(all_alerts)} Setup(s) Found</b>\n"
            f"Scanned {len(markets)} markets at {bst_now()}"
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
            f"Scanned {len(markets)} markets at {bst_now()}"
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
            f"Filters: Daily & 4H 200 EMA bias + SFP (4H & Daily)\n"
            f"Started at {bst_now()}"
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
