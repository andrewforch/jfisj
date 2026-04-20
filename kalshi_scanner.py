#!/usr/bin/env python3
"""
Kalshi Market Scanner
Scans all open Kalshi markets resolving within 7-8 days for mispriced contracts.
Identifies edge opportunities for a $48 bankroll.

Usage:
    python3 kalshi_scanner.py
    python3 kalshi_scanner.py --days 7 --bankroll 48 --min-edge 0.04
"""

import requests
import time
import math
import argparse
from datetime import datetime, timezone, timedelta

BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
HEADERS = {"Accept": "application/json"}


def fetch_markets(min_close_ts: int, max_close_ts: int) -> list[dict]:
    """Paginate through all open markets in the given close-time window."""
    markets = []
    cursor = None
    page = 0

    while True:
        params = {
            "status": "open",
            "min_close_ts": min_close_ts,
            "max_close_ts": max_close_ts,
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(
                f"{BASE_URL}/markets", headers=HEADERS, params=params, timeout=15
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [!] API error on page {page}: {e}")
            break

        data = resp.json()
        batch = data.get("markets", [])
        markets.extend(batch)
        page += 1

        cursor = data.get("cursor")
        if not cursor or not batch:
            break

        time.sleep(0.15)  # stay well under rate limits

    return markets


def parse_price(val) -> float | None:
    """Convert Kalshi price field to float (handles both string '0.6500' and int 65)."""
    if val is None:
        return None
    try:
        f = float(val)
        # Old API returned cents (0-100); new API returns dollars (0.0000-1.0000)
        return f if f <= 1.0 else f / 100.0
    except (TypeError, ValueError):
        return None


def score_market(m: dict) -> dict | None:
    """
    Return an opportunity dict if the market looks mispriced, else None.

    Signals checked:
      1. Sum arb: yes_ask + no_ask < 0.98  (risk-free if you can buy both sides)
      2. Wide-spread edge: yes_bid significantly below yes_ask with liquid market
      3. Last-price deviation: yes_last + no_last deviates from 1.00 by >3 cents
    """
    yes_bid = parse_price(m.get("yes_bid"))
    yes_ask = parse_price(m.get("yes_ask"))
    no_bid  = parse_price(m.get("no_bid"))
    no_ask  = parse_price(m.get("no_ask"))
    last    = parse_price(m.get("last_price"))

    # Need at minimum yes_ask and no_ask to evaluate
    if yes_ask is None or no_ask is None:
        return None

    yes_ask = min(yes_ask, 1.0)
    no_ask  = min(no_ask, 1.0)

    opportunities = []
    edge_score = 0.0

    # --- Signal 1: Sum arbitrage ---
    arb_sum = yes_ask + no_ask
    if arb_sum < 0.98:
        gap = 1.0 - arb_sum
        edge_score += gap * 3          # weight heavily — risk-free
        opportunities.append(f"SUM-ARB gap={gap:.3f} (yes_ask={yes_ask:.3f} no_ask={no_ask:.3f})")

    # --- Signal 2: Last-price deviation from 1.00 ---
    if last is not None:
        no_implied = 1.0 - last
        if no_bid is not None:
            deviation = abs(no_bid - no_implied)
            if deviation > 0.04:
                edge_score += deviation
                opportunities.append(
                    f"LAST-DEV deviation={deviation:.3f} (last={last:.3f} no_bid={no_bid:.3f})"
                )

    # --- Signal 3: Wide spread on YES side (liquidity mismatch) ---
    if yes_bid is not None and yes_ask is not None:
        spread = yes_ask - yes_bid
        volume = m.get("volume", 0) or 0
        if spread > 0.08 and volume > 500:
            edge_score += spread * 0.5
            opportunities.append(
                f"WIDE-SPREAD spread={spread:.3f} vol={volume}"
            )

    # --- Signal 4: YES deeply discounted vs fair value hint ---
    # If yes_ask < 0.10 and market seems liquid, possible mispricing
    open_interest = m.get("open_interest", 0) or 0
    if yes_ask < 0.10 and open_interest > 200:
        edge_score += 0.03
        opportunities.append(f"LOW-YES yes_ask={yes_ask:.3f} OI={open_interest}")

    if not opportunities or edge_score < 0.03:
        return None

    return {
        "ticker": m.get("ticker", ""),
        "title": m.get("title") or m.get("subtitle") or m.get("ticker", ""),
        "close_time": m.get("close_time", ""),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "last_price": last,
        "volume": m.get("volume", 0),
        "open_interest": open_interest,
        "edge_score": edge_score,
        "signals": opportunities,
    }


def calc_best_bet(opp: dict, bankroll: float) -> dict:
    """
    Given an opportunity, calculate the single best position and expected profit.
    Uses Kelly criterion (half-Kelly) for sizing within the bankroll.
    """
    ya, na = opp["yes_ask"], opp["no_ask"]

    # For sum-arb: split bankroll between YES and NO to guarantee profit
    if ya + na < 0.98:
        arb_gap = 1.0 - (ya + na)
        # Optimal split: buy YES fraction f, NO fraction (1-f)
        # Both pay $1 at resolution; we spend ya*f + na*(1-f) per dollar of cover
        # Lock-in profit = 1 - (ya*f + na*(1-f)) for any f; maximized by minimizing cost
        # Minimum cost is min(ya,na)*1 + the other side, but we need f such that
        # yes_payout = no_payout = 1 — just spend $bankroll total, split optimally
        yes_contracts = bankroll / (ya + na) * ya / (ya + na)  # proportional to yes_ask
        no_contracts  = bankroll / (ya + na) * na / (ya + na)
        # Simpler: buy X yes + Y no where X*ya + Y*na = bankroll, X=Y gives min cost
        contracts = bankroll / (ya + na)
        profit = contracts * arb_gap
        return {
            "strategy": "SUM-ARB",
            "position": f"Buy YES @ {ya:.3f} AND NO @ {na:.3f}",
            "bet_size": bankroll,
            "contracts": round(contracts, 1),
            "expected_profit": round(profit, 2),
            "roi_pct": round(profit / bankroll * 100, 1),
            "risk": "None (locked profit)",
        }

    # For directional edge: use half-Kelly on the better-priced side
    # Assume last_price is the "true" probability estimate
    p_yes = opp.get("last_price")
    if p_yes is None:
        p_yes = (opp.get("yes_bid") or ya + (opp.get("no_bid") or na) / 2)

    p_yes = max(0.01, min(0.99, p_yes or 0.5))
    p_no  = 1.0 - p_yes

    # Check which side has better edge
    yes_ev = p_yes / ya - 1 if ya > 0 else -1
    no_ev  = p_no  / na - 1 if na > 0 else -1

    if yes_ev >= no_ev and yes_ev > 0:
        side, price, prob = "YES", ya, p_yes
        ev = yes_ev
    elif no_ev > 0:
        side, price, prob = "NO", na, p_no
        ev = no_ev
    else:
        side, price, prob, ev = "YES", ya, p_yes, yes_ev

    # Half-Kelly fraction: f* = (edge / odds) / 2
    odds = (1 - price) / price  # net odds per $1 bet (payout - 1)
    kelly_f = max(0, (prob * (odds + 1) - 1) / odds / 2)
    bet = round(min(bankroll * kelly_f, bankroll), 2)
    contracts = bet / price if price > 0 else 0
    expected_profit = round(bet * ev, 2)

    return {
        "strategy": "DIRECTIONAL",
        "position": f"Buy {side} @ {price:.3f}",
        "bet_size": round(bet, 2),
        "contracts": round(contracts, 1),
        "expected_profit": expected_profit,
        "roi_pct": round(expected_profit / bankroll * 100, 1) if bankroll else 0,
        "risk": f"Lose ${bet:.2f} if wrong",
    }


def print_report(opportunities: list[dict], bankroll: float):
    sorted_opps = sorted(opportunities, key=lambda x: x["edge_score"], reverse=True)

    print("\n" + "=" * 70)
    print(f"  KALSHI MISPRICING SCANNER  |  Bankroll: ${bankroll:.2f}")
    print(f"  Scanned: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    if not sorted_opps:
        print("\n  No mispriced markets found in this window.")
        print("  Try widening --days or lowering --min-edge.\n")
        return

    print(f"\n  Found {len(sorted_opps)} opportunity/ies:\n")

    for i, opp in enumerate(sorted_opps[:20], 1):  # cap display at 20
        bet = calc_best_bet(opp, bankroll)
        close_dt = opp["close_time"]

        print(f"  {'─'*66}")
        print(f"  #{i:02d}  {opp['ticker']}")
        print(f"       {opp['title'][:65]}")
        print(f"       Closes: {close_dt}  |  Edge Score: {opp['edge_score']:.4f}")
        print(f"       Prices: YES ask={opp['yes_ask']:.3f}  NO ask={opp['no_ask']:.3f}"
              + (f"  Last={opp['last_price']:.3f}" if opp['last_price'] else ""))
        print(f"       Volume: {opp['volume']}  |  OI: {opp['open_interest']}")
        print(f"       Signals: {', '.join(opp['signals'])}")
        print(f"       >>> Strategy: {bet['strategy']}")
        print(f"           Position:  {bet['position']}")
        print(f"           Bet:       ${bet['bet_size']:.2f}  ({bet['contracts']:.1f} contracts)")
        print(f"           Exp Profit:${bet['expected_profit']:.2f}  (ROI {bet['roi_pct']:.1f}%)")
        print(f"           Risk:      {bet['risk']}")
        print()

    print("=" * 70)
    print("  DISCLAIMER: This is for informational purposes only.")
    print("  Prediction markets carry real financial risk.")
    print("  Verify all prices on Kalshi before placing any order.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Kalshi mispricing scanner")
    parser.add_argument("--days", type=float, default=8.0, help="Max days until resolution (default 8)")
    parser.add_argument("--min-days", type=float, default=0.0, help="Min days until resolution (default 0)")
    parser.add_argument("--bankroll", type=float, default=48.0, help="Available capital in USD (default 48)")
    parser.add_argument("--min-edge", type=float, default=0.03, help="Minimum edge score to report (default 0.03)")
    args = parser.parse_args()

    now = int(time.time())
    min_ts = now + int(args.min_days * 86400)
    max_ts = now + int(args.days * 86400)

    min_dt = datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    max_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"\nKalshi Scanner starting...")
    print(f"Window: {min_dt} → {max_dt}")
    print(f"Bankroll: ${args.bankroll:.2f}  |  Min edge: {args.min_edge}")
    print("Fetching markets...", end="", flush=True)

    markets = fetch_markets(min_ts, max_ts)
    print(f" {len(markets)} markets fetched.")

    if not markets:
        print("No markets returned. Check your network connection and try again.")
        return

    print("Scanning for mispricing...", end="", flush=True)
    opportunities = []
    for m in markets:
        opp = score_market(m)
        if opp and opp["edge_score"] >= args.min_edge:
            opportunities.append(opp)
    print(f" done.")

    print_report(opportunities, args.bankroll)


if __name__ == "__main__":
    main()
