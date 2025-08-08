#!/usr/bin/env python3
"""
options_bot.py — scan & (optionally) trade single-leg option income plays via Tradier.

What it does now:
- Loads TRADIER_* config from .env (python-dotenv)
- Auto-discovers your account (if not provided), pulls balances & positions
- Computes "deployable cash" from balances MINUS reserved collateral for open short puts
- Scans Short Puts (CSP) and Covered Calls (if you own >= 100 shares)
- Applies risk profiles that adjust filters & scoring
- Ranks & prints top 3; previews and (optionally) places the top trade

Usage (examples):
  python options_bot.py --tickers AAPL,MSFT,AMD --risk moderate
  python options_bot.py --watchlist "Income Candidates" --risk conservative
  python options_bot.py --tickers SOFI,F --risk extreme --live   # actually place after preview

Install:
  python -m venv .venv && . .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
  pip install -r requirements.txt
  cp .env.example .env   # fill in values
"""

import os
import sys
import time
import math
import json
import argparse
import logging
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtp
from typing import List, Dict, Any, Optional

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ----------------------------- Defaults (overridden by risk profile) -----------------------------
CFG = {
    "CASH_BUFFER": 0.90,       # fraction of deployable cash we’ll use
    "MIN_DTE": 14,
    "MAX_DTE": 60,
    "TARGET_DTE": 30,
    "MIN_OI": 50,
    "MIN_BID": 0.05,
    "MAX_SPREAD_BPS": 100,     # 100 bps = 1% of mid
    "VOL_PENALTY_SCALE": 1.0,
    "DELTA_TARGET_PUT": -0.25,
    "DELTA_TARGET_CCALL": 0.25,
    "EARNINGS_PENALTY": 0.70,  # <1 discounts score if earnings before expiration
    "EXCLUDE_EARNINGS": False, # if True, exclude any contract expiring after next earnings
}

DEFAULT_UNIVERSE = ["AAPL","MSFT","AMD","NVDA","KO","INTC","F","SOFI","PLTR","T","PFE"]
PREVIEW_BEFORE_PLACE = True
MAX_CHAIN_SLEEP = 0.15

# ----------------------------- Risk Profiles -----------------------------
RISK_PROFILES = {
    "extreme": {
        "CASH_BUFFER": 0.99,
        "MIN_DTE": 7,
        "MAX_DTE": 45,
        "TARGET_DTE": 21,
        "MIN_OI": 10,
        "MIN_BID": 0.03,
        "MAX_SPREAD_BPS": 300,
        "VOL_PENALTY_SCALE": 0.6,
        "DELTA_TARGET_PUT": -0.40,
        "DELTA_TARGET_CCALL": 0.40,
        "EARNINGS_PENALTY": 1.00,   # no penalty
        "EXCLUDE_EARNINGS": False,
    },
    "high": {
        "CASH_BUFFER": 0.95,
        "MIN_DTE": 10,
        "MAX_DTE": 50,
        "TARGET_DTE": 25,
        "MIN_OI": 25,
        "MIN_BID": 0.05,
        "MAX_SPREAD_BPS": 200,
        "VOL_PENALTY_SCALE": 0.8,
        "DELTA_TARGET_PUT": -0.30,
        "DELTA_TARGET_CCALL": 0.30,
        "EARNINGS_PENALTY": 0.85,
        "EXCLUDE_EARNINGS": False,
    },
    "moderate": {
        "CASH_BUFFER": 0.90,
        "MIN_DTE": 14,
        "MAX_DTE": 60,
        "TARGET_DTE": 30,
        "MIN_OI": 50,
        "MIN_BID": 0.05,
        "MAX_SPREAD_BPS": 100,
        "VOL_PENALTY_SCALE": 1.0,
        "DELTA_TARGET_PUT": -0.25,
        "DELTA_TARGET_CCALL": 0.25,
        "EARNINGS_PENALTY": 0.70,
        "EXCLUDE_EARNINGS": False,
    },
    "low": {
        "CASH_BUFFER": 0.88,
        "MIN_DTE": 21,
        "MAX_DTE": 75,
        "TARGET_DTE": 35,
        "MIN_OI": 100,
        "MIN_BID": 0.10,
        "MAX_SPREAD_BPS": 80,
        "VOL_PENALTY_SCALE": 1.2,
        "DELTA_TARGET_PUT": -0.20,
        "DELTA_TARGET_CCALL": 0.20,
        "EARNINGS_PENALTY": 0.60,
        "EXCLUDE_EARNINGS": False,
    },
    "conservative": {
        "CASH_BUFFER": 0.85,
        "MIN_DTE": 25,
        "MAX_DTE": 90,
        "TARGET_DTE": 45,
        "MIN_OI": 150,
        "MIN_BID": 0.10,
        "MAX_SPREAD_BPS": 60,
        "VOL_PENALTY_SCALE": 1.4,
        "DELTA_TARGET_PUT": -0.15,
        "DELTA_TARGET_CCALL": 0.15,
        "EARNINGS_PENALTY": 0.50,
        "EXCLUDE_EARNINGS": True,   # drop anything that crosses earnings
    },
}

# ----------------------------- Utils -----------------------------
def td_now():
    return datetime.now(timezone.utc)

def ensure_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def mid_price(bid, ask, last):
    try:
        if bid and ask and bid > 0 and ask > 0:
            return round((bid + ask) / 2.0, 2)
        if last and last > 0:
            return round(float(last), 2)
        if ask and ask > 0:
            return round(float(ask), 2)
        if bid and bid > 0:
            return round(float(bid), 2)
    except Exception:
        pass
    return None

def bps_spread(bid: float, ask: float) -> float:
    m = mid_price(bid, ask, None)
    if not m or m == 0:
        return float('inf')
    return abs(ask - bid) / m * 10000.0

def dte_days(expiration_date: str) -> int:
    exp = dtp.parse(expiration_date).date()
    return (exp - td_now().date()).days

def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def parse_occ_option(option_symbol: str) -> dict:
    """
    Parse OCC option symbol like 'AAPL250920P00190000' -> {'underlying','yymmdd','type','strike'}
    """
    # Len/format varies with underlying length; find last 15 chars pattern: [yymmdd][C|P][strike(8)]
    if not option_symbol or len(option_symbol) < 15:
        return {}
    tail = option_symbol[-15:]
    yymmdd = tail[:6]
    typ = tail[6]
    strike_raw = tail[7:]  # 8 digits; strike = int / 1000
    try:
        strike = int(strike_raw) / 1000.0
        # underlying is leading part
        underlying = option_symbol[:-15]
        return {
            "underlying": underlying,
            "date": yymmdd,
            "type": "put" if typ.upper() == "P" else "call",
            "strike": strike
        }
    except Exception:
        return {}
    
def deep_get(d, path, default=None):
    """deep_get({'a':{'b':1}}, 'a.b') -> 1"""
    cur = d
    for key in path.split('.'):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur

def print_balance_snapshot(balances: dict):
    b = balances.get("balances") or balances or {}
    snapshot = {
        "total_cash": b.get("total_cash"),
        "cash.cash_available": deep_get(b, "cash.cash_available"),
        "cash.unsettled_funds": deep_get(b, "cash.unsettled_funds"),
        "margin.option_buying_power": deep_get(b, "margin.option_buying_power"),
        "pdt.option_buying_power": deep_get(b, "pdt.option_buying_power"),
        "total_equity": b.get("total_equity"),
    }
    print("Balances snapshot:", ", ".join(f"{k}={snapshot[k]}" for k in snapshot))

def floor_to_tick(p: float, tick: float) -> float:
    return round(math.floor(p / tick) * tick, 2)

def guess_tick(bid: Optional[float], ask: Optional[float], last: Optional[float], mid: Optional[float]) -> float:
    """
    Heuristic:
    - If all visible quotes land on 0.05 ticks, assume 0.05.
    - Else assume 0.01.
    - As a safety fallback: if mid>=3 -> 0.05 else 0.01.
    """
    quotes = [x for x in [bid, ask, last, mid] if isinstance(x, (int, float)) and x > 0]
    if quotes:
        cents = [int(round(q * 100)) for q in quotes]
        if all(c % 5 == 0 for c in cents):
            return 0.05
        # if any quote already has penny precision not on a nickel, assume 0.01
        if any(c % 5 != 0 for c in cents):
            return 0.01
    # fallback by price level (conservative)
    m = mid or last or bid or ask or 1.0
    return 0.05 if m >= 3 else 0.01

# ----------------------------- Tradier Client -----------------------------
class Tradier:
    def __init__(self, token: str, account_id: Optional[str] = None, env: str = "live"):
        self.token = token
        self.base = "https://api.tradier.com"
        if str(env).lower() == "sandbox":
            self.base = "https://sandbox.tradier.com"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "User-Agent": "options-bot/1.1"
        })
        self.account_id = account_id

    def _get(self, path: str, params: dict = None):
        url = f"{self.base}{path}"
        r = self.session.get(url, params=params, timeout=20)
        if r.status_code == 401:
            raise RuntimeError("401 Unauthorized from Tradier (GET). Check token / env / account permissions.")
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict = None):
        url = f"{self.base}{path}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        r = self.session.post(url, data=data or {}, headers=headers, timeout=20)
        # dump error body so we can see why it failed (tick size, permissions, etc.)
        if r.status_code >= 400:
            try:
                print("\n--- Tradier order error body ---")
                print(r.text)
                print("--- end ---\n")
            except Exception:
                pass
            r.raise_for_status()
        return r.json()


    # --- account / balances / positions ---
    def get_profile(self) -> dict:
        return self._get("/v1/user/profile")

    def get_account_id(self) -> str:
        if self.account_id:
            return self.account_id
        prof = self.get_profile()
        accts = ensure_list(safe_get(prof, "profile", "accounts", "account", default=[]))
        if not accts:
            raise RuntimeError("No accounts found in /v1/user/profile.")
        self.account_id = accts[0].get("account")
        return self.account_id

    def get_balances(self, account_id: Optional[str] = None) -> dict:
        aid = account_id or self.get_account_id()
        return self._get(f"/v1/accounts/{aid}/balances")

    def get_positions(self, account_id: Optional[str] = None) -> List[dict]:
        aid = account_id or self.get_account_id()
        data = self._get(f"/v1/accounts/{aid}/positions")
        return ensure_list(safe_get(data, "positions", "position", default=[]))

    # --- watchlists ---
    def get_watchlists(self) -> List[dict]:
        data = self._get("/v1/watchlists")
        return ensure_list(safe_get(data, "watchlists", "watchlist", default=[]))

    def get_watchlist_symbols(self, name: str) -> List[str]:
        for wl in self.get_watchlists():
            if wl.get("name") == name:
                items = ensure_list(safe_get(wl, "items", "item", default=[]))
                return [i.get("symbol") for i in items if i.get("symbol")]
        return []

    # --- market data ---
    def get_quotes(self, symbols: List[str]) -> dict:
        return self._get("/v1/markets/quotes", params={"symbols": ",".join(symbols)})

    def get_history(self, symbol: str, days: int = 40) -> pd.DataFrame:
        end = td_now().date()
        start = end - timedelta(days=days * 2)  # buffer for wknds/holidays
        data = self._get("/v1/markets/history", params={
            "symbol": symbol, "interval": "daily",
            "start": start.isoformat(), "end": end.isoformat(), "session_filter": "all"
        })
        rows = ensure_list(safe_get(data, "history", "day", default=[]))
        return pd.DataFrame(rows)

    def get_expirations(self, symbol: str) -> List[str]:
        data = self._get("/v1/markets/options/expirations", params={
            "symbol": symbol, "includeAllRoots": "true"
        })
        return ensure_list(safe_get(data, "expirations", "date", default=[]))

    def get_chain(self, symbol: str, expiration: str, want_greeks: bool = True) -> List[dict]:
        data = self._get("/v1/markets/options/chains", params={
            "symbol": symbol, "expiration": expiration, "greeks": str(want_greeks).lower()
        })
        return ensure_list(safe_get(data, "options", "option", default=[]))

    def get_corporate_calendar(self, symbols: List[str]) -> Dict[str, Optional[datetime]]:
        joined = ",".join(symbols)
        out = {s: None for s in symbols}
        try:
            data = self._get("/beta/markets/fundamentals/calendars", params={"symbols": joined})
            for block in ensure_list(data):
                sym = block.get("request")
                calendars = safe_get(block, "results", default=[])
                next_earn = None
                for res in ensure_list(calendars):
                    tables = safe_get(res, "tables", default={})
                    for ev in ensure_list(tables.get("corporate_calendars", [])):
                        name = (ev.get("event") or "").lower()
                        if "earnings" in name:
                            when = ev.get("begin_date_time")
                            try:
                                dt = dtp.parse(when).date()
                                if dt >= td_now().date():
                                    if not next_earn or dt < next_earn:
                                        next_earn = dt
                            except Exception:
                                pass
                out[sym] = next_earn
        except Exception as e:
            logging.warning("Calendar fetch skipped (%s). Continuing without earnings.", str(e))
            # leave out as None for all
        return out


    # --- trading ---
    def preview_or_place_option_order(self, option_symbol: str, underlying: str, side: str,
                                      quantity: int, limit_price: float, duration: str = "day",
                                      live: bool = False, tag: str = "options-bot") -> dict:
        aid = self.get_account_id()
        payload = {
            "class": "option",
            "symbol": underlying,
            "option_symbol": option_symbol,
            "side": side,  # sell_to_open / buy_to_close etc.
            "quantity": str(quantity),
            "type": "limit",
            "duration": duration,
            "price": f"{limit_price:.2f}",
            "tag": tag
        }
        if PREVIEW_BEFORE_PLACE:
            payload["preview"] = "true"
            preview = self._post(f"/v1/accounts/{aid}/orders", data=payload)
            if not live:
                return {"preview": preview, "placed": False}
            payload.pop("preview", None)
        placed = self._post(f"/v1/accounts/{aid}/orders", data=payload)
        return {"preview": None, "placed": placed}

# ----------------------------- Analytics -----------------------------
def realized_vol_20d(hist_df: pd.DataFrame) -> Optional[float]:
    try:
        df = hist_df.copy().sort_values("date")
        rets = np.log(df["close"].astype(float)).diff().dropna()
        if len(rets) < 10:
            return None
        daily_std = np.std(rets, ddof=1)
        return float(daily_std * np.sqrt(252))
    except Exception:
        return None

def score_short_put(opt: dict, rv: Optional[float], earnings_by_exp: bool, cfg: dict) -> Optional[dict]:
    greeks = opt.get("greeks") or {}
    delta = greeks.get("delta")
    bid = float(opt.get("bid") or 0)
    ask = float(opt.get("ask") or 0)
    last = opt.get("last")
    m = mid_price(bid, ask, last)
    if not m or bid < cfg["MIN_BID"]:
        return None
    if bps_spread(bid, ask) > max(cfg["MAX_SPREAD_BPS"], 500):
        return None
    strike = float(opt["strike"])
    dte = dte_days(opt["expiration_date"])
    if dte < cfg["MIN_DTE"] or dte > cfg["MAX_DTE"]:
        return None
    oi = int(opt.get("open_interest") or 0)
    if oi < cfg["MIN_OI"]:
        return None
    if cfg["EXCLUDE_EARNINGS"] and earnings_by_exp:
        return None

    collateral = strike * 100.0
    credit = m * 100.0
    ann_yield = (credit / collateral) * (365.0 / max(dte, 1))

    delta_adj = 1.0 - min(abs((abs(delta or 0) - abs(cfg["DELTA_TARGET_PUT"]))) * 2.0, 0.6)
    vol = rv if (rv is not None) else 0.3
    vol_mult = 1.0 / (1.0 + cfg["VOL_PENALTY_SCALE"] * max(vol, 0.05))
    t_adj = 1.0 - min(abs(dte - cfg["TARGET_DTE"]) / 90.0, 0.3)
    earn_mult = (cfg["EARNINGS_PENALTY"] if earnings_by_exp else 1.0)

    score = ann_yield * delta_adj * vol_mult * t_adj * earn_mult

    return {
        "strategy": "SHORT_PUT",
        "underlying": opt["underlying"],
        "option_symbol": opt["symbol"],
        "expiration": opt["expiration_date"],
        "strike": strike,
        "dte": dte,
        "mid": m,
        "bid": bid,
        "ask": ask,
        "oi": oi,
        "delta": delta,
        "credit": round(credit, 2),
        "collateral": collateral,
        "annualized_yield": ann_yield,
        "score": score,
    }

def score_covered_call(opt: dict, rv: Optional[float], have_shares: int, px_now: float, earnings_by_exp: bool, cfg: dict) -> Optional[dict]:
    if have_shares < 100:
        return None
    strike = float(opt["strike"])
    if strike <= px_now:  # OTM only
        return None
    greeks = opt.get("greeks") or {}
    delta = greeks.get("delta")
    bid = float(opt.get("bid") or 0)
    ask = float(opt.get("ask") or 0)
    last = opt.get("last")
    m = mid_price(bid, ask, last)
    if not m or bid < cfg["MIN_BID"]:
        return None
    if bps_spread(bid, ask) > max(cfg["MAX_SPREAD_BPS"], 500):
        return None
    dte = dte_days(opt["expiration_date"])
    if dte < cfg["MIN_DTE"] or dte > cfg["MAX_DTE"]:
        return None
    oi = int(opt.get("open_interest") or 0)
    if oi < cfg["MIN_OI"]:
        return None
    if cfg["EXCLUDE_EARNINGS"] and earnings_by_exp:
        return None

    credit = m * 100.0
    base = px_now * 100.0
    ann_yield = (credit / base) * (365.0 / max(dte, 1))

    delta_adj = 1.0 - min(abs((abs(delta or 0) - abs(cfg["DELTA_TARGET_CCALL"]))) * 2.0, 0.6)
    vol = rv if (rv is not None) else 0.3
    vol_mult = 1.0 / (1.0 + cfg["VOL_PENALTY_SCALE"] * max(vol, 0.05))
    t_adj = 1.0 - min(abs(dte - cfg["TARGET_DTE"]) / 90.0, 0.3)
    earn_mult = (cfg["EARNINGS_PENALTY"] if earnings_by_exp else 1.0)

    score = ann_yield * delta_adj * vol_mult * t_adj * earn_mult

    return {
        "strategy": "COVERED_CALL",
        "underlying": opt["underlying"],
        "option_symbol": opt["symbol"],
        "expiration": opt["expiration_date"],
        "strike": strike,
        "dte": dte,
        "mid": m,
        "bid": bid,
        "ask": ask,
        "oi": oi,
        "delta": delta,
        "credit": round(credit, 2),
        "collateral": base,  # implicit via owning the shares
        "annualized_yield": ann_yield,
        "score": score,
    }

# ----------------------------- Cash / Collateral Logic -----------------------------
def compute_deployable_cash(balances: dict, positions: List[dict]) -> float:
    """
    Prefer true cash (cash accounts). Fall back to total_cash/sweep or
    buying power (margin/PDT). Exclude unsettled & pending cash, then
    subtract collateral reserved for open SHORT PUTS.
    """
    b = balances.get("balances") or balances or {}

    # Try a bunch of plausible fields in priority order
    candidates = [
        deep_get(b, "cash.cash_available"),
        b.get("total_cash"),
        deep_get(b, "cash.sweep"),
        deep_get(b, "pdt.option_buying_power"),
        deep_get(b, "margin.option_buying_power"),
        b.get("cash"),                     # legacy/fallback
        b.get("option_buying_power"),      # rare/fallback
    ]
    base_cash = next((float(v) for v in candidates if v not in (None, "", "NaN")), 0.0)

    unsettled = float(deep_get(b, "cash.unsettled_funds") or 0.0)
    pending   = float(b.get("pending_cash") or 0.0)
    usable_cash = max(base_cash - unsettled - pending, 0.0)

    # Reserve collateral for any open short puts
    reserved = 0.0
    for p in ensure_list(positions):
        instr = p.get("instrument", {})
        if instr.get("type") == "option" and int(p.get("quantity") or 0) < 0:
            occ = instr.get("option_symbol") or instr.get("symbol")
            meta = parse_occ_option(occ or "")
            if meta.get("type") == "put":
                reserved += float(meta.get("strike") or 0.0) * 100.0 * abs(int(p.get("quantity") or 0))

    return max(usable_cash - reserved, 0.0)


# ----------------------------- Main -----------------------------
def build_universe(args, t: Tradier) -> List[str]:
    syms: List[str] = []
    if args.watchlist:
        syms.extend(t.get_watchlist_symbols(args.watchlist))
    if args.tickers:
        syms.extend([s.strip().upper() for s in args.tickers.split(",") if s.strip()])
    if not syms:
        syms = DEFAULT_UNIVERSE
    return sorted(list(set(syms)))

def apply_risk_profile(name: str) -> dict:
    prof = RISK_PROFILES.get(name.lower(), RISK_PROFILES["moderate"])
    cfg = CFG.copy()
    cfg.update(prof)
    return cfg

def main():
    load_dotenv()  # .env
    ap = argparse.ArgumentParser(description="Scan & trade single-leg options via Tradier")
    ap.add_argument("--tickers", help="Comma-separated symbols to scan")
    ap.add_argument("--watchlist", help="Tradier Watchlist name to pull symbols from")
    ap.add_argument("--risk", choices=list(RISK_PROFILES.keys()), default="moderate",
                    help="Risk profile to apply")
    ap.add_argument("--sandbox", action="store_true",
                    help="Force sandbox (overrides TRADIER_ENV)")
    ap.add_argument("--live", action="store_true",
                    help="Place the top trade after preview")
    ap.add_argument("--min-oi", type=int, help="Override MIN_OI")
    ap.add_argument("--min-bid", type=float, help="Override MIN_BID")
    ap.add_argument("--max-spread-bps", type=float, help="Override MAX_SPREAD_BPS")
    args = ap.parse_args()

    token = os.environ.get("TRADIER_ACCESS_TOKEN")
    if not token:
        print("ERROR: TRADIER_ACCESS_TOKEN missing (set it in .env).", file=sys.stderr)
        sys.exit(2)

    env = (os.environ.get("TRADIER_ENV") or "live").lower()
    if args.sandbox:
        env = "sandbox"

    t = Tradier(token=token,
                account_id=os.environ.get("TRADIER_ACCOUNT_ID"),
                env=env)

    # Resolve account and pull balances/positions
    try:
        acct_id = t.get_account_id()
        balances = t.get_balances(acct_id)
        positions = t.get_positions(acct_id)
    except Exception as e:
        print(f"ERROR pulling account/balances/positions: {e}", file=sys.stderr)
        sys.exit(2)

    # Compute deployable cash
    deployable_cash = compute_deployable_cash(balances, positions)

    # Apply risk profile
    cfg = apply_risk_profile(args.risk)
    if args.min_oi is not None: cfg["MIN_OI"] = args.min_oi
    if args.min_bid is not None: cfg["MIN_BID"] = args.min_bid
    if args.max_spread_bps is not None: cfg["MAX_SPREAD_BPS"] = args.max_spread_bps

    print(f"Using account: {acct_id}  env={env}")
    print_balance_snapshot(balances)
    print(f"Open positions: {len(positions)} (stocks & options)")

    deployable_cash = compute_deployable_cash(balances, positions)
    print(f"Deployable cash BEFORE buffer: ${deployable_cash:,.2f}")

    cash_to_use = deployable_cash * cfg["CASH_BUFFER"]
    print(f"Cash to use (after buffer {cfg['CASH_BUFFER']:.0%}): ${cash_to_use:,.2f}")

    # Positions for covered calls
    shares_by_symbol = {}
    for p in ensure_list(positions):
        if p.get("instrument", {}).get("type") == "equity":
            sym = p["instrument"]["symbol"]
            shares_by_symbol[sym] = shares_by_symbol.get(sym, 0) + int(p.get("quantity") or 0)

    # Universe
    symbols = build_universe(args, t)
    if not symbols:
        print("No symbols to scan.")
        sys.exit(0)

    # Realized vol & earnings
    rv_by_symbol: Dict[str, Optional[float]] = {}
    for s in symbols:
        try:
            hist = t.get_history(s, days=40)
            rv_by_symbol[s] = realized_vol_20d(hist)
        except Exception:
            rv_by_symbol[s] = None

    earnings_dates = t.get_corporate_calendar(symbols)

    # Quotes
    quotes_raw = t.get_quotes(symbols)
    q_map = {}
    for q in ensure_list(safe_get(quotes_raw, "quotes", "quote", default=[])):
        q_map[q["symbol"]] = q

    # Scan
    candidates: List[dict] = []
    for sym in symbols:
        try:
            expirations = t.get_expirations(sym)
        except Exception as e:
            logging.warning("Expirations fetch failed for %s: %s", sym, e)
            continue

        expirations = [e for e in expirations if cfg["MIN_DTE"] <= dte_days(e) <= cfg["MAX_DTE"]]
        expirations = sorted(expirations, key=lambda e: dte_days(e))
        if not expirations:
            continue

        px_now = float(safe_get(q_map.get(sym, {}), "last", default=safe_get(q_map.get(sym, {}), "close", default=0)) or 0)
        next_earn = earnings_dates.get(sym)

        for exp in expirations:
            try:
                chain = t.get_chain(sym, exp, want_greeks=True)
            except Exception as e:
                logging.warning("Chain fetch failed for %s %s: %s", sym, exp, e)
                continue

            earnings_by_exp = bool(next_earn and (dtp.parse(exp).date() >= next_earn))

            # SHORT PUTS (respect cash_to_use)
            for opt in chain:
                if opt.get("option_type") != "put":
                    continue
                strike = float(opt["strike"])
                collateral = strike * 100.0
                if collateral > cash_to_use:
                    continue
                scored = score_short_put(opt, rv_by_symbol.get(sym), earnings_by_exp, cfg)
                if scored:
                    candidates.append(scored)

            # COVERED CALLS (if shares)
            have = shares_by_symbol.get(sym, 0)
            if have >= 100:
                for opt in chain:
                    if opt.get("option_type") != "call":
                        continue
                    scored = score_covered_call(opt, rv_by_symbol.get(sym), have, px_now, earnings_by_exp, cfg)
                    if scored:
                        candidates.append(scored)

            time.sleep(MAX_CHAIN_SLEEP)

    if not candidates:
        print("No viable option candidates found with current filters.")
        sys.exit(0)

    # Rank & report
    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    top3 = ranked[:3]

    print("\nTop 3 candidates:")
    for i, c in enumerate(top3, 1):
        delta_val = c.get("delta")
        delta_txt = f"{float(delta_val):.2f}" if isinstance(delta_val, (int, float)) else "NA"
        print(
            f"[{i}] {c['strategy']} {c['underlying']} {c['option_symbol']} "
            f"exp {c['expiration']} dte {c['dte']} strike {c['strike']} "
            f"mid {c['mid']} credit ${c['credit']:.2f} ann_yield {c['annualized_yield']:.2%} "
            f"delta {delta_txt} OI {c['oi']} score {c['score']:.4f}"
        )


    best = top3[0]
    print("\nTop pick:", json.dumps({
        k: best[k] for k in ("strategy","underlying","option_symbol","expiration","strike","dte","mid","credit","annualized_yield","score")
    }, indent=2))

    # Order
    side = "sell_to_open"  # both CSPs and CCs are STO
    qty = 1
    tick = guess_tick(best.get("bid"), best.get("ask"), None, best.get("mid"))
    limit_px = floor_to_tick(best["mid"], tick)
    print(f"Chosen limit price: {best['mid']} -> {limit_px} (tick {tick})")
    underlying = best["underlying"]
    option_symbol = best["option_symbol"]

    # Safety: earnings exclusion already handled by profile; still warn if crossing earnings
    warn = ""
    ne = earnings_dates.get(underlying)
    if ne and dtp.parse(best["expiration"]).date() >= ne:
        warn = f" (heads-up: earnings on or before {ne.isoformat()})"
    print(f"\nPreviewing order{warn}...")

    try:
        res = t.preview_or_place_option_order(
            option_symbol=option_symbol,
            underlying=underlying,
            side=side,
            quantity=qty,
            limit_price=limit_px,
            duration="day",
            live=args.live,
            tag=f"options-bot-{args.risk}"
        )
    except Exception as e:
        print(f"Order preview/place failed: {e}")
        sys.exit(1)

    if res.get("preview"):
        print("Preview response:")
        print(json.dumps(res["preview"], indent=2))

    if res.get("placed"):
        print("\nOrder placed:")
        print(json.dumps(res["placed"], indent=2))
    else:
        if args.live:
            print("Attempted to place order but no confirmation received.")
        else:
            print("\nDry-run complete. Re-run with --live to place after preview.")

if __name__ == "__main__":
    main()
