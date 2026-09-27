"""
ingest_round1.py
================
RUN THIS AFTER ROUND 1 OF WC 2026 IS COMPLETE (approx June 19, 2026).

This script:
  1. Fetches round 1 match statistics from API-Football
  2. Pulls xG data from StatsBomb (if released) or estimates from shots
  3. Aggregates to per-team stats: xG, possession, prog carries, defence
  4. Saves to data/live_stats.csv — Phase 2 feature set
  5. Refits the model with the live features now active

After this runs, every prediction uses real tournament data.
Run again after round 2 and round 3 for stronger signal each time.

Usage
-----
  python ingest_round1.py
  python ingest_round1.py --season 2026  # default
  python ingest_round1.py --dry-run      # show what would be fetched
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (DATA_DIR, LIVE_STATS_CACHE, STATSBOMB_CACHE,
                          API_FOOTBALL_KEY, API_FOOTBALL_BASE, WC_LEAGUE_ID,
                          aggregate_live_stats, _synthetic_live_stats)

# ══════════════════════════════════════════════════════════════════════════
# API-Football: fetch round 1 fixtures with statistics
# ══════════════════════════════════════════════════════════════════════════

def fetch_round1_fixture_ids(season: int = 2026) -> list[int]:
    """
    Get fixture IDs for round 1 group stage matches from API-Football.
    """
    headers = {
        "x-rapidapi-key": API_FOOTBALL_KEY,
        "x-rapidapi-host": "v3.football.api-sports.io",
    }
    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {
        "league": WC_LEAGUE_ID,
        "season": season,
        "round":  "Group Stage - 1",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return [fx["fixture"]["id"] for fx in data.get("response", [])]


def fetch_fixture_statistics(fixture_id: int) -> list[dict]:
    """
    Fetch in-match statistics for one fixture.
    Returns list of team stat blocks.
    """
    headers = {
        "x-rapidapi-key": API_FOOTBALL_KEY,
        "x-rapidapi-host": "v3.football.api-sports.io",
    }
    url = f"{API_FOOTBALL_BASE}/fixtures/statistics"
    resp = requests.get(url, headers=headers,
                        params={"fixture": fixture_id}, timeout=15)
    resp.raise_for_status()
    return resp.json().get("response", [])


def parse_team_stats(team_block: dict, match_id: int) -> dict | None:
    """
    Parse one team's statistics from an API-Football statistics block.

    xG: API-Football provides 'expected_goals' in the stats array.
    Possession: provided as a percentage string e.g. "54%".
    Progressive carries / defensive actions: not directly provided —
      we approximate from 'passes_accuracy', 'total_tackles', 'interceptions'.
    """
    team_name = team_block.get("team", {}).get("name", "Unknown")
    stats_list = team_block.get("statistics", [])

    stat_map = {}
    for s in stats_list:
        stat_map[s["type"]] = s["value"]

    def safe_float(key, default=0.0):
        v = stat_map.get(key)
        if v is None:
            return default
        if isinstance(v, str):
            v = v.replace("%", "").replace("-", "0")
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    xg         = safe_float("expected_goals")
    possession = safe_float("Ball Possession", 50.0)

    # Proxy progressive carries from forward passes + dribbles
    prog_carries = int(safe_float("Total passes", 300) * 0.08)  # ~8% are progressive

    # Defensive actions: tackles + interceptions + blocks
    def_actions = (int(safe_float("Total Tackles")) +
                   int(safe_float("Interceptions")) +
                   int(safe_float("Blocks")))

    return {
        "match_id":          match_id,
        "team":              team_name,
        "xg":                xg,
        "possession":        possession,
        "prog_carries":      prog_carries,
        "defensive_actions": def_actions,
    }


def fetch_live_stats_from_api(season: int = 2026,
                              dry_run: bool = False) -> pd.DataFrame:
    """
    Pull all round 1 match statistics from API-Football.
    Aggregates into per-team averages ready for Phase 2 features.
    """
    if API_FOOTBALL_KEY == "YOUR_KEY_HERE":
        print("⚠  No API key — using synthetic round 1 data.")
        print("  Set API_FOOTBALL_KEY env var to use real data.")
        return _synthetic_live_stats()

    print(f"Fetching round 1 fixture IDs for WC {season}...")
    fixture_ids = fetch_round1_fixture_ids(season)
    print(f"  Found {len(fixture_ids)} round 1 fixtures.")

    if dry_run:
        print(f"[dry-run] Would fetch stats for fixture IDs: {fixture_ids[:5]}...")
        return pd.DataFrame()

    rows = []
    for i, fx_id in enumerate(fixture_ids):
        print(f"  [{i+1}/{len(fixture_ids)}] Fixture {fx_id}...")
        try:
            stats_blocks = fetch_fixture_statistics(fx_id)
            for block in stats_blocks:
                row = parse_team_stats(block, fx_id)
                if row:
                    rows.append(row)
            time.sleep(0.8)   # rate limit
        except Exception as e:
            print(f"    ⚠  Failed: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(STATSBOMB_CACHE, index=False)   # overwrite with 2026 data
    print(f"✓ Saved {len(df)} team-match records.")
    return df


# ══════════════════════════════════════════════════════════════════════════
# StatsBomb live ingestion (if they release 2026 data)
# ══════════════════════════════════════════════════════════════════════════

def check_statsbomb_2026() -> bool:
    """
    Check whether StatsBomb has released 2026 WC data on their GitHub repo.
    They released 2022 data during the tournament — watch for 2026 release.

    Monitor: https://github.com/statsbomb/open-data/tree/master/data/matches
    """
    import requests
    try:
        url = ("https://api.github.com/repos/statsbomb/open-data"
               "/contents/data/matches")
        resp = requests.get(url, timeout=10)
        items = resp.json()
        # Competition ID for WC varies; check for new folder
        comp_ids = [int(item["name"]) for item in items
                    if item["name"].isdigit()]
        print(f"StatsBomb competitions available: {sorted(comp_ids)}")
        # 2022 WC = competition 43. If 2026 is released it will appear here.
        has_2026 = 43 in comp_ids   # update competition ID when known
        return has_2026
    except Exception as e:
        print(f"StatsBomb check failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════
# Main: ingest + refit
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ingest round 1 WC 2026 data")
    parser.add_argument("--season",  type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-refit", action="store_true",
                        help="Skip model refit after ingestion")
    args = parser.parse_args()

    print("=" * 55)
    print("ROUND 1 INGESTION — WC 2026")
    print("=" * 55)

    # Check StatsBomb first (preferred source for xG quality)
    print("\nChecking StatsBomb for 2026 data release...")
    has_sb = check_statsbomb_2026()
    if has_sb:
        print("  ✓ StatsBomb 2026 data available — use data_loader.py to pull it.")
    else:
        print("  StatsBomb 2026 not yet released — using API-Football stats.")

    # Fetch live stats
    raw_df = fetch_live_stats_from_api(season=args.season,
                                       dry_run=args.dry_run)
    if args.dry_run or len(raw_df) == 0:
        print("Dry run complete — no data written.")
        return

    # Aggregate to per-team
    live_stats = aggregate_live_stats(raw_df)
    live_stats.to_csv(LIVE_STATS_CACHE, index=False)
    print(f"\nLive stats aggregated: {len(live_stats)} teams")
    print(live_stats[["team", "xg_per90", "possession_avg",
                       "momentum_index", "games_played"]].to_string(index=False))

    # Refit model with Phase 2 features active
    if not args.no_refit:
        print("\n── Refitting model with live features ──")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "train.py"), "--live"],
            capture_output=False
        )
        if result.returncode == 0:
            print("\n✓ Model refitted with round 1 data.")
            print("  Run predict.py to generate round 2 predictions.")
        else:
            print("  ⚠  Refit failed — run:  python train.py --live")


if __name__ == "__main__":
    main()
