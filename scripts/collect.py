"""
Collects data from the MyTennis RTT public API (see ../openapi.yaml) into a
local SQLite database, throttled to at most one HTTP call per second.

Usage:
    python scripts/collect.py [--only handbooks,locations,tournaments,ratings] [--rate-limit 1.0]

Safe to re-run: every table is upserted by primary key, and tournaments /
rating rosters are skipped-and-reused when the server-reported record count
for that page/period already matches what's stored locally, so repeat runs
only pay for what actually changed.
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_URL = "https://apirtt.mytennis.online/api/v1/public"
DB_PATH = Path(__file__).resolve().parent.parent / "sqlite" / "mytennis.db"
RATE_LIMIT_SECONDS = 1.0
PAGE_SIZE = 1000  # server-enforced max observed for tour.roster / rating.roster

HANDBOOK_KEYS = [
    "tour_status",
    "tour_rank",
    "tour_format_calendar",
    "tour_category",
    "tour_system",
    "tour_access",
    "cover_type",
    "timezone",
]

# TourRecord fields that are real JSON integers in API responses; everything
# else in that payload comes back as a string (see openapi.yaml TourRecord).
TOUR_INT_FIELDS = {"cant_play", "cant_leave", "cant_request_wc", "org_sub_code"}

_last_call_at = 0.0
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5.0


def api_call(method: str, **params) -> dict:
    """POST one RPC call as multipart/form-data, enforcing the rate limit.

    Transient network errors (timeouts, connection resets) are retried with
    linear backoff, since a several-hundred-request sync will otherwise
    almost certainly get killed by one blip on someone else's network.
    """
    global _last_call_at

    fields = {"ruid": str(int(time.time() * 1000)), "method[0]": method}
    fields.update(params)
    files = {k: (None, str(v)) for k, v in fields.items()}

    for attempt in range(1, MAX_RETRIES + 1):
        wait = RATE_LIMIT_SECONDS - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.post(API_URL, files=files, headers={"Accept": "application/json"}, timeout=30)
            _last_call_at = time.monotonic()
            resp.raise_for_status()
            body = resp.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
            _last_call_at = time.monotonic()
            if attempt == MAX_RETRIES:
                raise
            backoff = RETRY_BACKOFF_SECONDS * attempt
            print(f"[retry] {method} attempt {attempt}/{MAX_RETRIES} failed ({exc!r}); retrying in {backoff:.0f}s")
            time.sleep(backoff)
            continue

        if body.get("status") != "success":
            raise RuntimeError(f"API error for method={method}: {body.get('code')} {body.get('desc')}")
        return body["data"]


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS handbook_items (
            handbook_key TEXT NOT NULL,
            item_id      TEXT NOT NULL,
            handbook_id  TEXT,
            parent_id    TEXT,
            position     INTEGER,
            enabled      INTEGER,
            selected     INTEGER,
            value        TEXT,
            value2       TEXT,
            name         TEXT,
            data         TEXT,
            PRIMARY KEY (handbook_key, item_id)
        );

        CREATE TABLE IF NOT EXISTS locations (
            location_id TEXT PRIMARY KEY,
            type        TEXT,
            name        TEXT,
            path        TEXT,
            parents     TEXT,
            lat         TEXT,
            lng         TEXT,
            timezone    TEXT
        );

        CREATE TABLE IF NOT EXISTS tournaments (
            tour_id TEXT PRIMARY KEY,
            wizard_id TEXT, name TEXT, subname TEXT, desc TEXT, number TEXT,
            tour_year TEXT, tour_month TEXT, tour_status TEXT, tour_format TEXT,
            tour_format_calendar TEXT, tour_type TEXT, tour_level TEXT, tour_serie TEXT,
            location_id TEXT, timezone TEXT, logo_file TEXT, is_public TEXT,
            tour_category TEXT, tour_rank TEXT, tour_access TEXT, tour_gender TEXT,
            player_age TEXT, fee TEXT, prize TEXT, currency TEXT,
            stop_requests TEXT, stop_wc_org TEXT, stop_wc_rtt TEXT, tour_system TEXT,
            org_id TEXT, org_full TEXT, org_short TEXT, org_paycard TEXT,
            covers TEXT, sob TEXT, location TEXT, location_full TEXT,
            referee_name TEXT, referee_id TEXT, courator_name TEXT,
            prioritypos_in_requests TEXT, prioritypos_in_entry TEXT,
            begin_date TEXT, end_date TEXT, tour_week TEXT, tour_members TEXT, sob_id TEXT,
            courts TEXT, pay_card TEXT, pay_invoice TEXT, pay_cash TEXT,
            entry_begin TEXT, entry_end TEXT, lateentry_end TEXT,
            lateleave_begin TEXT, lateleave_end TEXT, pub_ratinglist_dt TEXT,
            filter_geo TEXT, filter_rank TEXT, min_sport_rank TEXT, max_sport_rank TEXT,
            wc_player_start_dt TEXT, wc_player_stop_rtt_dt TEXT, wc_player_stop_rtt_dt_loc TEXT,
            wc_player_stop_org_dt TEXT, wc_public_rtt_stop_dt TEXT, wc_public_rtt_stop_dt_loc TEXT,
            wc_public_org_stop_dt TEXT, list_freezed TEXT, freeze_dt TEXT,
            min_team TEXT, max_team TEXT, cnt_team TEXT, ignorejoinlist TEXT, ignorelatejoin TEXT,
            ot_stage_id TEXT, ot_type TEXT, ot_begin_dt TEXT, ot_end_dt TEXT, ot_fee TEXT,
            ot_currency TEXT, ot_group_count TEXT, ot_member_count TEXT, ot_poe_count TEXT,
            ot_seed_count TEXT, ot_organizer_wc TEXT, ot_courator_wc TEXT, ot_lateleave_end TEXT,
            ot_online_bdt TEXT, ot_online_edt TEXT,
            oe_stage_id TEXT, oe_begin_dt TEXT, oe_end_dt TEXT, oe_fee TEXT,
            oe_currency TEXT, oe_group_count TEXT, oe_member_count TEXT, oe_seed_count TEXT,
            oe_organizer_wc TEXT, oe_courator_wc TEXT, oe_lateleave_end TEXT,
            oe_online_bdt TEXT, oe_online_edt TEXT,
            winner_name TEXT, cant_play INTEGER, cant_leave INTEGER, cant_request_wc INTEGER,
            reportf_files TEXT, org_sub_code INTEGER,
            is_imported TEXT, is_finished TEXT, report_status TEXT, tour_rating TEXT,
            tour_rating_category TEXT, same_as TEXT, original TEXT
        );

        CREATE TABLE IF NOT EXISTS rating_periods (
            rating_id   TEXT PRIMARY KEY,
            rank        TEXT,
            rating_date TEXT,
            start_date  TEXT,
            actual_date TEXT,
            comment     TEXT
        );

        CREATE TABLE IF NOT EXISTS rating_roster (
            rating_id  TEXT NOT NULL,
            player_id  TEXT NOT NULL,
            points     TEXT, counted TEXT, total TEXT,
            age TEXT, ag1 TEXT, ag2 TEXT, age_group TEXT,
            location_id TEXT, region_id TEXT, gender TEXT, position TEXT,
            name TEXT, rtt_number TEXT, birth_date TEXT, city TEXT, filter_pos TEXT,
            PRIMARY KEY (rating_id, player_id)
        );

        CREATE INDEX IF NOT EXISTS idx_tournaments_begin_date ON tournaments(begin_date);
        CREATE INDEX IF NOT EXISTS idx_rating_roster_rating_id ON rating_roster(rating_id);
        """
    )
    conn.commit()


def upsert(conn: sqlite3.Connection, table: str, row: dict) -> None:
    cols = list(row.keys())
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols)
    pk = {
        "handbook_items": "(handbook_key, item_id)",
        "locations": "(location_id)",
        "tournaments": "(tour_id)",
        "rating_periods": "(rating_id)",
        "rating_roster": "(rating_id, player_id)",
    }[table]
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT{pk} DO UPDATE SET {updates}"
    )
    conn.execute(sql, [row[c] for c in cols])


def sync_handbooks(conn: sqlite3.Connection) -> None:
    print(f"[handbooks] fetching {len(HANDBOOK_KEYS)} handbooks in one call...")
    params = {f"handbooks[{i}]": key for i, key in enumerate(HANDBOOK_KEYS)}
    data = api_call("handbook.items", **params)
    n = 0
    for key, items in data["handbooks"].items():
        for item in items:
            upsert(
                conn,
                "handbook_items",
                {
                    "handbook_key": key,
                    "item_id": item["item_id"],
                    "handbook_id": item.get("handbook_id"),
                    "parent_id": item.get("parent_id"),
                    "position": item.get("position"),
                    "enabled": item.get("enabled"),
                    "selected": item.get("selected"),
                    "value": item.get("value"),
                    "value2": item.get("value2"),
                    "name": item.get("name"),
                    "data": item.get("data"),
                },
            )
            n += 1
    conn.commit()
    print(f"[handbooks] stored {n} items across {len(data['handbooks'])} handbooks")


def sync_locations(conn: sqlite3.Connection) -> None:
    for type_id in ("1", "2"):
        print(f"[locations] fetching type={type_id} ...")
        data = api_call("location.search.term", limit=1000, **{"types[0]": type_id, "parent": "", "term": ""})
        items = data["location.search.term"]
        for loc in items:
            upsert(
                conn,
                "locations",
                {
                    "location_id": loc["location_id"],
                    "type": loc.get("type"),
                    "name": loc.get("name"),
                    "path": loc.get("path"),
                    "parents": loc.get("parents"),
                    "lat": loc.get("lat"),
                    "lng": loc.get("lng"),
                    "timezone": loc.get("timezone"),
                },
            )
        conn.commit()
        print(f"[locations] stored {len(items)} items of type={type_id}")
        if len(items) >= 1000:
            print(f"[locations] WARNING: type={type_id} hit the limit=1000 cap; results may be incomplete")


def _tournament_row(rec: dict) -> dict:
    row = {}
    for k, v in rec.items():
        if k in TOUR_INT_FIELDS:
            row[k] = int(v) if v is not None else None
        elif k == "reportf_files":
            row[k] = json.dumps(v, ensure_ascii=False)
        else:
            row[k] = v
    return row


def sync_tournaments(conn: sqlite3.Connection) -> None:
    local_count = conn.execute("SELECT COUNT(*) FROM tournaments").fetchone()[0]

    first = api_call(
        "tour.roster",
        page_no=1,
        per_page=PAGE_SIZE,
        sort_field="begin_date",
        sort_order="ASC",
        range="period",
    )["tour.roster"]
    total = first["navigator"]["count"]
    page_max = first["navigator"]["page_max"]

    if total == local_count:
        print(f"[tournaments] local count already matches server ({total}); skipping")
        return

    print(f"[tournaments] server has {total} records across {page_max} pages (local has {local_count}); syncing...")
    for rec in first["records"]:
        upsert(conn, "tournaments", _tournament_row(rec))
    conn.commit()

    for page in range(2, page_max + 1):
        data = api_call(
            "tour.roster",
            page_no=page,
            per_page=PAGE_SIZE,
            sort_field="begin_date",
            sort_order="ASC",
            range="period",
        )["tour.roster"]
        for rec in data["records"]:
            upsert(conn, "tournaments", _tournament_row(rec))
        conn.commit()
        print(f"[tournaments] page {page}/{page_max} synced")

    print("[tournaments] done")


def sync_ratings(conn: sqlite3.Connection) -> None:
    print("[ratings] fetching rating.list ...")
    periods = api_call("rating.list")["rating.list"]
    for p in periods:
        upsert(
            conn,
            "rating_periods",
            {
                "rating_id": p["rating_id"],
                "rank": p.get("rank"),
                "rating_date": p.get("rating_date"),
                "start_date": p.get("start_date"),
                "actual_date": p.get("actual_date"),
                "comment": p.get("comment"),
            },
        )
    conn.commit()
    print(f"[ratings] stored {len(periods)} rating periods; syncing rosters...")

    for i, p in enumerate(periods, 1):
        rating_id = p["rating_id"]
        rank = p["rank"]
        period_arg = p["rating_date"].replace("-", "")  # e.g. 2026-09-01 -> 20260901

        local_count = conn.execute(
            "SELECT COUNT(*) FROM rating_roster WHERE rating_id = ?", (rating_id,)
        ).fetchone()[0]

        first = api_call("rating.roster", page_no=1, per_page=PAGE_SIZE, rating=period_arg, rank=rank)["rating.roster"]
        total = first["navigator"]["count"]
        page_max = first["navigator"]["page_max"]

        if total == local_count:
            print(f"[ratings] {i}/{len(periods)} {rating_id}: already synced ({total} players), skipping")
            continue

        for rec in first["records"]:
            rec["rating_id"] = rating_id
            upsert(conn, "rating_roster", rec)

        for page in range(2, page_max + 1):
            data = api_call("rating.roster", page_no=page, per_page=PAGE_SIZE, rating=period_arg, rank=rank)[
                "rating.roster"
            ]
            for rec in data["records"]:
                rec["rating_id"] = rating_id
                upsert(conn, "rating_roster", rec)

        conn.commit()
        print(f"[ratings] {i}/{len(periods)} {rating_id}: synced {total} players ({page_max} page(s))")

    print("[ratings] done")


def main() -> None:
    global RATE_LIMIT_SECONDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="handbooks,locations,tournaments,ratings",
        help="Comma-separated subset to sync (default: all)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=RATE_LIMIT_SECONDS,
        help="Minimum seconds between API calls (default: 1.0)",
    )
    args = parser.parse_args()
    RATE_LIMIT_SECONDS = args.rate_limit

    steps = {
        "handbooks": sync_handbooks,
        "locations": sync_locations,
        "tournaments": sync_tournaments,
        "ratings": sync_ratings,
    }
    selected = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = set(selected) - set(steps)
    if unknown:
        parser.error(f"unknown --only value(s): {', '.join(sorted(unknown))}")

    print(f"DB: {DB_PATH}")
    print(f"Rate limit: {RATE_LIMIT_SECONDS}s between calls")

    conn = connect()
    try:
        init_db(conn)
        for name in selected:
            t0 = time.monotonic()
            steps[name](conn)
            print(f"[{name}] took {time.monotonic() - t0:.1f}s\n")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
