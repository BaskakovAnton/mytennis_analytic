"""
Exploratory analysis of the collected MyTennis RTT data (sqlite/mytennis.db,
built by collect.py). Loads each table into pandas and renders one PNG chart
per question into analysis/, grouped by data domain: tournaments, ratings,
locations.

Usage:
    python scripts/analyze.py
"""

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save PNGs instead of trying to pop up a window
import matplotlib.pyplot as plt
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "sqlite" / "mytennis.db"
OUT_DIR = Path(__file__).resolve().parent.parent / "analysis"

plt.rcParams["figure.autolayout"] = True
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]


def savefig(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path.relative_to(OUT_DIR.parent)}")


def handbook_map(conn: sqlite3.Connection, key: str) -> pd.Series:
    df = pd.read_sql_query(
        "SELECT item_id, name FROM handbook_items WHERE handbook_key = ?", conn, params=(key,)
    )
    return df.set_index("item_id")["name"]


def analyze_tournaments(conn: sqlite3.Connection) -> None:
    df = pd.read_sql_query(
        """
        SELECT tour_id, begin_date, end_date, location, tour_status,
               tour_category, tour_rank, tour_gender, fee, currency, tour_members
        FROM tournaments
        """,
        conn,
    )
    print(f"[tournaments] {len(df)} rows loaded")

    df["begin_date"] = pd.to_datetime(df["begin_date"], errors="coerce")
    df["year"] = df["begin_date"].dt.year

    # 1. Volume over time
    per_year = df.groupby("year").size()
    per_year = per_year[(per_year.index >= 2004) & (per_year.index <= pd.Timestamp.now().year)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(per_year.index.astype(int).astype(str), per_year.values, color=PALETTE[0])
    ax.set_title("Tournaments started per year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Tournament count")
    ax.tick_params(axis="x", rotation=90)
    savefig(fig, "tournaments_per_year.png")

    # 2. Where they happen
    top_loc = df["location"].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_loc.index, top_loc.values, color=PALETTE[1])
    ax.set_title("Top 15 cities by tournament count")
    ax.set_xlabel("Tournament count")
    savefig(fig, "tournaments_top_locations.png")

    # 3. Status breakdown ("0" is not a real handbook value — it means the
    # tournament record never had a status set, common for old imported data)
    status_labels = df["tour_status"].map(handbook_map(conn, "tour_status"))
    status_labels = status_labels.fillna(df["tour_status"].map({"0": "not set (0)"})).fillna("unknown")
    status_counts = status_labels.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(status_counts.index, status_counts.values, color=PALETTE[4])
    for y, v in enumerate(status_counts.values):
        ax.text(v, y, f" {v:,} ({v / status_counts.sum():.1%})", va="center", fontsize=8)
    ax.set_title("Tournament status breakdown")
    ax.set_xlabel("Tournament count")
    ax.margins(x=0.15)
    savefig(fig, "tournaments_status.png")

    # 4. Entry fee distribution
    fees = pd.to_numeric(df["fee"], errors="coerce")
    fees = fees[(fees > 0) & (fees <= fees.quantile(0.98))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(fees, bins=40, color=PALETTE[2])
    ax.set_title("Entry fee distribution (RUB, top 2% trimmed)")
    ax.set_xlabel("Fee")
    ax.set_ylabel("Tournament count")
    savefig(fig, "tournaments_fee_distribution.png")

    # 5. Category breakdown
    cat_counts = (
        df["tour_category"].map(handbook_map(conn, "tour_category")).fillna("unknown").value_counts().head(15)
    ).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(cat_counts.index, cat_counts.values, color=PALETTE[3])
    ax.set_title("Tournaments by category (top 15)")
    ax.set_xlabel("Tournament count")
    savefig(fig, "tournaments_by_category.png")


def analyze_ratings(conn: sqlite3.Connection) -> None:
    periods = pd.read_sql_query("SELECT rating_id, rank, rating_date FROM rating_periods", conn)
    periods["rating_date"] = pd.to_datetime(periods["rating_date"])
    periods = periods.sort_values("rating_date")
    print(f"[ratings] {len(periods)} periods loaded")

    counts = pd.read_sql_query(
        "SELECT rating_id, COUNT(*) AS n FROM rating_roster GROUP BY rating_id", conn
    )
    merged = periods.merge(counts, on="rating_id", how="left").fillna({"n": 0})

    # 1. Registered players over time, split by rank category
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (rank, grp) in enumerate(merged.groupby("rank")):
        ax.plot(grp["rating_date"], grp["n"], marker="o", markersize=3, label=f"rank {rank}", color=PALETTE[i % len(PALETTE)])
    ax.set_title("Registered players per rating snapshot over time")
    ax.set_xlabel("Rating date")
    ax.set_ylabel("Player count")
    ax.legend()
    savefig(fig, "ratings_players_over_time.png")

    # Deep-dive on the most recent snapshot per rank
    latest_ids = merged.sort_values("rating_date").groupby("rank").tail(1)["rating_id"].tolist()
    placeholders = ",".join("?" for _ in latest_ids)
    latest = pd.read_sql_query(
        f"SELECT * FROM rating_roster WHERE rating_id IN ({placeholders})", conn, params=latest_ids
    )
    latest["points"] = pd.to_numeric(latest["points"], errors="coerce")
    latest["age"] = pd.to_numeric(latest["age"], errors="coerce")
    print(f"[ratings] latest snapshots: {latest_ids} ({len(latest)} player rows)")

    # 2. Top 20 players by points across the latest snapshots
    top20 = latest.nlargest(20, "points")[["name", "points"]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["name"], top20["points"], color=PALETTE[0])
    ax.set_title(f"Top 20 players by points — {', '.join(latest_ids)}")
    ax.set_xlabel("Points")
    savefig(fig, "ratings_top20_players.png")

    # 3. Age distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(latest["age"].dropna(), bins=30, color=PALETTE[1])
    ax.set_title(f"Player age distribution — {', '.join(latest_ids)}")
    ax.set_xlabel("Age")
    ax.set_ylabel("Player count")
    savefig(fig, "ratings_age_distribution.png")

    # 4. Points distribution (log scale — heavily right-skewed)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(latest["points"].dropna(), bins=50, color=PALETTE[2])
    ax.set_yscale("log")
    ax.set_title(f"Points distribution (log y) — {', '.join(latest_ids)}")
    ax.set_xlabel("Points")
    ax.set_ylabel("Player count (log)")
    savefig(fig, "ratings_points_distribution.png")


def analyze_locations(conn: sqlite3.Connection) -> None:
    df = pd.read_sql_query("SELECT type, COUNT(*) AS n FROM locations GROUP BY type", conn)
    print(f"[locations] {df['n'].sum()} rows loaded across {len(df)} types")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(df["type"].astype(str), df["n"], color=PALETTE[3])
    ax.set_title("Locations by type (1=federal district, 2=region)")
    ax.set_xlabel("type")
    ax.set_ylabel("count")
    savefig(fig, "locations_by_type.png")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        analyze_tournaments(conn)
        analyze_ratings(conn)
        analyze_locations(conn)
    finally:
        conn.close()
    print(f"\nAll charts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
