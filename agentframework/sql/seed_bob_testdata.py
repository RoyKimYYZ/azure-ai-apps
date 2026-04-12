#!/usr/bin/env python3
"""
Seed test data for a user named 'Bob' in the fitness SQLite database.

Bob's profile
  - Born: January 1, 1985
  - Height: 5'10" (70 inches)
  - Starting weight: ~185 lbs (Oct 2025), trending to ~180 lbs (Mar 2026)
  - Diet: standard American / western diet
  - 5 months of meals + snacks: 2025-10-15 → 2026-03-14

Usage:
    # Uses the default DB path (agentframework.db next to this script's project root):
    python sql/seed_bob_testdata.py

    # Or specify a DB path explicitly:
    python sql/seed_bob_testdata.py --db /path/to/agentframework.db
"""

import argparse
import random
import sqlite3
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────

BOB_USER_ID = "bob-test-user-001"
BOB_EXTERNAL_KEY = "bob"

SEED_START = date(2025, 10, 15)
SEED_END = date(2026, 3, 14)

random.seed(42)  # reproducible

# ── meal templates (name, meal_type, kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg) ──

BREAKFASTS = [
    ("Bacon and eggs",                          "breakfast", 650, 35,  5,  52, 0.0,  1.0, 920),
    ("Bagel with cream cheese",                 "breakfast", 380, 12, 55,  12, 2.0, 10.0, 620),
    ("Pancakes with butter and maple syrup",    "breakfast", 650, 10, 95,  22, 2.0, 36.0, 780),
    ("Cereal with whole milk",                  "breakfast", 320,  9, 56,   6, 3.0, 18.0, 340),
    ("Sausage egg and cheese biscuit",          "breakfast", 520, 24, 38,  30, 1.0,  4.0,1240),
    ("Oatmeal with brown sugar and banana",     "breakfast", 340,  8, 64,   6, 5.0, 22.0, 180),
    ("Toast with peanut butter and jelly",      "breakfast", 420, 14, 54,  18, 3.0, 22.0, 480),
    ("Cheese omelette with hash browns",        "breakfast", 680, 30, 42,  40, 2.0,  2.0, 880),
    ("Cinnamon roll",                           "breakfast", 480,  7, 74,  18, 1.0, 36.0, 420),
    ("Breakfast burrito (egg sausage cheese)",  "breakfast", 560, 28, 44,  28, 2.0,  3.0,1140),
    ("French toast with powdered sugar",        "breakfast", 580, 14, 80,  22, 2.0, 28.0, 560),
    ("Waffles with whipped cream and berries",  "breakfast", 540, 10, 78,  22, 3.0, 26.0, 640),
]

LUNCHES = [
    ("Double cheeseburger and large fries",     "lunch", 1050, 38, 98,  55, 4.0, 14.0,1680),
    ("Turkey sub sandwich with chips",          "lunch",  720, 36, 74,  26, 4.0,  8.0,1800),
    ("Caesar salad with grilled chicken",       "lunch",  480, 34, 18,  28, 3.0,  4.0, 980),
    ("Pepperoni pizza two slices",              "lunch",  540, 24, 62,  22, 3.0,  6.0,1200),
    ("BLT sandwich with kettle chips",          "lunch",  720, 24, 68,  36, 3.0,  8.0,1380),
    ("Burrito bowl with steak and rice",        "lunch",  740, 40, 82,  24, 8.0,  4.0,1560),
    ("Grilled chicken sandwich combo",          "lunch",  740, 38, 72,  28, 3.0, 10.0,1480),
    ("Mac and cheese with hot dog",             "lunch",  780, 28, 86,  32, 2.0, 12.0,1640),
    ("Tuna melt sandwich",                      "lunch",  620, 36, 48,  26, 2.0,  4.0,1180),
    ("Chicken nuggets and fries",               "lunch",  880, 34, 90,  40, 3.0,  8.0,1520),
    ("Cobb salad without dressing",             "lunch",  520, 42, 14,  32, 4.0,  6.0, 860),
    ("Philly cheesesteak sub",                  "lunch",  820, 46, 62,  38, 3.0,  8.0,1740),
    ("Loaded baked potato soup and roll",       "lunch",  640, 22, 74,  28, 4.0, 10.0,1340),
]

DINNERS = [
    ("Spaghetti and meatballs with garlic bread",    "dinner",  920, 44, 98,  32, 5.0, 12.0,1480),
    ("Ribeye steak with mashed potatoes and salad",  "dinner",  980, 68, 46,  52, 4.0,  6.0, 980),
    ("Chicken Alfredo fettuccine",                   "dinner",  860, 48, 78,  32, 3.0,  6.0,1240),
    ("BBQ pork ribs with coleslaw and cornbread",    "dinner", 1200, 58, 74,  62, 4.0, 28.0,1820),
    ("Baked chicken thighs with roasted vegetables", "dinner",  680, 52, 32,  32, 5.0,  8.0, 780),
    ("Steak tacos three with guacamole",             "dinner",  820, 46, 60,  36, 6.0,  4.0,1240),
    ("Three-cheese pizza large",                     "dinner",  960, 40,104,  38, 4.0, 10.0,1680),
    ("Salmon with white rice and broccoli",          "dinner",  640, 48, 56,  18, 4.0,  4.0, 680),
    ("Pot roast with potatoes and carrots",          "dinner",  820, 58, 48,  36, 5.0, 10.0,1120),
    ("Beef stir fry with noodles",                   "dinner",  780, 42, 78,  28, 4.0, 12.0,1360),
    ("Fried chicken with biscuit and corn",          "dinner",  980, 52, 82,  44, 4.0, 14.0,1780),
    ("Lasagna with caesar salad",                    "dinner",  880, 46, 86,  34, 4.0, 12.0,1420),
    ("Chili with cheddar and crackers",              "dinner",  720, 42, 68,  24, 12.0, 8.0,1560),
    ("Grilled burger with sweet potato fries",       "dinner",  940, 44, 82,  44, 6.0, 22.0,1240),
    ("Shrimp fettuccine alfredo",                    "dinner",  820, 42, 74,  32, 3.0,  6.0,1120),
]

SNACKS = [
    ("Lay's potato chips single bag",           "snack", 280,  3, 28, 18, 2.0,  1.0, 380),
    ("Chocolate chip cookies three",            "snack", 240,  3, 34, 10, 1.0, 22.0, 210),
    ("Coca-Cola and pretzel bites",             "snack", 360,  6, 70,  4, 2.0, 40.0, 620),
    ("Chewy granola bar",                       "snack", 190,  4, 28,  7, 2.0, 14.0, 140),
    ("Apple slices with peanut butter",         "snack", 280,  8, 32, 14, 5.0, 18.0, 140),
    ("Trail mix with M&Ms",                     "snack", 360, 10, 38, 20, 3.0, 24.0, 120),
    ("Vanilla ice cream two scoops",            "snack", 300,  5, 34, 16, 0.0, 28.0, 120),
    ("Doritos nacho cheese bag",                "snack", 260,  4, 32, 14, 2.0,  2.0, 380),
    ("Oreo cookies four",                       "snack", 220,  2, 32, 10, 1.0, 16.0, 280),
    ("String cheese and crackers",              "snack", 240, 12, 24, 12, 1.0,  2.0, 560),
    ("Can of soda",                             "snack", 150,  0, 40,  0, 0.0, 40.0,  40),
    ("Banana and Nutella on toast",             "snack", 340,  6, 56, 12, 3.0, 28.0, 180),
    ("Mozzarella sticks three",                 "snack", 320, 16, 28, 16, 1.0,  2.0, 680),
]

# ── weight trajectory ─────────────────────────────────────────────────────────
# Bob starts at 185 lbs in mid-Oct 2025 and reaches 180 lbs by mid-Mar 2026.
# Holiday season in Dec bumps him up ~2 lbs, then he dips back down.

def _target_weight(d: date) -> float:
    """Piecewise linear weight target with holiday bump."""
    anchors = [
        (date(2025, 10, 15), 185.0),
        (date(2025, 12,  1), 184.0),
        (date(2026,  1,  1), 186.5),  # holiday eating peak
        (date(2026,  1, 15), 184.0),  # new-year resolve
        (date(2026,  2, 15), 182.0),
        (date(2026,  3, 14), 180.0),
    ]
    for i in range(len(anchors) - 1):
        d0, w0 = anchors[i]
        d1, w1 = anchors[i + 1]
        if d0 <= d <= d1:
            t = (d - d0).days / (d1 - d0).days
            return w0 + t * (w1 - w0)
    return anchors[-1][1]


# ── helper ────────────────────────────────────────────────────────────────────

def _iso(d: date, hour: int = 8, minute: int = 0) -> str:
    return datetime(d.year, d.month, d.day, hour, minute, 0, tzinfo=UTC).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


def _add_jitter(value: float, pct: float = 0.06) -> float:
    """Add ±pct random jitter to a value."""
    return round(value * (1 + random.uniform(-pct, pct)), 1)


# ── seed functions ────────────────────────────────────────────────────────────

def seed_user(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO users
            (user_id, external_user_key, name, birthday_mmddyyyy,
             height_value, height_unit, city, country, sex, timezone,
             created_at, updated_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """,
        (
            BOB_USER_ID,
            BOB_EXTERNAL_KEY,
            "Bob",
            "01/01/1985",
            70.0,          # 5'10"
            "in",
            "Toronto",
            "Canada",
            "male",
            "America/Toronto",
            "2025-10-15T00:00:00+00:00",
            "2025-10-15T00:00:00+00:00",
        ),
    )
    print(f"  ✓ user row inserted  user_id={BOB_USER_ID}")


def seed_body_metrics(conn: sqlite3.Connection) -> None:
    rows_inserted = 0

    # Weekly weigh-ins (every 7 days from start)
    d = SEED_START
    while d <= SEED_END:
        target = _target_weight(d)
        weight = round(target + random.uniform(-1.2, 1.2), 1)
        conn.execute(
            """
            INSERT OR IGNORE INTO body_metric_events
                (event_id, user_id, metric_type, value_primary, value_secondary,
                 unit, observed_at, source, confidence, notes, created_at)
            VALUES (?, ?, 'weight', ?, NULL, 'lbs', ?, 'manual', 0.95, NULL, ?)
            """,
            (_uid(), BOB_USER_ID, weight, _iso(d, 7, 0), _iso(d, 7, 1)),
        )
        rows_inserted += 1
        d += timedelta(days=7)

    # Monthly waist measurements (slight variation around 36")
    waist_vals = [36.5, 36.2, 36.8, 36.0, 35.8, 35.5]
    for i, waist in enumerate(waist_vals):
        m_date = SEED_START + timedelta(days=i * 30)
        if m_date > SEED_END:
            break
        conn.execute(
            """
            INSERT OR IGNORE INTO body_metric_events
                (event_id, user_id, metric_type, value_primary, value_secondary,
                 unit, observed_at, source, confidence, notes, created_at)
            VALUES (?, ?, 'waist', ?, NULL, 'in', ?, 'manual', 0.9, NULL, ?)
            """,
            (_uid(), BOB_USER_ID, waist, _iso(m_date, 8, 0), _iso(m_date, 8, 1)),
        )
        rows_inserted += 1

    # Two blood pressure readings (systolic/diastolic)
    bp_data = [
        (date(2025, 11, 10), 128.0, 82.0, "slightly elevated"),
        (date(2026,  1, 20), 124.0, 80.0, "improved after walking more"),
        (date(2026,  3,  5), 122.0, 78.0, None),
    ]
    for bp_date, systolic, diastolic, note in bp_data:
        conn.execute(
            """
            INSERT OR IGNORE INTO body_metric_events
                (event_id, user_id, metric_type, value_primary, value_secondary,
                 unit, observed_at, source, confidence, notes, created_at)
            VALUES (?, ?, 'blood_pressure', ?, ?, 'mmHg', ?, 'manual', 0.9, ?, ?)
            """,
            (_uid(), BOB_USER_ID, systolic, diastolic, _iso(bp_date, 9, 0), note, _iso(bp_date, 9, 1)),
        )
        rows_inserted += 1

    print(f"  ✓ body_metric_events inserted  count={rows_inserted}")


def seed_meals(conn: sqlite3.Connection) -> None:
    rows_inserted = 0
    current = SEED_START

    # Cycle indices — offset each meal type independently so menus don't repeat in lockstep
    b_idx = 0  # breakfast
    l_idx = 3  # lunch
    d_idx = 7  # dinner
    s_idx = 1  # snack

    while current <= SEED_END:
        dow = current.weekday()  # 0=Mon … 6=Sun

        # On weekends Bob sometimes skips breakfast and grabs brunch instead (modelled as a later breakfast)
        breakfast_hour = 9 if dow < 5 else 10
        lunch_hour = 12 if dow < 5 else 13
        dinner_hour = 19 if dow < 5 else 18
        snack_hour = 15

        # Occasionally skip a meal (~15% chance per meal)
        def maybe_insert(template_list: list, idx: int, hour: int, current=current) -> int:
            if random.random() < 0.12:  # skip ~12% of the time
                return idx
            tpl = template_list[idx % len(template_list)]
            name, mtype, kcal, prot, carbs, fat, fiber, sugar, sodium = tpl
            conn.execute(
                """
                INSERT OR IGNORE INTO meal_events
                    (meal_event_id, user_id, occurred_at, meal_type,
                     calories_kcal, protein_g, carbs_g, fat_g,
                     fiber_g, sugar_g, sodium_mg,
                     unit_system, confidence, model_name, model_version,
                     prompt_version, notes, created_at)
                VALUES (?, ?, ?, ?,  ?, ?, ?, ?,  ?, ?, ?,  'imperial', 0.9,
                        'seed-data', '1.0', 'v1', ?, ?)
                """,
                (
                    _uid(), BOB_USER_ID,
                    _iso(current, hour, random.randint(0, 30)),
                    mtype,
                    _add_jitter(float(kcal)),
                    _add_jitter(float(prot)),
                    _add_jitter(float(carbs)),
                    _add_jitter(float(fat)),
                    _add_jitter(float(fiber)),
                    _add_jitter(float(sugar)),
                    _add_jitter(float(sodium)),
                    name,
                    _iso(current, hour, random.randint(0, 30)),
                ),
            )
            return idx + 1

        b_idx = maybe_insert(BREAKFASTS, b_idx, breakfast_hour)
        l_idx = maybe_insert(LUNCHES,    l_idx, lunch_hour)
        d_idx = maybe_insert(DINNERS,    d_idx, dinner_hour)

        # Snack: skip ~30% of weekdays, ~50% of weekends (less routine)
        skip_snack_prob = 0.50 if dow >= 5 else 0.30
        if random.random() > skip_snack_prob:
            s_idx = maybe_insert(SNACKS, s_idx, snack_hour)

        # Extra snack on weekends ~25% of the time (late-night snacking)
        if dow >= 5 and random.random() < 0.25:
            s_idx = maybe_insert(SNACKS, s_idx, 21)

        rows_inserted += 1  # count days, not meals
        current += timedelta(days=1)

    print(f"  ✓ meal_events inserted  days={rows_inserted} (skips and extras applied)")


# ── entry point ───────────────────────────────────────────────────────────────

def _default_db_path() -> Path:
    # Mirror the app's _fitness_db_path() logic: look for agentframework.db
    # next to the project root (two levels up from this sql/ directory).
    candidate = Path(__file__).resolve().parent.parent / "agentframework.db"
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Bob test data into the fitness SQLite DB.")
    parser.add_argument("--db", default=None, help="Path to agentframework.db (default: auto-detect)")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else _default_db_path()
    print(f"\nTarget DB: {db_path}")
    if not db_path.exists():
        print("  ⚠  DB file not found — it will be created (schema must already be applied).")

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        print("\nSeeding Bob test data …")

        # Check schema exists
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        required = {"users", "body_metric_events", "meal_events"}
        if not required.issubset(tables):
            missing = required - tables
            raise RuntimeError(
                f"Schema tables missing: {missing}. "
                "Run sql/001_fitness_memory_sqlite.sql first."
            )

        seed_user(conn)
        seed_body_metrics(conn)
        seed_meals(conn)
        conn.commit()

    total_meals = sqlite3.connect(db_path).execute(
        "SELECT COUNT(*) FROM meal_events WHERE user_id = ?", (BOB_USER_ID,)
    ).fetchone()[0]
    total_metrics = sqlite3.connect(db_path).execute(
        "SELECT COUNT(*) FROM body_metric_events WHERE user_id = ?", (BOB_USER_ID,)
    ).fetchone()[0]

    print(f"\n✅ Done  meal_events={total_meals}  body_metric_events={total_metrics}")
    print("   Login with username 'bob' (external_user_key) in the chatbot.\n")


if __name__ == "__main__":
    main()
