#!/usr/bin/env python3
"""
Seed test data for user 'Bob' into the Azure SQL fitness database.

Bob's profile
  - Born: January 1, 1985
  - Height: 5'10" (70 inches)
  - Starting weight: ~185 lbs (Oct 2025), trending to ~180 lbs (Mar 2026)
  - Diet: standard American / western diet
  - 5 months of meals + snacks: 2025-10-15 → 2026-03-14

Usage:
    # DefaultAzureCredential (requires: az login or managed identity)
    uv run sql/seed_bob_testdata_azuresql.py

    # Admin username/password (reads AZURE_SQL_ADMIN_USER / AZURE_SQL_ADMIN_PASSWORD from .env)
    uv run sql/seed_bob_testdata_azuresql.py --auth adminpassword
"""

from __future__ import annotations

import argparse
import os
import random
import struct
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# ── load .env ─────────────────────────────────────────────────────────────────

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE, override=False)

# ── constants ────────────────────────────────────────────────────────────────

BOB_USER_ID = "bob-test-user-001"
BOB_EXTERNAL_KEY = "bob"

SEED_START = date(2025, 10, 15)
SEED_END = date(2026, 3, 14)

SQL_COPT_SS_ACCESS_TOKEN = 1256
AZURE_SQL_TOKEN_SCOPE = "https://database.windows.net/.default"

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

def _target_weight(d: date) -> float:
    """Piecewise linear weight target with holiday bump."""
    anchors = [
        (date(2025, 10, 15), 185.0),
        (date(2025, 12,  1), 184.0),
        (date(2026,  1,  1), 186.5),
        (date(2026,  1, 15), 184.0),
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _iso(d: date, hour: int = 8, minute: int = 0) -> str:
    return datetime(d.year, d.month, d.day, hour, minute, 0, tzinfo=UTC).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


def _add_jitter(value: float, pct: float = 0.06) -> float:
    return round(value * (1 + random.uniform(-pct, pct)), 1)


def _encode_sql_access_token(token: str) -> bytes:
    encoded = token.encode("utf-16-le")
    return struct.pack(f"<I{len(encoded)}s", len(encoded), encoded)


# ── Azure SQL connection ──────────────────────────────────────────────────────

def _get_connection(auth_mode: str):
    import pyodbc

    server = os.environ.get("AZURE_SQL_SERVER", "").strip()
    database = os.environ.get("AZURE_SQL_DATABASE", "").strip()
    driver = os.environ.get("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server").strip()
    encrypt = os.environ.get("AZURE_SQL_ENCRYPT", "true").strip().lower() in ("true", "1", "yes")
    trust_cert = os.environ.get("AZURE_SQL_TRUST_SERVER_CERTIFICATE", "false").strip().lower() in ("true", "1", "yes")
    timeout = int(os.environ.get("AZURE_SQL_CONNECTION_TIMEOUT", "30"))

    if not server:
        raise ValueError("AZURE_SQL_SERVER is not set. Check your .env file.")
    if not database:
        raise ValueError("AZURE_SQL_DATABASE is not set. Check your .env file.")
    if ".database.windows.net" not in server:
        server = f"{server}.database.windows.net"

    base_connstr = (
        f"Driver={{{driver}}};"
        f"Server=tcp:{server},1433;"
        f"Database={database};"
        f"Encrypt={'yes' if encrypt else 'no'};"
        f"TrustServerCertificate={'yes' if trust_cert else 'no'};"
        f"Connection Timeout={timeout};"
    )

    if auth_mode == "adminpassword":
        user = os.environ.get("AZURE_SQL_ADMIN_USER", "").strip()
        password = os.environ.get("AZURE_SQL_ADMIN_PASSWORD", "").strip()
        if not user or not password:
            raise ValueError(
                "AZURE_SQL_ADMIN_USER and AZURE_SQL_ADMIN_PASSWORD must be set for --auth adminpassword."
            )
        conn_str = base_connstr + f"UID={user};PWD={password};"
        print(f"  Auth: SQL admin user '{user}'")
        return pyodbc.connect(conn_str, autocommit=False)

    # defaultazurecredential
    from azure.identity import DefaultAzureCredential
    print("  Auth: DefaultAzureCredential (requires az login or managed identity)")
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    access_token = credential.get_token(AZURE_SQL_TOKEN_SCOPE).token
    token_struct = _encode_sql_access_token(access_token)
    return pyodbc.connect(base_connstr, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct}, autocommit=False)


# ── seed functions ────────────────────────────────────────────────────────────

def _t(name: str) -> str:
    schema = os.environ.get("AZURE_SQL_SCHEMA", "dbo").strip() or "dbo"
    return f"[{schema}].[{name}]"


def purge_bob(cursor) -> None:
    """Delete all existing Bob rows for clean, idempotent re-seeding."""
    cursor.execute(f"DELETE FROM {_t('meal_events')} WHERE user_id = ?", (BOB_USER_ID,))
    cursor.execute(f"DELETE FROM {_t('body_metric_events')} WHERE user_id = ?", (BOB_USER_ID,))
    cursor.execute(f"DELETE FROM {_t('users')} WHERE user_id = ?", (BOB_USER_ID,))
    print("  ✓ purged existing Bob rows")


def seed_user(cursor) -> None:
    now = _iso(SEED_START)
    cursor.execute(
        f"""
        INSERT INTO {_t('users')}
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
            70.0,
            "in",
            "Toronto",
            "Canada",
            "male",
            "America/Toronto",
            now,
            now,
        ),
    )
    print(f"  ✓ user row inserted  user_id={BOB_USER_ID}")


def seed_body_metrics(cursor) -> None:
    rows = 0

    # Weekly weigh-ins
    d = SEED_START
    while d <= SEED_END:
        target = _target_weight(d)
        weight = round(target + random.uniform(-1.2, 1.2), 1)
        cursor.execute(
            f"""
            INSERT INTO {_t('body_metric_events')}
                (event_id, user_id, metric_type, value_primary, value_secondary,
                 unit, observed_at, source, confidence, notes, created_at)
            VALUES (?, ?, 'weight', ?, NULL, 'lbs', ?, 'manual', 0.95, NULL, ?)
            """,
            (_uid(), BOB_USER_ID, weight, _iso(d, 7, 0), _iso(d, 7, 1)),
        )
        rows += 1
        d += timedelta(days=7)

    # Monthly waist measurements
    waist_vals = [36.5, 36.2, 36.8, 36.0, 35.8, 35.5]
    for i, waist in enumerate(waist_vals):
        m_date = SEED_START + timedelta(days=i * 30)
        if m_date > SEED_END:
            break
        cursor.execute(
            f"""
            INSERT INTO {_t('body_metric_events')}
                (event_id, user_id, metric_type, value_primary, value_secondary,
                 unit, observed_at, source, confidence, notes, created_at)
            VALUES (?, ?, 'waist', ?, NULL, 'in', ?, 'manual', 0.9, NULL, ?)
            """,
            (_uid(), BOB_USER_ID, waist, _iso(m_date, 8, 0), _iso(m_date, 8, 1)),
        )
        rows += 1

    # Blood pressure readings
    bp_data = [
        (date(2025, 11, 10), 128.0, 82.0, "slightly elevated"),
        (date(2026,  1, 20), 124.0, 80.0, "improved after walking more"),
        (date(2026,  3,  5), 122.0, 78.0, None),
    ]
    for bp_date, systolic, diastolic, note in bp_data:
        cursor.execute(
            f"""
            INSERT INTO {_t('body_metric_events')}
                (event_id, user_id, metric_type, value_primary, value_secondary,
                 unit, observed_at, source, confidence, notes, created_at)
            VALUES (?, ?, 'blood_pressure', ?, ?, 'mmHg', ?, 'manual', 0.9, ?, ?)
            """,
            (_uid(), BOB_USER_ID, systolic, diastolic, _iso(bp_date, 9, 0), note, _iso(bp_date, 9, 1)),
        )
        rows += 1

    print(f"  ✓ body_metric_events inserted  count={rows}")


def seed_meals(cursor) -> None:
    days = 0
    meal_rows = 0
    current = SEED_START

    b_idx, l_idx, d_idx, s_idx = 0, 3, 7, 1

    while current <= SEED_END:
        dow = current.weekday()
        breakfast_hour = 9 if dow < 5 else 10
        lunch_hour     = 12 if dow < 5 else 13
        dinner_hour    = 19 if dow < 5 else 18
        snack_hour     = 15

        def maybe_insert(template_list, idx, hour, current=current):
            nonlocal meal_rows
            if random.random() < 0.12:
                return idx
            tpl = template_list[idx % len(template_list)]
            name, mtype, kcal, prot, carbs, fat, fiber, sugar, sodium = tpl
            cursor.execute(
                f"""
                INSERT INTO {_t('meal_events')}
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
            meal_rows += 1
            return idx + 1

        b_idx = maybe_insert(BREAKFASTS, b_idx, breakfast_hour)
        l_idx = maybe_insert(LUNCHES,    l_idx, lunch_hour)
        d_idx = maybe_insert(DINNERS,    d_idx, dinner_hour)

        skip_snack_prob = 0.50 if dow >= 5 else 0.30
        if random.random() > skip_snack_prob:
            s_idx = maybe_insert(SNACKS, s_idx, snack_hour)
        if dow >= 5 and random.random() < 0.25:
            s_idx = maybe_insert(SNACKS, s_idx, 21)

        days += 1
        current += timedelta(days=1)

    print(f"  ✓ meal_events inserted  days={days}  meals={meal_rows}")


# ── check schema ──────────────────────────────────────────────────────────────

def _check_schema(cursor) -> None:
    schema = os.environ.get("AZURE_SQL_SCHEMA", "dbo").strip() or "dbo"
    required = {"users", "body_metric_events", "meal_events"}
    existing = {
        row[0]
        for row in cursor.execute(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'BASE TABLE'",
            (schema,),
        ).fetchall()
    }
    missing = required - existing
    if missing:
        raise RuntimeError(
            f"Required tables missing in schema '{schema}': {missing}. "
            "Run the Azure SQL migration scripts first."
        )


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed Bob test data into the Azure SQL fitness database."
    )
    parser.add_argument(
        "--auth",
        choices=["defaultazurecredential", "adminpassword"],
        default="defaultazurecredential",
        help="Authentication mode (default: defaultazurecredential)",
    )
    args = parser.parse_args()

    server = os.environ.get("AZURE_SQL_SERVER", "<not set>")
    database = os.environ.get("AZURE_SQL_DATABASE", "<not set>")
    print(f"\nTarget: {server} / {database}")

    print("Connecting …")
    conn = _get_connection(args.auth)

    try:
        cursor = conn.cursor()
        _check_schema(cursor)

        print("\nSeeding Bob test data …")
        purge_bob(cursor)
        seed_user(cursor)
        seed_body_metrics(cursor)
        seed_meals(cursor)
        conn.commit()
        print("  ✓ committed")

        meal_count = cursor.execute(
            f"SELECT COUNT(*) FROM {_t('meal_events')} WHERE user_id = ?", (BOB_USER_ID,)
        ).fetchone()[0]
        metric_count = cursor.execute(
            f"SELECT COUNT(*) FROM {_t('body_metric_events')} WHERE user_id = ?", (BOB_USER_ID,)
        ).fetchone()[0]
        print(f"\n✅ Done  meal_events={meal_count}  body_metric_events={metric_count}")
        print("   Login with username 'bob' in the chatbot.\n")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
