"""Source validation and point-in-time user features (DuckDB, China time)."""

from pathlib import Path
import json

import duckdb
import pandas as pd


START = '2017-11-25'
END = '2017-12-03'
FEATURES = [
    'pv_4d', 'cart_4d', 'fav_4d', 'buy_4d', 'active_days_4d',
    'pv_1d', 'cart_1d', 'fav_1d', 'buy_1d', 'recency_days', 'activity_4d',
]


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace('\\', '/').replace("'", "''")


def raw_relation(csv: Path) -> str:
    return f"""read_csv('{sql_path(csv)}', header=false, auto_detect=false,
        delim=',', columns={{'user_id':'BIGINT', 'item_id':'BIGINT',
        'category_id':'BIGINT', 'behavior_type':'VARCHAR', 'ts':'BIGINT'}})"""


VALID = """user_id > 0 AND item_id > 0 AND category_id > 0
           AND behavior_type IN ('pv','cart','fav','buy') AND ts IS NOT NULL"""

# China had no daylight-saving changes in the dataset period. Use UTC+8 explicitly.
LOCAL_DATE = "DATE '1970-01-01' + CAST(FLOOR((ts + 28800) / 86400.0) AS INTEGER)"


def connect_source(root: Path, out_dir: Path, memory: str = '600MB', threads: int = 2):
    """Read the existing cleaned database without mutating it; rebuild if absent."""
    cache = out_dir / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    temp = cache / 'duckdb_tmp'
    temp.mkdir(exist_ok=True)
    con = duckdb.connect(':memory:')
    con.execute("SET TimeZone='Asia/Shanghai'")
    con.execute(f"SET memory_limit='{memory}'")
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET temp_directory='{sql_path(temp)}'")
    con.execute("SET preserve_insertion_order=false")
    db = root / 'taobao_project.db'
    if not db.exists():
        db = cache / 'rebuilt_source.duckdb'
        if not db.exists():
            csv = root / 'UserBehavior.csv'
            if not csv.exists():
                raise FileNotFoundError('Place UserBehavior.csv in the project root.')
            con.execute(f"ATTACH '{sql_path(db)}' AS source")
            con.execute(f"""CREATE TABLE source.user_behavior_cleaned AS
                SELECT user_id, item_id, category_id, behavior_type,
                    to_timestamp(ts) AS action_time, {LOCAL_DATE} AS action_date,
                    CAST(FLOOR(((ts + 28800) % 86400) / 3600.0) AS INTEGER) AS action_hour
                FROM {raw_relation(csv)}
                WHERE ({VALID}) AND {LOCAL_DATE} BETWEEN DATE '{START}' AND DATE '{END}'
            """)
            con.execute('DETACH source')
    con.execute(f"ATTACH '{sql_path(db)}' AS source (READ_ONLY)")
    con.execute("""CREATE VIEW events AS SELECT user_id, item_id, category_id,
        behavior_type, action_time, action_date, action_hour
        FROM source.user_behavior_cleaned""")
    return con


def audit_source(con, root: Path, out_dir: Path) -> dict:
    """Reconcile raw and cleaned daily behavior counts; never silently ignore bad rows."""
    csv = root / 'UserBehavior.csv'
    if not csv.exists():
        raise FileNotFoundError('Raw CSV required to reconcile the cleaned database.')
    raw = con.execute(f"""
        SELECT {LOCAL_DATE} AS action_date, behavior_type, COUNT(*) AS n_raw,
            COUNT(*) FILTER (WHERE NOT COALESCE(({VALID}), FALSE)) AS n_invalid
        FROM {raw_relation(csv)} GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    clean = con.execute("""
        SELECT action_date, behavior_type, COUNT(*) AS n_clean,
            COUNT(*) FILTER (WHERE user_id IS NULL OR user_id <= 0 OR item_id IS NULL
                OR item_id <= 0 OR category_id IS NULL OR category_id <= 0
                OR behavior_type IS NULL OR behavior_type NOT IN ('pv','cart','fav','buy')
                OR action_time IS NULL OR action_date IS NULL OR action_hour IS NULL
                OR action_hour NOT BETWEEN 0 AND 23) AS n_invalid
        FROM events GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    in_window = raw['action_date'].between(pd.Timestamp(START), pd.Timestamp(END))
    expected = raw.loc[in_window].copy()
    expected['expected_clean'] = expected['n_raw'] - expected['n_invalid']
    check = expected[['action_date', 'behavior_type', 'expected_clean']].merge(
        clean[['action_date', 'behavior_type', 'n_clean']], how='outer',
        on=['action_date', 'behavior_type'], validate='one_to_one',
    ).fillna({'expected_clean': 0, 'n_clean': 0})
    check['difference'] = check['n_clean'] - check['expected_clean']
    raw.to_csv(out_dir / 'raw_date_behavior_counts.csv', index=False)
    check.to_csv(out_dir / 'source_reconciliation.csv', index=False)
    report = {
        'raw_events': int(raw.n_raw.sum()),
        'outside_window_events': int(raw.loc[~in_window, 'n_raw'].sum()),
        'invalid_events_inside_window': int(expected.n_invalid.sum()),
        'clean_events': int(clean.n_clean.sum()),
        'clean_invalid_events': int(clean.n_invalid.sum()),
        'mismatched_day_behavior_groups': int(check.difference.ne(0).sum()),
        'raw_bytes': csv.stat().st_size,
        'timezone': 'Asia/Shanghai (UTC+8)',
        'period': [START, END],
    }
    (out_dir / 'source_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if report['clean_invalid_events'] or report['mismatched_day_behavior_groups']:
        raise ValueError(f'Source validation failed: {report}. Check source_reconciliation.csv.')
    return report


def build_snapshots(con, out_dir: Path) -> Path:
    """One row per eligible user/score date; label never enters the feature window."""
    cache = out_dir / 'cache'
    cache.mkdir(exist_ok=True)
    daily_path = cache / 'sample_user_day.parquet'
    snapshot_path = cache / 'snapshots.parquet'
    # Modulo sampling retains each selected user's entire event sequence.
    con.execute(f"""COPY (
        SELECT user_id, action_date,
            COUNT(*) FILTER(WHERE behavior_type='pv')::INTEGER AS pv,
            COUNT(*) FILTER(WHERE behavior_type='cart')::INTEGER AS cart,
            COUNT(*) FILTER(WHERE behavior_type='fav')::INTEGER AS fav,
            COUNT(*) FILTER(WHERE behavior_type='buy')::INTEGER AS buy
        FROM events WHERE user_id % 5 = 0 GROUP BY 1, 2
    ) TO '{sql_path(daily_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)""")
    con.execute(f"""COPY (
        WITH daily AS MATERIALIZED (SELECT * FROM read_parquet('{sql_path(daily_path)}')),
        users AS (SELECT DISTINCT user_id FROM daily),
        dates AS (SELECT CAST(d AS DATE) AS score_date FROM
            generate_series(DATE '{START}', DATE '{END}', INTERVAL 1 DAY) t(d)),
        grid AS (
            SELECT u.user_id, d.score_date, COALESCE(pv,0) AS pv, COALESCE(cart,0) AS cart,
                COALESCE(fav,0) AS fav, COALESCE(buy,0) AS buy
            FROM users u CROSS JOIN dates d LEFT JOIN daily x
                ON u.user_id=x.user_id AND d.score_date=x.action_date
        ), features AS (
            SELECT user_id, score_date, CAST(buy>0 AS TINYINT) AS y,
                SUM(pv) OVER history::INTEGER AS pv_4d,
                SUM(cart) OVER history::INTEGER AS cart_4d,
                SUM(fav) OVER history::INTEGER AS fav_4d,
                SUM(buy) OVER history::INTEGER AS buy_4d,
                SUM(CASE WHEN pv+cart+fav+buy>0 THEN 1 ELSE 0 END) OVER history::INTEGER AS active_days_4d,
                LAG(pv) OVER user_days AS pv_1d, LAG(cart) OVER user_days AS cart_1d,
                LAG(fav) OVER user_days AS fav_1d, LAG(buy) OVER user_days AS buy_1d,
                DATE_DIFF('day', MAX(CASE WHEN pv+cart+fav+buy>0 THEN score_date END)
                    OVER history, score_date)::INTEGER AS recency_days,
                SUM(pv+cart+fav+buy) OVER history::INTEGER AS activity_4d
            FROM grid
            WINDOW user_days AS (PARTITION BY user_id ORDER BY score_date),
                history AS (PARTITION BY user_id ORDER BY score_date ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING)
        )
        SELECT *, CASE WHEN score_date < DATE '2017-12-02' THEN 'train'
                       WHEN score_date = DATE '2017-12-02' THEN 'validation' ELSE 'test' END AS fold
        FROM features WHERE score_date >= DATE '2017-11-29' AND active_days_4d > 0
    ) TO '{sql_path(snapshot_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)""")
    return snapshot_path


def load_snapshots(path: Path) -> pd.DataFrame:
    snapshots = pd.read_parquet(path)
    snapshots['score_date'] = pd.to_datetime(snapshots['score_date'])
    if snapshots.duplicated(['user_id', 'score_date']).any():
        raise ValueError('Duplicate user-date snapshots.')
    if snapshots[FEATURES + ['y']].isna().any().any():
        raise ValueError('Missing values in model inputs.')
    if not ((snapshots.recency_days.between(1, 4)) &
            (snapshots.active_days_4d.between(1, 4)) &
            (snapshots.activity_4d == snapshots[['pv_4d', 'cart_4d', 'fav_4d', 'buy_4d']].sum(axis=1))).all():
        raise ValueError('Invalid history window or counts.')
    return snapshots


def summarize_snapshots(snapshots: pd.DataFrame, out_dir: Path) -> dict:
    """Counts for chronological splits and mutually exclusive test-day segments."""
    import numpy as np

    splits = snapshots.groupby(['score_date', 'fold'], observed=True).agg(
        eligible_users=('user_id', 'size'), buyers=('y', 'sum'), buyer_share=('y', 'mean'),
    ).reset_index().sort_values('score_date')
    splits.to_csv(out_dir / 'snapshot_cohorts.csv', index=False)
    test = snapshots.loc[snapshots.fold.eq('test'), ['buy_4d', 'cart_4d', 'fav_4d', 'y']].copy()
    test['segment'] = np.select(
        [test.buy_4d.gt(0), test.cart_4d.gt(0), test.fav_4d.gt(0)],
        ['Recent buyer', 'Cart / no recent buy', 'Favorite only / no recent buy'],
        default='Browse only',
    )
    segment = test.groupby('segment').agg(users=('y', 'size'), buyers=('y', 'sum'), buyer_share=('y', 'mean')).reset_index()
    segment['share_of_users'] = segment.users / len(test)
    segment['share_of_buyers'] = segment.buyers / test.y.sum() if test.y.sum() else 0
    segment.sort_values('buyer_share', ascending=False).to_csv(out_dir / 'behavior_segments.csv', index=False)
    return {'snapshots': int(len(snapshots)), 'unique_sample_users': int(snapshots.user_id.nunique()),
            'test_users': int(len(test)), 'test_buyers': int(test.y.sum()),
            'test_buyer_share': float(test.y.mean())}
