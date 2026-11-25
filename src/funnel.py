"""Full-population activity metrics and a fixed-window, ordered item funnel."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _activity_summary(con) -> pd.DataFrame:
    """Compute three time grains in a single scan; never sum distinct users."""
    activity = con.execute("""
        SELECT
            CASE WHEN GROUPING(action_date) = 0 THEN 'day'
                 WHEN GROUPING(action_hour) = 0 THEN 'hour'
                 ELSE 'overall' END AS grain,
            action_date AS date, action_hour AS hour,
            COUNT(*) AS event_count,
            COUNT(DISTINCT user_id) AS active_users,
            COUNT(DISTINCT user_id) FILTER (WHERE behavior_type = 'buy') AS buyer_users,
            COUNT(*) FILTER (WHERE behavior_type = 'pv') AS pv_events,
            COUNT(*) FILTER (WHERE behavior_type = 'cart') AS cart_events,
            COUNT(*) FILTER (WHERE behavior_type = 'fav') AS fav_events,
            COUNT(*) FILTER (WHERE behavior_type = 'buy') AS buy_events
        FROM events
        GROUP BY GROUPING SETS ((action_date), (action_hour), ())
    """).df()
    activity["buy_to_pv_event_ratio"] = activity.buy_events.div(
        activity.pv_events.where(activity.pv_events > 0)
    )
    activity["buyer_user_share"] = activity.buyer_users.div(
        activity.active_users.where(activity.active_users > 0)
    )
    return activity


def _strict_counts(con, cutoff_date: str) -> dict:
    """Find same-item paths within 24h of first observed PV in a user sample."""
    sample_created = pv_created = False
    try:
        # A systematic user sample preserves each selected user's complete history.
        con.execute("""
            CREATE TEMP TABLE __funnel_sample AS
            SELECT user_id, item_id, behavior_type, action_time
            FROM events WHERE user_id % 10 = 0
        """)
        sample_created = True
        sample_events, sample_users = con.execute("""
            SELECT COUNT(*), COUNT(DISTINCT user_id) FROM __funnel_sample
        """).fetchone()
        con.execute("""
            CREATE TEMP TABLE __funnel_first_pv AS
            SELECT user_id, item_id, MIN(action_time) AS first_pv
            FROM __funnel_sample WHERE behavior_type = 'pv'
            GROUP BY user_id, item_id
        """)
        pv_created = True

        # Exclude pairs first viewed on the last observed date: they lack 24h follow-up.
        total_pv, eligible = con.execute("""
            SELECT COUNT(*), COUNT(*) FILTER (WHERE first_pv < CAST(? AS TIMESTAMPTZ))
            FROM __funnel_first_pv
        """, [cutoff_date]).fetchone()

        # An ordered path exists iff the earliest in-window intent precedes the
        # latest in-window purchase; a large many-to-many event join is unnecessary.
        counts = con.execute("""
            WITH paths AS (
                SELECT p.user_id, p.item_id,
                    MIN(s.action_time) FILTER (WHERE s.behavior_type IN ('cart', 'fav')) AS first_intent,
                    MIN(s.action_time) FILTER (WHERE s.behavior_type = 'cart') AS first_cart,
                    MIN(s.action_time) FILTER (WHERE s.behavior_type = 'fav') AS first_fav,
                    MAX(s.action_time) FILTER (WHERE s.behavior_type = 'buy') AS last_purchase
                FROM __funnel_sample s
                JOIN __funnel_first_pv p USING (user_id, item_id)
                WHERE p.first_pv < CAST(? AS TIMESTAMPTZ)
                  AND s.behavior_type IN ('cart', 'fav', 'buy')
                  AND s.action_time > p.first_pv
                  AND s.action_time <= p.first_pv + INTERVAL '24 hours'
                GROUP BY p.user_id, p.item_id
            )
            SELECT
                COUNT(*) FILTER (WHERE first_intent IS NOT NULL),
                COUNT(*) FILTER (WHERE first_intent < last_purchase),
                COUNT(*) FILTER (WHERE last_purchase IS NOT NULL),
                COUNT(*) FILTER (WHERE last_purchase IS NOT NULL AND
                    (first_intent IS NULL OR first_intent >= last_purchase)),
                COUNT(*) FILTER (WHERE first_cart IS NOT NULL),
                COUNT(*) FILTER (WHERE first_fav IS NOT NULL),
                COUNT(*) FILTER (WHERE first_cart < last_purchase),
                COUNT(*) FILTER (WHERE first_fav < last_purchase)
            FROM paths
        """, [cutoff_date]).fetchone()
        keys = (
            "intent_pairs", "intent_purchase_pairs", "any_purchase_pairs",
            "no_confirmed_intent_purchase_pairs", "cart_pairs", "fav_pairs",
            "cart_purchase_pairs", "fav_purchase_pairs",
        )
        result = dict(zip(keys, map(int, counts)))
        result.update(
            sample_rule="user_id % 10 = 0",
            sample_events=int(sample_events),
            sample_users=int(sample_users),
            all_observed_pv_pairs=int(total_pv),
            eligible_pv_pairs=int(eligible),
            excluded_incomplete_followup_pairs=int(total_pv - eligible),
            first_pv_cutoff_exclusive=cutoff_date,
            window_hours=24,
        )
        assert result["intent_purchase_pairs"] <= result["intent_pairs"] <= eligible
        assert result["any_purchase_pairs"] <= eligible
        assert (result["intent_purchase_pairs"] + result["no_confirmed_intent_purchase_pairs"]
                == result["any_purchase_pairs"])
        return result
    finally:
        # Only remove temporary, task-owned tables; do not change the source DB.
        if pv_created:
            con.execute("DROP TABLE temp.main.__funnel_first_pv")
        if sample_created:
            con.execute("DROP TABLE temp.main.__funnel_sample")


def _pct(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def _funnel_table(counts: dict) -> pd.DataFrame:
    eligible = counts["eligible_pv_pairs"]
    stages = [
        ("eligible_first_pv", "eligible_pv_pairs", "First observed PV with complete 24h follow-up"),
        ("pv_then_intent", "intent_pairs", "Later cart or fav, irrespective of purchase"),
        ("pv_then_intent_then_purchase", "intent_purchase_pairs", "PV < cart/fav < buy"),
        ("pv_then_any_purchase", "any_purchase_pairs", "Any later buy, irrespective of intent"),
        ("pv_then_purchase_no_confirmed_intent", "no_confirmed_intent_purchase_pairs",
         "Later buy with no strictly earlier post-PV cart/fav observed"),
        ("pv_then_cart", "cart_pairs", "Later cart; may overlap fav branch"),
        ("pv_then_fav", "fav_pairs", "Later fav; may overlap cart branch"),
        ("pv_then_cart_then_purchase", "cart_purchase_pairs", "PV < cart < buy; may overlap fav branch"),
        ("pv_then_fav_then_purchase", "fav_purchase_pairs", "PV < fav < buy; may overlap cart branch"),
    ]
    return pd.DataFrame([
        {"stage": stage, "user_item_pairs": counts[key], "eligible_pv_pairs": eligible,
         "eligible_share_pct": _pct(counts[key], eligible), "definition": definition}
        for stage, key, definition in stages
    ])


def _plot_activity(daily: pd.DataFrame, hourly: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    event_color = {"pv": "#356D9E", "cart": "#E49F38", "fav": "#7D70AB", "buy": "#3E947A"}
    for col, (frame, x, title) in enumerate((
        (daily, daily.date.dt.strftime("%m-%d"), "Daily"),
        (hourly, hourly.hour, "Hour of day (CST; pooled)"),
    )):
        ax = axes[0, col]
        for event, color in event_color.items():
            # Suppress zero counts on a logarithmic axis.
            y = frame[f"{event}_events"]
            ax.plot(x, y.where(y > 0), label=event.upper(), color=color, linewidth=1.7)
        ax.set_yscale("log")
        ax.set_title(f"{title}: event volume")
        ax.set_ylabel("Events (log scale)")
        ax.legend(ncol=2, fontsize=8, loc="best")
        ax = axes[1, col]
        ax.plot(x, 100 * frame.buyer_user_share, color=event_color["buy"],
                label="Buyer users / active users", linewidth=1.7)
        ax.plot(x, 100 * frame.buy_to_pv_event_ratio, color=event_color["pv"],
                label="Buy events / PV events", linewidth=1.7)
        ax.set_title(f"{title}: two different ratios")
        ax.set_ylabel("Percent")
        ax.legend(fontsize=8, loc="best")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Marketplace activity: full cleaned event population", fontsize=13)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_funnel(counts: dict, out: Path) -> None:
    eligible = counts["eligible_pv_pairs"]
    if not eligible:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = ["First observed PV", "Later cart or fav", "Later buy after intent"]
    values = [100.0, _pct(counts["intent_pairs"], eligible),
              _pct(counts["intent_purchase_pairs"], eligible)]
    bars = axes[0].barh(labels, values, color=["#356D9E", "#E49F38", "#3E947A"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 118)
    axes[0].bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    axes[0].set_title("Strictly ordered stages")
    axes[0].set_xlabel("% of eligible user-item pairs")
    branches = ["Buy after observed intent", "Buy, no confirmed prior intent"]
    branch_values = [_pct(counts["intent_purchase_pairs"], eligible),
                     _pct(counts["no_confirmed_intent_purchase_pairs"], eligible)]
    bars = axes[1].barh(branches, branch_values, color=["#3E947A", "#8997A3"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(1.0, max(branch_values) * 1.3))
    axes[1].bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    axes[1].set_title("Mutually exclusive purchase branches")
    axes[1].set_xlabel("% of eligible user-item pairs")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(f"24-hour item funnel | systematic 10% user sample | n={eligible:,}")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def analyze_funnel(con, out_dir: Path) -> dict:
    """Export descriptive population metrics and a temporally valid 24h funnel."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con.execute("SET TimeZone='Asia/Shanghai'")
    activity = _activity_summary(con)
    daily = activity.loc[activity.grain == "day"].drop(columns=["grain", "hour"]).sort_values("date")
    hourly = activity.loc[activity.grain == "hour"].drop(columns=["grain", "date"]).sort_values("hour")
    daily["date"] = pd.to_datetime(daily.date)
    hourly["hour"] = hourly.hour.astype(int)
    if daily.empty:
        raise ValueError("Cannot analyze an empty events view")
    overall_row = activity.loc[activity.grain == "overall"].iloc[0]
    overall = {key: int(overall_row[key]) for key in (
        "event_count", "active_users", "buyer_users", "pv_events", "cart_events", "fav_events", "buy_events"
    )}
    overall.update(
        observation_start=daily.date.min().strftime("%Y-%m-%d"),
        observation_end=daily.date.max().strftime("%Y-%m-%d"),
        buy_to_pv_event_ratio=(overall["buy_events"] / overall["pv_events"] if overall["pv_events"] else None),
        buyer_user_share=(overall["buyer_users"] / overall["active_users"] if overall["active_users"] else None),
    )
    daily.to_csv(out_dir / "daily_activity.csv", index=False)
    hourly.to_csv(out_dir / "hourly_activity.csv", index=False)
    _plot_activity(daily, hourly, out_dir / "activity_profile.png")

    cutoff_date = overall["observation_end"]
    counts = _strict_counts(con, cutoff_date)
    eligible = counts["eligible_pv_pairs"]
    for key in ("intent_pairs", "intent_purchase_pairs", "any_purchase_pairs",
                "no_confirmed_intent_purchase_pairs", "cart_pairs", "fav_pairs",
                "cart_purchase_pairs", "fav_purchase_pairs"):
        counts[key.replace("_pairs", "_pct")] = _pct(counts[key], eligible)
    counts["intent_to_purchase_pct"] = _pct(counts["intent_purchase_pairs"], counts["intent_pairs"])
    counts["no_confirmed_intent_share_of_purchases_pct"] = _pct(
        counts["no_confirmed_intent_purchase_pairs"], counts["any_purchase_pairs"]
    )
    counts["limitations"] = (
        "First observed PV, not first lifetime PV. Systematic user sample, not a randomized experiment. "
        "First PVs on the final observed date are excluded; assume all earlier dates are fully covered. "
        "Each subsequent stage must be strictly later and within 24 hours of first PV; same-second "
        "events do not establish order. Absence of confirmed intent is not proof of a direct path."
    )
    _funnel_table(counts).to_csv(out_dir / "strict_funnel.csv", index=False)
    _plot_funnel(counts, out_dir / "strict_funnel.png")
    files = ["daily_activity.csv", "hourly_activity.csv", "activity_profile.png", "strict_funnel.csv"]
    if eligible:
        files.append("strict_funnel.png")
    return {"overall": overall, "strict_funnel": counts, "files": files}
