"""Small deterministic tests: no dependency on the 3.67 GB source file."""

import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.data import build_snapshots, load_snapshots
from src.funnel import _strict_counts
from src.inference import _self_test
from src.modeling import _policy_metrics
from src.experiment import plan_trial


class CoreTests(unittest.TestCase):
    def test_features_cannot_see_target_day(self):
        con = duckdb.connect()
        try:
            con.execute('CREATE TABLE events(user_id BIGINT, action_date DATE, behavior_type VARCHAR)')
            con.executemany('INSERT INTO events VALUES (?,?,?)', [
                (5, '2017-11-25', 'pv'), (5, '2017-11-28', 'cart'),
                (5, '2017-11-29', 'buy'), (10, '2017-12-03', 'buy'),
                (11, '2017-11-28', 'pv'),
            ])
            with tempfile.TemporaryDirectory() as tmp:
                df = load_snapshots(build_snapshots(con, Path(tmp)))
                row = df.loc[df.score_date.eq(pd.Timestamp('2017-11-29'))].iloc[0]
                self.assertEqual((row.user_id, row.y, row.buy_4d, row.cart_4d), (5, 1, 0, 1))
                self.assertEqual(row.pv_4d, 1)
                self.assertEqual(row.active_days_4d, 2)
                self.assertEqual(set(df.user_id), {5})
                self.assertEqual(len(df), 5)
                # Rerun should faithfully replace the cached artifacts.
                self.assertEqual(len(load_snapshots(build_snapshots(con, Path(tmp)))), 5)
        finally:
            con.close()

    def test_order_window_and_same_item(self):
        con = duckdb.connect()
        con.execute("SET TimeZone='Asia/Shanghai'")
        con.execute('CREATE TABLE events(user_id BIGINT, item_id BIGINT, behavior_type VARCHAR, action_time TIMESTAMPTZ)')
        con.executemany('INSERT INTO events VALUES (?,?,?,?)', [
            (10, 1, 'pv', '2017-12-01 00:00:00+08'),
            (10, 1, 'cart', '2017-12-01 01:00:00+08'),
            (10, 1, 'buy', '2017-12-02 00:00:00+08'),  # inclusive 24-hour end
            (10, 2, 'pv', '2017-12-01 00:00:00+08'),
            (10, 2, 'cart', '2017-12-01 00:00:00+08'),  # tie cannot confirm order
            (10, 2, 'buy', '2017-12-01 03:00:00+08'),
            (10, 3, 'pv', '2017-12-01 00:00:00+08'),
            (10, 3, 'buy', '2017-12-02 00:00:01+08'),  # outside 24h
            (10, 4, 'pv', '2017-12-03 00:00:00+08'),  # incomplete follow-up
            (11, 1, 'pv', '2017-12-01 00:00:00+08'),  # outside sampling rule
        ])
        try:
            result = _strict_counts(con, '2017-12-03')
            self.assertEqual(result['eligible_pv_pairs'], 3)
            self.assertEqual(result['intent_pairs'], 1)
            self.assertEqual(result['intent_purchase_pairs'], 1)
            self.assertEqual(result['any_purchase_pairs'], 2)
            self.assertEqual(result['no_confirmed_intent_purchase_pairs'], 1)
        finally:
            con.close()

    def test_association_boundaries(self):
        _self_test()

    def test_targeting_random_reference(self):
        y = np.array([1, 0, 1, 0, 0])
        random = _policy_metrics(y, None, 'random')
        selected = _policy_metrics(y, np.arange(5), 'rule')
        self.assertTrue(all(np.isclose(x['lift'], 1.0) for x in random))
        self.assertEqual(random[0]['n_selected'], selected[0]['n_selected'])
        self.assertEqual(selected[0]['hits'], 1)

    def test_power_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            targets = pd.DataFrame({'selected_model': [True], 'budget_fraction': [.1], 'precision': [.25]})
            sizes = plan_trial(targets, Path(tmp))
            self.assertEqual(len(sizes), 3)
            self.assertTrue(sizes.users_per_arm.is_monotonic_decreasing)


if __name__ == '__main__':
    unittest.main()
