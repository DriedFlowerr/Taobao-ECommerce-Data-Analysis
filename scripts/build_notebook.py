"""Generate the guided review notebook and optionally embed its actual outputs."""

import argparse
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = [nbf.v4.new_markdown_cell('''# 淘宝用户行为分析：转化诊断、购买倾向预测与运营决策

业务问题：识别真实的行为路径与转化瓶颈；评估历史行为和未来购买的关联；在有限触达预算下优先筛选可能购买的活跃用户。

数据：2017-11-25 至 2017-12-03（北京时间），四类事件 `pv / cart / fav / buy`。本 Notebook 默认展示完整实验已生成的结果。需要重新计算时，将 `RECOMPUTE` 改为 `True` 后从第一格开始运行。核心实现位于 `src/`。''')]

    def section(title, code):
        cells.append(nbf.v4.new_markdown_cell(f'## {title}'))
        cells.append(nbf.v4.new_code_cell(code.strip()))

    section('0. 环境与路径', '''
from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
from run_analysis import main, ROOT, OUT

RECOMPUTE = False

def read_csv(name):
    return pd.read_csv(OUT / name)

def read_json(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))

def show_plot(name):
    display(Image(filename=str(OUT / name)))

print('Project:', ROOT)
print('Recompute:', RECOMPUTE)
''')

    section('1. 数据口径与质量审计', '''
if RECOMPUTE:
    main('data')

audit = read_json('source_audit.json')
display(pd.Series(audit, name='value').to_frame())
reconciliation = read_csv('source_reconciliation.csv')
display(reconciliation.head(8))
assert reconciliation['difference'].eq(0).all()
''')

    section('2. 全量业务指标与时间分布', '''
if RECOMPUTE:
    main('funnel')

summary = read_json('funnel_summary.json')
overall = summary['overall']
display(pd.Series(overall, name='value').to_frame())
daily = read_csv('daily_activity.csv')
daily['buyer_share_pct'] = 100 * daily.buyer_user_share
display(daily[['date', 'event_count', 'active_users', 'buyer_users', 'buyer_share_pct']].round(2))
show_plot('activity_profile.png')
''')

    section('3. 同用户、同商品的严格 24 小时漏斗', '''
strict = summary['strict_funnel']
funnel = read_csv('strict_funnel.csv')
display(funnel[['stage', 'user_item_pairs', 'eligible_share_pct']].round(4))
print('Sampled users:', f"{strict['sample_users']:,}")
print('Intent → buy:', f"{strict['intent_to_purchase_pct']:.2f}%")
print('No confirmed intermediate intent among purchases:',
      f"{strict['no_confirmed_intent_share_of_purchases_pct']:.2f}%")
show_plot('strict_funnel.png')
''')

    section('4. 预测样本与时间切分', '''
cohorts = read_csv('snapshot_cohorts.csv')
cohorts['buyer_share_pct'] = 100 * cohorts.buyer_share
display(cohorts.drop(columns='buyer_share').round(2))
print('Total labeled user-day snapshots:', f"{cohorts.eligible_users.sum():,}")
print('Feature window: previous 4 days; outcome: score date')
''')

    section('5. 可解释行为分层', '''
segments = read_csv('behavior_segments.csv')
for name in ['buyer_share', 'share_of_users', 'share_of_buyers']:
    segments[name + '_pct'] = 100 * segments[name]
display(segments[['segment', 'users', 'buyers', 'buyer_share_pct',
                  'share_of_users_pct', 'share_of_buyers_pct']].round(2))
''')

    section('6. 假设检验与分层标准化', '''
if RECOMPUTE:
    main('inference')

association = read_json('association_summary.json')
groups = read_csv('association_groups.csv')
display(groups.style.format({'purchase_rate': '{:.2%}', 'ci_low': '{:.2%}', 'ci_high': '{:.2%}'}))
effects = read_csv('association_effects.csv')
for name in ['risk_difference', 'ci_low', 'ci_high']:
    effects[name + '_pp'] = 100 * effects[name]
display(effects[['estimate', 'risk_difference_pp', 'ci_low_pp', 'ci_high_pp', 'p_value']].style.format(
    {'risk_difference_pp': '{:.2f}', 'ci_low_pp': '{:.2f}', 'ci_high_pp': '{:.2f}', 'p_value': '{:.2e}'}))
show_plot('association_cart.png')
''')

    section('7. 时间外预测与模型解释', '''
if RECOMPUTE:
    main('model')

model = read_json('model_summary.json')
metrics = read_csv('model_metrics.csv')
print('Selected on validation AP:', model['selected_model'])
display(metrics[['model', 'fold', 'ap', 'roc_auc', 'brier', 'base_rate', 'n']].round(4))
display(read_csv('model_importance.csv').head(8).round(4))
show_plot('model_diagnostics.png')
''')

    section('8. 相同触达预算下的离线运营决策', '''
policy = read_csv('targeting_policy.csv')
top10 = policy.loc[policy.budget_fraction.eq(.10)].copy()
display(top10[['policy', 'n_selected', 'hits', 'precision', 'recall', 'lift']].round(4))
selected = top10.loc[top10.selected_model].iloc[0]
rule = top10.loc[top10.policy.eq('Recent cart rule')].iloc[0]
print(f"Selected-model precision gain vs cart rule: {100*(selected.precision-rule.precision):.2f} pp")
print(f"Additional naturally observed purchasers captured: {int(selected.hits-rule.hits):,}")
display(policy.pivot(index='budget_fraction', columns='policy', values='lift').round(3))
''')

    section('9. 前瞻 A/B 实验设计', '''
ab = read_csv('ab_sample_size_plan.csv')
display(ab.style.format({'historical_targeted_purchase_rate': '{:.2%}',
                         'hypothetical_mde_pp': '{:.1f}', 'alpha_two_sided': '{:.2f}', 'power': '{:.0%}'}))
''')

    section('10. 结果闭环与边界', '''
best = metrics.loc[(metrics.fold.eq('test')) & (metrics.model.eq(model['selected_model']))].iloc[0]
standardized = association['standardized']
display(Markdown(f"""- 清洗后 **{audit['clean_events']:,}** 条事件、**{overall['active_users']:,}** 名活跃用户；事件比值与有序路径采用不同口径。
- **{strict['eligible_pv_pairs']:,}** 个具有完整观测期的用户—商品对中，24 小时内购买占 **{strict['any_purchase_pct']:.2f}%**；有意向后购买占全部合格 PV 对的 **{strict['intent_purchase_pct']:.2f}%**。
- 历史加购与未来一天购买的分层标准化率差为 **{standardized['risk_difference']*100:.2f} pp**（95% 近似区间 {standardized['ci_low']*100:.2f}–{standardized['ci_high']*100:.2f} pp），仍是关联而非因果效应。
- 最终模型测试集 AP **{best.ap:.4f}**（基准率 {best.base_rate:.2%}），ROC-AUC {best.roc_auc:.4f}；在 Top 10% 预算下命中率 **{selected.precision:.2%}**，相对于随机为 **{selected.lift:.2f}×**，相对加购规则高 **{(selected.precision-rule.precision)*100:.2f} pp**。
- 建议：把模型排序作为触达候选筛选方案，并在随机对照实验中检验增量购买。"""))
''')
    return nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'file_extension': '.py'},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Embed outputs from the actual local experiment')
    args = parser.parse_args()
    notebook = build()
    path = ROOT / 'analysis_v2.ipynb'
    if args.execute:
        from nbclient import NotebookClient
        NotebookClient(notebook, timeout=180, resources={'metadata': {'path': str(ROOT)}}).execute()
    nbf.write(notebook, path)
    print(path)
