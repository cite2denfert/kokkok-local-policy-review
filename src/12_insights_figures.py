"""인사이트 8개와 그림 1·2 (원본 CSV 필요). 결과는 output/ 에 저장한다(실제 수치가 들어 있어 공개 금지).

  python src/12_insights_figures.py            # 인사이트 표·metrics.json·그림 저장
  python src/12_insights_figures.py --no-fig   # 그림 없이 수치만

계산 정의는 09_make_evidence.py(근거 파일)와 11_threshold_decision.py(임계값 근거)와 같다.
인사이트 | 내용
  1 | 전국 외국인 결제금액과 외국인 비중의 인접 월 변화 방향
  2 | 대형할인점 결제가 없는 시군구(결제 없음은 하위 20%와 따로 표시)
  3 | 연령 구간별 편의점·슈퍼마켓 결제 비중
  4 | 외국인 결제 상위 50% 시군구의 업종 구성 3유형(R03)
  5 | 외국인 결제 비중의 지역 격차(R12 컷오프)
  6 | 전국 월 흐름과 지역 특이 지수, 규모별 흔들림과 R04 기준값
  7 | 성별·연령 미상 비중의 지역 차이
  8 | 일반한식 비중 분포, 군집 분리도와 유사 지자체 방식 선택
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import (AGES, INDS, MONTHS, DAYS, OUTPUT_DIR, load_config, load_script, read_raw, tidy,
                    setup_korean_font)

CFG = load_config()
TH = CFG['thresholds']
TD = load_script('11_threshold_decision')          # 임계값 근거 계산 함수 재사용
PEER_INDS = ['일반한식', '서양음식', '슈퍼마켓', '편의점', '대형할인점', '스넥', '제과점', '중국음식', '일식회집']


# ---------------------------------------------------------------- 1
def insight1_amount_vs_share(t):
    g = t.groupby('ym').agg(T=('amt', 'sum')).reindex(MONTHS)
    g['F'] = t[t.g == '3'].groupby('ym').amt.sum()
    g['U'] = t[t.g == 'x'].groupby('ym').amt.sum()
    g = g.fillna(0)
    g['foreign_share_pct'] = g.F / g['T'] * 100
    g['known_only_share_pct'] = g.F / (g['T'] - g.U) * 100
    cmp = pd.DataFrame({
        'period': [f'{a}->{b}' for a, b in zip(MONTHS[:-1], MONTHS[1:])],
        'foreign_amount_change_pct': (g.F.pct_change() * 100).values[1:],
        'foreign_share_change_pp': g.foreign_share_pct.diff().values[1:],
    })
    cmp['opposite'] = np.sign(cmp.foreign_amount_change_pct) * np.sign(cmp.foreign_share_change_pp) < 0
    m4, m5 = g.loc['2026-04'], g.loc['2026-05']
    summary = dict(opposite_periods=int(cmp.opposite.sum()), periods=len(cmp),
                   apr_to_may_amount_change_pct=float((m5.F / m4.F - 1) * 100),
                   apr_share_pct=float(m4.foreign_share_pct), may_share_pct=float(m5.foreign_share_pct),
                   smallest_share_change_pp=float(cmp.foreign_share_change_pp.abs().min()))
    return g, cmp, summary


# ---------------------------------------------------------------- 2
def industry_table(t):
    return t.pivot_table(index='key', columns='ind', values='amt', aggfunc='sum', fill_value=0).reindex(columns=INDS, fill_value=0)


def insight2_zero_hypermarket(t):
    ind = industry_table(t)
    sido = t.groupby('key').sido.first()
    zero = ind.index[ind['대형할인점'] == 0]
    by_sido = sido[zero].value_counts()
    summary = dict(zero_hypermarket_regions=int(len(zero)), share_of_regions=float(len(zero) / len(ind)),
                   no_payment_entries_all_industries=int((ind == 0).sum().sum()),
                   top_sido={k: int(v) for k, v in by_sido.head(5).items()})
    return by_sido, summary


# ---------------------------------------------------------------- 3
def insight3_age_industry(t):
    x = t[t.a.isin(AGES)]
    g = x.pivot_table(index='a', columns='ind', values='amt', aggfunc='sum', fill_value=0).reindex(index=AGES, columns=INDS, fill_value=0)
    sh = g.div(g.sum(axis=1), axis=0) * 100
    summary = {f'{i}_age{a}_pct': float(sh.loc[a, i]) for i in ['편의점', '슈퍼마켓'] for a in ['1', '6']}
    return sh, summary


# ---------------------------------------------------------------- 4
def foreign_mix(t, index):
    fo = t[t.g == '3'].pivot_table(index='key', columns='ind', values='amt', aggfunc='sum', fill_value=0).reindex(index=index, columns=INDS).fillna(0)
    tot = fo.sum(axis=1)
    life = (fo['편의점'] + fo['슈퍼마켓']) / tot.replace(0, np.nan) * 100
    kfood = fo['일반한식'] / tot.replace(0, np.nan) * 100
    return tot, life, kfood


def insight4_foreign_types(t, T):
    top = 1 - TH['R03_min_foreign_amount_percentile'] / 100
    cut = TH['R03_type_top_percentile'] / 100
    types = TD.r03_types(t, T.index, top, cut)
    tot, life, kfood = foreign_mix(t, T.index)
    elig = types != '없음'
    df = pd.DataFrame({'type': types, 'life_retail_pct': life, 'korean_food_pct': kfood, 'foreign_amount': tot})
    summary = dict(eligible_regions=int(elig.sum()),
                   types={k: int(v) for k, v in types[elig].value_counts().items()},
                   life_cut_pct=float(np.percentile(life[elig], 100 - TH['R03_type_top_percentile'])),
                   kfood_cut_pct=float(np.percentile(kfood[elig], 100 - TH['R03_type_top_percentile'])))
    return df, summary


# ---------------------------------------------------------------- 5
def insight5_foreign_share_gap(t, T):
    w = TD.foreign_share(t, T.index) * 100
    q = TH['R12_low_foreign_share_percentile']
    return w, dict(p10=float(w.quantile(.1)), median=float(w.median()), p90=float(w.quantile(.9)),
                   r12_cutoff_pct=float(np.percentile(w, q)), r12_low_regions=int((w <= np.percentile(w, q)).sum()))


# ---------------------------------------------------------------- 6
def insight6_season(t, T, idx):
    nat = t.groupby('ym').amt.sum().reindex(MONTHS)
    others = nat.drop('2026-05')
    daily = nat / pd.Series(DAYS)
    der = TD.r04_threshold_derivation(T, idx)
    alt = TD.r04_relative_alternative(idx)
    summary = dict(may_vs_other_months_pct=float((nat['2026-05'] / others.mean() - 1) * 100),
                   may_vs_other_months_daily_pct=float((daily['2026-05'] / daily.drop('2026-05').mean() - 1) * 100),
                   robust_sigma_by_band={r.band: round(r.robust_sigma, 4) for r in der['bands'].itertuples()},
                   derived_thresholds={r.band: r.rounded for r in der['bands'].itertuples()},
                   operating_thresholds=der['operating'],
                   r04_flagged_regions=der['flagged_regions'], r04_flagged_share=der['flagged_share'],
                   r04_flagged_by_group={g: v['flagged'] for g, v in der['by_group'].items()},
                   r04_flagged_by_month=der['flagged_by_month'],
                   rejected_same_month_top20_any_share=alt['any_month_share'])
    return der, summary


# ---------------------------------------------------------------- 7
def insight7_unknown(t, T):
    U = t[t.g == 'x'].groupby('key').amt.sum().reindex(T.index).fillna(0)
    u = U / T * 100
    return u, dict(min_pct=float(u.min()), max_pct=float(u.max()), median_pct=float(u.median()))


# ---------------------------------------------------------------- 8
def peer_features(t, T):
    """09_make_evidence.py 의 유사 지자체 특징: 업종 9 + 연령 5(코드 2~6) + 외국인 비중 + log 결제규모, 표준화."""
    ind = industry_table(t)
    ind_sh = ind.div(ind.sum(axis=1), axis=0) * 100
    age = t[t.a.isin(AGES)].pivot_table(index='key', columns='a', values='amt', aggfunc='sum', fill_value=0).reindex(index=T.index, columns=AGES, fill_value=0)
    age_sh = age.div(age.sum(axis=1), axis=0) * 100
    fs = TD.foreign_share(t, T.index) * 100
    X = pd.concat([ind_sh[PEER_INDS], age_sh[AGES[1:]], fs.rename('fs'), np.log(T).rename('logT')], axis=1)
    return (X - X.mean()) / X.std(ddof=0), ind_sh


def insight8_cluster(t, T, ks=(2, 3, 4, 5, 6)):
    Z, ind_sh = peer_features(t, T)
    kf = ind_sh['일반한식']
    sil = {}
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        for k in ks:
            lab = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(Z.values)
            sil[k] = float(silhouette_score(Z.values, lab))
    except ImportError:
        sil = {}
    return dict(korean_food_p10_pct=float(kf.quantile(.1)), korean_food_p90_pct=float(kf.quantile(.9)),
                silhouette=sil, silhouette_range=[min(sil.values()), max(sil.values())] if sil else None,
                decision='군집 분리도가 낮으면(실루엣 0.2 미만) 군집 대신 표준화 특징 공간의 최근접 이웃 5곳을 유사 지자체로 쓴다')


# ---------------------------------------------------------------- figures
COL = {'s1': '#2a78d6', 's2': '#eb6834', 's3': '#1baf7a', 'ink': '#0b0b0b', 'ink2': '#52514e', 'grid': '#e4e3df', 'muted': '#8a8984'}


def _style(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(COL['muted'])
    ax.tick_params(colors=COL['ink2'], labelsize=9)
    ax.grid(axis='y', color=COL['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(COL['ink'])


def make_figures(res, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    setup_korean_font()
    outdir.mkdir(parents=True, exist_ok=True)
    g, cmp = res['i1_table'], res['i1_cmp']
    sh3 = res['i3_table']
    der = res['i6_der']
    T = res['T']

    # 그림 1: (가) 금액과 비중의 방향 차이 (나) 연령별 생활유통 업종 비중 (다) 규모별 지수 흔들림과 R04 기준
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    x = [m[5:] + '월' for m in MONTHS]
    fi = g.F / g.F.iloc[0] * 100
    si = g.foreign_share_pct / g.foreign_share_pct.iloc[0] * 100
    ax.plot(x, fi, color=COL['s1'], lw=2, marker='o', ms=6, label='외국인 결제금액')
    ax.plot(x, si, color=COL['s2'], lw=2, marker='o', ms=6, label='외국인 결제 비중')
    ax.axhline(100, color=COL['muted'], lw=0.8, ls='--')
    ax.set_title('(가) 외국인 결제금액과 비중 (1월=100)', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    ax = axes[1]
    labels = ['20대 이하', '20대', '30대', '40대', '50대', '60대 이상']
    xx = np.arange(len(AGES))
    w = 0.38
    ax.bar(xx - w / 2 - 0.01, sh3['편의점'], w, color=COL['s1'], label='편의점')
    ax.bar(xx + w / 2 + 0.01, sh3['슈퍼마켓'], w, color=COL['s2'], label='슈퍼마켓')
    ax.set_xticks(xx, labels)
    ax.set_ylabel('연령 구간 결제 중 비중(%)', color=COL['ink2'], fontsize=9)
    ax.set_title('(나) 연령 구간별 편의점·슈퍼마켓 비중', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    ax = axes[2]
    flags, thr = der['flags'], der['thresholds']
    pct = T.rank(pct=True) * 100
    rng = np.random.default_rng(0)
    for k in flags.index:
        v = res['idx'].loc[k].values
        xs = np.full(len(v), pct[k])
        ax.scatter(xs + rng.uniform(-0.3, 0.3, len(v)), v, s=8, color=COL['muted'], alpha=0.35, linewidths=0)
        hit = v >= thr[k]
        if hit.any():
            ax.scatter(xs[hit], v[hit], s=16, color=COL['s2'], linewidths=0)
    for (lo, hi), y in [((0, TH['R04_very_small_percentile']), TH['R04_season_index_min_very_small']),
                        ((TH['R04_very_small_percentile'], TH['R04_small_percentile']), TH['R04_season_index_min_small']),
                        ((TH['R04_small_percentile'], 100), TH['R04_season_index_min'])]:
        ax.hlines(y, lo, hi, color=COL['s1'], lw=2)
        ax.text(hi - 1, y + 0.01, f'{y:.2f}', ha='right', va='bottom', fontsize=8, color=COL['ink2'])
    ax.set_xlabel('6개월 총결제 백분위(왼쪽이 소규모)', color=COL['ink2'], fontsize=9)
    ax.set_ylabel('지역 특이 지수(월)', color=COL['ink2'], fontsize=9)
    ax.set_title(f"(다) 규모별 지수 흔들림과 R04 기준 (발동 {der['flagged_regions']}곳)", fontsize=11, loc='left')
    _style(ax)
    fig.tight_layout()
    fig.savefig(outdir / 'figure1.png', dpi=180)
    plt.close(fig)

    # 그림 2: (라) 외국인 업종 구성 3유형 (마) 대형할인점 결제가 없는 시군구의 시도 분포
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    df, s4 = res['i4_df'], res['summary']['insight4']
    for name, c in [('생활유통형', COL['s1']), ('한식외식형', COL['s2']), ('복합형', COL['s3'])]:
        sub = df[df.type == name]
        ax.scatter(sub.life_retail_pct, sub.korean_food_pct, s=26, color=c, edgecolors='white', linewidths=1,
                   label=f'{name} {len(sub)}곳')
    ax.axvline(s4['life_cut_pct'], color=COL['muted'], ls='--', lw=1)
    ax.axhline(s4['kfood_cut_pct'], color=COL['muted'], ls='--', lw=1)
    ax.set_xlabel('외국인 결제 중 편의점+슈퍼마켓 비중(%)', color=COL['ink2'], fontsize=9)
    ax.set_ylabel('외국인 결제 중 일반한식 비중(%)', color=COL['ink2'], fontsize=9)
    ax.set_title(f"(라) 외국인 결제 상위 50% {s4['eligible_regions']}곳의 업종 구성", fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    ax = axes[1]
    by_sido = res['i2_by_sido'].sort_values()
    ax.barh(by_sido.index, by_sido.values, color=COL['s1'], height=0.6)
    for yv, v in enumerate(by_sido.values):
        ax.text(v + 0.2, yv, str(int(v)), va='center', fontsize=8, color=COL['ink2'])
    ax.set_xlabel('시군구 수', color=COL['ink2'], fontsize=9)
    ax.set_title(f"(마) 대형할인점 결제가 없는 시군구 {int(by_sido.sum())}곳의 시도 분포", fontsize=11, loc='left')
    _style(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=COL['grid'], linewidth=0.8)
    fig.tight_layout()
    fig.savefig(outdir / 'figure2.png', dpi=180)
    plt.close(fig)
    return [outdir / 'figure1.png', outdir / 'figure2.png']


# ---------------------------------------------------------------- run
def run_all(raw=None, outdir=OUTPUT_DIR, figures=True):
    t = tidy(read_raw(raw))
    T, idx = TD.base_tables(t)
    g, cmp, s1 = insight1_amount_vs_share(t)
    by_sido, s2 = insight2_zero_hypermarket(t)
    sh3, s3 = insight3_age_industry(t)
    df4, s4 = insight4_foreign_types(t, T)
    w5, s5 = insight5_foreign_share_gap(t, T)
    der, s6 = insight6_season(t, T, idx)
    u7, s7 = insight7_unknown(t, T)
    s8 = insight8_cluster(t, T)
    repro = TD.reproducibility_summary(t, T)
    summary = dict(insight1=s1, insight2=s2, insight3=s3, insight4=s4, insight5=s5, insight6=s6, insight7=s7, insight8=s8,
                   reproducibility=repro, reproducibility_reported=CFG['reproducibility']['reported'])
    res = dict(T=T, idx=idx, i1_table=g, i1_cmp=cmp, i2_by_sido=by_sido, i3_table=sh3, i4_df=df4, i6_der=der, summary=summary)
    outdir.mkdir(parents=True, exist_ok=True)
    g.to_csv(outdir / 'insight1_national_monthly.csv', encoding='utf-8-sig')
    cmp.to_csv(outdir / 'insight1_comparison.csv', index=False, encoding='utf-8-sig')
    by_sido.rename('regions').to_csv(outdir / 'insight2_zero_hypermarket_by_sido.csv', encoding='utf-8-sig')
    sh3.to_csv(outdir / 'insight3_age_industry_share.csv', encoding='utf-8-sig')
    der['bands'].to_csv(outdir / 'insight6_r04_bands.csv', index=False, encoding='utf-8-sig')
    (outdir / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    if figures:
        res['figures'] = make_figures(res, outdir)
    return res


def print_summary(s):
    rep = s['reproducibility_reported']
    i = s
    print('1. 금액·비중 반대 방향:', f"{i['insight1']['opposite_periods']}/{i['insight1']['periods']}구간,",
          f"4→5월 금액 {i['insight1']['apr_to_may_amount_change_pct']:+.1f}%, 비중 {i['insight1']['apr_share_pct']:.2f}%→{i['insight1']['may_share_pct']:.2f}%")
    print('2. 대형할인점 결제 없는 시군구:', i['insight2']['zero_hypermarket_regions'], f"({i['insight2']['share_of_regions']:.0%})", i['insight2']['top_sido'])
    print('3. 편의점 20대 이하→60대 이상:', f"{i['insight3']['편의점_age1_pct']:.1f}%→{i['insight3']['편의점_age6_pct']:.1f}%,",
          '슈퍼마켓', f"{i['insight3']['슈퍼마켓_age1_pct']:.1f}%→{i['insight3']['슈퍼마켓_age6_pct']:.1f}%")
    print('4. R03 대상', i['insight4']['eligible_regions'], '곳 유형:', i['insight4']['types'])
    print('5. 외국인 비중 p10/중앙/p90:', f"{i['insight5']['p10']:.1f}% / {i['insight5']['median']:.1f}% / {i['insight5']['p90']:.1f}%,",
          f"R12 컷오프 {i['insight5']['r12_cutoff_pct']:.1f}% ({i['insight5']['r12_low_regions']}곳)")
    print('6. 5월 다른 달 평균 대비:', f"{i['insight6']['may_vs_other_months_pct']:+.1f}% (일평균 {i['insight6']['may_vs_other_months_daily_pct']:+.1f}%)",
          '| 견고σ', i['insight6']['robust_sigma_by_band'], '| 1+2.5σ', i['insight6']['derived_thresholds'],
          '| R04 발동', i['insight6']['r04_flagged_regions'], f"({i['insight6']['r04_flagged_share']:.0%})")
    print('7. 미상 비중 범위:', f"{i['insight7']['min_pct']:.2f}% ~ {i['insight7']['max_pct']:.1f}%")
    s8 = i['insight8']
    print('8. 일반한식 p10/p90:', f"{s8['korean_food_p10_pct']:.1f}% / {s8['korean_food_p90_pct']:.1f}%", '| 실루엣', {k: round(v, 3) for k, v in s8['silhouette'].items()})
    print('반기 재현성(재계산 / 보고):', {k: f"{v:.2f} / {rep[k]}" for k, v in i['reproducibility'].items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', help='원본 CSV 경로')
    ap.add_argument('--out', default=str(OUTPUT_DIR))
    ap.add_argument('--no-fig', action='store_true')
    a = ap.parse_args()
    from pathlib import Path
    res = run_all(a.raw, Path(a.out), figures=not a.no_fig)
    print_summary(res['summary'])
    print('저장:', a.out)
