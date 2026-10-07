"""임계값 결정 근거 계산 (원본 CSV 필요, 외부 인구 자료는 선택). 출력은 집계값만.

docs/임계값_결정표.md 의 숫자(반기 재현성, R03 유형 분포·일치율, R04 규모 구간별 흔들림과 기준값, R12 컷오프, R11 발동 비율)를
이 파일로 다시 계산한다. 탐색 단계의 print 스크립트를 함수로 나눠 노트북·검증 스크립트에서도 불러 쓸 수 있게 했다.

  python src/11_threshold_decision.py
"""
import numpy as np
import pandas as pd
from common import read_raw, tidy, load_config, EXT_DIR, MONTHS, DAYS, CHECK_INDS

CFG = load_config()
TH = CFG['thresholds']
H1 = set(CFG['reproducibility']['split_1'])
H2 = set(CFG['reproducibility']['split_2'])


def jac(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def base_tables(t):
    """t: common.tidy() 결과 -> 시군구 총액 T, 지역 특이 지수 idx_adj (05_build_features.py 와 같은 정의)."""
    T = t.groupby('key').amt.sum().rename('T')
    m = t.pivot_table(index='key', columns='ym', values='amt', aggfunc='sum', fill_value=0).reindex(columns=MONTHS, fill_value=0)
    m_pd = m.div(pd.Series(DAYS), axis=1)                 # 일평균
    idx_raw = m_pd.div(m_pd.mean(axis=1), axis=0)         # 지역 자기 평균 대비
    nat = m_pd.sum(axis=0)
    idx_adj = idx_raw.div(nat / nat.mean(), axis=1)       # 전국 월 흐름 제거
    return T, idx_adj


def _halves(t):
    return t[t.ym.isin(H1)], t[t.ym.isin(H2)]


# ---------------------------------------------------------------- A1. R02
def r02_reproducibility(t, T, qs=(10, 15, 20, 25, 30, 40)):
    """업종 비중 하위 q% 시군구 집합의 반기(1~3월 vs 4~6월) Jaccard. 9개 업종 평균과 최저."""
    def share(x):
        p = x.pivot_table(index='key', columns='ind', values='amt', aggfunc='sum', fill_value=0).reindex(T.index).fillna(0)
        return p.div(p.sum(axis=1), axis=0)
    a, b = _halves(t)
    s1, s2 = share(a), share(b)
    out = {}
    for q in qs:
        js = [jac(s1[i] <= np.percentile(s1[i], q), s2[i] <= np.percentile(s2[i], q)) for i in CHECK_INDS if i in s1 and i in s2]
        out[q] = dict(mean=float(np.mean(js)), min=float(min(js)))
    return out


# ---------------------------------------------------------------- A2. R08
def r08_reproducibility(t, T, qs=(10, 20, 30)):
    """업종별 연령 구성비(연령 2~6) 하위 q% 시군구 집합의 반기 Jaccard 평균. 양쪽 모두 값이 있는 시군구가 50곳 미만인 쌍은 뺀다."""
    def ia(x):
        x = x[x.a != 'x']
        out = {}
        for i in CHECK_INDS:
            p = x[x.ind == i].pivot_table(index='key', columns='a', values='amt', aggfunc='sum', fill_value=0).reindex(T.index)
            out[i] = p.div(p.sum(axis=1), axis=0)
        return out
    a, b = _halves(t)
    a1, a2 = ia(a), ia(b)
    res = {}
    for q in qs:
        js = []
        for i in a1:
            for ag in ['2', '3', '4', '5', '6']:
                if ag not in a1[i] or ag not in a2[i]:
                    continue
                x, y = a1[i][ag], a2[i][ag]
                ok = x.notna() & y.notna()
                if ok.sum() < 50:
                    continue
                js.append(jac((x <= np.nanpercentile(x, q)) & ok, (y <= np.nanpercentile(y, q)) & ok))
        res[q] = dict(mean=float(np.mean(js)) if js else float('nan'), n_pairs=len(js),
                      min=float(min(js)) if js else float('nan'), max=float(max(js)) if js else float('nan'))
    return res


# ---------------------------------------------------------------- B·C. R04
def r04_sigma_by_decile(T, idx):
    """규모 10분위별 지역 특이 지수의 표준편차(월-지역 값 전체)."""
    dec = pd.qcut(T, 10, labels=False)
    return idx.groupby(dec).apply(lambda z: float(z.values.std())).to_dict()


def robust_sigma(v):
    v = np.asarray(v, float).ravel()
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med))), float(med)


def r04_threshold_derivation(T, idx):
    """R04 기준값 도출: 규모 구간별 견고 표준편차 -> 1 + 2.5σ -> 운영값(설정 파일)과 발동 지역 수.

    구간은 6개월 총결제 기준 하위 0~10, 10~20, 20~30, 30~50, 50~100%.
    운영값은 규모 하위 10% 이하 1.30, 10~30% 1.15, 나머지 1.10(설정 파일 thresholds).
    """
    cfg = CFG['rule_settings']['R04_season']
    bands = cfg['sigma_bands_percentile']
    k = cfg['multiplier']
    cuts = [np.percentile(T, p) for p in bands]
    rows = []
    for lo, hi, clo, chi in zip(bands[:-1], bands[1:], cuts[:-1], cuts[1:]):
        sel = (T > clo) & (T <= chi) if lo > 0 else (T <= chi)
        sig, med = robust_sigma(idx[sel].values)
        rows.append(dict(band=f'{lo}-{hi}%', n_regions=int(sel.sum()), median=med, robust_sigma=sig,
                         raw_threshold=1 + k * sig, rounded=round(1 + k * sig, 2)))
    # 운영 기준 적용
    vs_cut = np.percentile(T, TH['R04_very_small_percentile'])
    s_cut = np.percentile(T, TH['R04_small_percentile'])
    thr = pd.Series(TH['R04_season_index_min'], index=T.index)
    thr[T <= s_cut] = TH['R04_season_index_min_small']
    thr[T <= vs_cut] = TH['R04_season_index_min_very_small']
    flag = idx.ge(thr.reindex(idx.index), axis=0)
    any_flag = flag.any(axis=1)
    grp = pd.Series('rest', index=T.index)
    grp[T <= s_cut] = 'small'
    grp[T <= vs_cut] = 'very_small'
    by_group = {g: dict(n=int((grp == g).sum()), flagged=int(any_flag[grp == g].sum()),
                        share=float(any_flag[grp == g].mean())) for g in ['very_small', 'small', 'rest']}
    return dict(bands=pd.DataFrame(rows), operating={'very_small(<=10%)': TH['R04_season_index_min_very_small'],
                                                      'small(10~30%)': TH['R04_season_index_min_small'],
                                                      'rest(>30%)': TH['R04_season_index_min']},
                flagged_regions=int(any_flag.sum()), flagged_share=float(any_flag.mean()), by_group=by_group,
                flagged_by_month=flag.sum().astype(int).to_dict(), flags=flag, thresholds=thr)


def r04_relative_alternative(idx, top=20):
    """폐기한 대안: 같은 달 상위 top% 를 발동으로 볼 때 월별 발동 수와 한 번이라도 걸리는 비율."""
    fl = pd.DataFrame({c: idx[c] >= np.percentile(idx[c], 100 - top) for c in idx.columns})
    return dict(per_month=fl.sum().astype(int).to_dict(), any_month_share=float(fl.any(axis=1).mean()))


def r04_combo_table(T, idx):
    """이전 2구간 후보 조합별 발동(탐색 기록)."""
    small = T <= T.quantile(.2)
    rows = []
    for hi, lo in [(1.10, 1.25), (1.12, 1.25), (1.15, 1.25), (1.10, 1.20), (1.15, 1.30), (1.20, 1.30)]:
        thr = np.where(small, lo, hi)
        fl = pd.DataFrame({c: idx[c].values >= thr for c in idx.columns}, index=idx.index)
        cons = (fl.astype(int).values[:, 1:] & fl.astype(int).values[:, :-1]).any(axis=1)
        rows.append(dict(hi=hi, lo=lo, avg_per_month=round(fl.sum().mean(), 1), any_share=round(fl.any(axis=1).mean(), 3),
                         consecutive2=int(cons.sum()), small_share=round(fl[small].any(axis=1).mean(), 2),
                         rest_share=round(fl[~small].any(axis=1).mean(), 2)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- D. R12
def foreign_share(x, index):
    tt = x.groupby('key').amt.sum().reindex(index).fillna(0)
    F = x[x.g == '3'].groupby('key').amt.sum().reindex(index).fillna(0)
    return F / tt


def r12_reproducibility(t, T, qs=(10, 15, 20, 25, 30)):
    a, b = _halves(t)
    fa, fb, w = foreign_share(a, T.index), foreign_share(b, T.index), foreign_share(t, T.index)
    return {q: dict(jaccard=float(jac(fa <= np.percentile(fa, q), fb <= np.percentile(fb, q))),
                    cutoff_pct=float(np.percentile(w, q) * 100), n_low=int((w <= np.percentile(w, q)).sum())) for q in qs}


# ---------------------------------------------------------------- E. R03
def r03_types(x, index, top=0.5, cut=0.3):
    """외국인 결제 상위 top 지역 중 편의점+슈퍼 비중 상위 cut -> 생활유통형, 아니면 일반한식 상위 cut -> 한식외식형, 나머지 복합형.
    09_make_evidence.py 와 같은 순서(생활유통 우선)."""
    fo = x[x.g == '3'].pivot_table(index='key', columns='ind', values='amt', aggfunc='sum', fill_value=0).reindex(index).fillna(0)
    for c in ['슈퍼마켓', '편의점', '일반한식']:
        if c not in fo:
            fo[c] = 0
    tot = fo.sum(axis=1)
    life = (fo['슈퍼마켓'] + fo['편의점']) / tot.replace(0, np.nan)
    kr = fo['일반한식'] / tot.replace(0, np.nan)
    sel = tot >= tot.quantile(1 - top)
    t = pd.Series('없음', index=index)
    t[sel] = '복합형'
    t[sel & (kr >= kr[sel].quantile(1 - cut))] = '한식외식형'
    t[sel & (life >= life[sel].quantile(1 - cut))] = '생활유통형'
    return t


def r03_reproducibility(t, T, combos=((0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.4, 0.3), (0.6, 0.3))):
    a, b = _halves(t)
    out = {}
    for top, cut in combos:
        ta, tb, tc = r03_types(a, T.index, top, cut), r03_types(b, T.index, top, cut), r03_types(t, T.index, top, cut)
        both = (ta != '없음') & (tb != '없음')
        out[(top, cut)] = dict(distribution={k: int(v) for k, v in tc[tc != '없음'].value_counts().items()},
                               agreement=float((ta[both] == tb[both]).mean()) if both.any() else float('nan'), n=int(both.sum()))
    return out


# ---------------------------------------------------------------- F. R11
def r11_population(T, pop_path=EXT_DIR / 'ext_population.csv'):
    if not pop_path.exists():
        return None
    P = pd.read_csv(pop_path, encoding='utf-8-sig').set_index('region_key').pop_total.reindex(T.index)
    rows = []
    for V in [5000, 10000, 30000, 50000, 100000, 300000]:
        rows.append(dict(visitors=V, ratio_ge_1=int((V / P >= 1.0).sum()), ratio_ge_0_5=int((V / P >= 0.5).sum()),
                         ratio_ge_0_2=int((V / P >= 0.2).sum())))
    return dict(pop_quantiles=P.quantile([.1, .25, .5, .75, .9]).round(0).tolist(),
                per_capita_6m=(T / P).quantile([.1, .5, .9]).round(0).tolist(), table=pd.DataFrame(rows))


def reproducibility_summary(t, T):
    """결정표·요약서에 쓴 4개 값(R02·R12·R08 하위 20%, R03 상위 50%·30%)."""
    q = TH['R02_R08_low_percentile']
    top = 1 - TH['R03_min_foreign_amount_percentile'] / 100
    cut = TH['R03_type_top_percentile'] / 100
    return dict(R02=r02_reproducibility(t, T, (q,))[q]['mean'],
                R12=r12_reproducibility(t, T, (TH['R12_low_foreign_share_percentile'],))[TH['R12_low_foreign_share_percentile']]['jaccard'],
                R08=r08_reproducibility(t, T, (q,))[q]['mean'],
                R03=r03_reproducibility(t, T, ((top, cut),))[(top, cut)]['agreement'])


def main():
    pd.set_option('display.width', 220)
    t = tidy(read_raw())
    T, idx = base_tables(t)
    print('== A1. R02 업종 하위 q% 반기 재현성(H1 vs H2 Jaccard, 9업종 평균/최저) ==')
    for q, v in r02_reproducibility(t, T).items():
        print(q, round(v['mean'], 2), 'min', round(v['min'], 2))
    print('== A2. R08 업종x연령 하위 q% 반기 재현성(평균 Jaccard, 연령 2~6) ==')
    for q, v in r08_reproducibility(t, T).items():
        print(q, round(v['mean'], 2), 'n쌍', v['n_pairs'], '범위', round(v['min'], 2), '~', round(v['max'], 2))
    print('== B. 규모 10분위별 R04 지수 산포(월-지역 표준편차) ==')
    print({k: round(v, 3) for k, v in r04_sigma_by_decile(T, idx).items()})
    print('T 경계(억):', (T.quantile(np.arange(0, 1.01, .1)) / 1e8).round(0).tolist())
    print('== C. R04 기준값 도출(규모 구간별 견고 표준편차, 1 + 2.5σ) ==')
    d = r04_threshold_derivation(T, idx)
    print(d['bands'].round(3).to_string(index=False))
    print('운영값:', d['operating'])
    print('1개월 이상 발동:', d['flagged_regions'], f"({d['flagged_share']:.0%})", '| 구간별:',
          {g: f"{v['flagged']}/{v['n']} ({v['share']:.0%})" for g, v in d['by_group'].items()})
    print('월별 발동 지역 수:', d['flagged_by_month'])
    alt = r04_relative_alternative(idx)
    print('폐기한 대안(같은 달 상위 20%): 월별', alt['per_month'], '| 한 번 이상', f"{alt['any_month_share']:.0%}")
    print('== C2. 이전 2구간 후보 조합별 발동 ==')
    print(r04_combo_table(T, idx).to_string(index=False))
    print('== D. R12 외국인 비중 하위 q% 반기 재현성 ==')
    for q, v in r12_reproducibility(t, T).items():
        print(q, 'H1vsH2', round(v['jaccard'], 2), '| 컷오프 외국인비중 %', round(v['cutoff_pct'], 2), '| 대상', v['n_low'])
    print('== E. R03 유형 반기 일치율(top·cut 조합) ==')
    for (top, cut), v in r03_reproducibility(t, T).items():
        print(top, cut, '전체 분포', v['distribution'], '반기 일치율', round(v['agreement'], 2), 'n', v['n'])
    print('== F. R11 인구 분포 ==')
    f = r11_population(T)
    if f is None:
        print('data/external/ext_population.csv 가 없어 건너뜀')
    else:
        print('인구 분위(명) p10/p25/p50/p75/p90:', f['pop_quantiles'])
        print('1인당 6개월 결제 p10/p50/p90(원):', f['per_capita_6m'])
        print(f['table'].to_string(index=False))
    print('== 요약: 반기 재현성(재계산 vs 보고값) ==')
    s = reproducibility_summary(t, T)
    rep = CFG['reproducibility']['reported']
    for k in ['R02', 'R12', 'R08', 'R03']:
        print(k, round(s[k], 3), '| 보고값', rep[k])


if __name__ == '__main__':
    main()
