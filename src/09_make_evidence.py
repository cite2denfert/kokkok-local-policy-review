"""근거 파일 스키마 2.1 생성기.

사용:
  python src/09_make_evidence.py real       # BC 원본 -> private/review_evidence_real_v2.1.json (공개 금지)
  python src/09_make_evidence.py synthetic  # 가짜 세계에서 -> examples/review_evidence_demo_v2.1.json (공개 가능)

두 모드는 같은 build_evidence()를 쓰므로 구조가 똑같다. 스키마 2.0의 기존 키(records, evidence_id, data_kind,
scope, region_basis, applicability, amount_unit, months, limitations)는 뜻을 바꾸지 않고, 새 키만 더한다.
임계값은 config/rules_thresholds.json 의 thresholds 에서 읽는다(이 파일에 숫자를 직접 쓰지 않는다).
"""
import sys, json, math, argparse
from pathlib import Path
import numpy as np, pandas as pd
from common import load_config, read_raw, EXT_DIR, PRIVATE_DIR, EXAMPLES_DIR

CFG = load_config()
MONTHS = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06']
DAYS = dict(zip(MONTHS, [31, 28, 31, 30, 31, 30]))
AGES = ['1', '2', '3', '4', '5', '6']
AGE_LABEL = {'1': '20대 이하', '2': '20대', '3': '30대', '4': '40대', '5': '50대', '6': '60대 이상'}
IND_NAME = {'4004': '대형할인점', '4010': '편의점', '4020': '슈퍼마켓', '8001': '일반한식', '8002': '갈비전문점',
            '8003': '한정식', '8004': '일식회집', '8005': '중국음식', '8006': '서양음식', '8021': '스넥', '8301': '제과점'}
INDS = list(IND_NAME.values())
PEER_INDS = ['일반한식', '서양음식', '슈퍼마켓', '편의점', '대형할인점', '스넥', '제과점', '중국음식', '일식회집']
PCTS = [10, 20, 30, 50, 70, 80, 90]

# 임계값(운영 기준). 설정 파일에서 읽고, 근거 파일 안에도 실어 개발자가 설정과 대조할 수 있게 한다.
THRESHOLDS = dict(CFG['thresholds'])
REAL_VERSION = CFG['dataset_versions']['real']
DEMO_VERSION = CFG['dataset_versions']['demo']


def r(x, n=6):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), n)


# ---------------------------------------------------------------- tables
def tables_from_raw(d):
    """BC 원본(행 단위) -> 시군구별 집계 표."""
    d = d.copy()
    d['key'] = d.SIDO_NM + ' ' + d.CCG_NM
    d['g'] = d.GENDER_CD.astype(str).str.lower()
    d['a'] = d.AGE_CD.astype(str).str.lower()
    d['ind'] = d.TP_BUZ_NO.astype(str).map(IND_NAME)
    d['ym'] = d.STRD_YYMM.astype(str).str[:4] + '-' + d.STRD_YYMM.astype(str).str[4:]
    return build_tables(d[['key', 'ym', 'g', 'a', 'ind', 'amt', 'cnt']])


def build_tables(d):
    """d: key, ym, g, a, ind, amt, cnt  (g='3' 외국인, 'x' 미상 / a='1'..'6','x')"""
    keys = sorted(d.key.unique())
    mon = pd.DataFrame(index=pd.MultiIndex.from_product([keys, MONTHS], names=['key', 'ym']))
    mon['T'] = d.groupby(['key', 'ym']).amt.sum()
    mon['F'] = d[d.g == '3'].groupby(['key', 'ym']).amt.sum()
    mon['U'] = d[d.g == 'x'].groupby(['key', 'ym']).amt.sum()
    mon['C'] = d.groupby(['key', 'ym']).cnt.sum()
    mon['present'] = d.groupby(['key', 'ym']).size()
    present = mon['present'].notna()
    mon = mon.drop(columns='present')
    mon_f = mon.fillna(0)
    ind = d.pivot_table(index='key', columns='ind', values='amt', aggfunc='sum', fill_value=0).reindex(index=keys, columns=INDS, fill_value=0)
    age = d.pivot_table(index='key', columns='a', values='amt', aggfunc='sum', fill_value=0).reindex(index=keys, columns=AGES + ['x'], fill_value=0)
    dk = d[d.a != 'x']
    ind_age = dk.pivot_table(index='key', columns=['ind', 'a'], values='amt', aggfunc='sum', fill_value=0)
    ind_age = ind_age.reindex(index=keys, columns=pd.MultiIndex.from_product([INDS, AGES]), fill_value=0)
    fo = d[d.g == '3']
    f_ind = fo.pivot_table(index='key', columns='ind', values='amt', aggfunc='sum', fill_value=0).reindex(index=keys, columns=INDS, fill_value=0)
    f_age = fo.pivot_table(index='key', columns='a', values='amt', aggfunc='sum', fill_value=0).reindex(index=keys, columns=AGES, fill_value=0)
    return dict(keys=keys, mon=mon_f, present=present, ind=ind, age=age, ind_age=ind_age, f_ind=f_ind, f_age=f_age)


# ---------------------------------------------------------------- core
def dist_row(s):
    s = s.dropna()
    return {'n': int(len(s)), **{f'p{p}': r(np.percentile(s, p)) for p in PCTS}}


def rank_of(s, v):
    s = s.dropna()
    return {'value': r(v), 'percentile': r((s <= v).mean() * 100, 2), 'rank': int((s > v).sum() + 1)}


def build_evidence(tb, ext, kind, out_keys=None, ids=None, meta_extra=None, peer_k=5):
    keys = tb['keys']
    mon, ind, age, ind_age, f_ind, f_age = tb['mon'], tb['ind'], tb['age'], tb['ind_age'], tb['f_ind'], tb['f_age']
    present = tb['present']
    T = mon['T'].groupby('key').sum()
    F = mon['F'].groupby('key').sum()
    U = mon['U'].groupby('key').sum()
    C = mon['C'].groupby('key').sum()
    foreign_share = F / T * 100
    unknown_share = U / T * 100
    ind_sh = ind.div(ind.sum(axis=1), axis=0) * 100
    age_known = age[AGES].sum(axis=1)
    age_sh = age[AGES].div(age_known, axis=0) * 100
    ia_tot = ind_age.T.groupby(level=0).sum().T          # 업종별 연령 확인 가능 금액 합
    ia_sh = {}
    for i in INDS:
        ia_sh[i] = ind_age[i].div(ia_tot[i].replace(0, np.nan), axis=0) * 100
    f_tot = f_ind.sum(axis=1)
    f_ind_sh = f_ind.div(f_tot.replace(0, np.nan), axis=0) * 100
    f_age_sh = f_age.div(f_age.sum(axis=1).replace(0, np.nan), axis=0) * 100
    life = f_ind_sh['편의점'] + f_ind_sh['슈퍼마켓']
    kfood = f_ind_sh['일반한식']

    # 지역 특이 지수 = (월 일평균 / 6개월 일평균) / 전국 같은 값
    m_pd = mon['T'].unstack('ym')[MONTHS].div(pd.Series(DAYS))
    idx_raw = m_pd.div(m_pd.mean(axis=1), axis=0)
    nat = m_pd.sum(axis=0)
    idx_adj = idx_raw.div(nat / nat.mean(), axis=1)

    # 임계 컷오프
    small_cut = np.percentile(T, THRESHOLDS['small_size_percentile'])
    is_small = T <= small_cut
    r04_small_cut = np.percentile(T, THRESHOLDS['R04_small_percentile'])
    r04_vsmall_cut = np.percentile(T, THRESHOLDS['R04_very_small_percentile'])

    def r04_thr(k):
        if T[k] <= r04_vsmall_cut:
            return THRESHOLDS['R04_season_index_min_very_small']
        if T[k] <= r04_small_cut:
            return THRESHOLDS['R04_season_index_min_small']
        return THRESHOLDS['R04_season_index_min']
    f_cut = np.percentile(F, THRESHOLDS['R03_min_foreign_amount_percentile'])
    elig = F >= f_cut
    life_cut = np.percentile(life[elig], 100 - THRESHOLDS['R03_type_top_percentile'])
    kfood_cut = np.percentile(kfood[elig], 100 - THRESHOLDS['R03_type_top_percentile'])
    fs_cut = np.percentile(foreign_share, THRESHOLDS['R12_low_foreign_share_percentile'])
    low_p = THRESHOLDS['R02_R08_low_percentile']
    ind_cut = {i: np.percentile(ind_sh[i], low_p) for i in INDS}
    ia_cut = {i: {a: np.percentile(ia_sh[i][a].dropna(), low_p) for a in AGES} for i in INDS}

    def r03_type(k):
        if not elig[k]:
            return None
        if life[k] >= life_cut:
            return '생활유통형'
        if kfood[k] >= kfood_cut:
            return '한식외식형'
        return '복합형'
    r03 = {k: r03_type(k) for k in keys}

    # 유사 지자체: 업종 9 + 연령 5(코드 2~6) + 외국인 비중 + log 결제규모 를 표준화한 최근접 이웃
    X = pd.concat([ind_sh[PEER_INDS], age_sh[AGES[1:]], foreign_share.rename('fs'), np.log(T).rename('logT')], axis=1)
    Z = (X - X.mean()) / X.std(ddof=0)
    zv = Z.values
    pos = {k: i for i, k in enumerate(keys)}
    pool = out_keys if out_keys is not None else keys

    def peers_of(k):
        dd = np.sqrt(((zv[[pos[p] for p in pool]] - zv[pos[k]]) ** 2).sum(axis=1))
        order = [(pool[i], dd[i]) for i in np.argsort(dd) if pool[i] != k][:peer_k]
        return [{'region_key': p, 'distance': r(v, 4)} for p, v in order]

    # 전국 분포 메타
    national_distribution = {
        'total_amount': dist_row(T), 'foreign_share_pct': dist_row(foreign_share), 'unknown_share_pct': dist_row(unknown_share),
        'industry_share_pct': {i: dist_row(ind_sh[i]) for i in INDS},
        'age_share_pct': {a: dist_row(age_sh[a]) for a in AGES},
        'industry_age_share_pct': {i: {a: dist_row(ia_sh[i][a]) for a in AGES} for i in INDS},
        'foreign_life_retail_share_pct_among_R03_eligible': dist_row(life[elig]),
        'foreign_korean_food_share_pct_among_R03_eligible': dist_row(kfood[elig]),
    }

    def record(k, rid):
        rules = {}
        cal = []
        for ym in MONTHS:
            row = mon.loc[(k, ym)]
            t, f, u, c = row['T'], row['F'], row['U'], row['C']
            if not present.loc[(k, ym)]:
                cal.append(dict(month=ym, calculation_status='no_data', foreign_amount=None, total_amount=None, unknown_amount=None,
                                transaction_count=None, foreign_share_pct=None, known_only_share_pct=None, unknown_share_pct=None,
                                season_index=None, season_flag=None, warnings=['해당 월 관측 행 없음']))
                continue
            if t <= 0:
                cal.append(dict(month=ym, calculation_status='invalid_denominator', foreign_amount=int(f), total_amount=int(t), unknown_amount=int(u),
                                transaction_count=int(c), foreign_share_pct=None, known_only_share_pct=None, unknown_share_pct=None,
                                season_index=None, season_flag=None, warnings=['전체 금액 0으로 비중 계산 불가']))
                continue
            thr = r04_thr(k)
            si = idx_adj.loc[k, ym]
            cal.append(dict(month=ym, calculation_status='ok', foreign_amount=int(f), total_amount=int(t), unknown_amount=int(u),
                            transaction_count=int(c), foreign_share_pct=r(f / t * 100, 10),
                            known_only_share_pct=r(f / (t - u) * 100, 10) if t - u > 0 else None,
                            unknown_share_pct=r(u / t * 100, 10), season_index=r(si, 4), season_flag=bool(si >= thr), warnings=[]))
        flagged = [m['month'] for m in cal if m['season_flag']]
        zero_inds = [i for i in INDS if ind.loc[k, i] == 0]
        low_inds = [i for i in INDS if ind_sh.loc[k, i] <= ind_cut[i] and i not in zero_inds]
        low_ia = [f'{i}|{a}' for i in INDS for a in AGES
                  if not np.isnan(ia_sh[i].loc[k, a]) and ia_sh[i].loc[k, a] <= ia_cut[i][a]]
        ext_k = ext.get(k)
        rules['R02'] = dict(status='allowed', reason='시군구를 가맹점 소재지로 간주(주최 측 확인 전)')
        rules['R08'] = dict(status='allowed', display_level='reference', reason='시군구를 가맹점 소재지로 간주. 업종×연령 기준은 반기 재현성이 0.75(다른 규칙보다 낮음)라 화면에서 참고용으로 표시')
        rules['R03'] = dict(status='allowed', reason='유형 표시' if elig[k] else '외국인 결제 규모가 작아 유형 미표시')
        rules['R04'] = dict(status='allowed', reason='월 단위 집계. 시기 질문은 2026-01~06 안에서만')
        rules['R07'] = dict(status='allowed' if all(m['calculation_status'] == 'ok' for m in cal) else 'needs_review',
                            reason='F·T·U 계산 가능' if all(m['calculation_status'] == 'ok' for m in cal) else '누락·분모 0인 월 있음')
        rules['R09'] = dict(status='allowed', reason='프로필 안내(질문 없음)')
        rules['R12'] = dict(status='allowed', reason='외국인 결제 비중 계산 가능')
        rules['R11'] = dict(status='allowed' if ext_k else 'blocked', reason='주민등록 인구 있음' if ext_k else '주민등록 인구 없음')
        rules['R06'] = dict(status='blocked', reason='고급 소비 근거 검토 제외')
        rec = {
            'evidence_id': rid, 'data_kind': kind,
            'scope': dict(geographic_scope='sigungu', region_key=k, population='foreign_code_3', industry_scope='all_provided', age_scope='all',
                          period_start=MONTHS[0], period_end=MONTHS[-1]),
            'region_basis': 'merchant_location_assumed',
            'region_codes': (ext_k or {}).get('codes'),
            'applicability': rules, 'amount_unit': 'KRW', 'months': cal,
            'profile': {
                'total_amount': int(T[k]), 'transaction_count': int(C[k]),
                'industry_share_pct': {i: r(ind_sh.loc[k, i]) for i in INDS},
                'age_share_pct': {a: r(age_sh.loc[k, a]) for a in AGES},
                'age_unknown_share_pct': r(age.loc[k, 'x'] / T[k] * 100) if 'x' in age else None,
                'industry_age_share_pct': {i: {a: r(ia_sh[i].loc[k, a]) for a in AGES} for i in INDS},
            },
            'foreign_profile': {
                'foreign_amount_total': int(F[k]), 'foreign_share_pct': r(foreign_share[k]),
                'life_retail_share_pct': r(life[k]) if f_tot[k] > 0 else None,
                'korean_food_share_pct': r(kfood[k]) if f_tot[k] > 0 else None,
                'industry_share_pct': {i: r(f_ind_sh.loc[k, i]) for i in INDS},
                'age_share_pct': {a: r(f_age_sh.loc[k, a]) for a in AGES},
                'monthly_cv': r(mon.loc[k]['F'].std(ddof=0) / mon.loc[k]['F'].mean()) if mon.loc[k]['F'].mean() > 0 else None,
                'type': r03[k],
                'type_status': 'shown' if elig[k] else 'not_shown_small_foreign_amount',
            },
            'season_summary': dict(threshold=r04_thr(k),
                                   flagged_months=flagged),
            'national_rank': {
                'total_amount': rank_of(T, T[k]), 'foreign_share_pct': rank_of(foreign_share, foreign_share[k]),
                'unknown_share_pct': rank_of(unknown_share, unknown_share[k]),
                'industry_share_pct': {i: rank_of(ind_sh[i], ind_sh.loc[k, i]) for i in INDS},
                'age_share_pct': {a: rank_of(age_sh[a], age_sh.loc[k, a]) for a in AGES},
                'industry_age_share_percentile': {i: {a: (r((ia_sh[i][a].dropna() <= ia_sh[i].loc[k, a]).mean() * 100, 2) if not np.isnan(ia_sh[i].loc[k, a]) else None) for a in AGES} for i in INDS},
                'foreign_life_retail_share_pct': rank_of(life[elig], life[k]) if elig[k] else None,
                'foreign_korean_food_share_pct': rank_of(kfood[elig], kfood[k]) if elig[k] else None,
            },
            'size_flag': dict(is_small=bool(is_small[k]), total_amount_percentile=r((T <= T[k]).mean() * 100, 2),
                              unknown_share_pct=r(unknown_share[k]), unknown_share_percentile=r((unknown_share <= unknown_share[k]).mean() * 100, 2),
                              note='소액 지역이라 월별 변동이 클 수 있음' if is_small[k] else None),
            'derived_flags': dict(no_payment_industries=zero_inds, low_share_industries=low_inds, low_industry_age_pairs=low_ia,
                                  foreign_share_low=bool(foreign_share[k] <= fs_cut), r03_type=r03[k], season_flag_months=flagged),
            'peers': peers_of(k),
            'external': None,
            'limitations': ['시군구를 가맹점 소재지 기준으로 간주함(주최 측 확인 전)', 'BC카드 결제만 포함하며 현금·타사 카드는 빠져 있음',
                            '외국인은 성별 코드 3이며 관광객과 거주 외국인을 구분하지 못함',
                            '월 단위 집계이며 6개월 자료라 계절성으로 단정하지 않음',
                            '결제가 잡히지 않은 것을 상점이 없다는 뜻으로 해석하지 않음'],
        }
        if kind == 'synthetic':
            rec['limitations'] = ['합성 자료', '가상 시군구 예시이며 실제 지역이 아님'] + rec['limitations'][1:]
        if ext_k:
            pop = ext_k['pop_total']
            rf = ext_k['reg_foreign']
            rf_vals = [v for v in rf.values() if v is not None]
            rec['external'] = {
                'population_total': int(pop), 'population_base_date': ext_k['pop_base'],
                'population_by_age': {a: int(ext_k['pop_age'][a]) for a in AGES},
                'population_age_share_pct': {a: r(ext_k['pop_age'][a] / pop * 100) for a in AGES},
                'registered_foreign_by_month': rf,
                'depopulation_area_2021': bool(ext_k['depop']),
                'payments_per_resident_krw_6m': r(T[k] / pop, 0),
                'foreign_payment_per_registered_foreign_krw_6m': r(F[k] / np.mean(rf_vals), 0) if rf_vals else None,
                'caution': '결제액÷인구는 주민 1인당 소비가 아니라 상권 규모 감각용 값이며, 외국인 결제는 등록외국인만의 결제가 아님',
            }
        return rec

    recs = [record(k, (ids or {}).get(k, 'REAL-' + k)) for k in (out_keys if out_keys is not None else keys)]

    # 전국(ALL) 기록
    allT = mon.groupby(level='ym').sum().loc[MONTHS]
    cal_all = []
    for ym in MONTHS:
        t, f, u, c = allT.loc[ym, ['T', 'F', 'U', 'C']]
        cal_all.append(dict(month=ym, calculation_status='ok', foreign_amount=int(f), total_amount=int(t), unknown_amount=int(u), transaction_count=int(c),
                            foreign_share_pct=r(f / t * 100, 10), known_only_share_pct=r(f / (t - u) * 100, 10), unknown_share_pct=r(u / t * 100, 10),
                            season_index=None, season_flag=None, warnings=[]))
    ia_all = ind_age.sum()
    all_rec = {
        'evidence_id': (ids or {}).get('ALL', 'REAL-ALL'), 'data_kind': kind,
        'scope': dict(geographic_scope='national', region_key='ALL', population='foreign_code_3', industry_scope='all_provided', age_scope='all',
                      period_start=MONTHS[0], period_end=MONTHS[-1]),
        'region_basis': 'merchant_location_assumed', 'region_codes': None,
        'applicability': {'R07': dict(status='allowed', reason='전국 참고 값'), 'R06': dict(status='blocked', reason='고급 소비 근거 검토 제외')},
        'amount_unit': 'KRW', 'months': cal_all,
        'profile': {
            'total_amount': int(T.sum()), 'transaction_count': int(C.sum()),
            'industry_share_pct': {i: r(ind[i].sum() / ind.values.sum() * 100) for i in INDS},
            'age_share_pct': {a: r(age[a].sum() / age[AGES].values.sum() * 100) for a in AGES},
            'age_unknown_share_pct': r(age['x'].sum() / T.sum() * 100),
            'industry_age_share_pct': {i: {a: r(ia_all[(i, a)] / sum(ia_all[(i, b)] for b in AGES) * 100) if sum(ia_all[(i, b)] for b in AGES) else None for a in AGES} for i in INDS},
        },
        'foreign_profile': {
            'foreign_amount_total': int(F.sum()), 'foreign_share_pct': r(F.sum() / T.sum() * 100),
            'life_retail_share_pct': r((f_ind['편의점'].sum() + f_ind['슈퍼마켓'].sum()) / f_ind.values.sum() * 100),
            'korean_food_share_pct': r(f_ind['일반한식'].sum() / f_ind.values.sum() * 100),
            'industry_share_pct': {i: r(f_ind[i].sum() / f_ind.values.sum() * 100) for i in INDS},
            'age_share_pct': {a: r(f_age[a].sum() / f_age.values.sum() * 100) for a in AGES},
            'monthly_cv': None, 'type': None, 'type_status': 'not_applicable_national'},
        'season_summary': None, 'national_rank': None, 'size_flag': None, 'derived_flags': None, 'peers': [], 'external': None,
        'limitations': ['전국 참고 값이며 선택 지역 진단이 아님'],
    }
    if kind == 'synthetic':
        all_rec['limitations'] = ['합성 자료'] + all_rec['limitations']
    thresholds_out = dict(THRESHOLDS)
    thresholds_out['derived_cutoffs'] = dict(
        small_size_total_amount_max=r(small_cut, 0), R04_small_total_amount_max=r(r04_small_cut, 0), R04_very_small_total_amount_max=r(r04_vsmall_cut, 0), R03_min_foreign_amount=r(f_cut, 0), R03_life_retail_share_pct_min=r(life_cut),
        R03_korean_food_share_pct_min=r(kfood_cut), R12_foreign_share_pct_max=r(fs_cut),
        R02_low_industry_share_pct_max={i: r(ind_cut[i]) for i in INDS})
    doc = {
        'schema_version': CFG['dataset_versions']['schema'], 'dataset_version': (meta_extra or {}).get('dataset_version', REAL_VERSION), 'data_kind': kind,
        'amount_unit': 'KRW', 'percent_convention': '비중은 퍼센트 값 그대로(0~100)',
        'period': dict(start=MONTHS[0], end=MONTHS[-1]),
        'industry_codes': {k: v for k, v in IND_NAME.items()},
        'age_labels': AGE_LABEL,
        'age_note': '연령 코드 1은 설명서에 "20대 이하"로 표기되며 주민등록 인구의 20세 미만에 대응시킴(데이터 담당 확인 2026-09-20)',
        'display_policy': {'R08': {'level': 'reference', 'reason': '업종×연령 하위 20% 기준의 반기 재현성이 0.75(업종 단독 R02는 0.81)라 확정 판정이 아니라 참고용으로 표시하고 담당자가 확인하도록 한다'}},
        'external_sources': {
            'population': {'source': '행정안전부 행정동별 성별 연령별 주민등록 인구', 'base_date': '2026-08-31', 'note': '2026-07 인천 구 개편·전남광주 통합 반영. BC 자료 기간(옛 구) 기준으로 합산'},
            'registered_foreign': {'source': '법무부 시군구별 등록외국인', 'period': '2026-01~06'},
            'depopulation_area': {'basis': '2021년 지정 89곳 목록(필드명 depopulation_area_2021)', 'status': 'pending_redesignation',
                                  'note': '2026-06-30 보도 기준 행정안전부가 지표를 개편해 2026-10월부터 새로 지정할 계획이며 결과는 아직 발표되지 않음. 발표 후 region_code_map.csv의 depop_2021 열을 갱신하고 근거 파일을 다시 만든다'}},
        'thresholds': thresholds_out,
        'national_distribution': national_distribution,
        'records': recs + [all_rec],
    }
    if meta_extra:
        doc.update({k: v for k, v in meta_extra.items() if k != 'dataset_version'})
    return doc, dict(T=T, F=F, foreign_share=foreign_share, ind_sh=ind_sh, idx_adj=idx_adj, r03=r03, is_small=is_small)


# ---------------------------------------------------------------- real
def load_external(ext_dir=EXT_DIR):
    """공개 외부 자료(코드 매핑, 주민등록 인구, 등록외국인)를 시군구별 사전으로 읽는다."""
    ext_dir = Path(ext_dir)
    mp = pd.read_csv(ext_dir / 'region_code_map.csv', dtype=str, encoding='utf-8-sig').fillna('')
    pop = pd.read_csv(ext_dir / 'ext_population.csv', dtype={'sgg_code5': str}, encoding='utf-8-sig').set_index('region_key')
    rf = pd.read_csv(ext_dir / 'ext_registered_foreign.csv', dtype=str, encoding='utf-8-sig').set_index('region_key')
    ext, ids = {}, {}
    for row in mp.itertuples():
        k = row.region_key
        p = pop.loc[k]
        rfm = {}
        for ym in MONTHS:
            v = rf.loc[k].get('reg_foreign_' + ym.replace('-', ''))
            rfm[ym] = None if (v is None or v == '' or str(v) == 'nan') else int(float(v))
        code = row.sgg_code5_pre2026 or row.sgg_code5
        ids[k] = 'REAL-' + code.replace(';', '_')
        ext[k] = dict(codes=dict(admin_2026_08=row.sgg_code5, pre_2026=row.sgg_code5_pre2026 or None, match_type=row.match_type),
                      pop_total=p.pop_total, pop_base=str(p.base_date),
                      pop_age={a: p[f'pop_age{a}'] for a in AGES}, reg_foreign=rfm, depop=row.depop_2021 == 'True')
    return ext, ids, list(mp.region_key)


def real(raw=None, ext_dir=EXT_DIR, out=PRIVATE_DIR / 'review_evidence_real_v2.1.json', use_external=True):
    d = read_raw(raw)
    tb = tables_from_raw(d)
    if use_external:
        ext, ids, order = load_external(ext_dir)
        keys = [k for k in order if k in set(tb['keys'])]
    else:
        # 실행 점검용(가짜 원본 등). 외부 자료 없이 만들며 R11은 blocked가 된다.
        print('경고: 외부 자료 없이 만듭니다(--no-external). 실제 제출용 파일에는 쓰지 마세요.')
        ext, ids, keys = {}, {k: 'REAL-' + k for k in tb['keys']}, list(tb['keys'])
    ids['ALL'] = 'REAL-ALL'
    assert len(keys) == 255, len(keys)
    doc, aux = build_evidence(tb, ext, 'real', out_keys=keys, ids=ids,
                              meta_extra=dict(dataset_version=REAL_VERSION, generated_from='BC카드 제공 데이터 2026-01~06, 주민등록 인구 2026-08-31, 등록외국인 2026-01~06',
                                              handling='PRIVATE: 시군구별 실제 수치. 공개 저장소·공개 배포 금지. private/에만 둔다'))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('wrote', out, len(doc['records']), 'records')
    return doc


# ---------------------------------------------------------------- synthetic
def synthetic(out=EXAMPLES_DIR / 'review_evidence_demo_v2.1.json'):
    """가짜 세계(255개)를 만들어 전국 분포를 얻고, 시나리오별 가상 시군구 12곳만 내보낸다.
    모수는 실제 자료에서 맞춘 값이 아니라 임의로 정한 값이다."""
    rng = np.random.default_rng(20260920)
    N = 255
    wkeys = [f'W-{i:03d}' for i in range(N)]
    scen = {  # 이름: (설명, 규모 배율, 조정)
        'DEMO-SGG-A': ('대도시 구 유형: 규모 큼, 기본 구성', 30.0, {}),
        'DEMO-SGG-B': ('농촌 군 유형: 소액 지역, 대형할인점 결제 없음', 0.12, {'zero': ['대형할인점']}),
        'DEMO-SGG-C': ('외국인 결제가 편의점·슈퍼에 몰림(생활유통형)', 3.0, {'fs': 0.12, 'f_life': 0.62}),
        'DEMO-SGG-D': ('외국인 결제가 일반한식에 몰림(한식외식형)', 2.5, {'fs': 0.09, 'f_kfood': 0.34}),
        'DEMO-SGG-E': ('외국인 결제 균형(복합형)', 3.5, {'fs': 0.10, 'f_bal': True}),
        'DEMO-SGG-F': ('외국인 결제 비중이 낮은 지역(R12 대상)', 4.0, {'fs': 0.02}),
        'DEMO-SGG-G': ('특정 달에 결제가 뚜렷이 커지는 지역(R04 시기)', 2.0, {'spike': ('2026-02', 1.35)}),
        'DEMO-SGG-H': ('성별·연령 미상 비중이 매우 큰 지역', 1.5, {'unk': 0.30}),
        'DEMO-SGG-I': ('외국인 결제 규모가 작아 R03 유형 미표시', 0.5, {'fs': 0.02}),
        'DEMO-SGG-J': ('인구감소지역 예시, 한식 비중 높음', 0.35, {'kfood': 0.52, 'depop': True}),
        'DEMO-SGG-K': ('연령 대상 사업 예시: 60대 이상 비중이 낮은 지역', 6.0, {'age6': 0.10}),
        'DEMO-SGG-L': ('외부 자료(인구)가 없는 지역', 2.0, {'no_ext': True}),
    }
    demo_keys = list(scen)
    allk = wkeys + demo_keys
    rows = []
    age_alpha = np.array([1.0, 9, 15, 20, 24, 30])
    ind_alpha = np.array([28.0, 3, 5, 12, 10, 6, 15, 2, 8, 2, 9])  # 일반한식 등 INDS 순서와 무관한 임의 값
    ext_syn = {}
    for k in allk:
        sc = scen.get(k, ('', float(np.exp(rng.normal(0.5, 1.1))), {}))
        scale, adj = sc[1], sc[2]
        base_total = 4e8 * scale
        fs = adj.get('fs', float(np.clip(rng.beta(3, 45), 0.01, 0.2)))
        unk = adj.get('unk', float(np.clip(rng.beta(1.5, 45), 0.005, 0.15)))
        ish = rng.dirichlet(ind_alpha * 6)
        if 'kfood' in adj:
            ish = ish.copy(); ish[INDS.index('일반한식')] = adj['kfood']
            rest = [i for i in range(len(INDS)) if i != INDS.index('일반한식')]
            ish[rest] = ish[rest] / ish[rest].sum() * (1 - adj['kfood'])
        for z in adj.get('zero', []):
            j = INDS.index(z); ish[j] = 0; ish = ish / ish.sum()
        ash = rng.dirichlet(age_alpha * 8)
        if 'age6' in adj:
            ash = ash.copy(); ash[5] = adj['age6']; ash[:5] = ash[:5] / ash[:5].sum() * (1 - adj['age6'])
        fish = rng.dirichlet(ind_alpha * 4)
        if 'f_life' in adj:
            fish = np.zeros(len(INDS)); li, si = INDS.index('편의점'), INDS.index('슈퍼마켓')
            fish[li], fish[si] = adj['f_life'] * 0.55, adj['f_life'] * 0.45
            other = [j for j in range(len(INDS)) if j not in (li, si)]
            fish[other] = rng.dirichlet(np.ones(len(other)) * 5) * (1 - adj['f_life'])
        if 'f_kfood' in adj:
            fish = rng.dirichlet(ind_alpha * 4); kj = INDS.index('일반한식')
            fish[kj] = adj['f_kfood']; oth = [j for j in range(len(INDS)) if j != kj]
            fish[oth] = fish[oth] / fish[oth].sum() * (1 - adj['f_kfood'])
            lj = [INDS.index('편의점'), INDS.index('슈퍼마켓')]
            fish[lj] = fish[lj] * 0.3; fish = fish / fish.sum()
        if adj.get('f_bal'):
            fish = np.full(len(INDS), 0.0)
            for nm, v in [('일반한식', 0.09), ('편의점', 0.03), ('슈퍼마켓', 0.03)]:
                fish[INDS.index(nm)] = v
            oth = [j for j, nm in enumerate(INDS) if nm not in ('일반한식', '편의점', '슈퍼마켓')]
            fish[oth] = rng.dirichlet(np.ones(len(oth)) * 6) * 0.85
        for z in adj.get('zero', []):
            j = INDS.index(z); fish = fish.copy(); fish[j] = 0; fish = fish / fish.sum()
        fash = rng.dirichlet(np.array([2.0, 12, 12, 10, 8, 6]))
        mw = rng.normal(1.0, 0.02, 6)
        if 'spike' in adj:
            mw[MONTHS.index(adj['spike'][0])] *= adj['spike'][1]
        monthly_w = mw * np.array([DAYS[m] for m in MONTHS]) * np.array([0.96, 1.0, 0.97, 0.99, 1.09, 1.0])
        monthly_w = monthly_w / monthly_w.sum()
        for mi, ym in enumerate(MONTHS):
            tm = base_total * monthly_w[mi]
            fm, um = tm * fs, tm * unk
            km = tm - fm - um   # 내국인 성별·연령 확인분
            for ii, ind_name in enumerate(INDS):
                # 내국인 확인분
                for ai, a in enumerate(AGES):
                    amt = km * ish[ii] * ash[ai]
                    if amt > 0: rows.append((k, ym, '1', a, ind_name, amt, amt / 45000))
                famt = fm * fish[ii]
                for ai, a in enumerate(AGES):
                    if famt * fash[ai] > 0: rows.append((k, ym, '3', a, ind_name, famt * fash[ai], famt * fash[ai] / 60000))
                uamt = um * ish[ii]
                if uamt > 0: rows.append((k, ym, 'x', 'x', ind_name, uamt, uamt / 45000))
        if not sc[2].get('no_ext'):
            popn = int(max(6000, base_total / 6 / 1500 * rng.uniform(0.7, 1.3)))
            pa = rng.dirichlet(np.array([12.0, 11, 12, 15, 17, 30]) * 10)
            pop_age = {a: int(popn * pa[i]) for i, a in enumerate(AGES)}
            popn = sum(pop_age.values())
            regf = int(max(30, popn * fs * rng.uniform(0.15, 0.5)))
            ext_syn[k] = dict(codes=dict(admin_2026_08=None, pre_2026=None, match_type='synthetic'), pop_total=popn, pop_base='2026-08-31',
                              pop_age=pop_age, reg_foreign={m: int(regf * (1 + 0.004 * i)) for i, m in enumerate(MONTHS)},
                              depop=bool(sc[2].get('depop', False)))
    d = pd.DataFrame(rows, columns=['key', 'ym', 'g', 'a', 'ind', 'amt', 'cnt'])
    d['amt'] = d.amt.round().astype('int64'); d['cnt'] = d.cnt.round().astype('int64')
    # 누락 월 예시: DEMO-SGG-B 의 2026-03 관측 행을 지운다(no_data)
    d = d[~((d.key == 'DEMO-SGG-B') & (d.ym == '2026-03'))]
    tb = build_tables(d)
    ids = {k: k for k in demo_keys}; ids['ALL'] = 'DEMO-ALL'
    doc, aux = build_evidence(tb, ext_syn, 'synthetic', out_keys=demo_keys, ids=ids,
                              meta_extra=dict(dataset_version=DEMO_VERSION,
                                              handling='합성 자료. 공개 저장소에 둘 수 있음. 전국 분포는 임의로 만든 가상 255개 시군구 기준',
                                              demo_scenarios={k: v[0] for k, v in scen.items()}))
    # 합성 파일에는 가상 세계 개별 값이 없고, 전국 분포·전국 기록은 가상 세계에서 계산된 값임을 표시
    doc['note_synthetic'] = '전국(ALL) 기록과 national_distribution은 임의 모수로 만든 가상 255개 시군구에서 계산했다'
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('wrote', out, len(doc['records']), 'records')
    return doc


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='근거 파일 스키마 2.1 생성기')
    ap.add_argument('mode', nargs='?', default='real', choices=['real', 'synthetic'])
    ap.add_argument('--raw', help='원본 CSV 경로(기본 data/ABP_CONTEST_DATA.csv)')
    ap.add_argument('--ext-dir', default=str(EXT_DIR), help='공개 외부 자료 폴더')
    ap.add_argument('--out', help='출력 경로')
    ap.add_argument('--no-external', action='store_true', help='외부 자료 없이 만들기(실행 점검용)')
    a = ap.parse_args()
    if a.mode == 'real':
        real(a.raw, a.ext_dir, a.out or PRIVATE_DIR / 'review_evidence_real_v2.1.json', use_external=not a.no_external)
    else:
        synthetic(a.out or EXAMPLES_DIR / 'review_evidence_demo_v2.1.json')
