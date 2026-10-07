"""근거 파일 자동 검증 35항목 (검증 절차 가이드 A2).

  python src/10_verify_evidence.py                 # 실제 원본 + private/ 실제 근거 + examples/ 합성 근거
  python src/10_verify_evidence.py --mode check    # 보고 집계값(57건·30곳 등) 대신 원본 재계산값과만 비교(가짜 원본 점검용)

확인하는 것
  S  구조: 버전, 기록 수, 실제·합성 파일의 키 구성, 설정 파일 임계값과 일치
  M  합계: 비중 합 100, 월 합 = 총액, 전국 기록 = 시군구 합, 값 범위
  R  원본 재계산: 원본 행 수·지역·월·업종, 시군구 14곳의 월별 F·T·U·C와 업종 비중을 다른 방식(행 단위 누적)으로 다시 계산
  A  집계값: 결제 없음, R03 유형 분포, 소액 지역, R04 발동, R12 대상, R04 기준·발동월·R02 컷오프의 내부 일관성
  D  합성 시나리오 A~L, 합성 파일에 실제 지역 이름이 없는지

FAIL이 나오면 원인(원본 변경, 임계값 변경, 스크립트 수정)을 먼저 찾는다. 원인을 모른 채 기대값을 고쳐 통과시키지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (AGES, INDS, IND_NAME, MONTHS, DAYS, EXAMPLES_DIR, OUTPUT_DIR, PRIVATE_DIR, RAW_CSV, load_config)

CFG = load_config()
TH = CFG['thresholds']
EPS = 1e-6
META_KEYS = {'generated_from', 'handling', 'demo_scenarios', 'note_synthetic'}   # 파일 설명용 키(실제·합성이 달라도 됨)
RESULTS: list[dict] = []


def check(cid, name, ok, detail=''):
    RESULTS.append(dict(id=cid, name=name, ok=bool(ok), detail=str(detail)))


def safe(cid, name, fn):
    """검사 하나가 예외로 멈추면 그 항목만 FAIL로 남기고 계속한다."""
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f'예외: {type(e).__name__}: {e}'
    check(cid, name, ok, detail)


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sgg(doc):
    return [r for r in doc['records'] if r['scope']['region_key'] != 'ALL']


def all_rec(doc):
    return next(r for r in doc['records'] if r['scope']['region_key'] == 'ALL')


def keyset(d):
    return sorted(d.keys()) if isinstance(d, dict) else None


# ---------------------------------------------------------------- 원본 행 단위 누적(pandas 피벗과 다른 방식)
def read_raw_rows(path):
    """csv 모듈로 원본을 한 줄씩 읽어 시군구-월, 시군구-업종 합계를 사전에 누적한다."""
    for enc in ('utf-8-sig', 'cp949'):
        try:
            f = open(path, encoding=enc, newline='')
            f.readline(); f.seek(0)
            break
        except UnicodeDecodeError:
            continue
    mon = defaultdict(lambda: [0, 0, 0, 0])            # (key, ym) -> F, T, U, C
    ind = defaultdict(int)                             # (key, ind) -> amt
    nat = defaultdict(lambda: [0, 0, 0, 0])
    fo_ind = defaultdict(int)
    gender_amt = defaultdict(int)
    keys, months, inds = set(), set(), set()
    n = 0
    with f:
        for row in csv.DictReader(f):
            n += 1
            k = f"{row['SIDO_NM']} {row['CCG_NM']}"
            ym = f"{row['STRD_YYMM'][:4]}-{row['STRD_YYMM'][4:6]}"
            g = row['GENDER_CD'].strip().lower()
            amt, cnt = int(float(row['amt'])), int(float(row['cnt']))
            nm = IND_NAME[str(int(float(row['TP_BUZ_NO'])))]
            keys.add(k); months.add(ym); inds.add(nm)
            for acc in (mon[(k, ym)], nat[ym]):
                acc[1] += amt; acc[3] += cnt
                if g == '3':
                    acc[0] += amt
                elif g == 'x':
                    acc[2] += amt
            ind[(k, nm)] += amt
            if g == '3':
                fo_ind[(k, nm)] += amt
            gender_amt[g] += amt
    return dict(n=n, keys=keys, months=months, inds=inds, mon=mon, ind=ind, nat=nat, fo_ind=fo_ind)


def pct(values, p):
    return float(np.percentile(np.asarray(values, float), p))


def recompute_aggregates(raw):
    """원본 누적값에서 집계값을 근거 파일과 독립적으로 다시 계산한다(numpy만 사용)."""
    keys = sorted(raw['keys'])
    T = {k: sum(raw['mon'][(k, m)][1] for m in MONTHS) for k in keys}
    F = {k: sum(raw['mon'][(k, m)][0] for m in MONTHS) for k in keys}
    Tv = np.array([T[k] for k in keys], float)
    # 결제 없음
    zero = sum(1 for k in keys for i in INDS if raw['ind'].get((k, i), 0) == 0)
    # 소액 지역
    small = int((Tv <= pct(Tv, TH['small_size_percentile'])).sum())
    # R12
    fs = np.array([F[k] / T[k] for k in keys])
    r12 = int((fs <= pct(fs, TH['R12_low_foreign_share_percentile'])).sum())
    # R03
    Fv = np.array([F[k] for k in keys], float)
    elig = Fv >= pct(Fv, TH['R03_min_foreign_amount_percentile'])
    life, kf = [], []
    for k in keys:
        ft = sum(raw['fo_ind'].get((k, i), 0) for i in INDS)
        life.append((raw['fo_ind'].get((k, '편의점'), 0) + raw['fo_ind'].get((k, '슈퍼마켓'), 0)) / ft * 100 if ft else np.nan)
        kf.append(raw['fo_ind'].get((k, '일반한식'), 0) / ft * 100 if ft else np.nan)
    life, kf = np.array(life), np.array(kf)
    lc = pct(life[elig], 100 - TH['R03_type_top_percentile'])
    kc = pct(kf[elig], 100 - TH['R03_type_top_percentile'])
    types = defaultdict(int)
    for e, l, q in zip(elig, life, kf):
        if e:
            types['생활유통형' if l >= lc else '한식외식형' if q >= kc else '복합형'] += 1
    # R04
    daily = np.array([[raw['mon'][(k, m)][1] / DAYS[m] for m in MONTHS] for k in keys])
    idx_raw = daily / daily.mean(axis=1, keepdims=True)
    nat = daily.sum(axis=0)
    idx = idx_raw / (nat / nat.mean())
    vs, s = pct(Tv, TH['R04_very_small_percentile']), pct(Tv, TH['R04_small_percentile'])
    thr = np.where(Tv <= vs, TH['R04_season_index_min_very_small'], np.where(Tv <= s, TH['R04_season_index_min_small'], TH['R04_season_index_min']))
    r04 = int((idx >= thr[:, None]).any(axis=1).sum())
    return dict(no_payment_entries=zero, small_regions=small, r12_low_regions=r12, r03_types=dict(types), r04_flagged_regions=r04)


def json_aggregates(doc):
    recs = sgg(doc)
    types = defaultdict(int)
    for r in recs:
        if r['foreign_profile']['type']:
            types[r['foreign_profile']['type']] += 1
    return dict(no_payment_entries=sum(len(r['derived_flags']['no_payment_industries']) for r in recs),
                small_regions=sum(r['size_flag']['is_small'] for r in recs),
                r12_low_regions=sum(r['derived_flags']['foreign_share_low'] for r in recs),
                r03_types=dict(types),
                r04_flagged_regions=sum(bool(r['season_summary']['flagged_months']) for r in recs))


# ---------------------------------------------------------------- 검사
def run(raw_path, real_path, demo_path, mode):
    real = load_json(real_path)
    demo = load_json(demo_path)
    vers = CFG['dataset_versions']
    exp = CFG['expected_real']
    R, D = sgg(real), sgg(demo)
    dr = {r['scope']['region_key']: r for r in D}

    # ---- S. 구조
    safe('S01', '실제 근거 파일 버전', lambda: (real['dataset_version'] == vers['real'] and real['schema_version'] == vers['schema'] and real['data_kind'] == 'real',
                                     f"{real['dataset_version']} / schema {real['schema_version']} / {real['data_kind']}"))
    safe('S02', '합성 근거 파일 버전', lambda: (demo['dataset_version'] == vers['demo'] and demo['schema_version'] == vers['schema'] and demo['data_kind'] == 'synthetic',
                                     f"{demo['dataset_version']} / schema {demo['schema_version']} / {demo['data_kind']}"))
    safe('S03', '기록 수(실제 255+전국, 합성 12+전국)', lambda: (len(R) == 255 and len(D) == 12 and len(real['records']) == 256 and len(demo['records']) == 13,
                                                     f'실제 {len(R)}+1, 합성 {len(D)}+1'))
    safe('S04', '시군구 기록 최상위 키가 실제·합성에서 같음', lambda: (
        len({tuple(keyset(r)) for r in R + D}) == 1 and
        sorted(set(keyset(real)) - META_KEYS) == sorted(set(keyset(demo)) - META_KEYS),
        f'키 구성 {len({tuple(keyset(r)) for r in R + D})}종'))

    def s05():
        mk = {tuple(sorted(m)) for r in R + D for m in r['months']}
        return len(mk) == 1 and all(len(r['months']) == 6 for r in R + D), f'월 기록 키 구성 {len(mk)}종'
    safe('S05', 'months[] 키 구성 같음·6개월 모두 있음', s05)

    def s06():
        parts = ['profile', 'foreign_profile', 'season_summary', 'size_flag', 'derived_flags', 'applicability']
        diff = [p for p in parts if len({tuple(keyset(r[p])) for r in R + D}) != 1]
        r08 = all(r['applicability']['R08'].get('display_level') == 'reference' for r in R + D)
        return not diff and r08 and real['display_policy']['R08']['level'] == 'reference', f'다른 키 구성: {diff or "없음"}, R08 참고 표시 {r08}'
    safe('S06', '하위 블록 키 구성 같음·R08 참고용 표시', s06)

    def s07():
        bad = [k for k, v in TH.items() if real['thresholds'].get(k) != v or demo['thresholds'].get(k) != v]
        return not bad, f'설정 파일과 다른 임계값: {bad or "없음"}'
    safe('S07', '근거 파일 임계값 = config/rules_thresholds.json', s07)

    # ---- M. 합계
    def share_sum(get):
        bad = []
        for r in R:
            vals = [v for v in get(r).values() if v is not None]
            if vals and abs(sum(vals) - 100) > 1e-3:
                bad.append(r['scope']['region_key'])
        return not bad, f'합이 100이 아닌 시군구 {len(bad)}곳'
    safe('M01', '업종 비중 합 100', lambda: share_sum(lambda r: r['profile']['industry_share_pct']))
    safe('M02', '연령 비중 합 100(미상 제외)', lambda: share_sum(lambda r: r['profile']['age_share_pct']))
    safe('M03', '외국인 업종 비중 합 100', lambda: share_sum(lambda r: r['foreign_profile']['industry_share_pct']))

    def msum(field, total):
        bad = [r['scope']['region_key'] for r in R if sum(m[field] or 0 for m in r['months']) != r['profile'][total]]
        return not bad, f'불일치 {len(bad)}곳'
    safe('M04', '월별 T 합 = 6개월 총결제', lambda: msum('total_amount', 'total_amount'))
    safe('M05', '월별 C 합 = 6개월 거래건수', lambda: msum('transaction_count', 'transaction_count'))

    def m06():
        a = all_rec(real)
        bad = []
        for i, ym in enumerate(MONTHS):
            for f in ['foreign_amount', 'total_amount', 'unknown_amount', 'transaction_count']:
                if sum(r['months'][i][f] or 0 for r in R) != a['months'][i][f]:
                    bad.append(f'{ym}:{f}')
        return not bad, f'불일치 {bad or "없음"}'
    safe('M06', '전국(ALL) 월별 값 = 시군구 합', m06)

    def m07():
        bad = 0
        for r in R:
            for m in r['months']:
                if m['calculation_status'] != 'ok':
                    continue
                if not (0 <= m['foreign_amount'] <= m['total_amount'] and 0 <= m['unknown_amount'] <= m['total_amount']):
                    bad += 1
                for f in ['foreign_share_pct', 'unknown_share_pct', 'known_only_share_pct']:
                    if m[f] is not None and not (0 <= m[f] <= 100):
                        bad += 1
        return bad == 0, f'범위 밖 {bad}건'
    safe('M07', 'F≤T, U≤T, 비중 0~100', m07)

    # ---- R. 원본 재계산
    raw = read_raw_rows(raw_path)

    def r01():
        ok = len(raw['keys']) == 255 and len(raw['months']) == 6 and len(raw['inds']) == 11
        if mode == 'real':
            ok = ok and raw['n'] == exp['raw_rows']
        return ok, f"행 {raw['n']:,} / 시군구 {len(raw['keys'])} / 월 {len(raw['months'])} / 업종 {len(raw['inds'])}"
    safe('R01', '원본 규모(행·시군구·월·업종)', r01)

    # 표본 14곳: 총결제 순위에서 고르게 + 미상 비중 최대 + 결제 없음이 있는 곳
    by_T = sorted(R, key=lambda r: r['profile']['total_amount'])
    picks = [by_T[int(i)]['scope']['region_key'] for i in np.linspace(0, 254, 12)]
    picks.append(max(R, key=lambda r: r['size_flag']['unknown_share_pct'])['scope']['region_key'])
    zero_rec = [r for r in R if r['derived_flags']['no_payment_industries']]
    picks.append((zero_rec[0] if zero_rec else by_T[128])['scope']['region_key'])
    picks = list(dict.fromkeys(picks))
    rr = {r['scope']['region_key']: r for r in R}

    def r02():
        bad = []
        for k in picks:
            for i, ym in enumerate(MONTHS):
                F, T, U, C = raw['mon'].get((k, ym), [0, 0, 0, 0])
                m = rr[k]['months'][i]
                if m['calculation_status'] == 'no_data':
                    if T != 0:
                        bad.append(f'{k} {ym} no_data')
                    continue
                if (m['foreign_amount'], m['total_amount'], m['unknown_amount'], m['transaction_count']) != (F, T, U, C):
                    bad.append(f'{k} {ym}')
        return not bad, f'표본 {len(picks)}곳 x 6개월, 불일치 {len(bad)}건'
    safe('R02', '표본 14곳 월별 F·T·U·C 원 단위 일치', r02)

    def r03():
        bad = 0
        for k in picks:
            tot = sum(raw['ind'].get((k, i), 0) for i in INDS)
            for i in INDS:
                if abs(raw['ind'].get((k, i), 0) / tot * 100 - rr[k]['profile']['industry_share_pct'][i]) > EPS:
                    bad += 1
        return bad == 0, f'표본 {len(picks)}곳 x 11업종, 불일치 {bad}건'
    safe('R03', '표본 14곳 업종 비중 일치', r03)

    def r04():
        a = all_rec(real)
        bad = [ym for i, ym in enumerate(MONTHS)
               if tuple(raw['nat'][ym]) != (a['months'][i]['foreign_amount'], a['months'][i]['total_amount'],
                                            a['months'][i]['unknown_amount'], a['months'][i]['transaction_count'])]
        return not bad, f'불일치 월 {bad or "없음"}'
    safe('R04', '전국 월별 F·T·U·C = 원본 합계', r04)

    # ---- A. 집계값: JSON = 원본 재계산 (= 보고값, mode real)
    agg_raw = recompute_aggregates(raw)
    agg_js = json_aggregates(real)

    def agg_check(key):
        def f():
            ok = agg_js[key] == agg_raw[key]
            detail = f'근거 파일 {agg_js[key]} / 원본 재계산 {agg_raw[key]}'
            if mode == 'real':
                ok = ok and agg_js[key] == exp[key]
                detail += f' / 보고값 {exp[key]}'
            return ok, detail
        return f
    safe('A01', '결제 없음 업종 건수', agg_check('no_payment_entries'))
    safe('A02', 'R03 유형 분포', agg_check('r03_types'))
    safe('A03', '소액 지역 수', agg_check('small_regions'))
    safe('A04', 'R04 시기 발동 지역 수', agg_check('r04_flagged_regions'))
    safe('A05', 'R12 외국인 비중 낮음 지역 수', agg_check('r12_low_regions'))

    def a06():
        dc = real['thresholds']['derived_cutoffs']
        bad = 0
        for r in R:
            T = r['profile']['total_amount']
            want = (TH['R04_season_index_min_very_small'] if T <= dc['R04_very_small_total_amount_max'] + 0.5 else
                    TH['R04_season_index_min_small'] if T <= dc['R04_small_total_amount_max'] + 0.5 else TH['R04_season_index_min'])
            bad += r['season_summary']['threshold'] != want
        return bad == 0, f'규모 구간과 다른 기준 {bad}곳'
    safe('A06', 'R04 기준(1.30·1.15·1.10)이 규모 구간대로 붙음', a06)

    def a07():
        bad = 0
        for r in R:
            thr = r['season_summary']['threshold']
            for m in r['months']:
                if m['season_index'] is not None and m['season_flag'] != (m['season_index'] >= thr - 5e-5):
                    bad += 1
            if r['season_summary']['flagged_months'] != [m['month'] for m in r['months'] if m['season_flag']]:
                bad += 1
        return bad == 0, f'불일치 {bad}건'
    safe('A07', 'season_flag = 지수 ≥ 기준, 발동월 목록 일치', a07)

    def a08():
        cut = real['thresholds']['derived_cutoffs']['R02_low_industry_share_pct_max']
        bad = 0
        for r in R:
            sh, z, low = r['profile']['industry_share_pct'], set(r['derived_flags']['no_payment_industries']), set(r['derived_flags']['low_share_industries'])
            want = {i for i in INDS if sh[i] <= cut[i] + 1e-6 and i not in z}
            bad += want != low
            bad += any(sh[i] != 0 for i in z)
        return bad == 0, f'불일치 {bad}곳'
    safe('A08', 'R02 하위 20% 업종 = 컷오프 이하(결제 없음 제외)', a08)

    # ---- D. 합성 시나리오
    def d01():
        names = set(json.dumps(demo, ensure_ascii=False).split('"'))
        hits = [k for k in raw['keys'] if k in names or k.split(' ', 1)[-1] in names]
        return not hits, f'합성 파일에서 발견된 실제 시군구 이름 {len(hits)}개'
    safe('D01', '합성 파일에 실제 시군구 이름 없음', d01)

    def d02():
        a = dr['DEMO-SGG-A']
        return len(a['peers']) == 5 and a['profile']['total_amount'] == max(r['profile']['total_amount'] for r in D), f"유사 지자체 {len(a['peers'])}곳"
    safe('D02', 'A: 대도시 구 기본 구성, 유사 지자체 5곳', d02)

    def d03():
        b = dr['DEMO-SGG-B']
        mar = b['months'][MONTHS.index('2026-03')]
        ok = b['size_flag']['is_small'] and '대형할인점' in b['derived_flags']['no_payment_industries'] and mar['calculation_status'] == 'no_data'
        return ok, f"소액 {b['size_flag']['is_small']}, 결제 없음 {b['derived_flags']['no_payment_industries']}, 3월 {mar['calculation_status']}"
    safe('D03', 'B: 소액 지역·대형할인점 결제 없음·3월 no_data', d03)

    def d04():
        t = [dr[k]['foreign_profile']['type'] for k in ['DEMO-SGG-C', 'DEMO-SGG-D', 'DEMO-SGG-E']]
        return t == ['생활유통형', '한식외식형', '복합형'], f'C·D·E 유형 {t}'
    safe('D04', 'C·D·E: R03 유형 순서', d04)

    def d05():
        f, i = dr['DEMO-SGG-F'], dr['DEMO-SGG-I']
        ok = f['derived_flags']['foreign_share_low'] and i['derived_flags']['foreign_share_low'] and i['foreign_profile']['type'] is None \
            and i['foreign_profile']['type_status'] == 'not_shown_small_foreign_amount'
        return ok, f"F 낮음 {f['derived_flags']['foreign_share_low']}, I 낮음 {i['derived_flags']['foreign_share_low']}, I 유형 {i['foreign_profile']['type_status']}"
    safe('D05', 'F·I: R12 켜짐, I는 R03 유형 미표시', d05)

    def d06():
        g = dr['DEMO-SGG-G']
        return g['season_summary']['flagged_months'] == ['2026-02'], f"발동월 {g['season_summary']['flagged_months']}"
    safe('D06', 'G: 2월만 R04 시기 발동', d06)

    def d07():
        h = dr['DEMO-SGG-H']
        top = max(D, key=lambda r: r['size_flag']['unknown_share_pct'])['scope']['region_key']
        ok = top == 'DEMO-SGG-H' and all(m['known_only_share_pct'] is not None and m['known_only_share_pct'] > m['foreign_share_pct'] for m in h['months'])
        return ok, f"미상 비중 최대 {top}, 미상 {h['size_flag']['unknown_share_pct']:.1f}%"
    safe('D07', 'H: 미상 비중이 가장 크고 두 비중(전체·미상 제외)이 함께 있음', d07)

    def d08():
        j, k = dr['DEMO-SGG-J'], dr['DEMO-SGG-K']
        j_ok = j['external']['depopulation_area_2021'] and j['national_rank']['industry_share_pct']['일반한식']['percentile'] >= 80
        k_ok = any(p.endswith('|6') for p in k['derived_flags']['low_industry_age_pairs']) and k['applicability']['R08']['display_level'] == 'reference'
        return j_ok and k_ok, f'J 인구감소지역·한식 상위 {j_ok}, K 60대 이상 낮은 쌍·참고 표시 {k_ok}'
    safe('D08', 'J·K: 인구감소지역 라벨, R08 60대 이상(참고용)', d08)

    def d09():
        l = dr['DEMO-SGG-L']
        ok = l['external'] is None and l['applicability']['R11']['status'] == 'blocked' and l['region_codes'] is None
        return ok, f"external {l['external']}, R11 {l['applicability']['R11']['status']}"
    safe('D09', 'L: 외부 자료 없음 -> external null, R11 blocked', d09)


def main():
    ap = argparse.ArgumentParser(description='근거 파일 자동 검증 35항목')
    ap.add_argument('--raw', default=str(RAW_CSV))
    ap.add_argument('--real', default=str(PRIVATE_DIR / 'review_evidence_real_v2.1.json'))
    ap.add_argument('--demo', default=str(EXAMPLES_DIR / 'review_evidence_demo_v2.1.json'))
    ap.add_argument('--mode', choices=['real', 'check'], default='real',
                    help='real: 보고 집계값까지 비교 / check: 원본 재계산값과만 비교(가짜 원본 실행 점검)')
    ap.add_argument('--report', default=str(OUTPUT_DIR / 'validation_35.json'))
    a = ap.parse_args()
    run(a.raw, a.real, a.demo, a.mode)
    for r in RESULTS:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['id']}  {r['name']}  | {r['detail']}")
    n_ok = sum(r['ok'] for r in RESULTS)
    print(f'\nPASS {n_ok}개, FAIL {len(RESULTS) - n_ok}개 (전체 {len(RESULTS)}항목, 모드 {a.mode})')
    print('전체 결과:', '통과' if n_ok == len(RESULTS) == 35 else '실패')
    out = Path(a.report); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(mode=a.mode, results=RESULTS), ensure_ascii=False, indent=2), encoding='utf-8')
    sys.exit(0 if n_ok == len(RESULTS) == 35 else 1)


if __name__ == '__main__':
    main()
