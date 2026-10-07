"""실행 점검용 가짜 원본 CSV 만들기.

공모전 원본은 재배포할 수 없어서, 같은 열 구조(9열)를 가진 가짜 자료로 파이프라인 전체가 돌아가는지 확인한다.
지역 이름은 실제와 겹치지 않는 '가상도01 가상시001' 형식이고, 모든 수치는 임의 분포에서 뽑았다.
실제 분석 결과와 아무 관계가 없다.

  python src/make_synthetic_raw.py                       # -> data/synthetic/ABP_SYNTHETIC.csv
  BC_RAW_CSV=data/synthetic/ABP_SYNTHETIC.csv python src/12_insights_figures.py --out output/synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR, IND_NAME, MONTHS, DAYS

RAW_NM = {'4004': '대형할인점', '4010': '편 의 점', '4020': '슈퍼 마켓', '8001': '일반한식', '8002': '갈비전문점', '8003': '한정식',
          '8004': '일식회집', '8005': '중국음식', '8006': '서양음식', '8021': '스넥', '8301': '제 과 점'}   # 원본 표기(띄어쓰기 포함)


def make(n_regions=255, seed=7):
    rng = np.random.default_rng(seed)
    codes = list(IND_NAME)
    sidos = [f'가상도{i:02d}' for i in range(1, 18)]
    regions = [(sidos[i % 17], f'가상시{i + 1:03d}') for i in range(n_regions)]
    ind_alpha = np.array([10, 8, 9, 30, 3, 2, 4, 5, 6, 3, 5], float)
    rows = []
    for si, (sido, ccg) in enumerate(regions):
        scale = float(np.exp(rng.normal(23.5, 1.2)))                  # 6개월 총결제 규모
        fs = float(np.clip(rng.beta(3, 40), 0.01, 0.3))               # 외국인 비중
        unk = float(np.clip(rng.beta(1.2, 40), 0.003, 0.35))          # 성별 미상 비중
        ish = rng.dirichlet(ind_alpha * 5)
        if scale < np.exp(22.6) and rng.random() < 0.6:               # 소규모 지역 일부는 대형할인점 결제 없음
            ish[0] = 0; ish /= ish.sum()
        fish = rng.dirichlet(ind_alpha * 2)
        fish[0] = 0 if ish[0] == 0 else fish[0]; fish /= fish.sum()
        ash = rng.dirichlet(np.array([1, 9, 15, 20, 24, 30], float) * 6)
        fash = rng.dirichlet(np.array([2, 12, 12, 10, 8, 6], float) * 3)
        noise = 0.02 + 0.25 / np.log(scale)                           # 작을수록 월 변동이 큼
        mw = rng.normal(1.0, noise, 6) * np.array([DAYS[m] for m in MONTHS]) * np.array([0.96, 1.0, 0.97, 0.99, 1.10, 1.0])
        if rng.random() < 0.08:
            mw[rng.integers(0, 6)] *= 1.3
        mw = np.clip(mw, 0.2, None); mw /= mw.sum()
        for mi, ym in enumerate(MONTHS):
            tm = scale * mw[mi]
            fm, um = tm * fs, tm * unk
            km = tm - fm - um
            for ii, code in enumerate(codes):
                for ai, a in enumerate(['1', '2', '3', '4', '5', '6']):
                    v = km * ish[ii] * ash[ai]
                    if v >= 1:
                        male = rng.uniform(0.4, 0.6)
                        rows.append((ym, sido, ccg, code, '1', a, v * male))
                        rows.append((ym, sido, ccg, code, '2', a, v * (1 - male)))
                    fv = fm * fish[ii] * fash[ai]
                    if fv >= 1:
                        rows.append((ym, sido, ccg, code, '3', a, fv))
                uv = um * ish[ii]
                if uv >= 1:
                    rows.append((ym, sido, ccg, code, 'x', 'x', uv))
    d = pd.DataFrame(rows, columns=['ym', 'SIDO_NM', 'CCG_NM', 'TP_BUZ_NO', 'GENDER_CD', 'AGE_CD', 'amt'])
    d['amt'] = (d.amt * rng.lognormal(0, 0.35, len(d))).round().astype('int64')   # 행 단위 잡음(반기 재현성이 1이 되지 않게)
    d = d[d.amt > 0]
    d['cnt'] = np.maximum(1, (d.amt / rng.uniform(9000, 30000, len(d))).round()).astype('int64')
    d['STRD_YYMM'] = d.ym.str.replace('-', '').astype(int)
    d['TP_BUZ_NM'] = d.TP_BUZ_NO.map(RAW_NM)
    d['TP_BUZ_NO'] = d.TP_BUZ_NO.astype(int)
    return d[['STRD_YYMM', 'SIDO_NM', 'CCG_NM', 'TP_BUZ_NO', 'TP_BUZ_NM', 'GENDER_CD', 'AGE_CD', 'amt', 'cnt']]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(DATA_DIR / 'synthetic' / 'ABP_SYNTHETIC.csv'))
    a = ap.parse_args()
    d = make()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False, encoding='utf-8-sig')
    print('wrote', out, d.shape, 'regions', d[['SIDO_NM', 'CCG_NM']].drop_duplicates().shape[0])
