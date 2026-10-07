"""시군구별 소비 프로필 특징 계산 (집계값만 산출; 원본 행 출력 없음)
입력: data/ABP_CONTEST_DATA.csv (BC 제공 원본)  출력: work/features.pkl (로컬 작업용, 공개 금지)
08_threshold_sensitivity.py, 11_threshold_decision.py 가 이 파일을 읽는다.
"""
import pandas as pd, numpy as np
from common import read_raw, WORK_DIR

d = read_raw()
d['key'] = d.SIDO_NM + ' ' + d.CCG_NM
T = d.groupby('key').amt.sum().rename('T')
F = d[d.GENDER_CD.astype(str) == '3'].groupby('key').amt.sum().rename('F')
U = d[d.GENDER_CD.astype(str).str.lower() == 'x'].groupby('key').amt.sum().rename('U')
C = d.groupby('key').cnt.sum().rename('C')
f = pd.concat([T, F, U, C], axis=1).fillna(0)
f['foreign_share'] = f.F / f['T']
f['unk_share'] = f.U / f['T']

# 업종 비중(전체 결제 대비)
ind = d.pivot_table(index='key', columns='TP_BUZ_NM', values='amt', aggfunc='sum', fill_value=0)
ind_share = ind.div(ind.sum(axis=1), axis=0)

# 월별 총액 및 국가 패턴 보정 지수
m = d.pivot_table(index='key', columns='STRD_YYMM', values='amt', aggfunc='sum', fill_value=0)
days = pd.Series({202601: 31, 202602: 28, 202603: 31, 202604: 30, 202605: 31, 202606: 30})
m_pd = m.div(days, axis=1)                       # 일평균
idx_raw = m_pd.div(m_pd.mean(axis=1), axis=0)    # 지역 자기 평균 대비(일수 보정)
nat = m_pd.sum(axis=0); nat_idx = nat / nat.mean()
idx_adj = idx_raw.div(nat_idx, axis=1)           # 전국 월 패턴 제거 = 지역 특이 월

# 외국인 결제 구조
fo = d[d.GENDER_CD.astype(str) == '3'].pivot_table(index='key', columns='TP_BUZ_NM', values='amt', aggfunc='sum', fill_value=0)
fo_share = fo.div(fo.sum(axis=1), axis=0)

WORK_DIR.mkdir(exist_ok=True)
pd.to_pickle(dict(f=f, ind=ind, ind_share=ind_share, m=m, idx_raw=idx_raw, idx_adj=idx_adj,
                  fo=fo, fo_share=fo_share, nat_idx=nat_idx), WORK_DIR / 'features.pkl')
print(f.shape, ind_share.shape, m.shape, fo.shape)
print(sorted(ind.columns))
print(nat_idx.round(3).to_dict())
