"""외부 데이터 가공: 주민등록 인구(2026-08-31, 행정동) + 등록외국인(시군구) -> BC 255개 시군구 기준 표
입력: data/external/raw/ext_pop.csv, data/external/raw/ext_foreign.csv (공공데이터 원본, cp949),
      data/external/region_code_map.csv(06b가 만든 초안)
출력: data/external/region_code_map.csv(확정), ext_population.csv, ext_registered_foreign.csv
"""
import pandas as pd, numpy as np, re, warnings
from common import EXT_DIR, EXT_RAW_DIR, WORK_DIR
warnings.filterwarnings('ignore')
m=pd.read_csv(EXT_DIR / 'region_code_map.csv',dtype=str,encoding='utf-8-sig').fillna('')
m=m.rename(columns={'sgg_code5':'sgg_code5_legal_draft'})
p=pd.read_csv(EXT_RAW_DIR / 'ext_pop.csv',encoding='cp949',dtype={'행정기관코드':str})
p['sgg5']=p.행정기관코드.str[:5]
p['시군구명']=p.시군구명.fillna('')
# --- 1. 행정동 -> BC region_key
def psido(s): return '전남광주통합특별시' if s in ('광주광역시','전라남도') else s
m['psido']=m.sido_nm.map(psido)
key_of={(r.psido,r.ccg_nm):r.region_key for r in m.itertuples()}
key_of[('세종특별자치시','')]='세종특별자치시 세종특별자치시'
# 인천 2026-07 구 개편: BC(2026-01~06)는 옛 구 기준 -> 옛 구 구성 행정동으로 되돌린다
JUNG_INLAND=['신포동','연안동','신흥동','도원동','율목동','동인천동','개항동']
def region(r):
    if r.시도명=='세종특별자치시': return '세종특별자치시 세종특별자치시'
    if r.시도명=='인천광역시':
        if r.시군구명 in ('영종구',): return '인천광역시 중구'
        if r.시군구명=='제물포구': return '인천광역시 중구' if r.읍면동명 in JUNG_INLAND else '인천광역시 동구'
        if r.시군구명 in ('서해구','검단구'): return '인천광역시 서구'
    return key_of.get((r.시도명,r.시군구명))
p['region_key']=p.apply(region,axis=1)
print('미매핑 행정동:',p.region_key.isna().sum())
# --- 2. 연령 구간(BC AGE_CD): 1=20세 미만 2=20대 3=30대 4=40대 5=50대 6=60대 이상
cols=[c for c in p.columns if re.match(r'^\d+세(남자|여자)$|^110세이상',c)]
age_of={}
for c in cols:
    n=int(re.match(r'^(\d+)',c).group(1)); age_of[c]=n
def code(n): return 1 if n<20 else 2 if n<30 else 3 if n<40 else 4 if n<50 else 5 if n<60 else 6
bins={k:[c for c in cols if code(age_of[c])==k] for k in range(1,7)}
assert sum(len(v) for v in bins.values())==len(cols)==222, len(cols)
for k,v in bins.items(): p[f'pop_age{k}']=p[v].sum(axis=1)
p['pop_total']=p['계']
assert (p[[f'pop_age{k}' for k in range(1,7)]].sum(axis=1)==p.pop_total).all()
g=p.groupby('region_key')[['pop_total']+[f'pop_age{k}' for k in range(1,7)]].sum()
g['base_date']='2026-08-31'
# 코드 열
m['sgg_code5']=m.region_key.map(lambda k:'')
code_pop=p.groupby('region_key').sgg5.agg(lambda s:';'.join(sorted(set(s))))
m['sgg_code5']=m.region_key.map(code_pop)
m['sgg_code5_pre2026']=m.sgg_code5_legal_draft   # BC 자료 기간(2026-01~06) 시점 코드(광주 29xxx/전남 46xxx/인천 옛 구)
def match(r):
    if ';' in r.sgg_code5 or r.region_key.startswith('인천광역시 중구') or r.region_key.startswith('인천광역시 동구') : return '합산·분할(2026-07 개편)'
    if r.sido_nm in ('광주광역시','전라남도'): return '코드 변경(전남광주 통합)'
    return '일치'
m['match_type']=m.apply(match,axis=1)
m.loc[m.region_key=='인천광역시 중구','pop_note']='영종구 전체 + 제물포구 중 옛 중구 7개 동'
m.loc[m.region_key=='인천광역시 동구','pop_note']='제물포구 중 옛 동구 11개 동'
m.loc[m.region_key=='인천광역시 서구','pop_note']='서해구 + 검단구 합산'
m['pop_note']=m.pop_note.fillna('')
m.loc[m.ccg_nm.str.startswith('화성시'),'note']='2026-07 신설 일반구. 코드는 2026-08 주민등록 인구 파일의 행정기관코드로 확정'
m.loc[m.sido_nm.isin(['광주광역시','전라남도']),'note']='2026-07 전남광주통합특별시 출범으로 행정기관코드가 12xxx로 바뀜. sgg_code5_pre2026은 통합 전 코드'
m.loc[m.region_key=='세종특별자치시 세종특별자치시','note']='단일 시(구 없음)'
m['code_status']='대조 완료(2026-08 행정기관코드)'
# 인천 옛 구 인구 대조(2026-09-20): 인천시 행정체제 개편 자료(2026-07-01 기준 168,271·58,162·635,553명, 영종구 126,145명)와 비교
m.loc[m.region_key=='인천광역시 중구','note']='옛 구 인구 대조: 영종구를 뺀 원도심 7개 동 41,342명(2026-08-31)은 인천시 자료(168,271-126,145=42,126명, 2026-07-01)보다 1.9% 적음. 영종구는 성장 중이라 8월 138,659명이 7월 자료 126,145명보다 큼'
m.loc[m.region_key=='인천광역시 동구','note']='옛 구 인구 대조: 11개 동 57,875명(2026-08-31)은 인천시 자료 58,162명(2026-07-01)보다 0.5% 적음'
m.loc[m.region_key=='인천광역시 서구','note']='옛 구 인구 대조: 서해구+검단구 668,150명(2026-08-31)은 인천시 자료 635,553명(2026-07-01)보다 5.1% 많음. 검단·영종의 인구 증가와 자료 기준 시점 차이로 추정'
# --- 3. 인구 표
pop=m[['region_key','sgg_code5']].merge(g.reset_index(),on='region_key',how='left')
for k in range(1,7): pop[f'share_age{k}']=(pop[f'pop_age{k}']/pop.pop_total).round(4)
pop.to_csv(EXT_DIR / 'ext_population.csv',index=False,encoding='utf-8-sig')
# --- 4. 등록외국인 (BC 기간 2026-01~06, 옛 지역 기준으로 255개 일치)
f=pd.read_csv(EXT_RAW_DIR / 'ext_foreign.csv',encoding='cp949',dtype=str)
f['시도']=f.시도.str.replace('　','').str.strip().replace({'경기':'경기도','강원도':'강원특별자치도','전라북도':'전북특별자치도'}); f['시군구']=f.시군구.str.strip()
# 원본 오류 보정: 2026-01~05 목포시·여수시가 '전라북도'로 적재됨 -> 전라남도
f.loc[f.시군구.isin(['목포시','여수시'])&(f.시도=='전북특별자치도'),'시도']='전라남도'
f=f[(f.년=='2026')&(f.월.isin(['01','02','03','04','05','06']))].copy()
f['ym']=f.년+f.월; f['n']=f.등록외국인수.astype(int)
f['region_key']=f.시도+' '+f.시군구
piv=f.pivot_table(index='region_key',columns='ym',values='n',aggfunc='sum')
piv.columns=[f'reg_foreign_{c}' for c in piv.columns]
rf=m[['region_key','sgg_code5_pre2026']].merge(piv.reset_index(),on='region_key',how='left')
print('등록외국인 결측 시군구:',rf.iloc[:,2:].isna().any(axis=1).sum(), '/',len(rf))
rf.to_csv(EXT_DIR / 'ext_registered_foreign.csv',index=False,encoding='utf-8-sig')
# --- 5. 매핑표 확정 저장
final=m[['region_key','sido_nm','ccg_nm','sgg_code5','sgg_code5_pre2026','match_type','pop_note','depop_2021','code_status','note']]
final.to_csv(EXT_DIR / 'region_code_map.csv',index=False,encoding='utf-8-sig')
print(final.match_type.value_counts().to_dict(), len(final), final.region_key.is_unique)
print('인구 합계:',int(pop.pop_total.sum()),'| 결측:',pop.pop_total.isna().sum())
WORK_DIR.mkdir(exist_ok=True)
pd.to_pickle(dict(pop=pop,rf=rf,map=final),WORK_DIR / 'ext.pkl')
