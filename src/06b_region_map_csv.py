"""BC 시도·시군구명 -> 행정구역 코드 매핑표 초안 + 인구감소지역(2021 지정) 표시.
입력: work/map_draft.pkl (시군구명별 법정동 코드 후보를 모은 초안, 공개 코드표에서 만든 중간 파일)
출력: data/external/region_code_map.csv (초안. 07_process_external.py 가 행정기관코드로 확정한다)
"""
import pandas as pd
from common import WORK_DIR, EXT_DIR

m=pd.read_pickle(WORK_DIR / 'map_draft.pkl')
m['region_key']=m.SIDO_NM+' '+m.CCG_NM
m['sido_code']=m.sgg_code5.where(m.sgg_code5!='NA').str[:2]
sido_code={'서울특별시':'11','부산광역시':'26','대구광역시':'27','인천광역시':'28','광주광역시':'29','대전광역시':'30','울산광역시':'31','세종특별자치시':'36','경기도':'41','강원특별자치도':'51','충청북도':'43','충청남도':'44','전북특별자치도':'52','전라남도':'46','경상북도':'47','경상남도':'48','제주특별자치도':'50'}
m['sido_code']=m.SIDO_NM.map(sido_code)
m['is_gu_of_city']=m.CCG_NM.str.contains(' ')
m['parent_city']=m.CCG_NM.where(m.is_gu_of_city).str.split(' ').str[0]
# 인구감소지역(2021.10 지정 89곳; 군위군은 2023.7 경북→대구 편입 후에도 유지)
depop={'부산광역시':['동구','서구','영도구'],'대구광역시':['남구','서구','군위군'],'인천광역시':['강화군','옹진군'],
'경기도':['가평군','연천군'],
'강원특별자치도':['고성군','삼척시','양구군','양양군','영월군','인제군','정선군','철원군','태백시','평창군','홍천군','화천군'],
'충청북도':['괴산군','단양군','보은군','영동군','옥천군','제천시'],
'충청남도':['공주시','금산군','논산시','보령시','부여군','서천군','예산군','청양군','태안군'],
'전북특별자치도':['고창군','김제시','남원시','무주군','부안군','순창군','임실군','장수군','정읍시','진안군'],
'전라남도':['강진군','고흥군','곡성군','구례군','담양군','보성군','신안군','영광군','영암군','완도군','장성군','장흥군','진도군','함평군','해남군','화순군'],
'경상북도':['고령군','문경시','봉화군','상주시','성주군','안동시','영덕군','영양군','영주시','영천시','울릉군','울진군','의성군','청도군','청송군'],
'경상남도':['거창군','고성군','남해군','밀양시','산청군','의령군','창녕군','하동군','함안군','함양군','합천군']}
dp=set((s,c) for s,l in depop.items() for c in l)
print('depop total',len(dp))
m['depop_2021']=[ (s,c) in dp for s,c in zip(m.SIDO_NM,m.CCG_NM)]
print('depop in BC:',m.depop_2021.sum())
def status(r):
    if r.sgg_code5=='NA': return '미확정'
    if r.is_gu_of_city and r.parent_city=='부천시': return '확인 필요'
    return '작성(대조 전)'
m['code_status']=m.apply(status,axis=1)
m['sgg_code5']=m.sgg_code5.where(m.sgg_code5!='NA','')
notes={'대구광역시 군위군':'2023-07 경북→대구 편입, 코드 27720',
'세종특별자치시 세종특별자치시':'단일 시(구 없음)',
'인천광역시 미추홀구':'구 남구(2018 개칭)',
}
def note(r):
    k=r.region_key
    if k in notes: return notes[k]
    if r.SIDO_NM=='경기도' and r.CCG_NM.startswith('화성시'): return '일반구 신설(BC 자료에 4개 구로 등장). 표준코드 미확인, 인구자료 대조로 확정'
    if r.SIDO_NM=='경기도' and r.CCG_NM.startswith('부천시'): return '일반구 코드 재확인 필요(폐지 후 재설치 여부)'
    if r.SIDO_NM=='강원특별자치도': return '2023-06 개칭, 코드 접두 42→51'
    if r.SIDO_NM=='전북특별자치도': return '2024-01 개칭, 코드 접두 45→52'
    return ''
m['note']=m.apply(note,axis=1)
out=m[['region_key','SIDO_NM','CCG_NM','sido_code','sgg_code5','parent_city','depop_2021','code_status','note']].rename(columns={'SIDO_NM':'sido_nm','CCG_NM':'ccg_nm'}).sort_values(['sido_code','sgg_code5','ccg_nm'])
EXT_DIR.mkdir(parents=True,exist_ok=True)
out.to_csv(EXT_DIR / 'region_code_map.csv',index=False,encoding='utf-8-sig')
print(out.code_status.value_counts().to_dict(), len(out), out.region_key.is_unique)
