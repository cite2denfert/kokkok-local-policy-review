"""임계 기준 민감도 분석(원본 CSV, work/features.pkl 필요). 출력은 집계값만.
05_build_features.py 를 먼저 실행한다. 탐색 단계에서 쓴 세 스크립트(sens, sens2, sens3)를 한 파일로 묶었다.
"""
from common import read_raw, WORK_DIR
FEATURES = WORK_DIR / 'features.pkl'

# ===== sens.py =====
import pandas as pd, numpy as np
D=pd.read_pickle(FEATURES); f=D['f']; ish=D['ind_share']
d=read_raw(); d['key']=d.SIDO_NM+' '+d.CCG_NM
pd.set_option('display.width',200)
# --- A. 월별 업종 비중 안정성: 업종별로 6개월 전체 하위20% 집합 vs 개별 월 하위20% 집합 Jaccard
inds=['일반한식','서양음식','슈퍼 마켓','편 의 점','대형할인점','스넥','제 과 점','중국음식','일식회집']
d['ind']=d.TP_BUZ_NM
mi=d.pivot_table(index=['STRD_YYMM','key'],columns='ind',values='amt',aggfunc='sum',fill_value=0)
ms=mi.div(mi.sum(axis=1),axis=0)
print("== A. 하위 q% 집합의 월간 재현성 (평균 Jaccard, 전체6개월 vs 각 월) ==")
rows=[]
for q in [10,20,30]:
    r={'q':q}
    for i in inds:
        base=ish[i]<=np.percentile(ish[i],q)
        js=[]
        for ym in sorted(ms.index.get_level_values(0).unique()):
            s=ms.loc[ym][i].reindex(ish.index).fillna(0)
            fl=s<=np.percentile(s,q)
            js.append((base&fl).sum()/max((base|fl).sum(),1))
        r[i]=round(np.mean(js),2)
    rows.append(r)
print(pd.DataFrame(rows).set_index('q').T)
# --- B. 절대 수준: 하위 q 컷오프가 중앙값 대비 몇 %인지
print("\n== B. 업종별 컷오프(비중 %) / 중앙값 ==")
rows=[]
for i in inds:
    x=ish[i]*100
    rows.append(dict(ind=i,zero=(x==0).sum(),p10=x.quantile(.1),p20=x.quantile(.2),p30=x.quantile(.3),p50=x.median(),p80=x.quantile(.8),p20_over_p50=x.quantile(.2)/x.median() if x.median()>0 else np.nan))
print(pd.DataFrame(rows).round(2).set_index('ind'))
# --- C. 규모와 월간 변동: 업종비중 CV(월별) vs T 분위
print("\n== C. 규모별(T 5분위) 월별 업종 비중 변동(평균 CV) ==")
T=f['T']; qb=pd.qcut(T,5,labels=['Q1소','Q2','Q3','Q4','Q5대'])
cv=(ms.groupby(level=1).std()/ms.groupby(level=1).mean().replace(0,np.nan))[inds].reindex(T.index)
print(cv.groupby(qb).mean().round(2).assign(mean=lambda x:x.mean(axis=1)).round(2))
print("T 5분위 경계(백만원):",(T.quantile([0,.1,.2,.4,.6,.8,1])/1e6).round(0).tolist())
# --- D. 시기 지수: raw vs adj, 임계별 지역별 발동 월 수
print("\n== D. R04 시기(지수) 임계 민감도 ==")
for name,idx in [('raw(일수보정)',D['idx_raw']),('adj(전국월패턴 제거)',D['idx_adj'])]:
    print(name)
    for q in [70,80,90]:
        # 월별: 그 월 지수가 255개 시군구 분포 상위(100-q)%
        fl=pd.DataFrame({c:idx[c]>=np.percentile(idx[c],q) for c in idx.columns})
        print(f"  상위{100-q}%: 월별 발동 시군구 수={fl.sum().to_dict()}, 1개월이상 발동={fl.any(axis=1).mean():.0%}")
# 국가 패턴이 raw 지수에 미치는 영향: 5월 raw 상위 20%가 몇%가 adj에서도 상위 20%인가
for c in D['idx_raw'].columns:
    a=D['idx_raw'][c]>=np.percentile(D['idx_raw'][c],80); b=D['idx_adj'][c]>=np.percentile(D['idx_adj'][c],80)
    print(c,'raw/adj 상위20% 겹침',round((a&b).sum()/(a|b).sum(),2), '| raw 평균', round(D['idx_raw'][c].mean(),3))
# --- E. 임계 절대값 기준 대안: 지수 >=1.10 / 1.15 / 1.20
print("\n== E. 절대 지수 기준 (adj) 월별 발동 비율 ==")
for th in [1.05,1.10,1.15,1.20,1.30]:
    print(th, {c:round((D['idx_adj'][c]>=th).mean(),2) for c in D['idx_adj'].columns})
# 지수 변동의 노이즈 크기: 규모 Q1 vs Q5 adj 지수 표준편차
print("\n규모 5분위별 adj 지수 표준편차 평균:", D['idx_adj'].std(axis=1).groupby(qb).mean().round(3).to_dict())

# ===== sens2.py =====
T=f['T']
print("연령코드별 결제 비중(%):", (d.groupby(d.AGE_CD.astype(str)).amt.sum()/d.amt.sum()*100).round(2).to_dict())
idx=D['idx_adj']
size_low=T<=T.quantile(.2)
print("size_low n=",size_low.sum())
for th_hi,th_lo in [(1.10,1.10),(1.10,1.20),(1.10,1.25),(1.15,1.25),(1.15,1.30)]:
    thr=np.where(size_low,th_lo,th_hi)
    fl=pd.DataFrame({c:idx[c].values>=thr for c in idx.columns},index=idx.index)
    print(th_hi,th_lo,'월별 발동 수',fl.sum().to_dict(),'| 1개월이상 지역 비율',round(fl.any(axis=1).mean(),2),'| 소규모 발동 비율',round(fl[size_low].any(axis=1).mean(),2))
# 발동한 월 분포(1.10/1.25)
thr=np.where(size_low,1.25,1.10)
fl=pd.DataFrame({c:idx[c].values>=thr for c in idx.columns},index=idx.index)
print("지역당 발동 월 수 분포:",fl.sum(axis=1).value_counts().sort_index().to_dict())
# R03 유형 안정성: 전반(1-3월) vs 후반(4-6월)
fo=d[d.GENDER_CD.astype(str)=='3'].copy()
fo['half']=np.where(fo.STRD_YYMM<=202603,'H1','H2')
def types(x, top=0.5, cut=0.3):
    piv=x.pivot_table(index='key',columns='TP_BUZ_NM',values='amt',aggfunc='sum',fill_value=0)
    tot=piv.sum(axis=1); life=(piv['슈퍼 마켓']+piv['편 의 점'])/tot
    kr2=piv['일반한식']/tot
    sel=tot>=tot.quantile(1-top)
    lc=life[sel].quantile(1-cut); kc=kr2[sel].quantile(1-cut)
    t=pd.Series('기타(하위50%)',index=piv.index)
    t[sel]='혼합'
    t[sel&(kr2>=kc)]='한식'
    t[sel&(life>=lc)]='생활'
    return t,life,kr2,tot
tt,life,kr,tot=types(fo)
print("\n전체 유형 분포:",tt.value_counts().to_dict())
a,_,_,_=types(fo[fo.half=='H1']); b,_,_,_=types(fo[fo.half=='H2'])
a,b=a.reindex(tt.index).fillna('기타(하위50%)'),b.reindex(tt.index).fillna('기타(하위50%)')
both=(a!='기타(하위50%)')&(b!='기타(하위50%)')&(tt!='기타(하위50%)')
print("반기 일치율(양쪽 다 대상):",round((a[both]==b[both]).mean(),2),'n=',both.sum())
# 컷오프 민감도
for top,cut in [(0.5,0.2),(0.5,0.3),(0.5,0.4),(0.4,0.3),(0.6,0.3)]:
    t,_,_,_=types(fo,top,cut); print(top,cut,t.value_counts().to_dict())
piv=fo.pivot_table(index='key',columns='TP_BUZ_NM',values='amt',aggfunc='sum',fill_value=0)
print("외국인 결제 규모: 하위50% 경계(백만원)",round(piv.sum(axis=1).median()/1e6,1),"최소",round(piv.sum(axis=1).min()/1e6,2))
sel=tot>=tot.quantile(.5)
print("생활&한식 동시 상위30%:",((life[sel]>=life[sel].quantile(.7))&(kr[sel]>=kr[sel].quantile(.7))).sum())
# R08 연령 비중 안정성 (업종x연령). 원본의 연령 미상 코드는 소문자 x
a=d.copy(); a['ag']=a.AGE_CD.astype(str).str.lower()
a=a[a.ag!='x']
print("\n연령 x(누락) 행 비중:", round((d.AGE_CD.astype(str).str.lower()=='x').mean(),3))
for ind in ['일반한식','편 의 점']:
    s=a[a.TP_BUZ_NM==ind].pivot_table(index='key',columns='ag',values='amt',aggfunc='sum',fill_value=0)
    sh=s.div(s.sum(axis=1),axis=0)
    print(ind,'연령별 비중 p20/p50/p80(%)',{c:(round(sh[c].quantile(.2)*100,1),round(sh[c].median()*100,1),round(sh[c].quantile(.8)*100,1)) for c in sh.columns})

# ===== sens3.py =====
d['g']=d.GENDER_CD.astype(str).str.lower()
# R12 외국인 비중 하위20% 반기 재현성
def fs(x):
    t=x.groupby('key').amt.sum(); F=x[x.g=='3'].groupby('key').amt.sum().reindex(t.index).fillna(0); return F/t
a=fs(d[d.STRD_YYMM<=202603]); b=fs(d[d.STRD_YYMM>202603]); w=fs(d)
for q in [10,20,30]:
    A=a<=np.percentile(a,q);B=b<=np.percentile(b,q);W=w<=np.percentile(w,q)
    print(q,'H1vsH2 Jaccard',round((A&B).sum()/(A|B).sum(),2),'| 전체와 H1',round((W&A).sum()/(W|A).sum(),2))
low=w<=np.percentile(w,20); print('하위20% 중 규모하위20% 비율',round((low&(T<=T.quantile(.2))).sum()/low.sum(),2))
print('foreign share p10/20/50/80(%)',[round(w.quantile(q)*100,1) for q in [.1,.2,.5,.8]])
# 대형할인점 0 지역의 규모
ind=D['ind']; z=ind['대형할인점']==0
print('대형할인점 0 n=',z.sum(),'중 규모 하위20% 비율',round((z&(T<=T.quantile(.2))).sum()/max(z.sum(),1),2),'| 규모 중앙값 대비(0인 지역 T 중앙값/전체)',round(T[z].median()/T.median(),2) if z.any() else None)
# R02 발동 수: 업종별 하위20% (0 포함) & 규모 하위20% 겹침
for i in ['일반한식','서양음식','슈퍼 마켓','편 의 점','스넥','제 과 점','중국음식','일식회집']:
    fl=ish[i]<=ish[i].quantile(.2); print(i,'발동',fl.sum(),'그중 소규모',(fl&(T<=T.quantile(.2))).sum())
