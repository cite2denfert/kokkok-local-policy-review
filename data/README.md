# data/

| 위치 | 내용 | 저장소 |
| --- | --- | --- |
| `data/ABP_CONTEST_DATA.csv` | 공모전 제공 원본(BC카드 2026-01~06) | **올리지 않음**(재배포 불가, `.gitignore`) |
| `data/synthetic/` | `src/make_synthetic_raw.py`가 만드는 가짜 원본(실행 점검용) | 올리지 않음(다시 만들 수 있음) |
| `data/external/` | 공공데이터 가공본: `region_code_map.csv`, `ext_population.csv`, `ext_registered_foreign.csv` | 공개 자료라 넣으면 함께 올라감(09 real·11 R11에 필요) |
| `data/external/raw/` | 공공데이터 원본(`ext_pop.csv`, `ext_foreign.csv`) | 올리지 않음 |

외부 자료 출처
- 주민등록 인구: 행정안전부 행정동별 성별·연령별 주민등록 인구(2026-08-31)
- 등록외국인: 법무부 시군구별 등록외국인 현황(공공데이터포털 15100022, 2026-01~06)
- 인구감소지역: 2021년 지정 89곳(행정안전부 고시)
