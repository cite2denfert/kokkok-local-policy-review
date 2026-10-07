from pathlib import Path
import pandas as pd

# 프로젝트 폴더 위치
BASE_DIR = Path(__file__).resolve().parent.parent

# CSV 위치
CSV_PATH = BASE_DIR / "data" / "ABP_CONTEST_DATA.csv"

# CSV 읽기
df = pd.read_csv(
    CSV_PATH,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig"
)

print("=" * 50)
print("1. 데이터 크기")
print("=" * 50)

print("행:", len(df))
print("열:", len(df.columns))

print("\n컬럼 목록")
print(df.columns.tolist())


print("\n" + "=" * 50)
print("2. 기간")
print("=" * 50)

print(sorted(df["STRD_YYMM"].unique()))


print("\n" + "=" * 50)
print("3. GENDER_CD")
print("=" * 50)

print(sorted(df["GENDER_CD"].unique()))


print("\n" + "=" * 50)
print("4. AGE_CD")
print("=" * 50)

print(sorted(df["AGE_CD"].unique()))


print("\n" + "=" * 50)
print("5. 지역")
print("=" * 50)

print("시도 수:", df["SIDO_NM"].nunique())

region_count = (
    df[["SIDO_NM", "CCG_NM"]]
    .drop_duplicates()
    .shape[0]
)

print("시도-시군구 조합:", region_count)


print("\n" + "=" * 50)
print("6. 업종")
print("=" * 50)

industry_count = (
    df[["TP_BUZ_NO", "TP_BUZ_NM"]]
    .drop_duplicates()
    .shape[0]
)

print("업종 수:", industry_count)


print("\n" + "=" * 50)
print("7. 숫자 검사")
print("=" * 50)

for col in ["amt", "cnt"]:

    numeric = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    print(
        col,
        "변환 실패:",
        numeric.isna().sum()
    )


print("\n" + "=" * 50)
print("8. 중복 검사")
print("=" * 50)

print(
    "완전히 동일한 중복 행:",
    df.duplicated(keep=False).sum()
)


print("\n검사 완료!")
