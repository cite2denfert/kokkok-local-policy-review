from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "ABP_CONTEST_DATA.csv"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# 1. 데이터 읽기
df = pd.read_csv(
    CSV_PATH,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig"
)


# 2. GENDER_CD 정리
df["gender_norm"] = (
    df["GENDER_CD"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# 3. 코드 검증
allowed_gender = {"1", "2", "3", "x"}

actual_gender = set(df["gender_norm"].unique())

unexpected_gender = actual_gender - allowed_gender

if unexpected_gender:
    raise ValueError(
        f"예상하지 못한 GENDER_CD: {unexpected_gender}"
    )


# 4. 금액과 건수를 숫자로 변환
df["amt_num"] = pd.to_numeric(
    df["amt"],
    errors="raise"
).astype("int64")

df["cnt_num"] = pd.to_numeric(
    df["cnt"],
    errors="raise"
).astype("int64")


# 5. 월별 집계
results = []

for month, group in df.groupby("STRD_YYMM", sort=True):

    # F = 외국인 결제금액
    F = int(
        group.loc[
            group["gender_norm"] == "3",
            "amt_num"
        ].sum()
    )

    # T = 전체 결제금액
    T = int(group["amt_num"].sum())

    # U = 성별 미상 결제금액
    U = int(
        group.loc[
            group["gender_norm"] == "x",
            "amt_num"
        ].sum()
    )

    # C = 전체 거래건수
    C = int(group["cnt_num"].sum())


    # 기본 검증
    if not (0 <= F <= T):
        raise ValueError(f"{month}: F 범위 이상")

    if not (0 <= U <= T):
        raise ValueError(f"{month}: U 범위 이상")

    if F + U > T:
        raise ValueError(f"{month}: F + U > T")


    # 비중 계산
    foreign_share = F / T * 100
    unknown_share = U / T * 100
    known_only_share = F / (T - U) * 100


    results.append(
        {
            "month": month,
            "foreign_amount": F,
            "total_amount": T,
            "unknown_amount": U,
            "transaction_count": C,
            "foreign_share_pct": foreign_share,
            "unknown_share_pct": unknown_share,
            "known_only_share_pct": known_only_share
        }
    )


# 6. 표로 변환
monthly_df = pd.DataFrame(results)


# 7. 저장
monthly_df.to_csv(
    OUTPUT_DIR / "monthly_r07.csv",
    index=False,
    encoding="utf-8-sig"
)


# 8. 터미널에 출력
print("\n===== 월별 R07 결과 =====")

print(
    monthly_df.to_string(index=False)
)

print("\n저장 완료:")
print(OUTPUT_DIR / "monthly_r07.csv")
