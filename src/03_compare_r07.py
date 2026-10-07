from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "output" / "monthly_r07.csv"
OUTPUT_PATH = BASE_DIR / "output" / "monthly_comparison.csv"


# 1. 월별 R07 결과 읽기
df = pd.read_csv(INPUT_PATH, dtype={"month": str})


# 2. 방향 판정 함수
def get_direction(value):
    if value > 0:
        return "up"
    elif value < 0:
        return "down"
    else:
        return "flat"


# 3. 인접 월 비교
results = []

for i in range(1, len(df)):

    before = df.iloc[i - 1]
    after = df.iloc[i]

    month0 = before["month"]
    month1 = after["month"]

    F0 = before["foreign_amount"]
    F1 = after["foreign_amount"]

    T0 = before["total_amount"]
    T1 = after["total_amount"]

    share0 = before["foreign_share_pct"]
    share1 = after["foreign_share_pct"]

    known_share0 = before["known_only_share_pct"]
    known_share1 = after["known_only_share_pct"]


    # 외국인 금액 증감률
    if F0 > 0:
        foreign_amount_change_pct = (
            (F1 - F0) / F0 * 100
        )
    else:
        foreign_amount_change_pct = None


    # 전체 금액 증감률
    if T0 > 0:
        total_amount_change_pct = (
            (T1 - T0) / T0 * 100
        )
    else:
        total_amount_change_pct = None


    # 비중 변화 (%p)
    foreign_share_change_pp = (
        share1 - share0
    )

    known_only_share_change_pp = (
        known_share1 - known_share0
    )


    # 방향
    amount_direction = get_direction(F1 - F0)

    share_direction = get_direction(
        foreign_share_change_pp
    )


    # 금액과 비중이 서로 반대 방향인가?
    opposite = (
        amount_direction in ["up", "down"]
        and
        share_direction in ["up", "down"]
        and
        amount_direction != share_direction
    )


    results.append(
        {
            "period": f"{month0}->{month1}",
            "foreign_amount_change_pct":
                foreign_amount_change_pct,
            "total_amount_change_pct":
                total_amount_change_pct,
            "foreign_share_change_pp":
                foreign_share_change_pp,
            "known_only_share_change_pp":
                known_only_share_change_pp,
            "foreign_amount_direction":
                amount_direction,
            "foreign_share_direction":
                share_direction,
            "opposite":
                opposite
        }
    )


# 4. 표 생성
comparison_df = pd.DataFrame(results)


# 5. 파일 저장
comparison_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# 6. 보기 편하게 출력
display_df = comparison_df.copy()

for col in [
    "foreign_amount_change_pct",
    "total_amount_change_pct",
    "foreign_share_change_pp",
    "known_only_share_change_pp"
]:
    display_df[col] = display_df[col].round(6)


print("\n===== 월별 비교 결과 =====")

print(
    display_df.to_string(index=False)
)

print("\n저장 완료:")
print(OUTPUT_PATH)
