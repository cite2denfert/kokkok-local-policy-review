from pathlib import Path
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MONTHLY_PATH = BASE_DIR / "output" / "monthly_r07.csv"
OUTPUT_PATH = BASE_DIR / "output" / "review_evidence.json"


# --------------------------------------------------
# 1. 월별 분석 결과 읽기
# --------------------------------------------------

monthly_df = pd.read_csv(
    MONTHLY_PATH,
    dtype={"month": str}
)


# --------------------------------------------------
# 2. 월별 결과를 개발 JSON 형식으로 변환
# --------------------------------------------------

months = []

for _, row in monthly_df.iterrows():

    raw_month = str(row["month"])

    # 202601 -> 2026-01
    formatted_month = f"{raw_month[:4]}-{raw_month[4:]}"

    months.append(
        {
            "month": formatted_month,
            "calculation_status": "ok",

            "foreign_amount":
                int(row["foreign_amount"]),

            "total_amount":
                int(row["total_amount"]),

            "unknown_amount":
                int(row["unknown_amount"]),

            "transaction_count":
                int(row["transaction_count"]),

            "foreign_share_pct":
                float(row["foreign_share_pct"]),

            "known_only_share_pct":
                float(row["known_only_share_pct"]),

            "unknown_share_pct":
                float(row["unknown_share_pct"]),

            "warnings": []
        }
    )


# --------------------------------------------------
# 3. 최종 개발 전달 JSON
# --------------------------------------------------

review_evidence = {

    "schema_version": "2.0",

    "dataset_version": "real-20260917-v1",

    "records": [

        {
            "evidence_id":
                "R07-NATIONAL-2026H1",

            "data_kind":
                "real",

            "scope": {

                "geographic_scope":
                    "national",

                "region_key":
                    "ALL",

                "population":
                    "foreign_code_3",

                "industry_scope":
                    "all_provided",

                "age_scope":
                    "all",

                "period_start":
                    "2026-01",

                "period_end":
                    "2026-06"
            },

            "region_basis":
                "unknown",

            "applicability": {

                "R07": {
                    "status":
                        "allowed",

                    "reason":
                        "전국 2026년 1~6월 금액·비중 및 분모 비교용"
                },

                "R06": {
                    "status":
                        "blocked",

                    "reason":
                        "현재 파일은 R07 분석 결과이며 R06 실행 자료가 아님"
                },

                "R02": {
                    "status":
                        "needs_review",

                    "reason":
                        "지역 기준·업종 후보·운영 조건 추가 확인 필요"
                }
            },

            "amount_unit":
                "KRW",

            "months":
                months,

            "limitations": [
                "전국 범위의 참고 결과",
                "특정 시도·시군구의 소비 결과로 해석하지 않음",
                "외국인 결제금액과 외국인 결제비중은 서로 다른 지표",
                "미상 제외 비중을 더 정확한 정답으로 해석하지 않음",
                "사업의 인과효과를 증명하는 자료가 아님"
            ]
        }
    ]
}


# --------------------------------------------------
# 4. 저장
# --------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        review_evidence,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n===== review_evidence.json 생성 완료 =====")
print("dataset_version: real-20260917-v1")
print("월별 자료 수:", len(months))
print("저장 위치:", OUTPUT_PATH)
