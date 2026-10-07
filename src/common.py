"""공통 경로·설정·원본 읽기.

모든 스크립트가 같은 위치를 쓰도록 경로를 여기서 정한다. 원본 경로는 환경 변수 BC_RAW_CSV로 바꿀 수 있다.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CONFIG_PATH = ROOT / "config" / "rules_thresholds.json"
DATA_DIR = ROOT / "data"
EXT_DIR = DATA_DIR / "external"          # 공개 외부 자료(가공본)
EXT_RAW_DIR = EXT_DIR / "raw"            # 공개 외부 자료(내려받은 원본)
WORK_DIR = ROOT / "work"                 # features.pkl 등 중간 산출물(공개 금지)
PRIVATE_DIR = ROOT / "private"           # 실제 근거 파일(공개 금지)
OUTPUT_DIR = ROOT / "output"             # 분석 결과·그림(실제 수치 포함, 공개 금지)
EXAMPLES_DIR = ROOT / "examples"         # 합성 근거 파일(공개 가능)

RAW_CSV = Path(os.environ.get("BC_RAW_CSV", DATA_DIR / "ABP_CONTEST_DATA.csv"))

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
DAYS = dict(zip(MONTHS, [31, 28, 31, 30, 31, 30]))
AGES = ["1", "2", "3", "4", "5", "6"]
IND_NAME = {"4004": "대형할인점", "4010": "편의점", "4020": "슈퍼마켓", "8001": "일반한식", "8002": "갈비전문점",
            "8003": "한정식", "8004": "일식회집", "8005": "중국음식", "8006": "서양음식", "8021": "스넥", "8301": "제과점"}
INDS = list(IND_NAME.values())
# 11_threshold_decision.py 의 재현성 점검에 쓰는 9개 업종(갈비전문점·한정식 제외)
CHECK_INDS = ["일반한식", "서양음식", "슈퍼마켓", "편의점", "대형할인점", "스넥", "제과점", "중국음식", "일식회집"]


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_raw(path: Path | str | None = None) -> pd.DataFrame:
    """BC 원본 CSV를 읽는다. 받은 파일은 cp949였고 작업 중 utf-8로 다시 저장한 사본도 있어 둘 다 시도한다."""
    path = Path(path or RAW_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"원본 CSV가 없습니다: {path}\n"
            "공모전 제공 원본은 재배포할 수 없어 저장소에 없습니다. data/ABP_CONTEST_DATA.csv 에 두거나 "
            "BC_RAW_CSV 환경 변수로 경로를 지정하세요. 실행 점검만 하려면 src/make_synthetic_raw.py 로 가짜 원본을 만드세요."
        )
    last = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise last


def tidy(d: pd.DataFrame) -> pd.DataFrame:
    """원본 행 -> key, sido, ym, g, a, ind, amt, cnt. 업종은 업종 코드로 이름을 붙인다(원본 업종명은 '편 의 점'처럼 띄어 쓰여 있음)."""
    out = pd.DataFrame({
        "key": d.SIDO_NM.astype(str) + " " + d.CCG_NM.astype(str),
        "sido": d.SIDO_NM.astype(str),
        "ym": d.STRD_YYMM.astype(str).str[:4] + "-" + d.STRD_YYMM.astype(str).str[4:6],
        "g": d.GENDER_CD.astype(str).str.strip().str.lower(),
        "a": d.AGE_CD.astype(str).str.strip().str.lower(),
        "ind": d.TP_BUZ_NO.astype(str).str.strip().map(IND_NAME),
        "amt": pd.to_numeric(d.amt, errors="raise").astype("int64"),
        "cnt": pd.to_numeric(d.cnt, errors="raise").astype("int64"),
    })
    if out.ind.isna().any():
        bad = sorted(d.loc[out.ind.isna(), "TP_BUZ_NO"].astype(str).unique())
        raise ValueError(f"알 수 없는 업종 코드: {bad}")
    return out


def load_script(name: str):
    """'09_make_evidence' 처럼 숫자로 시작하는 스크립트를 모듈로 불러온다."""
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SRC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_korean_font():
    """그림 한글 표시. 설치된 폰트 중 먼저 찾은 것을 쓴다."""
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans KR"]:
        if cand in names:
            matplotlib.rcParams["font.family"] = cand
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
