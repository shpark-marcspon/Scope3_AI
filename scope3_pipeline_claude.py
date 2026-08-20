# -*- coding: utf-8 -*-
"""
Scope3 AI 자동 산정 파이프라인 (웹/서버용, Colab 비의존 버전)
================================================================
원본: Scope3_AI_테마_그룹핑_집계v1 (Google Colab 노트북)을 그대로 변환.
- Colab 전용 부분(파일 업로드 위젯, Google Drive 마운트, Colab 시크릿)만
  일반 파일 경로 / 환경변수 방식으로 교체했고, 계산 로직은 전혀 건드리지 않았습니다.
- 웹 서버(백엔드)에서는 run_scope3_pipeline()을 호출해서 사용하면 됩니다.
- 커맨드라인에서 바로 테스트하려면: python scope3_pipeline.py <입력파일.xlsx> [--report-year 2025]

필요한 패키지 (사전 설치)
--------------------------
pip install pandas numpy openpyxl geopy anthropic rapidfuzz

필요한 환경변수
----------------
- ANTHROPIC_API_KEY : Anthropic API 키 (필수 — 품목 분류·거리 추정 등에 Claude Sonnet 5 사용)
- SCOPE3_BASE_PATH  : 배출계수_통합_DB.xlsx 가 들어있는 폴더 경로
                       (지정 안 하면 이 파일과 같은 폴더의 'data' 폴더를 기본값으로 사용)

참고: 이 파일은 scope3_pipeline.py(OpenAI GPT-4.1-mini 버전)와 계산 로직은 완전히 동일하고,
품목 분류·도시명 변환·이동수단 분류 등에 쓰던 LLM 호출만 Claude Sonnet 5로 교체한 버전입니다.
"""
from dotenv import load_dotenv
load_dotenv()  # .env 파일을 읽어서 os.environ에 넣어줌
import os

# ── Anthropic API 키 ───────────────────────────────────────────
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError(
        "환경변수 ANTHROPIC_API_KEY가 설정되어 있지 않습니다. "
        "예) export ANTHROPIC_API_KEY=sk-ant-...  (또는 배포 환경의 시크릿/환경변수 설정에서 지정)"
    )
os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

# ── pandas 버전 호환 헬퍼 ────────────────────────────────────────
# 원본 노트북은 pd.to_numeric(..., errors="ignore")를 사용하는데,
# 이 옵션은 pandas 2.2+에서 지원 중단(pandas 3.x에서 완전히 제거)되었습니다.
# 배포 환경의 pandas 버전에 상관없이 동작하도록 동일한 동작을 직접 구현합니다.
def _to_numeric_ignore(series):
    """pandas 2.x의 pd.to_numeric(series, errors='ignore')와 동일하게 동작:
    전체가 숫자로 변환 가능하면 변환하고, 하나라도 실패하면 원본을 그대로 반환합니다."""
    import pandas as pd
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        return series


import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from anthropic import Anthropic

client = Anthropic()  # ANTHROPIC_API_KEY 환경변수를 자동으로 읽습니다

geolocator = Nominatim(user_agent="scope3_ai", timeout=10)

# ── 배출계수 통합 DB 등 참조 파일 위치 ─────────────────────────
# Colab에서는 Google Drive를 마운트해서 base_path를 잡았지만,
# 서버 환경에서는 SCOPE3_BASE_PATH 환경변수(또는 기본값 data/ 폴더)를 사용합니다.
# 이 폴더 안에 "배출계수_통합_DB.xlsx" 파일이 있어야 합니다.
base_path = os.environ.get(
    "SCOPE3_BASE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
if not os.path.isdir(base_path):
    raise FileNotFoundError(
        f"base_path 폴더를 찾을 수 없습니다: {base_path}\n"
        f"환경변수 SCOPE3_BASE_PATH로 '배출계수_통합_DB.xlsx'가 있는 폴더를 지정해주세요."
    )

# ── 기본 설정값 (run_scope3_pipeline() 호출 시 override 가능) ──
# report_year: 배출계수_통합_DB의 기계산 KRW EF 컬럼 선택 기준 (예: 2025 → "EF (kgCO2e/KRW) 2025년말 기준" 컬럼 사용)
report_year = 2025
# theme: 새 v1 템플릿이면 None으로 유지
theme = None

# ── [원본 셀 4] ─────────────────────────────────────────────
def infer_scope3_category_llm(row, sheet_name):
    sheet_hint_map = {
        "01_구매자산_Template": ["Category1", "Category2"],
        "02_운송_Template": ["Category4", "Category9"],
        "03_출장통근_Template": ["Category6", "Category7"],
        "04_기타확장_Template": [
            "Category3", "Category5", "Category8",
            "Category10", "Category11", "Category12",
            "Category13", "Category14", "Category15"
        ],
    }

    allowed = sheet_hint_map.get(
        sheet_name,
        ["Category1", "Category2", "Category3", "Category4", "Category5", "Category6", "Category7",
         "Category8", "Category9", "Category10", "Category11", "Category12", "Category13", "Category14", "Category15"]
    )

    prompt = f"""
다음 데이터를 Scope 3 카테고리로 분류하라.

업로드 시트명: {sheet_name}
이 시트에서 우선 고려할 카테고리 후보: {allowed}

데이터:
{row.to_dict()}

규칙:
- 반드시 후보 카테고리 중 하나를 우선 선택하라
- 단, 데이터가 후보와 명백히 맞지 않으면 Unknown 출력
- 설명하지 말고 카테고리명만 출력
"""

    res = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=30,
        messages=[{"role": "user", "content": prompt}],
    )

    result = _extract_response_text(res).strip()

    valid = set(allowed + ["Unknown"])
    if result not in valid:
        return "Unknown"

    return result

# ── [원본 셀 8] ─────────────────────────────────────────────
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from anthropic import Anthropic
from functools import lru_cache

client = Anthropic()  # ANTHROPIC_API_KEY 환경변수를 자동으로 읽습니다

geolocator = Nominatim(user_agent="scope3_ai", timeout=10)


def _extract_response_text(res):
    """
    Claude 응답에서 첫 번째 텍스트 블록을 추출한다.
    (extended thinking이 켜져 있으면 res.content[0]이 텍스트가 아니라
    ThinkingBlock(.text가 아닌 .thinking 속성만 가짐)일 수 있어,
    content[0]을 무조건 텍스트라고 가정하면 안 된다.)
    """
    for block in res.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""

# ── 공용 AI 호출 헬퍼 (rule 테이블에 없는 값에 대한 AI 판단 폴백들이 공유) ──
# PATCH: 단위환산 등 "규칙에 없으면 무조건 AI가 판단" 해야 하는 지점에서
# 네트워크/일시적 오류로 AI 호출이 한 번 실패했다고 바로 포기하고
# "[미지원 단위: ...]" 같은 실패 메시지를 반환하는 문제를 막기 위해,
# 재시도(retry) + JSON 파싱 실패 시 숫자만 정규식으로 추출하는 fallback을 추가한다.
import re
import time


def _call_claude_text(prompt, system=None, max_tokens=150, retries=3, retry_delay=1.0):
    """Claude 호출. 일시적 오류(레이트리밋/네트워크 등)에 대비해 최대 retries회 재시도한다."""
    last_err = None
    for attempt in range(retries):
        try:
            res = client.messages.create(
                model="claude-sonnet-5", max_tokens=max_tokens,
                system=system, messages=[{"role": "user", "content": prompt}],
            )
            text = _extract_response_text(res).strip()
            if text:
                return text
        except Exception as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(retry_delay * (attempt + 1))  # 점진적 백오프
    if last_err is not None:
        print(f"  [AI 호출 경고] {retries}회 재시도 후에도 실패: {last_err}")
    return None


def _extract_json_object(text):
    """모델이 JSON 앞뒤에 다른 텍스트를 덧붙였을 때를 대비해 {...} 블록만 추출."""
    if not text:
        return None
    try:
        return safe_json_loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return safe_json_loads(m.group())
        except Exception:
            return None
    return None


def _call_claude_json(prompt, max_tokens=150, retries=3):
    """
    JSON 응답을 요구하는 AI 호출.
    1) 정상 JSON 파싱 시도
    2) 실패 시 응답 안에서 {...} 블록만 골라 재파싱
    3) 그래도 실패하면 재시도(다음 attempt에서 프롬프트에 형식 재강조)
    """
    for attempt in range(retries):
        sys_prompt = "Return only valid JSON, with no other text before or after it."
        p = prompt if attempt == 0 else (
            prompt + "\n\n반드시 JSON 객체 하나만 출력하라. 다른 설명/문장은 절대 포함하지 마라."
        )
        text = _call_claude_text(p, system=sys_prompt, max_tokens=max_tokens, retries=1)
        parsed = _extract_json_object(text)
        if parsed is not None:
            return parsed
        if attempt < retries - 1:
            time.sleep(0.5)
    return None

# ── [원본 셀 12] ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
# 배출계수 통합 DB 로드  |  배출계수_통합_DB.xlsx
# ══════════════════════════════════════════════════════════════
_EF_DB_FILE = os.path.join(base_path, "배출계수_통합_DB.xlsx")

import re

def _fix_cached_krw_ef_columns(df):
    """
    '배출계수_통합_DB.xlsx'의 'EF (kgCO2e/KRW) ...년말 기준' 컬럼은 엑셀 수식(예: =F4/1439)이라,
    파일이 한 번도 재계산되지 않은 상태로 저장되어 있으면 pandas/openpyxl로 읽을 때 빈 값(NaN)이 됩니다.
    이 경우 같은 시트의 USD 기준 EF 컬럼(kgCO2e/USD) ÷ 컬럼명에 적힌 환율로 직접 계산해 채웁니다.
    캐시된 값이 이미 있으면 그대로 사용합니다.
    """
    usd_col = next((c for c in df.columns if "kgCO2" in str(c) and "USD" in str(c)), None)
    if usd_col is None:
        return df
    usd_series = pd.to_numeric(df[usd_col], errors="coerce")
    for c in list(df.columns):
        cs = str(c)
        if "kgCO2e/KRW" in cs.replace(" ", "") and "년말" in cs:
            cached = pd.to_numeric(df[c], errors="coerce")
            if cached.notna().sum() == 0:
                m = re.search(r"([\d,]+\.?\d*)\s*KRW/USD", cs)
                if m:
                    rate = float(m.group(1).replace(",", ""))
                    df[c] = usd_series / rate
                    print(f"  [보정] '{cs.splitlines()[0]}...' 컬럼이 비어있어 USD EF÷{rate}로 재계산했습니다.")
    return df


def _load_ef_sheet(sheet_name, header_row=1, str_cols=None):
    """통합 DB 시트 로드 헬퍼"""
    try:
        df = pd.read_excel(_EF_DB_FILE, sheet_name=sheet_name,
                           header=header_row, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        # 숫자 컬럼 변환 (str_cols 제외)
        for col in df.columns:
            if str_cols and col in str_cols:
                continue
            df[col] = _to_numeric_ignore(df[col])
        df = _fix_cached_krw_ef_columns(df)
        df = df.dropna(how="all").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[경고] {sheet_name} 로드 실패: {e}")
        return pd.DataFrame()

# ── 에너지 DB (C3, C8, C13) ──────────────────────────────────
energy_ef_db = _load_ef_sheet("에너지", header_row=3)  # PATCH: 실제 헤더 행(0-index=3)
print(f"[OK] energy_ef_db: {len(energy_ef_db)}행  컬럼: {energy_ef_db.columns.tolist()}")

# ── 운송 DB (C4, C9) ─────────────────────────────────────────
transport_ef_db = _load_ef_sheet("운송", header_row=1)
print(f"[OK] transport_ef_db: {len(transport_ef_db)}행")

# ── 출장·통근 DB (C6, C7) ────────────────────────────────────
travel_ef_db = _load_ef_sheet("출장_통근", header_row=1)
print(f"[OK] travel_ef_db: {len(travel_ef_db)}행")

# ── 폐기물_C5 DB ─────────────────────────────────────────────
waste_c5_db = _load_ef_sheet("폐기물_C5", header_row=2)
print(f"[OK] waste_c5_db: {len(waste_c5_db)}행")

# ── 폐기물_C12 DB ─────────────────────────────────────────────
waste_c12_db = _load_ef_sheet("폐기물_C12", header_row=1,
                               str_cols=["대분류\n코드","중분류\n코드","소분류\n코드"])
print(f"[OK] waste_c12_db: {len(waste_c12_db)}행")
# ── 운송·출장 지출기반 배출계수 DB 로드 ──────────────────────
transport_spend_db = _load_ef_sheet("운송_Spend", header_row=2)
travel_spend_db    = _load_ef_sheet("출장_Spend",  header_row=2)
print(f"[OK] transport_spend_db: {len(transport_spend_db)}행")
print(f"[OK] travel_spend_db:    {len(travel_spend_db)}행")


# ── 환경부 전체 평균 EF (처리방법 미입력/매핑 없음 시) ──────
WASTE_AVG_EF = {"매립": 0.175965929, "소각": 1.176142131, "재활용": 0.029390317}

# ── 운송 EF 맵 (코드 → kgCO2eq/ton·km) ──────────────────────
def _build_transport_ef_map_from_db():
    if transport_ef_db.empty:
        return {"road":0.120,"parcel":0.180,"rail":0.030,"sea":0.015,"air":0.600}
    code_col = next((c for c in transport_ef_db.columns if "표준코드" in c), None)
    ef_col   = next((c for c in transport_ef_db.columns if "배출계수" in c), None)
    if not code_col or not ef_col:
        return {"road":0.120,"parcel":0.180,"rail":0.030,"sea":0.015,"air":0.600}
    result = {}
    for _, row in transport_ef_db.iterrows():
        code = str(row.get(code_col,"")).strip()
        ef   = pd.to_numeric(row.get(ef_col), errors="coerce")
        if code and pd.notna(ef):
            result[code] = float(ef)
    return result

TRANSPORT_EF_MAP = _build_transport_ef_map_from_db()
print(f"[OK] TRANSPORT_EF_MAP: {TRANSPORT_EF_MAP}")

# ── 출장·통근 EF 맵 (코드 → kgCO2eq/km·인) ──────────────────
def _build_travel_ef_map_from_db():
    # ── DEFRA 2025 기준 기본값 ──
    defaults = {
        "flight":0.14253,"flight_dom":0.22928,"flight_short":0.12786,
        "flight_short_eco":0.12576,"flight_short_biz":0.18863,
        "flight_long":0.15282,"flight_long_eco":0.11704,
        "flight_long_prem":0.18726,"flight_long_biz":0.33940,
        "flight_long_first":0.46814,"flight_eco":0.10916,
        "flight_biz":0.31656,"flight_first":0.43663,
        "ferry":0.11270,"ferry_foot":0.01871,"ferry_car":0.12933,
        "bus":0.10385,"coach":0.02776,
        "train":0.03546,"train_intl":0.00446,"tram":0.02860,"subway":0.02780,
        "taxi":0.14861,   # passenger·km 고정
        "car":0.17304,"car_small":0.14340,"car_medium":0.17174,"car_large":0.21007,
        "car_hybrid":0.09167,"car_ev":0.0,
        "motorcycle":0.11367,"motorcycle_small":0.08319,"motorcycle_large":0.13252,
        "walk":0.0,"bicycle":0.0,
    }
    if travel_ef_db.empty:
        return defaults
    # 통합DB 단일 EF 컬럼 로드
    code_col = next((c for c in travel_ef_db.columns if "표준코드" in c), None)
    ef_col   = next((c for c in travel_ef_db.columns
                     if "EF" in c and "kgCO2" in c), None)
    if not code_col or not ef_col:
        return defaults
    result = {}
    for _, row in travel_ef_db.iterrows():
        code = str(row.get(code_col, "")).strip()
        ef   = pd.to_numeric(row.get(ef_col), errors="coerce")
        if code and pd.notna(ef):
            result[code] = float(ef)
    return {**defaults, **result}


TRAVEL_EF_MAP = _build_travel_ef_map_from_db()
BUSINESS_TRAVEL_EF = TRAVEL_EF_MAP   # 기존 변수명 호환
COMMUTE_EF = TRAVEL_EF_MAP
print(f"[OK] TRAVEL_EF_MAP 로드 완료")

# ── 에너지 EF 검색 함수 ───────────────────────────────────────
def lookup_energy_ef(energy_name: str, unit_hint: str = None):
    """
    에너지원 명칭으로 EF 조회.
    에너지 시트 컬럼: 구분, 배출활동, 활동유형(에너지원), 투입량 기준단위,
                     순발열량, CO2계수, CO2만, Non-CO2, CO2eq ★
    반환: (활동유형명, ef_co2eq, 단위)
    """
    if energy_ef_db.empty:
        return None, None, None

    # 컬럼명 탐색
    name_col = next((c for c in energy_ef_db.columns if "활동유형" in c or "에너지원" in c), None)
    ef_col   = next((c for c in energy_ef_db.columns if "CO2eq" in c and "★" in c), None) or                next((c for c in energy_ef_db.columns if "CO2eq" in c), None)
    unit_col = next((c for c in energy_ef_db.columns if "기준단위" in c or ("단위" in c and "발열량" not in c)), None)

    if not name_col or not ef_col:
        return None, None, None

    q = str(energy_name).strip().lower()

    # 1) 정확 일치
    exact = energy_ef_db[energy_ef_db[name_col].astype(str).str.strip().str.lower() == q]
    if not exact.empty:
        r = exact.iloc[0]
        ef = pd.to_numeric(r[ef_col], errors="coerce")
        return r[name_col], ef if pd.notna(ef) else None, r.get(unit_col)

    # 2) 부분 일치 (쌍방향)
    for _, r in energy_ef_db.iterrows():
        rname = str(r[name_col]).strip().lower()
        if q in rname or rname in q:
            ef = pd.to_numeric(r[ef_col], errors="coerce")
            return r[name_col], ef if pd.notna(ef) else None, r.get(unit_col)

    # 3) 키워드 alias 매칭
    ALIAS = {
        "전기": "전력", "electricity": "전력", "electric": "전력",
        "lng": "도시가스(lng)", "천연가스": "천연가스(lng)",
        "cng": "cng(차량)", "도시가스": "도시가스(lng)",
        "lpg": "액화석유가스(lpg)", "프로판": "액화석유가스(lpg)",
        "diesel": "경유", "gasoline": "휘발유", "kerosene": "보일러 등유",
        "스팀": "열(스팀)", "steam": "열(스팀)", "열": "열(스팀)",
        "항공": "항공유", "jet": "항공유",
    }
    mapped = ALIAS.get(q)
    if mapped:
        for _, r in energy_ef_db.iterrows():
            if mapped in str(r[name_col]).lower():
                ef = pd.to_numeric(r[ef_col], errors="coerce")
                return r[name_col], ef if pd.notna(ef) else None, r.get(unit_col)

    return None, None, None

# ── [원본 셀 14] ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
# 구매/자산 지출기반 배출계수 DB (spend_db)
# 배출계수_통합_DB.xlsx > 구매_USEPA 시트만 사용 (외부 USEEIO CSV 미사용)
# ══════════════════════════════════════════════════════════════
_usepa_raw = pd.read_excel(_EF_DB_FILE, sheet_name="구매_USEPA", header=2)
_usepa_raw.columns = [str(c).strip() for c in _usepa_raw.columns]
_usepa_raw = _fix_cached_krw_ef_columns(_usepa_raw)
_usepa_raw = _usepa_raw.dropna(how="all").reset_index(drop=True)

_title_col = next((c for c in _usepa_raw.columns if "NAICS Title" in c or "Title" in c), None)
_code_col  = next((c for c in _usepa_raw.columns if "NAICS 코드" in c or ("NAICS" in c and "코드" in c)), None)
_cat_col   = next((c for c in _usepa_raw.columns if "품목군" in c), None)
_sub_col   = next((c for c in _usepa_raw.columns if "상세 카테고리" in c or "카테고리" in c), None)
_kw_col    = next((c for c in _usepa_raw.columns if "키워드" in c), None)
_usd_col   = next((c for c in _usepa_raw.columns if "kgCO2" in c and "USD" in c), None)

# 보고연도(report_year)에 맞는 "기계산 KRW EF" 컬럼 우선 사용
_target_year = str(globals().get("report_year") or 2025)
_krw_candidates = [c for c in _usepa_raw.columns if "kgCO2e/KRW" in c.replace(" ", "")]
_krw_col = next((c for c in _krw_candidates if f"{_target_year}년말" in c.replace(" ", "")), None)
if _krw_col is None and _krw_candidates:
    _krw_col = sorted(_krw_candidates)[-1]  # 보고연도 컬럼이 없으면 가장 최근 컬럼 사용

_krw_series = pd.to_numeric(_usepa_raw[_krw_col], errors="coerce") if _krw_col else None
if _krw_series is None or _krw_series.notna().sum() == 0:
    # KRW 컬럼이 여전히 비어있으면(예상치 못한 경우) USD EF를 그대로 사용 (최후 안전장치)
    _krw_series = pd.to_numeric(_usepa_raw[_usd_col], errors="coerce") if _usd_col else None

spend_db = pd.DataFrame({
    "name": _usepa_raw[_title_col].astype(str).str.strip(),
    "ef": _krw_series,
})
if _usd_col:
    spend_db["ef_usd"] = pd.to_numeric(_usepa_raw[_usd_col], errors="coerce")
if _code_col:
    spend_db["naics_code"] = _usepa_raw[_code_col]
if _cat_col:
    spend_db["품목군"] = _usepa_raw[_cat_col]
if _sub_col:
    spend_db["상세카테고리"] = _usepa_raw[_sub_col]
if _kw_col:
    spend_db["한국어매핑키워드"] = _usepa_raw[_kw_col]

spend_db = spend_db.dropna(subset=["name"]).drop_duplicates(subset=["name"]).reset_index(drop=True)
print(f"[OK] spend_db (배출계수_통합_DB.xlsx > 구매_USEPA): {len(spend_db)}행, EF 컬럼: {_krw_col or _usd_col}, "
      f"유효 EF 값: {spend_db['ef'].notna().sum()}/{len(spend_db)}")
spend_db.head()

# ── [원본 셀 16] ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
# C5 / C12 폐기물 배출계수 검색 함수
# ══════════════════════════════════════════════════════════════

# 템플릿 폐기물 종류 → waste_db 폐기물 명칭 매핑 키워드
# 형식: {템플릿에 나오는 키워드: waste_db 소분류 코드나 명칭 키워드}
_WASTE_NAME_ALIAS = {
    # 종이류
    "폐지류": "폐지",      "신문지": "신문지류",  "골판지": "골판지류",
    "복사용지": "백상지류", "a4용지": "백상지류",  "잡지": "잡지류",
    # 플라스틱
    "폐플라스틱": "혼합폐플라스틱", "플라스틱": "혼합폐플라스틱",
    "폐pet": "폐PET",      "폐pp": "폐PP",        "폐pe": "폐PE",
    "폐pvc": "폐PVC",      "폐ps": "폐PS",        "스티로폼": "폐EPS(스티로폼)",
    "폐abs": "폐ABS수지",
    # 금속
    "폐금속": "폐금속류",   "폐철": "폐철류",      "폐알루미늄": "폐알루미늄류",
    "폐동": "폐동류",       "캔류": "폐철캔",
    # 목재
    "폐목재": "폐목재류",   "폐목": "폐목재류",    "팔레트": "폐팔레트",
    # 유리
    "폐유리": "폐유리류",   "유리병": "폐병류",    "판유리": "폐판유리",
    # 음식물
    "음식물": "음식물류 폐기물", "식품": "음식물류 폐기물",
    # 고무
    "폐고무": "폐고무류",   "타이어": "폐타이어",
    # 전자
    "전자제품": "산업용폐전기전자제품", "가전": "폐가전제품",
    "토너": "프린트토너 및 카트리지폐부속품",
    # 유기성
    "오니": "폐수처리오니", "슬러지": "폐수처리오니",
    # 건설
    "콘크리트": "폐콘크리트",
}

# 처리방법 정규화
_TREATMENT_ALIAS = {
    "매립": "매립", "land": "매립", "landfill": "매립",
    "소각": "소각", "incineration": "소각", "incinerator": "소각", "inc": "소각",
    "재활용": "재활용", "recycling": "재활용", "recycle": "재활용", "rec": "재활용",
    # PATCH: "기타"가 별칭에 없으면 _normalize_treatment()가 None을 반환하고,
    # lookup_waste_ef()의 처리방법 필터가 빈 문자열과 비교돼 DB에 폐기물종류별로
    # 정확히 정의된 "기타" 행(예: 폐콘크리트류+기타→매립평균)을 아예 찾지 못한 채
    # 무조건 WASTE_AVG_EF["소각"]로 떨어지는 문제가 있었다.
    "기타": "기타", "other": "기타", "etc": "기타",
}

def _normalize_waste_name(raw: str) -> str:
    """템플릿 폐기물 종류명 → waste_db 검색용 정규화"""
    s = str(raw).strip().lower()
    for alias, target in _WASTE_NAME_ALIAS.items():
        if alias in s:
            return target
    return str(raw).strip()

def _normalize_treatment(raw: str) -> str:
    """처리방법 → 매립/소각/재활용"""
    s = str(raw).strip().lower()
    for alias, std in _TREATMENT_ALIAS.items():
        if alias in s:
            return std
    return None

def lookup_waste_ef(waste_name: str, treatment: str) -> float:
    """
    C5 폐기물 배출계수 조회.
    waste_c5_db (폐기물_C5 시트)에서 (폐기물종류, 처리방법) 조합으로 직접 매핑.
    매핑 없으면 해당 처리방법 전체 평균 반환.
    """
    treatment_std = _normalize_treatment(treatment) if treatment else None
    fallback = WASTE_AVG_EF.get(treatment_std or "소각")

    if waste_c5_db.empty:
        return fallback

    # 컬럼명 파악
    type_col  = next((c for c in waste_c5_db.columns if "폐기물 종류" in c), None)
    treat_col = next((c for c in waste_c5_db.columns if "처리방법" in c), None)
    kw_col    = next((c for c in waste_c5_db.columns if "키워드" in c), None)
    ef_col    = next((c for c in waste_c5_db.columns if "배출계수" in c), None)

    if not type_col or not treat_col or not ef_col:
        return fallback

    # 처리방법 필터
    treat_mask = waste_c5_db[treat_col].astype(str).str.strip() == (treatment_std or "")
    treat_df   = waste_c5_db[treat_mask]

    if treat_df.empty:
        return fallback

    name_q = str(waste_name).strip().lower()

    # 1) 폐기물종류 정확 일치
    exact = treat_df[treat_df[type_col].astype(str).str.strip().str.lower() == name_q]
    if not exact.empty:
        val = pd.to_numeric(exact.iloc[0][ef_col], errors="coerce")
        if pd.notna(val):
            return float(val)

    # 2) 폐기물종류 부분 일치
    partial = treat_df[treat_df[type_col].astype(str).str.lower().str.contains(
        name_q, na=False, regex=False)]
    if not partial.empty:
        val = pd.to_numeric(partial.iloc[0][ef_col], errors="coerce")
        if pd.notna(val):
            return float(val)

    # 3) 키워드 컬럼에서 매칭
    if kw_col:
        for _, row in treat_df.iterrows():
            kws = str(row.get(kw_col,"")).lower().split(",")
            if any(k.strip() and k.strip() in name_q for k in kws):
                val = pd.to_numeric(row[ef_col], errors="coerce")
                if pd.notna(val):
                    return float(val)

    # 4) 전체 DB에서 폐기물종류 부분 일치 (처리방법 무시)
    all_partial = waste_c5_db[waste_c5_db[type_col].astype(str).str.lower().str.contains(
        name_q, na=False, regex=False)]
    if not all_partial.empty:
        treat_match = all_partial[all_partial[treat_col].astype(str).str.strip() == (treatment_std or "")]
        target = treat_match if not treat_match.empty else all_partial
        val = pd.to_numeric(target.iloc[0][ef_col], errors="coerce")
        if pd.notna(val):
            return float(val)

    return fallback


def lookup_waste_ef_by_code(대분류코드=None, 중분류코드=None, 소분류코드=None, treatment="소각") -> float:
    """
    C12용: 대분류/중분류/소분류 코드로 waste_c12_db에서 EF 조회.
    treatment: 매립 / 소각 / 재활용
    """
    treatment_std = _normalize_treatment(treatment) if treatment else "소각"
    fallback = WASTE_AVG_EF.get(treatment_std)

    if waste_c12_db.empty:
        return fallback

    # 컬럼명 파악
    big_col = next((c for c in waste_c12_db.columns if "대분류" in c and "코드" in c), None)
    mid_col = next((c for c in waste_c12_db.columns if "중분류" in c and "코드" in c), None)
    sub_col = next((c for c in waste_c12_db.columns if "소분류" in c and "코드" in c), None)
    ef_map  = {
        "매립":   next((c for c in waste_c12_db.columns if "매립 EF" in c or ("매립" in c and "EF" in c)), None),
        "소각":   next((c for c in waste_c12_db.columns if "소각 EF" in c or ("소각" in c and "EF" in c)), None),
        "재활용": next((c for c in waste_c12_db.columns if "재활용 EF" in c or ("재활용" in c and "EF" in c)), None),
    }
    ef_col = ef_map.get(treatment_std)
    if not ef_col:
        return fallback

    db = waste_c12_db.copy()

    # 소분류 → 중분류 → 대분류 순으로 필터 (우선순위)
    if 소분류코드 and sub_col:
        filtered = db[db[sub_col].astype(str).str.strip() == str(소분류코드).strip()]
        if not filtered.empty:
            val = pd.to_numeric(filtered.iloc[0][ef_col], errors="coerce")
            return float(val) if pd.notna(val) else fallback

    if 중분류코드 and mid_col:
        filtered = db[db[mid_col].astype(str).str.strip() == str(중분류코드).strip()]
        if not filtered.empty:
            val = pd.to_numeric(filtered.iloc[0][ef_col], errors="coerce")
            return float(val) if pd.notna(val) else fallback

    if 대분류코드 and big_col:
        filtered = db[db[big_col].astype(str).str.strip() == str(대분류코드).strip()]
        if not filtered.empty:
            val = pd.to_numeric(filtered.iloc[0][ef_col], errors="coerce")
            return float(val) if pd.notna(val) else fallback

    return fallback


def calc_c12_emission(row) -> float:
    """
    C12 판매제품 폐기 배출량 계산.
    산정식: 판매량 × 제품중량(kg) × Σ(처리방법EF × 처리비중%)
    처리비중 미입력 시 → 평균 EF 합산
    """
    qty      = to_float(row.get("판매량"))
    weight   = to_float(row.get("제품중량"))
    unit     = str(row.get("제품중량 단위") or "kg").strip().lower()
    대분류    = str(row.get("폐기물 대분류(선택)") or "").strip()
    중분류    = str(row.get("폐기물 중분류(선택)") or "").strip()
    소분류    = str(row.get("폐기물 소분류(선택)") or "").strip()
    제품명    = str(row.get("제품명") or "").strip()

    if qty is None or weight is None:
        return None

    # 단위 변환 → kg
    if unit in ["g", "gram"]:
        weight = weight / 1000
    elif unit in ["ton", "t"]:
        weight = weight * 1000

    total_weight_kg = qty * weight

    # 처리비중 컬럼
    pct_lf  = to_float(row.get("재활용 비중(%)(선택)".replace("재활용", "매립"))) or               to_float(row.get("매립 비중(%)")) or to_float(row.get("매립비중(%)"))
    pct_inc = to_float(row.get("소각 비중(%)(선택)")) or               to_float(row.get("소각 비중(%)")) or to_float(row.get("소각비중(%)"))
    pct_rec = to_float(row.get("재활용 비중(%)(선택)")) or               to_float(row.get("재활용 비중(%)")) or to_float(row.get("재활용비중(%)"))

    def _pct(v):
        """% → 소수 변환 (50 → 0.5, 0.5 → 0.5 둘 다 처리)"""
        if v is None:
            return None
        return v / 100 if v > 1 else v

    pct_lf  = _pct(pct_lf)
    pct_inc = _pct(pct_inc)
    pct_rec = _pct(pct_rec)

    # EF 조회 (소분류 > 중분류 > 대분류 > 제품명 순)
    def _ef(treatment):
        if 소분류 and 소분류 != "-":
            return lookup_waste_ef_by_code(소분류코드=소분류, treatment=treatment)
        elif 중분류:
            return lookup_waste_ef_by_code(중분류코드=중분류, treatment=treatment)
        elif 대분류:
            return lookup_waste_ef_by_code(대분류코드=대분류, treatment=treatment)
        elif 제품명:
            return lookup_waste_ef(제품명, treatment)
        return WASTE_AVG_EF.get(_normalize_treatment(treatment))

    ef_lf  = _ef("매립")
    ef_inc = _ef("소각")
    ef_rec = _ef("재활용")

    # 처리비중 모두 미입력 → 평균 EF 3개 단순 합산 방식
    if pct_lf is None and pct_inc is None and pct_rec is None:
        ef_avg = (
            WASTE_AVG_EF["매립"] +
            WASTE_AVG_EF["소각"] +
            WASTE_AVG_EF["재활용"]
        ) / 3
        return total_weight_kg * ef_avg / 1000  # kg→tCO2e

    emission = 0.0
    if pct_lf  is not None and ef_lf  is not None:
        emission += ef_lf  * pct_lf
    if pct_inc is not None and ef_inc is not None:
        emission += ef_inc * pct_inc
    if pct_rec is not None and ef_rec is not None:
        emission += ef_rec * pct_rec

    return total_weight_kg * emission / 1000  # kg→tCO2e


# ══════════════════════════════════════════════════════════════
# 운송·출장 지출기반 EF 검색 함수 (US EPA NAICS 기반)
# ══════════════════════════════════════════════════════════════

def _get_spend_ef_col(db: pd.DataFrame):
    """EF 컬럼명 반환"""
    return next((c for c in db.columns if "EF" in c and "kgCO2" in c), None)

def _get_spend_kw_col(db: pd.DataFrame):
    return next((c for c in db.columns if "키워드" in c), None)

def _get_spend_ko_col(db: pd.DataFrame):
    return next((c for c in db.columns if "한국어 품목명" in c or "한국어명" in c), None)

def _get_spend_naics_col(db: pd.DataFrame):
    return next((c for c in db.columns if "NAICS 코드" in c or "NAICS코드" in c), None)

def _get_spend_title_col(db: pd.DataFrame):
    return next((c for c in db.columns if "NAICS Title" in c or "Title" in c), None)


def lookup_transport_spend_ef(mode_or_item: str) -> tuple:
    """
    C4/C9 운송 지출기반 EF 조회.
    mode_or_item: 운송수단명 또는 품목명 (한국어 or 영어)
    반환: (NAICS_title, ef_kgCO2e_per_USD, naics_code)
    """
    if transport_spend_db.empty:
        return None, None, None

    ef_col    = _get_spend_ef_col(transport_spend_db)
    kw_col    = _get_spend_kw_col(transport_spend_db)
    ko_col    = _get_spend_ko_col(transport_spend_db)
    title_col = _get_spend_title_col(transport_spend_db)
    naics_col = _get_spend_naics_col(transport_spend_db)

    if not ef_col or not title_col:
        return None, None, None

    q = str(mode_or_item).strip().lower()

    # 1) 한국어 키워드 매칭
    if kw_col:
        for _, row in transport_spend_db.iterrows():
            kws = str(row.get(kw_col, "")).lower().split(",")
            if any(k.strip() and k.strip() in q for k in kws):
                ef = pd.to_numeric(row[ef_col], errors="coerce")
                if pd.notna(ef):
                    return row[title_col], float(ef), str(row.get(naics_col, ""))

    # 2) 한국어명 부분 일치
    if ko_col:
        for _, row in transport_spend_db.iterrows():
            if q in str(row.get(ko_col, "")).lower():
                ef = pd.to_numeric(row[ef_col], errors="coerce")
                if pd.notna(ef):
                    return row[title_col], float(ef), str(row.get(naics_col, ""))

    # 3) NAICS Title 부분 일치
    for _, row in transport_spend_db.iterrows():
        if q in str(row.get(title_col, "")).lower():
            ef = pd.to_numeric(row[ef_col], errors="coerce")
            if pd.notna(ef):
                return row[title_col], float(ef), str(row.get(naics_col, ""))

    # fallback: General Freight Trucking 평균
    default = transport_spend_db[
        transport_spend_db[title_col].astype(str).str.contains(
            "General Freight Trucking, Local", na=False)]
    if not default.empty:
        ef = pd.to_numeric(default.iloc[0][ef_col], errors="coerce")
        return default.iloc[0][title_col], float(ef) if pd.notna(ef) else None, None

    return None, None, None


def lookup_travel_spend_ef(travel_item: str) -> tuple:
    """
    C6 출장 지출기반 EF 조회.
    travel_item: 출장 항목 (항공권, 택시비, 호텔, 렌터카 등)
    반환: (NAICS_title, ef_kgCO2e_per_USD, naics_code)
    """
    if travel_spend_db.empty:
        return None, None, None

    ef_col    = _get_spend_ef_col(travel_spend_db)
    kw_col    = _get_spend_kw_col(travel_spend_db)
    ko_col    = _get_spend_ko_col(travel_spend_db)
    title_col = _get_spend_title_col(travel_spend_db)
    naics_col = _get_spend_naics_col(travel_spend_db)

    if not ef_col or not title_col:
        return None, None, None

    q = str(travel_item).strip().lower()

    # 1) 키워드 매칭
    if kw_col:
        for _, row in travel_spend_db.iterrows():
            kws = str(row.get(kw_col, "")).lower().split(",")
            if any(k.strip() and k.strip() in q for k in kws):
                ef = pd.to_numeric(row[ef_col], errors="coerce")
                if pd.notna(ef):
                    return row[title_col], float(ef), str(row.get(naics_col, ""))

    # 2) 한국어명 부분 일치
    if ko_col:
        for _, row in travel_spend_db.iterrows():
            if q in str(row.get(ko_col, "")).lower():
                ef = pd.to_numeric(row[ef_col], errors="coerce")
                if pd.notna(ef):
                    return row[title_col], float(ef), str(row.get(naics_col, ""))

    # 3) NAICS Title 부분 일치
    for _, row in travel_spend_db.iterrows():
        if q in str(row.get(title_col, "")).lower():
            ef = pd.to_numeric(row[ef_col], errors="coerce")
            if pd.notna(ef):
                return row[title_col], float(ef), str(row.get(naics_col, ""))

    # fallback: 항공 EF (가장 흔한 출장 항목)
    default = travel_spend_db[
        travel_spend_db[title_col].astype(str).str.contains(
            "Scheduled Passenger Air", na=False)]
    if not default.empty:
        ef = pd.to_numeric(default.iloc[0][ef_col], errors="coerce")
        return default.iloc[0][title_col], float(ef) if pd.notna(ef) else None, None

    return None, None, None


def calc_transport_spend_emission(row) -> dict:
    """
    C4/C9 운송 지출기반 배출량 계산.
    산정식: 운송금액(KRW) ÷ 환율(KRW/USD) × EF(kgCO2eq/USD) / 1000 → tCO2e
    """
    spend_krw = to_float(
        row.get("운송금액") or row.get("월운송금액") or row.get("연간운송금액(원)"))
    mode = _safe_text(row.get("운송수단") or row.get("자재명") or "")

    if spend_krw is None:
        return {"배출량(tCO2e)": None, "EF(kgCO2e/USD)": None, "Spend방식": "운송"}

    # 환율 변환
    월 = to_float(row.get("발생월"))
    # 월 없으면 연평균 자동 사용 (get_bok_usd_krw_monthly_avg에서 처리)
    fx = get_bok_usd_krw_monthly_avg(int(월) if 월 else None, year=FX_REFERENCE_YEAR)
    spend_usd = spend_krw / fx if fx else spend_krw / 1400

    title, ef, naics = lookup_transport_spend_ef(mode)
    emission = (spend_usd * ef / 1000) if (ef and spend_usd) else None

    return {
        "배출량(tCO2e)":  emission,
        "EF(kgCO2e/USD)": ef,
        "NAICS":           naics,
        "NAICS_Title":     title,
        "Spend방식":       "운송_Spend",
        "적용환율(KRW/USD)": fx,
        "USD환산금액":      spend_usd,
    }


def calc_travel_spend_emission(row) -> dict:
    """
    C6 출장 지출기반 배출량 계산.
    산정식: 출장비(KRW) ÷ 환율(KRW/USD) × EF(kgCO2eq/USD) / 1000 → tCO2e
    """
    spend_krw = to_float(
        row.get("출장비용") or row.get("월출장비") or row.get("연간지출금액(원)"))
    item = _safe_text(
        row.get("출장항목") or row.get("이동수단") or row.get("출장수단") or "")

    if spend_krw is None:
        return {"배출량(tCO2e)": None, "EF(kgCO2e/USD)": None, "Spend방식": "출장"}

    월 = to_float(row.get("발생월"))
    # 월 없으면 연평균 자동 사용 (get_bok_usd_krw_monthly_avg에서 처리)
    fx = get_bok_usd_krw_monthly_avg(int(월) if 월 else None, year=FX_REFERENCE_YEAR)
    spend_usd = spend_krw / fx if fx else spend_krw / 1400

    title, ef, naics = lookup_travel_spend_ef(item)
    emission = (spend_usd * ef / 1000) if (ef and spend_usd) else None

    return {
        "배출량(tCO2e)":  emission,
        "EF(kgCO2e/USD)": ef,
        "NAICS":           naics,
        "NAICS_Title":     title,
        "Spend방식":       "출장_Spend",
        "적용환율(KRW/USD)": fx,
        "USD환산금액":      spend_usd,
    }

# ── [원본 셀 20] ─────────────────────────────────────────────
@lru_cache(maxsize=1024)
def normalize_city(city):

    prompt = f"""
    다음 도시를 영어 도시명으로 변환하라.

    {city}

    예
    서울 → Seoul
    부산 → Busan

    영어 도시명만 출력
    """

    res = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )

    return _extract_response_text(res).strip()

# ── [원본 셀 22] ─────────────────────────────────────────────
@lru_cache(maxsize=1024)
def _cached_geocode(query: str):
    """
    geolocator.geocode() 결과 캐시.
    같은 지명(예: '서울, KR')이 데이터에 여러 번 나올 때마다 Nominatim에
    매번 새로 조회하면 행 수가 많을수록 매우 느려지고, 요청 간 딜레이가 없어
    레이트리밋에 걸릴 위험도 커진다. 지명→좌표는 실행 중 바뀌지 않으므로 캐시로 재사용한다.
    """
    return geolocator.geocode(query)


# ── [원본 셀 24] ─────────────────────────────────────────────

# -----------------------------------
# 공통 헬퍼
# -----------------------------------
def _safe_text2(x):
    if x is None:
        return ""
    if isinstance(x, float) and pd.isnull(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s

def to_float(x):
    if x is None:
        return None
    try:
        s = str(x).replace(",", "").strip()
        if s == "":
            return None
        return float(s)
    except:
        return None

def _contains_any(text, keywords):
    text = _safe_text2(text).lower()
    return any(k.lower() in text for k in keywords)

def _concat_row_text(row, fields):
    vals = [_safe_text2(row.get(f)) for f in fields if f in row.index]
    return " ".join([v for v in vals if v]).lower()

# -----------------------------------
# 키워드 사전
# -----------------------------------
ROAD_KEYWORDS = [
    "road", "truck", "lorry", "van", "delivery", "freight", "cargo truck",
    "트럭", "화물차", "카고", "배송", "납품", "상차", "하차", "육상",
    "윙바디", "탑차", "냉동탑차", "5t", "11t", "25t"
]

SEA_KEYWORDS = [
    "sea", "ocean", "vessel", "voyage", "fcl", "lcl", "bl", "mbl", "hbl",
    "port", "terminal", "cy", "cfs", "선박", "해상", "항만", "부산항", "인천항", "평택항"
]

AIR_KEYWORDS = [
    "air", "air cargo", "flight", "awb", "mawb", "hawb", "airport", "cargo",
    "항공", "항공화물", "공항", "대한항공 cargo", "asiana cargo", "icn", "gmp", "pvg", "nrt"
]

RAIL_KEYWORDS = [
    "rail", "railway", "train", "intermodal rail",
    "철도", "철송", "기차"
]

PARCEL_KEYWORDS = [
    "parcel", "courier", "express", "small parcel",
    "택배", "특송", "퀵", "cj대한통운", "한진택배", "롯데택배", "fedex", "dhl", "ups"
]

PORT_HINTS = ["항", "port", "terminal", "cy", "부산항", "인천항", "평택항"]
AIRPORT_HINTS = ["공항", "airport", "icn", "gmp", "pvg", "nrt", "lax", "jfk"]


# -----------------------------------
# 1) 운송수단 추정
# -----------------------------------
def infer_transport_mode(row):
    """
    return:
      mode: road / sea / air / rail / parcel / unknown
      mode_source: column / keyword / geo / default
      mode_confidence: high / medium / low / none
    """

    # 1순위: 명시 컬럼
    direct_mode_text = _concat_row_text(row, ["운송수단", "배송수단", "mode", "shipment mode", "transport_mode"])
    if direct_mode_text:
        if _contains_any(direct_mode_text, PARCEL_KEYWORDS):
            return "parcel", "column", "high"
        if _contains_any(direct_mode_text, AIR_KEYWORDS):
            return "air", "column", "high"
        if _contains_any(direct_mode_text, SEA_KEYWORDS):
            return "sea", "column", "high"
        if _contains_any(direct_mode_text, RAIL_KEYWORDS):
            return "rail", "column", "high"
        if _contains_any(direct_mode_text, ROAD_KEYWORDS):
            return "road", "column", "high"

    # 2순위: 적요/거래처/운송사명 등 키워드
    keyword_text = _concat_row_text(
        row,
        ["설명/적요", "비고", "거래처명", "운송사명", "송장번호", "B/L", "AWB", "Booking", "운송구분"]
    )

    # parcel 먼저
    if _contains_any(keyword_text, PARCEL_KEYWORDS):
        return "parcel", "keyword", "medium"
    if _contains_any(keyword_text, AIR_KEYWORDS):
        return "air", "keyword", "medium"
    if _contains_any(keyword_text, SEA_KEYWORDS):
        return "sea", "keyword", "medium"
    if _contains_any(keyword_text, RAIL_KEYWORDS):
        return "rail", "keyword", "medium"
    if _contains_any(keyword_text, ROAD_KEYWORDS):
        return "road", "keyword", "medium"

    # 3순위: 출발지/도착지 패턴
    origin = _safe_text2(row.get("출발지")).lower()
    dest = _safe_text2(row.get("도착지")).lower()
    geo_text = f"{origin} {dest}"

    if _contains_any(geo_text, AIRPORT_HINTS):
        return "air", "geo", "low"
    if _contains_any(geo_text, PORT_HINTS):
        return "sea", "geo", "low"

    # 국내 일반 사업장/공장/창고/고객사 간 이동이면 road로 추정
    if any(k in geo_text for k in ["공장", "창고", "물류센터", "고객사", "warehouse", "factory", "dc"]):
        return "road", "geo", "low"

    # 4순위: 기본값
    return "unknown", "default", "none"


# -----------------------------------
# 2) 직선거리 추정 (sea/air/rail용 보조)
# -----------------------------------
def estimate_straight_distance(origin, dest, country_hint=None):
    try:
        if not origin or not dest:
            return None

        origin = normalize_city(origin)
        dest = normalize_city(dest)

        q_origin = f"{origin}, {country_hint}" if country_hint else origin
        q_dest = f"{dest}, {country_hint}" if country_hint else dest

        o = _cached_geocode(q_origin)
        d = _cached_geocode(q_dest)

        if o and d:
            return geodesic((o.latitude, o.longitude), (d.latitude, d.longitude)).km
    except:
        pass
    return None


# -----------------------------------
# 3) 거리 이상치 검증
# -----------------------------------
def validate_distance_km(distance_km, mode="road", country_hint="KR"):
    distance_km = to_float(distance_km)
    if distance_km is None:
        return None

    # 국내 기준 대략적 상한선
    if country_hint == "KR":
        if mode in ["road", "parcel"] and distance_km > 800:
            return None
        if mode == "rail" and distance_km > 1000:
            return None
        if mode == "sea" and distance_km > 3000:
            return None
        if mode == "air" and distance_km > 3000:
            return None

    # 너무 작은 음수/0 방지
    if distance_km <= 0:
        return None

    return distance_km

# ── [원본 셀 25] ─────────────────────────────────────────────
def coalesce_row_fields(row, field_names):
    """
    row에서 field_names 순서대로 보면서
    비어있지 않은 첫 값을 반환
    """
    for field in field_names:
        if field in row.index:
            val = row.get(field)
            if pd.notnull(val):
                text = str(val).strip()
                if text and text.lower() != "nan":
                    return text
    return ""

def coalesce_row_numeric(row, field_names):
    for field in field_names:
        if field in row.index:
            val = row.get(field)
            if pd.notnull(val):
                try:
                    return float(str(val).replace(",", "").strip())
                except:
                    pass
    return None

# ── [원본 셀 27] ─────────────────────────────────────────────
def calc_transport(row):
    """
    기본 계산:
      배출량 = 거리(km) × 중량(ton) × mode별 EF
    """
    mode, mode_source, mode_confidence = infer_transport_mode(row)
    distance_km, distance_source = infer_distance(row, mode)

    weight_kg = to_float(row.get("화물중량(kg)"))
    weight_ton = _kg_to_ton(weight_kg)

    # row에 추정값 기록이 필요하면 process_template_inventory에서 별도 저장
    # 여기서는 계산만 담당
    if distance_km is None or weight_ton is None:
        return None, None

    ef = TRANSPORT_EF_MAP.get(mode)
    if ef is None:
        return None, None

    emission = distance_km * weight_ton * ef
    return ef, emission

# ── [원본 셀 29] ─────────────────────────────────────────────
# ==========================================
# Category 6/7 compatibility patch
# 맨 마지막 셀에 추가해서 실행
# ==========================================
import re
import pandas as pd

def _safe_text_travel(x):
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s

# -----------------------------------
# 공통 헬퍼 보완
# -----------------------------------
# -----------------------------------
# 출장/통근 이동수단 정규화
# -----------------------------------
TRAVEL_MODE_MAP = {
    "flight": [
        "flight", "air", "airplane", "plane", "air cargo",
        "항공", "비행기", "항공편", "국내선", "국제선",
        "대한항공", "아시아나", "제주항공", "진에어", "티웨이", "에어부산",
        "ke", "oz", "lj", "tw"
    ],
    "train": [
        "train", "rail", "railway",
        "기차", "철도", "열차", "ktx", "srt", "무궁화", "새마을"
    ],
    "bus": [
        "bus", "coach", "shuttle",
        "버스", "고속버스", "시외버스", "전세버스", "셔틀", "통근버스", "광역버스", "공항버스"
    ],
    "car": [
        "car", "auto", "vehicle", "rental car", "rent-a-car",
        "자동차", "차량", "승용차", "자가용", "렌터카", "렌트카",
        "법인차", "업무차량", "개인차량"
    ],
    "taxi": [
        "taxi", "cab",
        "택시", "카카오택시", "카카오t"
    ],
    "subway": [
        "subway", "metro", "underground",
        "지하철", "전철", "도시철도", "line 1", "line 2", "line 3", "line 4"
    ],
    "walk": [
        "walk", "walking",
        "도보", "걷기"
    ],
    "bicycle": [
        "bike", "bicycle", "cycling",
        "자전거", "따릉이"
    ]
}

def detect_travel_mode_rule(mode_text):
    text = _safe_text_travel(mode_text).lower()
    if not text:
        return "unknown"

    text = re.sub(r"[\(\)\[\]\/,_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if any(k in text for k in ["coach","코치","장거리버스"]):
        return "coach"
    if any(k in text for k in [
        "bus", "shuttle",
        "버스", "고속버스", "시외버스", "전세버스", "공항버스", "셔틀", "광역버스"
    ]):
        return "bus"

    # 항공 클래스 세분화 (순서 중요: 세부 → 일반 순)
    if any(k in text for k in ["퍼스트","first class","first"]) and        any(k in text for k in ["항공","flight","비행"]):
        if any(k in text for k in ["장거리","long","long-haul","long haul"]):
            return "flight_long_first"
        return "flight_first"
    if any(k in text for k in ["비즈니스","business class"]) and        any(k in text for k in ["항공","flight","비행"]):
        if any(k in text for k in ["장거리","long","long-haul"]):
            return "flight_long_biz"
        if any(k in text for k in ["단거리","short","short-haul","유럽"]):
            return "flight_short_biz"
        return "flight_biz"
    if any(k in text for k in ["프리미엄이코노미","premium economy"]):
        if any(k in text for k in ["장거리","long"]):
            return "flight_long_prem"
    if any(k in text for k in ["장거리","long haul","long-haul"]) and        any(k in text for k in ["항공","flight","비행"]):
        return "flight_long"
    if any(k in text for k in ["단거리","short haul","short-haul","유럽"]) and        any(k in text for k in ["항공","flight","비행"]):
        return "flight_short"
    if any(k in text for k in ["국내선","국내항공","domestic flight"]):
        return "flight_dom"
    if any(k in text for k in [
        "flight", "air", "airplane", "plane",
        "항공", "비행기", "항공편", "국제선",
        "대한항공", "아시아나", "제주항공", "진에어", "티웨이",
        "ke", "oz", "lj", "tw", "공항"
    ]):
        return "flight"

    if any(k in text for k in [
        "ktx", "srt", "train", "rail", "railway",
        "기차", "철도", "열차", "무궁화", "새마을"
    ]):
        return "train"

    if any(k in text for k in ["경전철","트램","tram","light rail"]):
        return "tram"
    if any(k in text for k in [
        "지하철", "전철", "subway", "metro"
    ]):
        return "subway"

    if any(k in text for k in [
        "hybrid", "하이브리드", "hev", "phev", "플러그인하이브리드",
    ]):
        return "car_hybrid"

    if any(k in text for k in [
        "전기차", "전기자동차", "electric car", "electric vehicle", "ev",
        "테슬라", "아이오닉", "ev6", "bev", "제네시스ev",
    ]):
        return "car_ev"

    if any(k in text for k in ["소형차","경차","small car","minicar"]):
        return "car_small"
    if any(k in text for k in ["중형차","medium car","중형"]):
        return "car_medium"
    if any(k in text for k in ["대형차","large car","대형suv","대형"]):
        return "car_large"
    if any(k in text for k in [
        "car", "auto", "vehicle", "렌터카", "렌트카", "자가용", "승용차", "자동차", "차량"
    ]):
        return "car"

    if any(k in text for k in ["스쿠터","경량오토바이","소형오토바이","moped","scooter","small motorcycle"]):
        return "motorcycle_small"
    if any(k in text for k in ["대형오토바이","리터바이크","large motorcycle","할리","harley"]):
        return "motorcycle_large"
    if any(k in text for k in ["오토바이","motorcycle","motorbike"]):
        return "motorcycle"

    if any(k in text for k in [
        "taxi", "cab", "카카오t", "카카오택시", "택시"
    ]):
        return "taxi"

    if any(k in text for k in ["ferry foot","foot passenger","도보페리"]):
        return "ferry_foot"
    if any(k in text for k in ["ferry","페리","선박","배"]):
        return "ferry"
    if any(k in text for k in ["도보", "walk", "walking"]):
        return "walk"

    if any(k in text for k in ["자전거", "따릉이", "bike", "bicycle", "cycling"]):
        return "bicycle"

    return "unknown"

def normalize_travel_mode(mode, return_meta=False):
    text = _safe_text_travel(mode).lower()

    if not text:
        return ("unknown", "empty") if return_meta else "unknown"

    # 1차 규칙 기반
    rule_mode = detect_travel_mode_rule(text)
    if rule_mode != "unknown":
        if rule_mode in ["walk", "bicycle"]:
            norm = "car"   # 출장에서는 fallback
        else:
            norm = rule_mode

        return (norm, "rule") if return_meta else norm

    # 2차 LLM fallback
    try:
        prompt = f"""
다음 출장 이동수단을 아래 6개 중 하나의 영어 transport mode로 변환하라.

입력값:
{text}

가능한 값:
flight
train
subway
taxi
bus
car

규칙:
- 반드시 위 6개 중 하나만 출력
- 설명 금지
- 공항/항공사/비행기 관련이면 flight
- KTX/SRT/철도 관련이면 train
- 지하철/매트로 관련이면 subway
- 버스/셔틀 관련이면 bus
- 택시/카카오택시 관련이면 taxi
- 렌터카/자가용/차량 관련이면 car
"""

        res = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=10,
            system="Return only one word: flight, train, bus, or car.",
            messages=[{"role": "user", "content": prompt}],
        )

        out = _extract_response_text(res).strip().lower()

        # 혹시 불필요한 문자 제거
        out = out.replace(".", "").replace(",", "").strip()

        if out in ["flight", "train", "bus", "car", "subway", "taxi"]:
            return (out, "llm") if return_meta else out

        return ("unknown", "llm_invalid") if return_meta else "unknown"

    except Exception:
        return ("unknown", "llm_error") if return_meta else "unknown"

# ── [원본 셀 30] ─────────────────────────────────────────────
# ---------------------------
# 기본 EF 맵 (예시값)
# 필요하면 나중에 실제 EF DB/공식계수로 교체
# 단위: kgCO2e / ton-km 가정
# ---------------------------
# ── 운송 EF 맵: 통합 DB(배출계수_통합_DB.xlsx > 운송 시트) 기반 ──
# Cell 10에서 TRANSPORT_EF_MAP 이 이미 빌드됨.
# 여기서는 하위 호환 패치만 적용.

def refresh_transport_ef_map():
    """통합 DB에서 운송 EF 재로드"""
    global TRANSPORT_EF_MAP
    TRANSPORT_EF_MAP = _build_transport_ef_map_from_db()
    print(f"[OK] TRANSPORT_EF_MAP 갱신: {TRANSPORT_EF_MAP}")

def _kg_to_ton(weight_kg):
    w = to_float(weight_kg)
    if w is None:
        return None
    return w / 1000.0

@lru_cache(maxsize=512)
def _ai_estimate_distance_km(origin: str, dest: str, mode: str = "road", country_hint: str = "KR"):
    """
    지오코딩 기반 거리 추정(estimate_distance_ai/estimate_straight_distance)이 실패했을 때
    (도시/지명을 인식하지 못하는 등) AI에게 직접 두 지점 간 이동 거리를 추정하게 하는 최종 폴백.
    """
    mode_label = {"road": "도로(트럭)", "parcel": "택배", "sea": "해상", "air": "항공", "rail": "철도"}.get(mode, mode)
    prompt = f"""두 지점 간 실제 이동 거리를 편도 기준 km로 추정하라.
출발지: {origin}
도착지: {dest}
운송수단: {mode_label}
국가 힌트: {country_hint}

JSON으로만 답하라: {{"distance_km": <숫자>}}"""
    parsed = _call_claude_json(prompt, max_tokens=60)
    d = to_float(parsed.get("distance_km")) if parsed else None
    return validate_distance_km(d, mode=mode, country_hint=country_hint)

@lru_cache(maxsize=512)
def _ai_estimate_straight_km(origin: str, dest: str, country_hint: str = "KR"):
    """
    지오코딩(estimate_straight_distance)이 실패했을 때, AI에게 두 지점 간
    "직선거리(대권거리)"를 추정하게 한다. get_distance_km에서 지오코딩 성공 시와
    동일한 방식(직선거리 × 이동수단별 보정계수)으로 처리하기 위한 것으로,
    이동수단은 여기서 고려하지 않는다 (보정계수는 get_distance_km에서 곱함).
    """
    prompt = f"""두 지점 간 직선거리(대권거리, great-circle distance)를 km로 추정하라.
출발지: {origin}
도착지: {dest}
국가 힌트: {country_hint}

JSON으로만 답하라: {{"distance_km": <숫자>}}"""
    parsed = _call_claude_json(prompt, max_tokens=60)
    return to_float(parsed.get("distance_km")) if parsed else None

_DISTANCE_MODE_FACTOR = {"road": 1.3, "parcel": 1.35, "air": 1.09, "sea": 1.30, "rail": 1.20}
# air: 1.05 -> 1.09로 상향. ICAO Carbon Calculator / DEFRA 방법론에서 실제 비행경로가
# 관제 대기·우회 등으로 대권거리(great-circle distance)보다 약 8~9% 더 길다고 보는
# 우회보정(circuity/uplift)을 반영 (Climatiq/ICAO 벤치마킹 결과).

_AIRPORT_TEXT_HINTS = ["공항", "airport", "국제공항", "terminal"]

def _looks_like_airport(text: str) -> bool:
    """출발지/도착지 텍스트가 이미 '공항'을 가리키는지 판별 (공항명 또는 3자리 IATA 코드)."""
    t = _safe_text2(text)
    if not t:
        return False
    if _contains_any(t.lower(), _AIRPORT_TEXT_HINTS):
        return True
    return bool(re.fullmatch(r"[A-Za-z]{3}", t.strip()))


@lru_cache(maxsize=512)
def _ai_nearest_airport(location: str, country_hint: str = "KR"):
    """
    항공편 출발지/도착지가 공항이 아니라 주소·지명으로 입력됐을 때,
    그 지역에서 실제로 정기 여객노선을 운항하는 가장 가까운 주요 공항을 AI로 추정한다.
    (단순 최단거리 활주로가 아니라, 실제로 항공권을 끊어 이용할 만한 공항을 우선)
    이후 이 공항 이름을 다시 지오코딩해서 좌표를 구하고, 그 좌표 간 대권거리로
    항공 이동거리를 계산한다 (주소-대-주소 직선거리를 쓰지 않기 위함).
    """
    prompt = f"""아래 지역에서 국내선/국제선 항공편을 이용한다면 가장 가까운, 실제로 정기 여객노선이 있는
주요 공항의 이름을 알려줘. 단순히 지리적으로 제일 가까운 소형 비행장이 아니라
사람들이 실제로 항공권을 구매해 이용하는 공항을 골라라.

지역: {location}
국가 힌트: {country_hint}

JSON으로만 답하라: {{"airport_name": "<공항 정식 명칭, 국가 포함>", "iata": "<IATA 코드, 모르면 빈 문자열>"}}"""
    parsed = _call_claude_json(prompt, max_tokens=80)
    if not parsed:
        return None
    name = _safe_text2(parsed.get("airport_name"))
    return name or None


def _resolve_air_location(text: str, country_hint: str = "KR") -> str:
    """항공 모드일 때 출발지/도착지를 '공항명'으로 정규화 (이미 공항이면 그대로 둠)."""
    text = _safe_text2(text)
    if not text or _looks_like_airport(text):
        return text
    airport = _ai_nearest_airport(text, country_hint=country_hint)
    return airport or text


# ---------------------------
# 기존 코드 호환용 get_distance_km
# 기존 process_template_inventory / 출장 / 통근 함수에서 그대로 호출 가능
# ---------------------------
def get_distance_km(origin, dest, mode="road", country_hint="KR"):
    """
    PATCH: 이전에는 지오코딩 성공 시엔 "직선거리 × 보정계수"를, 지오코딩 실패로
    AI 폴백에 들어가면 "AI가 추정한 실이동거리를 보정계수 없이 그대로" 써서
    행마다 서로 다른 기준(직선거리 기반 vs 실거리 기반)이 섞여 있었다.
    지오코딩 성공 여부와 무관하게 "직선거리를 구한 뒤 동일한 보정계수를 곱하는"
    한 가지 방식으로 통일한다 (지오코딩 우선, 실패 시에만 AI가 직선거리를 추정).

    PATCH2: mode="air"인데 출발지/도착지가 공항이 아니라 주소·지명이면,
    그 좌표를 그대로 지오코딩해 직선거리를 재는 대신(과대/과소산정 원인),
    먼저 각 지점에서 가장 가까운 주요 공항으로 치환한 뒤 그 공항들 사이의
    거리를 계산한다 (Climatiq/ICAO 벤치마킹 결과 반영).
    """
    origin = _safe_text2(origin)
    dest = _safe_text2(dest)

    if not origin or not dest:
        return None

    if mode == "air":
        origin = _resolve_air_location(origin, country_hint=country_hint)
        dest = _resolve_air_location(dest, country_hint=country_hint)

    factor = _DISTANCE_MODE_FACTOR.get(mode, 1.0)

    straight = estimate_straight_distance(
        origin, dest,
        country_hint=country_hint if country_hint == "KR" else None
    )
    if straight is None:
        straight = _ai_estimate_straight_km(origin, dest, country_hint=country_hint)

    if straight is not None:
        validated = validate_distance_km(straight * factor, mode=mode, country_hint=country_hint)
        if validated is not None:
            return validated

    # 직선거리 기반 추정이 모두 실패한 극히 예외적인 경우에만,
    # AI에게 실이동거리를 직접 추정하게 하는 최종 폴백을 사용한다.
    return _ai_estimate_distance_km(origin, dest, mode=mode, country_hint=country_hint)

# ---------------------------
# infer_distance 내부도 get_distance_km 기반으로 통일
# ---------------------------
def infer_distance(row, mode):
    """
    return:
      distance_km
      distance_source: input / road_geo / parcel_geo / air_geo / sea_geo / rail_geo / unknown
    """
    input_distance = to_float(row.get("운송거리(km)"))
    country_hint = _safe_text2(row.get("국가")) or "KR"

    if input_distance is not None:
        validated = validate_distance_km(input_distance, mode=mode, country_hint=country_hint)
        if validated is not None:
            return validated, "input"

    origin = _safe_text2(row.get("출발지"))
    dest = _safe_text2(row.get("도착지"))

    if not origin or not dest:
        return None, "unknown"

    d = get_distance_km(origin, dest, mode=mode, country_hint=country_hint)
    if d is None:
        return None, "unknown"

    if mode == "road":
        return d, "road_geo"
    elif mode == "parcel":
        return d, "parcel_geo"
    elif mode == "air":
        return d, "air_geo"
    elif mode == "sea":
        return d, "sea_geo"
    elif mode == "rail":
        return d, "rail_geo"
    else:
        return d, "unknown"



# ---------------------------
# 디버그용: 운송 행 확인 함수
# ---------------------------
def debug_transport_row(row):
    mode, mode_source, mode_confidence = infer_transport_mode(row)
    distance_km, distance_source = infer_distance(row, mode)
    weight_kg = to_float(row.get("화물중량(kg)"))
    weight_ton = _kg_to_ton(weight_kg)
    ef = TRANSPORT_EF_MAP.get(mode)

    emission = None
    if distance_km is not None and weight_ton is not None and ef is not None:
        emission = distance_km * weight_ton * ef

    return {
        "mode": mode,
        "mode_source": mode_source,
        "mode_confidence": mode_confidence,
        "distance_km": distance_km,
        "distance_source": distance_source,
        "weight_kg": weight_kg,
        "weight_ton": weight_ton,
        "ef": ef,
        "emission": emission,
    }

# ── [원본 셀 32] ─────────────────────────────────────────────
def _has_val(v):
    return pd.notnull(v) and str(v).strip() != ""


def _norm(v):
    if not _has_val(v):
        return ""
    return str(v).strip().lower()


def _is_yes(v):
    return _norm(v) in {
        "y", "yes", "true", "1", "o",
        "예", "네", "사용", "있음", "유", "oui"
    }


def _safe_text_04(v):
    if not _has_val(v):
        return ""
    return str(v).strip().lower()


def _join_row_text_04(row):
    cols = [
        "데이터유형", "설명/적요", "품목명", "자산명", "판매제품명",
        "폐기물종류", "에너지원/연료명", "투자대상명", "비고"
    ]
    texts = []
    for c in cols:
        if c in row.index and _has_val(row.get(c)):
            texts.append(str(row.get(c)).strip().lower())
    return " ".join(texts)


def infer_scope3_category_04_detail(row):
    categories = [
        "Category3", "Category5", "Category8",
        "Category10", "Category11", "Category12",
        "Category13", "Category14", "Category15"
    ]

    scores = {cat: 0 for cat in categories}
    reasons = {cat: [] for cat in categories}

    def add(cat, pts, why):
        scores[cat] += pts
        reasons[cat].append(f"+{pts} {why}")

    def subtract(cat, pts, why):
        scores[cat] -= pts
        reasons[cat].append(f"-{pts} {why}")

    text = _join_row_text_04(row)
    data_type = _norm(row.get("데이터유형"))
    unit = _norm(row.get("단위"))

    item_name = _safe_text_04(row.get("품목명"))
    asset_name = _safe_text_04(row.get("자산명"))
    sold_name = _safe_text_04(row.get("판매제품명"))
    waste_name = _safe_text_04(row.get("폐기물종류"))
    energy_name = _safe_text_04(row.get("에너지원/연료명"))
    desc_text = _safe_text_04(row.get("설명/적요"))
    note_text = _safe_text_04(row.get("비고"))

    has_asset = _has_val(row.get("자산명"))
    has_sold_product = _has_val(row.get("판매제품명"))
    has_waste = _has_val(row.get("폐기물종류"))
    has_energy = _has_val(row.get("에너지원/연료명"))
    has_investment = _has_val(row.get("투자대상명")) or _has_val(row.get("지분율(%)"))
    is_leased = _is_yes(row.get("임차자산여부"))
    is_downstream = _is_yes(row.get("다운스트림여부"))
    is_franchise = _is_yes(row.get("프랜차이즈여부"))

    # -----------------------------
    # 1) 데이터유형 직접 신호
    # -----------------------------
    if "에너지" in data_type:
        add("Category3", 8, "데이터유형=에너지")
    if "폐기물" in data_type:
        add("Category5", 8, "데이터유형=폐기물")
    if "투자" in data_type:
        add("Category15", 8, "데이터유형=투자")
    if "임차" in data_type:
        if is_downstream:
            add("Category13", 8, "데이터유형=임차 + 다운스트림여부=Y")
        else:
            add("Category8", 8, "데이터유형=임차")
    if "프랜차이즈" in data_type:
        add("Category14", 8, "데이터유형=프랜차이즈")

    # 판매제품 계열 prior 강화
    if "판매제품" in data_type or "다운스트림" in data_type:
        add("Category10", 4, "데이터유형에 판매제품/다운스트림 포함")
        add("Category11", 4, "데이터유형에 판매제품/다운스트림 포함")
        add("Category12", 4, "데이터유형에 판매제품/다운스트림 포함")

    # -----------------------------
    # 2) 강한 컬럼 신호
    # -----------------------------
    if has_energy:
        add("Category3", 7, "에너지원/연료명 존재")
    if has_waste:
        add("Category5", 7, "폐기물종류 존재")
    if has_investment:
        add("Category15", 9, "투자대상명 또는 지분율 존재")

    if is_franchise:
        add("Category14", 9, "프랜차이즈여부=Y")

    if is_leased and is_downstream:
        add("Category13", 9, "임차자산여부=Y + 다운스트림여부=Y")
    elif is_leased:
        add("Category8", 9, "임차자산여부=Y")

    if has_sold_product:
        add("Category10", 4, "판매제품명 존재")
        add("Category11", 4, "판매제품명 존재")
        add("Category12", 4, "판매제품명 존재")

    if is_downstream:
        add("Category10", 3, "다운스트림여부=Y")
        add("Category11", 3, "다운스트림여부=Y")
        add("Category12", 3, "다운스트림여부=Y")

    if has_asset and not is_downstream:
        add("Category8", 2, "자산명 존재")
    if has_asset and is_downstream:
        add("Category13", 2, "자산명 존재 + 다운스트림여부=Y")

    # -----------------------------
    # 3) 키워드 신호
    # -----------------------------
    energy_keywords = [
        "전력", "전기", "스팀", "증기", "열", "냉열",
        "lng", "lpg", "도시가스", "연료", "fuel",
        "diesel", "gasoline", "휘발유", "경유", "가스"
    ]
    waste_keywords = [
        "폐기", "폐기물", "폐유", "폐수", "스크랩",
        "재활용", "소각", "매립", "처리", "scrap", "disposal"
    ]
    lease_keywords = [
        "임차", "리스", "lease", "rent", "렌탈", "임대"
    ]
    process_keywords = [
        "가공", "조립", "반제품", "중간재", "후공정", "processing",
        "원재료", "소재", "부품", "반가공", "혼합", "성형", "압출", "도금"
    ]
    use_keywords = [
        "사용", "충전", "사용전력", "전력소비", "연료소비", "사용시간",
        "수명", "내구연한", "운행", "사용단계", "전력사용", "운전", "가동",
        "cycle", "hours", "runtime"
    ]
    eol_keywords = [
        "폐기", "회수", "재활용", "매립", "소각", "분해",
        "end-of-life", "eol", "포장폐기", "폐포장", "폐배터리",
        "scrap", "disposal"
    ]
    franchise_keywords = ["가맹", "가맹점", "franchise"]
    investment_keywords = ["투자", "지분", "펀드", "portfolio", "equity"]

    if any(k in text for k in energy_keywords):
        add("Category3", 4, "텍스트에 에너지 관련 키워드 존재")

    if any(k in text for k in waste_keywords):
        add("Category5", 4, "텍스트에 폐기물 관련 키워드 존재")

    if any(k in text for k in lease_keywords):
        if is_downstream:
            add("Category13", 4, "텍스트에 임차/리스 키워드 + 다운스트림")
        else:
            add("Category8", 4, "텍스트에 임차/리스 키워드 존재")

    if any(k in text for k in franchise_keywords):
        add("Category14", 5, "텍스트에 프랜차이즈 관련 키워드 존재")

    if any(k in text for k in investment_keywords):
        add("Category15", 5, "텍스트에 투자 관련 키워드 존재")

    # -----------------------------
    # 3-1) 판매제품 계열 추가 prior / 세부 분류
    # -----------------------------
    if has_sold_product or is_downstream:
        # 기본적으로 downstream sold product family 점수
        add("Category10", 2, "판매제품/다운스트림 계열 기본 prior")
        add("Category11", 2, "판매제품/다운스트림 계열 기본 prior")
        add("Category12", 2, "판매제품/다운스트림 계열 기본 prior")

        intermediate_keywords = ["반제품", "중간재", "원재료", "소재", "부품"]
        final_product_keywords = ["완제품", "제품", "기기", "장비", "가전", "차량", "배터리", "노트북", "모니터"]

        if any(k in sold_name for k in intermediate_keywords) or any(k in item_name for k in intermediate_keywords):
            add("Category10", 4, "판매제품/품목명이 중간재·반제품 성격")

        if any(k in sold_name for k in final_product_keywords) or any(k in item_name for k in final_product_keywords):
            add("Category11", 2, "판매제품/품목명이 최종제품 성격")
            add("Category12", 2, "판매제품/품목명이 최종제품 성격")

        if any(k in text for k in process_keywords):
            add("Category10", 6, "텍스트에 판매제품 가공 관련 키워드 존재")

        if any(k in text for k in use_keywords):
            add("Category11", 6, "텍스트에 판매제품 사용 관련 키워드 존재")

        if any(k in text for k in eol_keywords):
            add("Category12", 6, "텍스트에 판매제품 폐기 관련 키워드 존재")

        # 사용단계는 에너지/연료 정보와 같이 오면 더 강함
        if has_energy:
            add("Category11", 4, "판매제품 + 에너지원/연료명 존재 → 사용단계 가능성")

        # 가공단계는 재료형 단위에서 자주 보임
        if unit in {"kg", "ton", "t", "m2", "m3"}:
            add("Category10", 2, f"판매제품 + 단위={unit} → 추가 가공형 가능성")

        # 폐기단계는 질량 단위와 자주 같이 옴
        if unit in {"kg", "ton", "t"}:
            add("Category12", 2, f"판매제품 + 단위={unit} → 폐기단계 산정 가능성")

        # -----------------------------
        # 3-2) 약한 감점 로직
        # -----------------------------
        if any(k in text for k in use_keywords):
            subtract("Category10", 2, "사용단계 신호가 강해 가공단계 가능성 낮춤")
            subtract("Category12", 1, "사용단계 신호가 강해 폐기단계 가능성 일부 낮춤")

        if any(k in text for k in eol_keywords):
            subtract("Category10", 2, "폐기단계 신호가 강해 가공단계 가능성 낮춤")
            subtract("Category11", 1, "폐기단계 신호가 강해 사용단계 가능성 일부 낮춤")

        if any(k in text for k in process_keywords):
            subtract("Category11", 1, "가공단계 신호가 강함")
            subtract("Category12", 1, "가공단계 신호가 강함")

    # -----------------------------
    # 4) 단위 보조 신호
    # -----------------------------
    if unit in {"kwh", "mwh", "gwh", "nm3", "sm3", "m3", "gj", "mj", "toe"}:
        add("Category3", 3, f"단위={unit} → 에너지형 데이터 가능성")

    if unit in {"kg", "ton", "t", "l"} and has_waste:
        add("Category5", 2, f"단위={unit} + 폐기물종류 존재")

    # -----------------------------
    # 5) 규칙 결과 정리
    # -----------------------------
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top1_cat, top1_score = ranked[0]
    top2_cat, top2_score = ranked[1]
    top3 = ranked[:3]

    # -----------------------------
    # 6) 최종 결정
    # -----------------------------
    use_llm = False
    if top1_score < 5:
        use_llm = True
    elif (top1_score - top2_score) <= 1:
        use_llm = True

    if use_llm:
        final_cat = infer_scope3_category_llm(row, "04_기타확장_Template")
        decision_type = "llm_fallback"
        reason_text = (
            f"규칙기반 분류가 모호하여 LLM fallback 사용. "
            f"규칙1순위={top1_cat}({top1_score}), 규칙2순위={top2_cat}({top2_score}), "
            f"TOP3={top3}"
        )
    else:
        final_cat = top1_cat
        decision_type = "rule"
        reason_text = (
            f"규칙기반 분류. 최종={top1_cat}({top1_score}). "
            f"주요근거: {' / '.join(reasons[top1_cat]) if reasons[top1_cat] else '없음'}"
        )

    score_map_text = " | ".join([f"{k}:{v}" for k, v in ranked])
    top3_text = " | ".join([f"{k}:{v}" for k, v in top3])

    return {
        "category": final_cat,
        "decision_type": decision_type,
        "rule_top_category": top1_cat,
        "rule_top_score": top1_score,
        "top3": top3,
        "top3_text": top3_text,
        "reason_text": reason_text,
        "score_map_text": score_map_text,
        "reasons_by_top1": " / ".join(reasons[top1_cat]) if reasons[top1_cat] else ""
    }

# ── [원본 셀 37] ─────────────────────────────────────────────
KNOWN_TEMPLATE_HEADERS = {
    "사업장", "법인명", "품목명", "자산명", "수량", "단위", "구매금액", "금액",
    "설명/적요", "비고", "거래처명", "운송수단", "운송거리(km)", "화물중량(kg)",
    "운송구분", "출발지", "도착지", "출장비용", "출장인원", "출장수단", "이동거리(km)",
    "교통수단", "직원수", "평균통근거리(km)", "데이터유형", "판매제품명", "폐기물종류",
    "에너지원/연료명", "투자대상명", "지분율(%)", "임차자산여부", "다운스트림여부", "프랜차이즈여부"
}


def _normalize_header_token(v):
    if pd.isnull(v):
        return ""
    return str(v).strip().lower()


def detect_template_header_row(input_file, sheet_name, max_rows=20):
    """
    헤더 행 감지:
    - KNOWN_TEMPLATE_HEADERS 매칭 (*, 공백 제거 후 비교)
    - * 로 시작하는 셀이 2개 이상인 행도 헤더로 인식
    - 매칭 점수가 같으면 더 아래 행(실제 헤더가 아래에 있음) 우선
    """
    preview = pd.read_excel(input_file, sheet_name=sheet_name, header=None, nrows=max_rows)
    best_row = 0
    best_score = -1

    # * 제거 후 소문자 비교용 set
    normalized_known = {_normalize_header_token(x) for x in KNOWN_TEMPLATE_HEADERS}
    # * 포함 원본도 추가 (예: *국가, *사업장)
    normalized_known |= {_normalize_header_token(x).lstrip("*").strip()
                         for x in KNOWN_TEMPLATE_HEADERS}

    for i in range(min(max_rows, len(preview))):
        row_vals_raw = [str(v).strip() if pd.notnull(v) else "" for v in preview.iloc[i].tolist()]

        # * 로 시작하는 필수 입력 셀 수 카운트
        star_count = sum(1 for v in row_vals_raw if v.startswith("*") and len(v) > 1)

        # KNOWN_TEMPLATE_HEADERS 매칭 (*, 공백 제거 후)
        row_vals_norm = [v.lstrip("*").strip().lower() for v in row_vals_raw if v]
        match_score = sum(1 for v in row_vals_norm if v in normalized_known)

        # 종합 점수: 매칭 + * 셀 가중치
        score = match_score + star_count * 2

        if score > best_score:
            best_score = score
            best_row = i

    return best_row if best_score >= 2 else 0


def read_template_sheet(input_file, sheet_name):
    """
    v1 템플릿 시트 읽기.

    구조:
      - 상단 작성가이드 행 (카테고리 정의, 작성기준 등)  ← 완전히 무시
      - 주헤더행: *국가, *사업장 ... | 연간 입력 | 월별 지출금액 | 비고(선택)
      - 서브헤더행: 연간지출금액(원) | 1월 | 2월 | ... | 12월
      - 데이터행: 실제 입력값

    핵심 로직:
      1) * 로 시작하는 컬럼이 2개 이상 있는 행 = 주헤더행으로 감지
      2) 주헤더 + 서브헤더 합쳐서 최종 컬럼명 구성
      3) 연간 입력 행 / 월별 입력 행 분리 (중복 방지)
      4) 월별 행은 melt 하지 않고 그대로 반환 (표준화 함수에서 처리)
    """
    df_raw = pd.read_excel(input_file, sheet_name=sheet_name, header=None)
    if df_raw.empty:
        return pd.DataFrame()

    MONTH_SET  = {f"{i}월" for i in range(1, 13)}
    ANNUAL_SET = {
        "연간지출금액(원)", "연간취득금액(원)", "연간사용량",
        "연간운송량", "연간운송금액(원)", "연간처리량",
        "연간이동횟수", "연간근무일수",
    }
    # 연간/월별 입력 그룹을 나타내는 병합셀 헤더 키워드
    GROUP_HEADERS = {
        "연간 입력", "월별 지출금액", "월별 취득금액", "월별 사용량",
        "월별 운송량", "월별 운송금액", "월별 처리량", "월별 이동횟수",
        "월별 근무일수", "건물전체 월별사용량", "출도착지",
    }

    # ── Step 1: 주헤더 행 찾기 ──────────────────────────────────
    # *국가, *사업장 등 * 로 시작하는 필수컬럼이 2개 이상인 행
    main_header_row = None
    for i in range(min(25, len(df_raw))):
        row_vals = [str(v).strip() for v in df_raw.iloc[i].tolist()
                    if pd.notnull(v) and str(v).strip()]
        star_count = sum(1 for v in row_vals if v.startswith("*"))
        # * 컬럼 2개 이상이거나, "연간 입력"/"월별 ~" 같은 그룹헤더 포함
        group_count = sum(1 for v in row_vals if v in GROUP_HEADERS)
        if star_count >= 2 or (star_count >= 1 and group_count >= 1):
            main_header_row = i
            break

    if main_header_row is None:
        # fallback: KNOWN_TEMPLATE_HEADERS 매칭
        main_header_row = detect_template_header_row(input_file, sheet_name)

    if main_header_row is None or main_header_row >= len(df_raw):
        return pd.DataFrame()

    # ── Step 2: 서브헤더 행 확인 ────────────────────────────────
    main_vals = [str(v).strip() if pd.notnull(v) else ""
                 for v in df_raw.iloc[main_header_row].tolist()]
    sub_row = main_header_row + 1
    has_sub = False

    if sub_row < len(df_raw):
        sub_vals_raw = [str(v).strip() if pd.notnull(v) else ""
                        for v in df_raw.iloc[sub_row].tolist()]
        month_cnt  = sum(1 for v in sub_vals_raw if v in MONTH_SET)
        annual_cnt = sum(1 for v in sub_vals_raw if v in ANNUAL_SET)
        has_sub = (month_cnt >= 2 or annual_cnt >= 1)

    # ── Step 3: 최종 컬럼명 조합 ────────────────────────────────
    if has_sub:
        sub_vals = sub_vals_raw
        final_cols = []
        for ci, (main, sub) in enumerate(zip(main_vals, sub_vals)):
            if main in GROUP_HEADERS and sub:
                # 그룹헤더 → 서브헤더 값으로 교체
                final_cols.append(sub)
            elif not main and sub:
                final_cols.append(sub)
            elif main.startswith("Unnamed"):
                final_cols.append(sub if sub else f"_col{ci}")
            else:
                final_cols.append(main if main else f"_col{ci}")
        data_start = sub_row + 1
    else:
        final_cols = [v if v else f"_col{ci}"
                      for ci, v in enumerate(main_vals)]
        data_start = main_header_row + 1

    # ── Step 4: 데이터 읽기 ────────────────────────────────────
    df = df_raw.iloc[data_start:].copy()
    # 컬럼 수 맞추기
    n_cols = min(len(final_cols), len(df.columns))
    df = df.iloc[:, :n_cols]
    df.columns = final_cols[:n_cols]

    # ── Step 5: 완전 빈 행 제거 ───────────────────────────────
    df = df.dropna(how="all").reset_index(drop=True)

    # ── Step 6: * 제거, 불필요 컬럼 제거 ─────────────────────
    df.columns = [str(c).lstrip("*").strip() for c in df.columns]
    # _col~ 로 시작하는 컬럼: 의미있는 값 없으면 제거
    # "AI 추천", "AI 자동산정" 같은 템플릿 안내 텍스트만 있는 컬럼도 제거
    AI_GUIDE_VALS = {"AI 추천", "AI 자동산정", "ai 추천", "ai 자동산정"}
    def _col_has_real_data(col):
        vals = df[col].dropna()
        real = [v for v in vals if str(v).strip() not in ("", "nan") + tuple(AI_GUIDE_VALS)]
        return len(real) > 0

    keep_cols = []
    for c in df.columns:
        if str(c).startswith("_col"):
            if _col_has_real_data(c):
                keep_cols.append(c)
        else:
            keep_cols.append(c)
    df = df[keep_cols]

    # 템플릿 안내 컬럼 제거 (배출계수, 배출량 컬럼에 "AI 추천" 같은 값만 있는 경우)
    cols_to_drop = []
    for c in df.columns:
        if any(k in str(c) for k in ["배출계수", "배출량"]):
            vals = df[c].dropna().astype(str).str.strip()
            if vals.empty or vals.isin(AI_GUIDE_VALS).all():
                cols_to_drop.append(c)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # ── Step 7: 연간/월별 행 구분 태깅 ───────────────────────
    annual_cols = [c for c in df.columns if c in ANNUAL_SET]
    month_cols  = [c for c in df.columns if c in MONTH_SET]

    if annual_cols and month_cols:
        def _has_monthly(r):
            return any(
                pd.notnull(r.get(m)) and str(r.get(m)).strip() not in ("", "nan", "0")
                for m in month_cols
            )
        monthly_mask = df.apply(_has_monthly, axis=1)
        df.loc[~monthly_mask, "입력방식"] = "연간"
        df.loc[monthly_mask,  "입력방식"] = "월별"
    elif month_cols:
        df["입력방식"] = "월별"
    else:
        df["입력방식"] = "연간"

    # ── Step 8: 작성가이드 잔재 행 제거 ─────────────────────
    # 첫 번째 컬럼(국가 등)에 한국어 가이드 텍스트가 들어온 행 제거
    GUIDE_PATTERNS = ["카테고리", "작성 기준", "※", "1)", "2)", "3)", "4)", "5)",
                      "- ", "Scope", "배출계수는", "온실가스 산정"]
    first_col = df.columns[0] if len(df.columns) > 0 else None
    if first_col:
        def _is_guide(v):
            s = str(v).strip()
            return any(s.startswith(p) for p in GUIDE_PATTERNS) or len(s) > 80
        guide_mask = df[first_col].apply(_is_guide)
        if guide_mask.any():
            df = df[~guide_mask].reset_index(drop=True)

    return df


def get_first_existing_value(row, candidates):
    for col in candidates:
        if col in row.index:
            return row.get(col)
    return None
# === 새 v1 템플릿 헤더 추가 ===
KNOWN_TEMPLATE_HEADERS = KNOWN_TEMPLATE_HEADERS | {
    # C1/C2 공통
    "국가", "사업장", "품목군", "자산분류",
    "상세 품목명/서비스명", "상세 자산명",
    "계정과목(선택)", "거래처명(선택)",
    "연간지출금액(원)", "연간취득금액(원)", "자산번호(선택)",
    # C3
    "연료/에너지명",
    # C4/C9
    "운송수단(선택)", "자재명/품목명", "중량단위",
    "운송거리(km)(선택)", "연간운송량", "연간운송금액(원)",
    # C5
    "폐기물 종류", "상세 폐기물명(선택)", "처리방법", "연간처리량",
    # C6/C7
    "출장주체", "편도거리(km)(선택)", "연간이동횟수", "연간근무일수",
    "출장항목",
    # C8/C13
    "임차자산명", "임대자산명", "에너지종류",
    "건물전체 면적", "사용 면적",
    "건물전체 인원수", "자사 인원수",
    "건물전체 연간사용량",
    # C11
    "보고연도", "에너지명",
    "제품당 연간 에너지 사용량",
    "수명(개월) 혹은 평균 사용기간(개월)",
    # C12
    "폐기물 대분류(선택)", "폐기물 중분류(선택)", "폐기물 소분류(선택)",
    "판매량", "제품중량", "제품중량 단위",
    # C14
    "프랜차이즈명", "가맹점 수", "연료 종류", "연료 사용량", "연료 단위",
    # C15
    "피투자기업명 (필수)", "지분율(%) (선택)",
    "피투자기업 Scope1배출량", "피투자기업 Scope2배출량",
    "비고(선택)",
}

# ── [원본 셀 38] ─────────────────────────────────────────────

# =========================================
# 템플릿 품목/자산명 -> curated NAICS shortlist 기반 Spend EF 매칭 보강
# 위치: 템플릿용 Category1/2 지출기반 EF 매칭 보강
# =========================================

# (노트북 셀 명령 제거됨 — 배포 전 미리 설치 필요: pip install rapidfuzz)
from rapidfuzz import process, fuzz
from functools import lru_cache
import json
import re
import pandas as pd

SPEND_DB_NAMES = (
    spend_db["name"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)
SPEND_DB_NAME_SET = set(SPEND_DB_NAMES)

# 1) 실무 shortlist: 사무용품/비품/탕비/폐기물 중심
CURATED_SPEND_CANDIDATES = [
    "Office Supplies (except Paper) Manufacturing",
    "Stationery Product Manufacturing",
    "Printing and Writing Paper Merchant Wholesalers",
    "Stationery and Office Supplies Merchant Wholesalers",
    "Office Supplies and Stationery Stores",
    "Office Furniture (except Wood) Manufacturing",
    "Wood Office Furniture Manufacturing",
    "Showcase, Partition, Shelving, and Locker Manufacturing",
    "Furniture Merchant Wholesalers",
    "Electronic Computer Manufacturing",
    "Computer Terminal and Other Computer Peripheral Equipment Manufacturing",
    "Computer and Computer Peripheral Equipment and Software Merchant Wholesalers",
    "Office Equipment Merchant Wholesalers",
    "Coffee and Tea Manufacturing",
    "Soft Drink Manufacturing",
    "Bottled Water Manufacturing",
    "General Line Grocery Merchant Wholesalers",
    "Other Grocery and Related Products Merchant Wholesalers",
    "Sanitary Paper Product Manufacturing",
    "Commercial Printing (except Screen and Books)",
    "Sign Manufacturing",
    "Hazardous Waste Treatment and Disposal",
    "Other Nonhazardous Waste Treatment and Disposal",
]

CURATED_SPEND_CANDIDATES = [x for x in CURATED_SPEND_CANDIDATES if x in SPEND_DB_NAME_SET]

GENERIC_NOISE_WORDS = [
    "구입", "구매", "구입대", "구입건", "구입비", "추가", "의 건",
    "관련", "정산", "품의", "사용", "용", "등", "외"
]

NOISE_PATTERNS = [
    r"\([^)]*\)",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}/\d{1,2}\b",
    r"\b\d+\.\d+\b",
    r"[=:,/.\-*]+",
    r"\b\d+\b",
]

# ══════════════════════════════════════════════════════════════════
# ERP_TO_NAICS_HINTS — 한국 기업 구매 품목 → NAICS 매핑 (86개 항목)
# 배출계수_통합_DB.xlsx > 구매_USEPA 시트와 동기화
# 우선순위: 1) 구매_USEPA 시트 직접 로드  2) 아래 하드코딩 fallback
# ══════════════════════════════════════════════════════════════════

def _load_usepa_hints_from_db():
    """통합 DB 구매_USEPA 시트 → ERP_TO_NAICS_HINTS 포맷으로 변환"""
    try:
        _ef_db = os.path.join(base_path, "배출계수_통합_DB.xlsx")
        df = pd.read_excel(_ef_db, sheet_name="구매_USEPA", header=2)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all").reset_index(drop=True)

        hints = []
        # 컬럼명 파악
        cat_col  = next((c for c in df.columns if "품목군" in c), None)
        sub_col  = next((c for c in df.columns if "상세 카테고리" in c or "카테고리" in c), None)
        kw_col   = next((c for c in df.columns if "키워드" in c), None)
        code_col = next((c for c in df.columns if "NAICS 코드" in c or ("NAICS" in c and "코드" in c)), None)
        title_col= next((c for c in df.columns if "NAICS Title" in c or "Title" in c), None)
        ef_col   = next((c for c in df.columns if "EF" in c and ("kgCO2" in c or "USD" in c)), None)

        if not all([cat_col, kw_col, title_col]):
            return None

        for _, row in df.iterrows():
            품목군 = str(row.get(cat_col,"")).strip()
            상세   = str(row.get(sub_col,"")).strip() if sub_col else ""
            kw_raw = str(row.get(kw_col,"")).strip()
            title  = str(row.get(title_col,"")).strip()
            naics  = str(row.get(code_col,"")).strip() if code_col else ""
            ef     = pd.to_numeric(row.get(ef_col), errors="coerce") if ef_col else None

            if not 품목군 or not title or 품목군.startswith("["):
                continue

            kws    = [k.strip() for k in kw_raw.split(",") if k.strip()]
            # 정규식 패턴 생성 (각 키워드를 OR 패턴으로)
            patterns = [rf"(?i){k}" for k in kws if k]

            # 기존 hints에 동일 title이 있으면 병합, 없으면 추가
            existing = next((h for h in hints if title in h.get("titles",[])), None)
            if existing:
                existing["patterns"].extend(patterns)
                existing["keywords"] = list(set(existing.get("keywords",[]) + kws))
            else:
                hints.append({
                    "family":     f"{품목군}_{상세}".replace(" ","_").replace("/","_").replace("·","_")[:40],
                    "category":   품목군,
                    "sub":        상세,
                    "patterns":   patterns,
                    "keywords":   kws,
                    "titles":     [title],
                    "naics_code": naics,
                    "ef_usd":     float(ef) if pd.notna(ef) else None,
                    "generic_ko": 상세 or 품목군,
                    "generic_en": title.split("(")[0].strip()[:40],
                })
        return hints
    except Exception as e:
        print(f"[경고] 구매_USEPA DB 로드 실패: {e}")
        return None

# DB에서 로드 시도, 실패 시 하드코딩 fallback
ERP_TO_NAICS_HINTS = _load_usepa_hints_from_db() or [
    # ── fallback: 핵심 12개 family ──
    {"family":"office_supplies","category":"사무용품","patterns":[r"사무용품",r"문구",r"소모품",r"스테플러"],"titles":["Office Supplies (except Paper) Manufacturing","Stationery and Office Supplies Merchant Wholesalers"],"generic_ko":"사무용품","generic_en":"office supplies"},
    {"family":"paper","category":"사무용품","patterns":[r"a4용지",r"복사지",r"용지",r"copy.?paper"],"titles":["Printing and Writing Paper Merchant Wholesalers"],"generic_ko":"복사용지","generic_en":"copy paper"},
    {"family":"toner","category":"사무용품","patterns":[r"토너",r"잉크",r"카트리지"],"titles":["Office Equipment Merchant Wholesalers"],"generic_ko":"토너/잉크","generic_en":"toner"},
    {"family":"office_furniture","category":"사무가구·비품","patterns":[r"의자",r"책상",r"파티션",r"캐비닛"],"titles":["Office Furniture (except Wood) Manufacturing","Showcase, Partition, Shelving, and Locker Manufacturing"],"generic_ko":"사무가구","generic_en":"office furniture"},
    {"family":"computer","category":"IT장비","patterns":[r"노트북",r"데스크탑",r"컴퓨터",r"pc\b",r"서버"],"titles":["Electronic Computer Manufacturing"],"generic_ko":"컴퓨터","generic_en":"computer"},
    {"family":"monitor","category":"IT장비","patterns":[r"모니터",r"디스플레이"],"titles":["Audio and Video Equipment Manufacturing"],"generic_ko":"모니터","generic_en":"monitor"},
    {"family":"coffee","category":"탕비·식음료","patterns":[r"커피",r"원두"],"titles":["Coffee and Tea Manufacturing"],"generic_ko":"커피","generic_en":"coffee"},
    {"family":"water","category":"탕비·식음료","patterns":[r"생수",r"정수기"],"titles":["Bottled Water Manufacturing"],"generic_ko":"생수","generic_en":"water"},
    {"family":"raw_material","category":"원재료/부자재","patterns":[r"철강",r"알루미늄",r"수지",r"원재료"],"titles":["Iron and Steel Mills and Ferroalloy Manufacturing"],"generic_ko":"원재료","generic_en":"raw material"},
    {"family":"packaging","category":"포장재","patterns":[r"포장재",r"박스",r"비닐"],"titles":["Corrugated and Solid Fiber Box Manufacturing"],"generic_ko":"포장재","generic_en":"packaging"},
    {"family":"outsourcing","category":"외주·서비스","patterns":[r"외주",r"용역",r"컨설팅"],"titles":["Management Consulting Services"],"generic_ko":"외주/용역","generic_en":"consulting"},
    {"family":"misc","category":"기타","patterns":[r"기타"],"titles":["Other Miscellaneous Durable Goods Merchant Wholesalers"],"generic_ko":"기타","generic_en":"misc"},
]

if ERP_TO_NAICS_HINTS:
    print(f"[OK] ERP_TO_NAICS_HINTS 로드: {len(ERP_TO_NAICS_HINTS)}개 항목")
    # spend_db에 실제 존재하는 title만 필터링
    if "SPEND_DB_NAME_SET" in dir() and SPEND_DB_NAME_SET:
        for hint in ERP_TO_NAICS_HINTS:
            hint["titles"] = [t for t in hint.get("titles",[]) if t in SPEND_DB_NAME_SET]
        print(f"  → spend_db 매칭 후 유효 hint: {sum(1 for h in ERP_TO_NAICS_HINTS if h['titles'])}개")

GENERIC_NOISE_WORDS = [
    "구입", "구매", "구입대", "구입건", "구입비", "추가", "의 건",
    "관련", "정산", "품의", "사용", "용", "등", "외"
]

NOISE_PATTERNS = [
    r"\([^)]*\)",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}/\d{1,2}\b",
    r"\b\d+\.\d+\b",
    r"[=:,/.\-*]+",
    r"\b\d+\b",
]

ERP_TO_NAICS_HINTS = [
    {
        "family": "office_supplies",
        "patterns": [r"사무용품", r"문구", r"소모품", r"스테플러", r"계산기", r"바인더", r"서류봉투", r"도장", r"고무인", r"라벨지", r"화이트보드", r"자석테이프", r"펜심", r"책꽂이", r"결재판"],
        "titles": [
            "Office Supplies (except Paper) Manufacturing",
            "Stationery and Office Supplies Merchant Wholesalers",
            "Office Supplies and Stationery Stores",
        ],
        "generic_ko": "사무용품",
        "generic_en": "office supplies",
    },
    {
        "family": "paper",
        "patterns": [r"a4용지", r"복사지", r"copy\s*paper", r"용지", r"복사\s*지"],
        "titles": [
            "Printing and Writing Paper Merchant Wholesalers",
            "Stationery Product Manufacturing",
        ],
        "generic_ko": "복사용지",
        "generic_en": "copy paper",
    },
    {
        "family": "printer_supplies",
        "patterns": [r"토너", r"잉크", r"카트리지", r"프린터\s*토너", r"복사기\s*토너", r"드럼\s*유닛"],
        "titles": [
            "Office Supplies (except Paper) Manufacturing",
            "Office Equipment Merchant Wholesalers",
        ],
        "generic_ko": "프린터 소모품",
        "generic_en": "printer supplies",
    },
    {
        "family": "office_furniture",
        "patterns": [r"의자", r"책상", r"보조책상", r"파티션", r"옷장", r"캐비닛", r"비품", r"수납장"],
        "titles": [
            "Office Furniture (except Wood) Manufacturing",
            "Wood Office Furniture Manufacturing",
            "Showcase, Partition, Shelving, and Locker Manufacturing",
            "Furniture Merchant Wholesalers",
        ],
        "generic_ko": "사무용 비품/가구",
        "generic_en": "office furniture",
    },
    {
        "family": "computer",
        "patterns": [r"노트북", r"데스크탑", r"컴퓨터", r"pc\b", r"서버"],
        "titles": [
            "Electronic Computer Manufacturing",
            "Computer and Computer Peripheral Equipment and Software Merchant Wholesalers",
        ],
        "generic_ko": "컴퓨터 장비",
        "generic_en": "computer equipment",
    },
    {
        "family": "monitor",
        "patterns": [r"모니터", r"디스플레이", r"태블릿모니터"],
        "titles": [
            "Computer Terminal and Other Computer Peripheral Equipment Manufacturing",
            "Computer and Computer Peripheral Equipment and Software Merchant Wholesalers",
        ],
        "generic_ko": "모니터/주변기기",
        "generic_en": "computer monitor",
    },
    {
        "family": "coffee",
        "patterns": [r"커피원두", r"원두", r"캡슐\s*커피", r"커피"],
        "titles": [
            "Coffee and Tea Manufacturing",
            "Other Grocery and Related Products Merchant Wholesalers",
        ],
        "generic_ko": "커피/원두",
        "generic_en": "coffee",
    },
    {
        "family": "water",
        "patterns": [r"생수", r"식수대", r"정수기\s*물", r"정수기", r"물통"],
        "titles": [
            "Bottled Water Manufacturing",
            "Other Grocery and Related Products Merchant Wholesalers",
        ],
        "generic_ko": "생수/식수",
        "generic_en": "drinking water",
    },
    {
        "family": "beverages_snacks",
        "patterns": [r"주방음료", r"탕비", r"간식", r"음료", r"자판기\s*재료", r"다과"],
        "titles": [
            "Other Grocery and Related Products Merchant Wholesalers",
            "General Line Grocery Merchant Wholesalers",
            "Soft Drink Manufacturing",
        ],
        "generic_ko": "주방음료/간식",
        "generic_en": "beverages and snacks",
    },
    {
        "family": "paper_cups_tissue",
        "patterns": [r"종이컵", r"휴지", r"티슈", r"키친타월"],
        "titles": [
            "Sanitary Paper Product Manufacturing",
            "Stationery Product Manufacturing",
        ],
        "generic_ko": "위생 종이류",
        "generic_en": "paper hygiene products",
    },
    {
        "family": "printing_sign",
        "patterns": [r"상장케이스", r"상장", r"명함", r"인쇄", r"출력물", r"현수막", r"사인", r"표지판", r"기념품"],
        "titles": [
            "Commercial Printing (except Screen and Books)",
            "Sign Manufacturing",
        ],
        "generic_ko": "인쇄물/표지/기념품",
        "generic_en": "printed materials",
    },
    {
        "family": "waste",
        "patterns": [r"폐기물", r"오니", r"폐수", r"처리비", r"처리\s*용역"],
        "titles": [
            "Hazardous Waste Treatment and Disposal",
            "Other Nonhazardous Waste Treatment and Disposal",
        ],
        "generic_ko": "폐기물 처리",
        "generic_en": "waste treatment",
    },
]

def clean_product_text(product: str) -> str:
    if pd.isnull(product):
        return ""

    s = str(product).strip()
    if not s:
        return ""

    for p in NOISE_PATTERNS:
        s = re.sub(p, " ", s, flags=re.IGNORECASE)

    for w in GENERIC_NOISE_WORDS:
        s = s.replace(w, " ")

    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_hint_rules(product: str):
    text = str(product or "")
    matched = []
    for rule in ERP_TO_NAICS_HINTS:
        if any(re.search(p, text, flags=re.IGNORECASE) for p in rule["patterns"]):
            matched.append(rule)
    return matched

def build_allowed_titles(product: str):
    matched = detect_hint_rules(product)
    titles = []
    if matched:
        for rule in matched:
            titles.extend(rule["titles"])
    else:
        titles.extend(CURATED_SPEND_CANDIDATES)

    # broad fallback 몇 개 항상 포함
    broad_defaults = [
        "Office Supplies (except Paper) Manufacturing",
        "Stationery and Office Supplies Merchant Wholesalers",
        "Other Grocery and Related Products Merchant Wholesalers",
        "Furniture Merchant Wholesalers",
        "Office Equipment Merchant Wholesalers",
    ]
    titles.extend([x for x in broad_defaults if x in SPEND_DB_NAME_SET])

    deduped = []
    seen = set()
    for t in titles:
        if t in SPEND_DB_NAME_SET and t not in seen:
            deduped.append(t)
            seen.add(t)
    return deduped[:12]

def guess_generic_from_rules(product: str):
    matched = detect_hint_rules(product)
    if not matched:
        return None
    # 첫 규칙 우선
    rule = matched[0]
    return {
        "generic_korean": rule["generic_ko"],
        "generic_english": rule["generic_en"],
        "family": rule["family"],
        "rule_titles": rule["titles"],
    }

def safe_json_loads(text: str):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

def build_prompt_for_curated_naics(product: str, cleaned_product: str, allowed_titles):
    allowed_text = "\n".join([f"- {x}" for x in allowed_titles]) if allowed_titles else "- (no candidates)"
    return f"""
You map Korean ERP purchase text to one of the allowed NAICS-like spend factor titles.

Rules:
1) Ignore dates, payment details, voucher info, quantity formatting, and internal accounting wording.
2) Infer the generic purchased item or service.
3) Choose up to 3 best titles ONLY from the allowed titles below.
4) If the text is broad, choose a broad title.
5) Do not invent a title outside the allowed list.

Original ERP text:
{product}

Lightly cleaned text:
{cleaned_product}

Allowed titles:
{allowed_text}

Return JSON only:
{{
  "generic_korean": "...",
  "generic_english": "...",
  "selected_titles": ["title1", "title2", "title3"],
  "confidence": 0.0,
  "reason_short": "..."
}}

Examples:
- 사무용품 -> office supplies -> Office Supplies (except Paper) Manufacturing
- 복사지 -> copy paper -> Printing and Writing Paper Merchant Wholesalers
- 토너 -> printer supplies -> Office Equipment Merchant Wholesalers
- 의자 -> office furniture -> Office Furniture (except Wood) Manufacturing
- 노트북 -> computer equipment -> Electronic Computer Manufacturing
- 생수 -> drinking water -> Bottled Water Manufacturing
- 오니 처리비 -> waste treatment -> Hazardous Waste Treatment and Disposal
"""

@lru_cache(maxsize=2048)
def translate_product_for_search_info(product):
    if pd.isnull(product):
        return None

    product = str(product).strip()
    if not product:
        return None

    cleaned_product = clean_product_text(product)
    allowed_titles = build_allowed_titles(product)
    rule_guess = guess_generic_from_rules(product)

    prompt = build_prompt_for_curated_naics(
        product=product,
        cleaned_product=cleaned_product,
        allowed_titles=allowed_titles,
    )

    res = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        system="Return only valid JSON, with no other text before or after it. "
               "Select titles only from the allowed list.",
        messages=[{"role": "user", "content": prompt}],
    )

    parsed = safe_json_loads(_extract_response_text(res))
    selected_titles = parsed.get("selected_titles", [])
    if not isinstance(selected_titles, list):
        selected_titles = []

    # allowed list 안의 값만 유지
    selected_titles = [x for x in selected_titles if x in allowed_titles]

    # LLM이 못 고르면 규칙 기반 후보를 우선 fallback
    if not selected_titles and rule_guess:
        selected_titles = [x for x in rule_guess["rule_titles"] if x in allowed_titles][:3]

    generic_korean = str(parsed.get("generic_korean", "")).strip()
    generic_english = str(parsed.get("generic_english", "")).strip()
    confidence = parsed.get("confidence", 0.0)
    reason_short = str(parsed.get("reason_short", "")).strip()

    if rule_guess:
        if not generic_korean:
            generic_korean = rule_guess["generic_korean"]
        if not generic_english:
            generic_english = rule_guess["generic_english"]

    candidate_pool = []
    for idx, title in enumerate(selected_titles):
        # LLM 선택 우선순위에 따른 인위적 score
        score = max(95 - idx * 5, 80)
        candidate_pool.append((title, score))

    # 혹시 selected_titles가 1개도 없으면 allowed title 안에서만 fuzzy
    if not candidate_pool and generic_english:
        fuzzy_hits = process.extract(
            generic_english,
            allowed_titles,
            scorer=fuzz.WRatio,
            limit=5
        )
        for name, score, _ in fuzzy_hits:
            if score >= 75:
                candidate_pool.append((name, score))

    return {
        "generic_korean": generic_korean,
        "generic_english": generic_english,
        "selected_titles": selected_titles,
        "allowed_titles": allowed_titles,
        "candidate_pool": candidate_pool,
        "confidence": confidence,
        "reason_short": reason_short,
    }

@lru_cache(maxsize=2048)
def translate_product_for_search(product):
    info = translate_product_for_search_info(product)
    if not info:
        return None

    # 검색/디버그용 문자열
    if info.get("generic_english"):
        return info["generic_english"]

    selected = info.get("selected_titles", [])
    if selected:
        return selected[0]

    return None

def get_naics_candidates(product, choices=None, top_k=10):
    info = translate_product_for_search_info(product)
    if not info:
        return []

    pool = info.get("candidate_pool", [])
    if pool:
        return [(name, score, None) for name, score in pool[:top_k]]

    return []

def normalize_product_to_naics(product, top_k=10):
    candidates = get_naics_candidates(product, top_k=top_k)
    if not candidates:
        return None
    best_title, best_score, _ = candidates[0]
    if best_score < 78:
        return None
    return best_title

@lru_cache(maxsize=2048)
def search_spend_ef(product):
    if pd.isnull(product):
        return None, None

    info = translate_product_for_search_info(product)
    if not info:
        return None, None

    # 1) LLM/규칙이 선택한 shortlist 우선
    for title in info.get("selected_titles", []):
        if title in SPEND_DB_NAME_SET:
            match = spend_db[spend_db["name"] == title]
            if len(match) > 0:
                best = match.iloc[0]
                return best["name"], best["ef"]

    # 2) candidate_pool 기준
    candidate_pool = info.get("candidate_pool", [])
    if candidate_pool:
        best_name, best_score = candidate_pool[0]
        if best_score >= 78 and best_name in SPEND_DB_NAME_SET:
            match = spend_db[spend_db["name"] == best_name]
            if len(match) > 0:
                best = match.iloc[0]
                return best["name"], best["ef"]

    # 3) 정말 마지막 fallback: allowed titles 안에서 generic_english로만 fuzzy
    generic_english = info.get("generic_english", "")
    allowed_titles = info.get("allowed_titles", [])
    if generic_english and allowed_titles:
        fallback = process.extractOne(
            generic_english,
            allowed_titles,
            scorer=fuzz.WRatio
        )
        if fallback and fallback[1] >= 82:
            title = fallback[0]
            match = spend_db[spend_db["name"] == title]
            if len(match) > 0:
                best = match.iloc[0]
                return best["name"], best["ef"]

    return None, None

# ── [원본 셀 41] ─────────────────────────────────────────────
def _safe_text(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


TITLE_TYPE_REASON = {
    "Manufacturing": "제조 기준 후보",
    "Merchant Wholesalers": "도매 유통 기준 후보",
    "Stores": "소매 구매 기준 후보",
    "Electronic Shopping and Mail-Order Houses": "온라인 구매 기준 후보",
    "Treatment and Disposal": "처리/처분 서비스 기준 후보",
    "Collection": "수거 서비스 기준 후보",
    "Facilities": "처리 시설 기준 후보",
}

FAMILY_LABELS = {
    "office_supplies": "일반 사무용품",
    "paper": "복사용지/문서용 종이",
    "printer_supplies": "프린터 소모품",
    "office_furniture": "사무가구/비품",
    "computer": "컴퓨터/IT 장비",
    "monitor": "모니터/디스플레이",
    "coffee": "커피/원두",
    "water": "생수/식수",
    "beverage": "주방음료/탕비",
    "sanitary_paper": "위생/종이류",
    "waste": "폐기물 처리",
}

FAMILY_KEYWORDS = {
    "office_supplies": ["사무용품", "문구", "소모품", "office supplies", "stationery"],
    "paper": ["a4", "복사지", "복사용지", "copy paper", "용지", "paper"],
    "printer_supplies": ["토너", "잉크", "cartridge", "카트리지", "드럼", "printer", "프린터"],
    "office_furniture": ["의자", "chair", "책상", "desk", "파티션", "partition", "캐비닛", "cabinet", "옷장", "비품"],
    "computer": ["노트북", "laptop", "데스크탑", "desktop", "computer", "pc"],
    "monitor": ["모니터", "monitor", "display"],
    "coffee": ["커피", "원두", "coffee", "tea", "캡슐"],
    "water": ["생수", "식수", "정수기", "water", "식수대"],
    "beverage": ["주방음료", "탕비", "간식", "음료", "beverage", "snack"],
    "sanitary_paper": ["종이컵", "휴지", "티슈", "paper cup", "tissue"],
    "waste": ["폐기물", "오니", "waste", "sludge", "처리비", "처리"],
}

SPEND_FAMILY_TITLES = {
    "office_supplies": [
        "Office Supplies (except Paper) Manufacturing",
        "Stationery and Office Supplies Merchant Wholesalers",
        "Office Supplies and Stationery Stores",
        "Stationery Product Manufacturing",
        "Printing and Writing Paper Merchant Wholesalers",
    ],
    "paper": [
        "Printing and Writing Paper Merchant Wholesalers",
        "Stationery Product Manufacturing",
        "Stationery and Office Supplies Merchant Wholesalers",
        "Office Supplies and Stationery Stores",
        "All Other Converted Paper Product Manufacturing",
    ],
    "printer_supplies": [
        "Office Supplies (except Paper) Manufacturing",
        "Stationery and Office Supplies Merchant Wholesalers",
        "Office Equipment Merchant Wholesalers",
        "Computer and Computer Peripheral Equipment and Software Merchant Wholesalers",
        "Other Electronic Parts and Equipment Merchant Wholesalers",
    ],
    "office_furniture": [
        "Office Furniture (except Wood) Manufacturing",
        "Wood Office Furniture Manufacturing",
        "Furniture Merchant Wholesalers",
        "Showcase, Partition, Shelving, and Locker Manufacturing",
        "Institutional Furniture Manufacturing",
    ],
    "computer": [
        "Electronic Computer Manufacturing",
        "Computer and Computer Peripheral Equipment and Software Merchant Wholesalers",
        "Office Equipment Merchant Wholesalers",
        "Electronics Stores",
        "Electronic Shopping and Mail-Order Houses",
    ],
    "monitor": [
        "Computer Terminal and Other Computer Peripheral Equipment Manufacturing",
        "Computer and Computer Peripheral Equipment and Software Merchant Wholesalers",
        "Office Equipment Merchant Wholesalers",
        "Electronics Stores",
        "Electronic Shopping and Mail-Order Houses",
    ],
    "coffee": [
        "Coffee and Tea Manufacturing",
        "Other Grocery and Related Products Merchant Wholesalers",
        "General Line Grocery Merchant Wholesalers",
        "Electronic Shopping and Mail-Order Houses",
        "Warehouse Clubs and Supercenters",
    ],
    "water": [
        "Bottled Water Manufacturing",
        "Other Grocery and Related Products Merchant Wholesalers",
        "General Line Grocery Merchant Wholesalers",
        "Electronic Shopping and Mail-Order Houses",
        "Warehouse Clubs and Supercenters",
    ],
    "beverage": [
        "Other Grocery and Related Products Merchant Wholesalers",
        "General Line Grocery Merchant Wholesalers",
        "Soft Drink Manufacturing",
        "Coffee and Tea Manufacturing",
        "Warehouse Clubs and Supercenters",
    ],
    "sanitary_paper": [
        "Sanitary Paper Product Manufacturing",
        "All Other Converted Paper Product Manufacturing",
        "Office Supplies (except Paper) Manufacturing",
        "Stationery and Office Supplies Merchant Wholesalers",
        "Office Supplies and Stationery Stores",
    ],
    "waste": [
        "Hazardous Waste Treatment and Disposal",
        "Other Nonhazardous Waste Treatment and Disposal",
        "Hazardous Waste Collection",
        "Other Waste Collection",
        "Materials Recovery Facilities",
    ],
}


def _norm_title_key(x):
    return _safe_text(x).strip().lower()


def _unique_recommendations(recs, exclude_title=None, limit=4):
    out = []
    seen = set()
    exclude_key = _norm_title_key(exclude_title)

    for rec in recs or []:
        title = _safe_text(rec.get("title"))
        if not title:
            continue
        key = _norm_title_key(title)
        if exclude_key and key == exclude_key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": title, "ef": to_float(rec.get("ef"))})
        if len(out) >= limit:
            break

    return out


def _existing_spend_titles(titles):
    out = []
    db_names = spend_db["name"].dropna().astype(str).str.strip()
    for t in titles:
        match = spend_db[db_names == str(t).strip()]
        if len(match) > 0:
            ef = to_float(match.iloc[0]["ef"])
            out.append({"title": t, "ef": ef})
    return out


def detect_spend_family(product="", desc="", vendor=""):
    text = " ".join([_safe_text(product), _safe_text(desc), _safe_text(vendor)]).lower()

    if any(k in text for k in ["폐기물", "오니", "waste", "sludge", "처리비", "처리"]):
        return "waste"
    if any(k in text for k in ["토너", "ink", "잉크", "cartridge", "카트리지", "드럼", "printer", "프린터"]):
        return "printer_supplies"
    if any(k in text for k in ["a4", "복사지", "복사용지", "copy paper", "용지", "paper"]):
        return "paper"
    if any(k in text for k in ["의자", "chair", "책상", "desk", "파티션", "partition", "캐비닛", "cabinet", "옷장", "locker", "비품"]):
        return "office_furniture"
    if any(k in text for k in ["모니터", "monitor", "display"]):
        return "monitor"
    if any(k in text for k in ["노트북", "laptop", "데스크탑", "desktop", "pc computer", "computer", "pc"]):
        return "computer"
    if any(k in text for k in ["커피", "원두", "coffee", "tea", "캡슐"]):
        return "coffee"
    if any(k in text for k in ["생수", "식수", "정수기", "water", "식수대"]):
        return "water"
    if any(k in text for k in ["주방음료", "탕비", "간식", "음료", "beverage", "snack"]):
        return "beverage"
    if any(k in text for k in ["종이컵", "휴지", "티슈", "paper cup", "tissue"]):
        return "sanitary_paper"
    return "office_supplies"


def _detect_reason_keywords(product="", desc="", vendor="", family=None, max_n=2):
    text = " ".join([_safe_text(product), _safe_text(desc), _safe_text(vendor)]).lower()
    kws = FAMILY_KEYWORDS.get(family, [])
    found = []
    for kw in kws:
        if kw.lower() in text and kw not in found:
            found.append(kw)
        if len(found) >= max_n:
            break
    return found


def _title_basis_reason(title):
    for suffix, reason in TITLE_TYPE_REASON.items():
        if suffix in str(title):
            return reason
    return "관련 구매 분류 후보"


def _compose_short_reason(title, family, product="", desc="", vendor="", applied=False, method=None):
    keywords = _detect_reason_keywords(product, desc, vendor, family=family, max_n=2)
    keyword_txt = ", ".join([f"'{k}'" for k in keywords]) if keywords else "텍스트 패턴"
    family_label = FAMILY_LABELS.get(family, "관련 품목군")
    basis_reason = _title_basis_reason(title)
    tail = "실제 적용 후보." if applied else "가장 우선 검토 후보."

    if method == "Activity":
        return f"키워드 {keyword_txt} 기반, 수량 기준 품목으로 판단. {tail}"

    return f"키워드 {keyword_txt} 기반, {family_label}으로 판단. {basis_reason}. {tail}"


def make_ef_db_reason(ef_db_name, method, product="", desc="", vendor="", family=None, pick_source=None):
    if not _safe_text(ef_db_name):
        return ""
    return _compose_short_reason(
        ef_db_name,
        family=family,
        product=product,
        desc=desc,
        vendor=vendor,
        applied=True,
        method=method,
    )


def make_recommendation_reason(title, family, product="", desc="", vendor="", rank=1):
    family_label = FAMILY_LABELS.get(family, "관련 품목군")
    keywords = _detect_reason_keywords(product, desc, vendor, family=family, max_n=2)
    keyword_txt = ", ".join([f"'{k}'" for k in keywords]) if keywords else "텍스트 패턴"
    basis_reason = _title_basis_reason(title)

    if rank == 1:
        rank_txt = "가장 우선 검토 후보"
    elif rank <= 3:
        rank_txt = "상위 대안 후보"
    else:
        rank_txt = "대체 후보"

    return f"키워드 {keyword_txt} 기반, {family_label}으로 판단. {basis_reason}. {rank_txt}."


def get_spend_recommendations(product="", desc="", vendor="", top_k=8):
    family = detect_spend_family(product, desc, vendor)
    family_titles = SPEND_FAMILY_TITLES.get(family, SPEND_FAMILY_TITLES["office_supplies"])
    recs = _existing_spend_titles(family_titles)

    if len(recs) < top_k:
        base_query = _safe_text(product) or _safe_text(desc) or _safe_text(vendor)
        generic_english = ""
        try:
            info = translate_product_for_search_info(base_query)
            if info:
                generic_english = (
                    _safe_text(info.get("generic_english"))
                    or _safe_text(info.get("english_search_query"))
                    or _safe_text(info.get("english_db_name"))
                )
        except Exception:
            generic_english = ""

        if generic_english:
            try:
                choices = spend_db["name"].dropna().astype(str).unique().tolist()
                fuzzy_hits = process.extract(
                    generic_english,
                    choices,
                    scorer=fuzz.WRatio,
                    limit=10,
                )
                for title, score, _ in fuzzy_hits:
                    if score >= 78 and title not in [r["title"] for r in recs]:
                        match = spend_db[spend_db["name"].astype(str).str.strip() == str(title).strip()]
                        if len(match) > 0:
                            ef = to_float(match.iloc[0]["ef"])
                            recs.append({"title": title, "ef": ef})
                    if len(recs) >= top_k:
                        break
            except Exception:
                pass

    recs = _unique_recommendations(recs, exclude_title=None, limit=top_k)
    return recs[:top_k], family


def select_primary_spend_candidate(product="", desc="", vendor=""):
    base_query = _safe_text(product) or _safe_text(desc) or _safe_text(vendor)

    direct_name, direct_ef = None, None
    if base_query:
        try:
            direct_name, direct_ef = search_spend_ef(base_query)
            direct_ef = to_float(direct_ef)
        except Exception:
            direct_name, direct_ef = None, None

    recs, family = get_spend_recommendations(product=product, desc=desc, vendor=vendor, top_k=8)
    family_titles = SPEND_FAMILY_TITLES.get(family, [])
    family_title_keys = {_norm_title_key(x) for x in family_titles}

    if family == "printer_supplies" and recs:
        top = recs[0]
        return {
            "name": top["title"],
            "ef": to_float(top["ef"]),
            "family": family,
            "pick_source": "family_priority",
            "recommendations": recs,
        }

    if direct_name and direct_ef is not None:
        if _norm_title_key(direct_name) in family_title_keys or not recs:
            return {
                "name": direct_name,
                "ef": direct_ef,
                "family": family,
                "pick_source": "direct_search",
                "recommendations": recs,
            }

    if recs:
        top = recs[0]
        return {
            "name": top["title"],
            "ef": to_float(top["ef"]),
            "family": family,
            "pick_source": "top_recommendation",
            "recommendations": recs,
        }

    if direct_name and direct_ef is not None:
        return {
            "name": direct_name,
            "ef": direct_ef,
            "family": family,
            "pick_source": "direct_search",
            "recommendations": recs,
        }

    return {
        "name": None,
        "ef": None,
        "family": family,
        "pick_source": "none",
        "recommendations": [],
    }


def format_emission_formula(method, qty=None, unit=None, spend=None, ef=None, emission=None):
    ef = to_float(ef)
    emission = to_float(emission)
    qty = to_float(qty)
    spend = to_float(spend)

    if method == "Activity" and qty is not None and ef is not None and emission is not None:
        unit_txt = _safe_text(unit)
        unit_txt = f" {unit_txt}" if unit_txt else ""
        return f"배출량 = 수량 × 배출계수 = {qty:g}{unit_txt} × {ef:g} = {emission:g}"
    if method == "Spend" and spend is not None and ef is not None and emission is not None:
        return f"배출량 = 구매금액 × 배출계수 = {spend:g} × {ef:g} = {emission:g}"
    return ""


def build_recommendation_columns(recs, spend_amount, product="", desc="", vendor="", family=None, recommend_n=4):
    """카테고리 1/2 구매·자산용 추천 EF 컬럼 생성.
    기존 3개 추천에서 4개 추천으로 확장했습니다.
    """
    out = {}
    spend_amount = to_float(spend_amount)

    recs = recs or []
    for i in range(recommend_n):
        idx = i + 1
        if i < len(recs):
            ef_name = recs[i]["title"]
            ef_val = to_float(recs[i]["ef"])
            alt_emission = None
            if spend_amount is not None and ef_val is not None:
                alt_emission = spend_amount * ef_val

            out[f"추천EF{idx}"] = ef_name
            out[f"추천계수{idx}"] = ef_val
            out[f"추천배출량{idx}"] = alt_emission
            out[f"추천사유{idx}"] = make_recommendation_reason(
                ef_name, family, product=product, desc=desc, vendor=vendor, rank=idx
            )
        else:
            out[f"추천EF{idx}"] = None
            out[f"추천계수{idx}"] = None
            out[f"추천배출량{idx}"] = None
            out[f"추천사유{idx}"] = None

    return out



# ── 출장·통근 EF: 통합 DB(배출계수_통합_DB.xlsx > 출장_통근 시트) 기반 ──
# Cell 10에서 TRAVEL_EF_MAP / BUSINESS_TRAVEL_EF / COMMUTE_EF 빌드됨.
# 여기서는 기존 변수명 호환을 위해 재할당.

# TRAVEL_EF_MAP가 아직 정의 안된 경우를 위한 fallback
if "TRAVEL_EF_MAP" not in dir():
    # DEFRA 2025 fallback (DB 로드 실패 시)
    TRAVEL_EF_MAP = {
        "flight":0.14253,"flight_dom":0.22928,"flight_short":0.12786,
        "flight_short_eco":0.12576,"flight_short_biz":0.18863,
        "flight_long":0.15282,"flight_long_eco":0.11704,
        "flight_long_prem":0.18726,"flight_long_biz":0.33940,
        "flight_long_first":0.46814,"flight_eco":0.10916,
        "flight_biz":0.31656,"flight_first":0.43663,
        "bus":0.10385,"coach":0.02776,
        "train":0.03546,"train_intl":0.00446,"tram":0.02860,
        "subway":0.02780,
        "taxi":0.20806,"taxi_black":0.30604,
        "car":0.17304,"car_small":0.14340,"car_medium":0.17174,"car_large":0.21007,
        "car_hybrid":0.09167,"car_ev":0.0,
        "motorcycle":0.11367,"motorcycle_small":0.08319,"motorcycle_large":0.13252,
        "walk":0.0,"bicycle":0.0,
        "ferry":0.11270,"ferry_foot":0.01871,"ferry_car":0.12933,
    }
    BUSINESS_TRAVEL_EF = TRAVEL_EF_MAP
    COMMUTE_EF = TRAVEL_EF_MAP

TRAVEL_EF = TRAVEL_EF_MAP  # 하위 호환
BUS_SUBWAY_AVG_EF = (TRAVEL_EF_MAP.get("bus", 0.10385) + TRAVEL_EF_MAP.get("subway", 0.0278)) / 2


def normalize_commute_mode(mode):
    rule_mode = detect_travel_mode_rule(mode)
    if rule_mode == "taxi":
        return "car"
    # car_hybrid / car_ev는 그대로 유지 (별도 EF 적용)
    return rule_mode


def calc_business_travel(row):
    origin = row.get("출발지")
    dest = row.get("도착지")
    distance = to_float(row.get("이동거리(km)"))
    people = to_float(row.get("출장인원"))
    mode_raw = row.get("출장수단")
    cost = to_float(row.get("출장비용"))

    if people is None:
        people = 1

    mode = normalize_travel_mode(mode_raw)
    mode_text = _safe_text_travel(mode_raw).lower()

    if distance is None and pd.notnull(origin) and pd.notnull(dest):
        if mode == "flight":
            distance = get_distance_km(origin, dest, mode="air")
        elif mode in ["train", "subway"]:
            distance = get_distance_km(origin, dest, mode="rail")
        else:
            distance = get_distance_km(origin, dest, mode="road")

    if ("버스" in mode_text or "bus" in mode_text) and ("지하철" in mode_text or "subway" in mode_text or "전철" in mode_text):
        ef = BUS_SUBWAY_AVG_EF
    else:
        ef = BUSINESS_TRAVEL_EF.get(mode)

    if ef is not None and distance is not None:
        emission = distance * people * ef
        return distance, ef, emission

    if cost is not None:
        try:
            _, spend_ef = search_spend_ef("business travel")
            spend_ef = to_float(spend_ef)
            if spend_ef is not None:
                emission = cost * spend_ef
                return distance, spend_ef, emission
        except Exception:
            pass

    return distance, None, None


def calc_commute(row):
    employees = to_float(row.get("직원수"))
    distance = to_float(row.get("평균통근거리(km)"))
    if distance is None:
        distance = to_float(row.get("이동거리(km)"))

    mode_raw = row.get("교통수단") or row.get("이동수단")
    mode = normalize_commute_mode(mode_raw)
    mode_text = _safe_text_travel(mode_raw).lower()
    origin = row.get("출발지")
    dest = row.get("도착지")

    if employees is None:
        employees = 1

    if distance is None and pd.notnull(origin) and pd.notnull(dest):
        if mode in ["car", "bus", "walk", "bicycle"]:
            distance = get_distance_km(origin, dest, mode="road")
        elif mode in ["train", "subway"]:
            distance = get_distance_km(origin, dest, mode="rail")
        else:
            distance = get_distance_km(origin, dest, mode="road")

    if ("버스" in mode_text or "bus" in mode_text) and ("지하철" in mode_text or "subway" in mode_text or "전철" in mode_text):
        ef = BUS_SUBWAY_AVG_EF
    else:
        ef = COMMUTE_EF.get(mode)

    if ef is not None and employees is not None and distance is not None:
        emission = employees * distance * ef
        return distance, ef, emission

    return distance, None, None

# ── [원본 셀 43] ─────────────────────────────────────────────
# ==========================================
# User patch: 2025 월평균 환율 + 분리 템플릿 대응
# - 구매: 월별 행형/월별 가로형 둘 다 지원
# - 운송: 발생월/자재명 컬럼 반영
# - 출장/통근: 분리 시트 지원
# - USEPA spend EF(USD 기준) 계산을 위해 입력금액을 USD 기준으로 정규화
#   * KRW 입력 -> 2025년 월평균 원/달러 환율로 USD 환산
#   * USD 입력 -> 그대로 USD 사용
#   * 감사용으로 KRW 환산금액도 별도 저장
# ==========================================
import os
import re
import pandas as pd

KNOWN_TEMPLATE_HEADERS = set(KNOWN_TEMPLATE_HEADERS) | {
    "구매 월", "구매월", "발생월", "분류", "자재명",
    "비용", "이동거리", "인원수",
    "03_출장_Template", "04_통근_Template"
}

FX_REFERENCE_YEAR = 2025

# 2025년 공식 월평균 원/달러 환율(원/USD)을 여기에 등록해서 사용합니다.
# 예시:
# BOK_USD_KRW_MONTHLY_AVG_2025 = {1: 1350.1, 2: 1342.8, ..., 12: 1470.2}
BOK_USD_KRW_MONTHLY_AVG_2025 =  {
    1: 1455.79,
    2: 1445.56,
    3: 1456.95,
    4: 1444.31,
    5: 1394.49,
    6: 1366.95,
    7: 1375.22,
    8: 1389.66,
    9: 1391.83,
    10: 1423.36,
    11: 1457.77,
    12: 1467.40,
}

_BOK_USD_KRW_MONTHLY_AVG_CACHE = {}
if BOK_USD_KRW_MONTHLY_AVG_2025:
    _BOK_USD_KRW_MONTHLY_AVG_CACHE[FX_REFERENCE_YEAR] = {
        int(k): float(v) for k, v in BOK_USD_KRW_MONTHLY_AVG_2025.items()
    }

MONTH_NAME_TO_NUM = {f"{i}월": i for i in range(1, 13)}
PURCHASE_WIDE_MONTH_COLS = list(MONTH_NAME_TO_NUM.keys())

def register_bok_usd_krw_monthly_avg(year, month_avg_dict):
    year = int(year)
    cleaned = {}
    for k, v in (month_avg_dict or {}).items():
        try:
            if isinstance(k, str):
                k = k.replace("월", "").strip()
            mm = int(k)
            vv = float(v)
            if 1 <= mm <= 12:
                cleaned[mm] = vv
        except Exception:
            continue
    _BOK_USD_KRW_MONTHLY_AVG_CACHE[year] = cleaned
    return cleaned

def load_bok_usd_krw_monthly_avg_from_table(path_or_df, year=FX_REFERENCE_YEAR):
    if isinstance(path_or_df, pd.DataFrame):
        fx_df = path_or_df.copy()
    else:
        ext = str(path_or_df).lower()
        if ext.endswith(".csv"):
            fx_df = pd.read_csv(path_or_df)
        else:
            fx_df = pd.read_excel(path_or_df)

    month_col = None
    rate_col = None
    for c in fx_df.columns:
        cl = str(c).strip().lower()
        if month_col is None and cl in {"월", "month", "purchase_month", "구매 월"}:
            month_col = c
        if rate_col is None and ("환율" in cl or "rate" in cl):
            rate_col = c

    if month_col is None or rate_col is None:
        raise ValueError("월평균 환율 테이블에는 '월'과 '환율' 컬럼이 필요합니다.")

    month_avg = {}
    for _, r in fx_df.iterrows():
        try:
            mm = _coerce_month_value(r[month_col])
            rr = float(r[rate_col])
            if mm is not None and pd.notnull(rr):
                month_avg[int(mm)] = rr
        except Exception:
            continue

    return register_bok_usd_krw_monthly_avg(year, month_avg)



def _normalize_currency_unit(value):
    s = _safe_text(value).strip().upper()
    mapping = {
        "원": "KRW", "KRW": "KRW", "KRW": "KRW", "WON": "KRW", "₩": "KRW",
        "USD": "USD", "US$": "USD", "$": "USD", "달러": "USD", "미국달러": "USD",
        "U$": "USD", "USDOLLAR": "USD", "US DOLLAR": "USD",
    }
    return mapping.get(s, s)

def _coerce_month_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        mm = int(value)
        return mm if 1 <= mm <= 12 else None

    s = str(value).strip()
    if not s:
        return None

    if s in MONTH_NAME_TO_NUM:
        return MONTH_NAME_TO_NUM[s]

    m = re.search(r"([1-9]|1[0-2])\s*월", s)
    if m:
        return int(m.group(1))

    m = re.search(r"^\s*([1-9]|1[0-2])\s*$", s)
    if m:
        return int(m.group(1))

    if "구매일자" in s or "date" in s.lower():
        return None

    # YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD
    m = re.search(r"20\d{2}[./-](\d{1,2})[./-]\d{1,2}", s)
    if m:
        mm = int(m.group(1))
        return mm if 1 <= mm <= 12 else None

    return None

def _normalize_purchase_classification(value):
    s = _safe_text(value).strip().lower()
    if not s:
        return None

    cat1_keys = ["category1", "cat1", "카테고리1", "구매", "상품", "원재료", "소모품", "일반구매", "구입"]
    cat2_keys = ["category2", "cat2", "카테고리2", "자산", "자본재", "설비", "비품", "장비"]

    if any(k in s for k in cat2_keys):
        return "Category2"
    if any(k in s for k in cat1_keys):
        return "Category1"
    return None

def _reshape_purchase_wide_to_long(df):
    df = df.copy()
    month_cols = [c for c in df.columns if str(c).strip() in PURCHASE_WIDE_MONTH_COLS]
    if not month_cols:
        return df

    id_cols = [c for c in df.columns if c not in month_cols + ["합계"]]
    melted = df.melt(
        id_vars=id_cols,
        value_vars=month_cols,
        var_name="구매 월",
        value_name="구매금액",
    )
    melted["구매 월"] = melted["구매 월"].map(lambda x: MONTH_NAME_TO_NUM.get(str(x).strip()))
    melted["구매금액"] = melted["구매금액"].apply(to_float)
    melted = melted[melted["구매 월"].notnull()]
    melted = melted[melted["구매금액"].notnull()]
    melted = melted[melted["구매금액"] != 0]
    melted = melted.reset_index(drop=True)
    return melted


def _get_purchase_month(row):
    for key in ["구매 월", "구매월", "발생월"]:
        mm = _coerce_month_value(row.get(key))
        if mm is not None:
            return mm

    date_val = row.get("구매일자")
    if pd.notnull(date_val):
        s = str(date_val)
        m = re.search(r"20\d{2}[./-](\d{1,2})[./-]\d{1,2}", s)
        if m:
            mm = int(m.group(1))
            if 1 <= mm <= 12:
                return mm
    return None

def get_bok_usd_krw_monthly_avg(month, year=FX_REFERENCE_YEAR):
    """
    월평균 원/달러 환율 반환.
    - 월 정보 있으면 월평균 사용
    - 월 없으면 연평균 자동 fallback (에러 없음)
    - 연평균도 없으면 기본값 1400 사용
    """
    month = _coerce_month_value(month)
    fx_map = _BOK_USD_KRW_MONTHLY_AVG_CACHE.get(int(year), {})

    if month is not None:
        rate = fx_map.get(int(month))
        if rate is not None:
            return float(rate)

    # 연평균 fallback
    if fx_map:
        return round(sum(fx_map.values()) / len(fx_map), 2)

    # 최후 fallback
    return 1400.0


def get_bok_usd_krw_annual_avg(year=FX_REFERENCE_YEAR) -> float:
    """연평균 환율 반환 (연간 입력 데이터용)"""
    return get_bok_usd_krw_monthly_avg(month=None, year=year)



# ══════════════════════════════════════════════════════════════
# 단위 환산 — C1/C2 구매금액 기준단위(KRW/USD) 환산
# ══════════════════════════════════════════════════════════════
# 금액 단위 정규화 (→ KRW 기준)
_CURRENCY_ALIAS = {
    # 원화 계열
    "원": "KRW", "krw": "KRW", "₩": "KRW", "한화": "KRW",
    "won": "KRW", "korean won": "KRW",
    # 달러 계열
    "usd": "USD", "달러": "USD", "미달러": "USD", "$": "USD",
    "us dollar": "USD", "us$": "USD",
    # 기타 외화 — USD로 간주 (환율 보정 필요)
    "eur": "EUR", "유로": "EUR", "€": "EUR",
    "jpy": "JPY", "엔": "JPY", "¥": "JPY",
    "cny": "CNY", "위안": "CNY", "인민폐": "CNY",
    "vnd": "VND", "동": "VND",
    "sgd": "SGD", "싱가포르달러": "SGD",
    "gbp": "GBP", "파운드": "GBP", "£": "GBP",
    "": "KRW",  # 미입력 → KRW 기본
}

# 중량 단위 정규화 (→ kg 기준)
_WEIGHT_TO_KG = {
    "kg": 1.0, "킬로그램": 1.0, "kilogram": 1.0,
    "g": 0.001, "gram": 0.001, "그램": 0.001,
    "ton": 1000.0, "t": 1000.0, "톤": 1000.0, "mt": 1000.0, "metric ton": 1000.0,
    "lb": 0.453592, "lbs": 0.453592, "pound": 0.453592,
    "oz": 0.028350, "ounce": 0.028350,
}

# 부피 단위 정규화 (→ L 기준)
_VOLUME_TO_L = {
    "l": 1.0, "liter": 1.0, "litre": 1.0, "리터": 1.0, "ℓ": 1.0,
    "ml": 0.001, "milliliter": 0.001, "밀리리터": 0.001,
    "kl": 1000.0, "kiloliter": 1000.0,
    "m3": 1000.0, "m³": 1000.0, "cubic meter": 1000.0, "세제곱미터": 1000.0,
    "gal": 3.78541, "gallon": 3.78541, "갤런": 3.78541,
}

# 에너지/가스 단위 정규화 (→ Nm³ 또는 kWh 기준)
_ENERGY_TO_NM3 = {
    "nm3": 1.0, "nm³": 1.0, "normal cubic meter": 1.0, "표준세제곱미터": 1.0,
    "m3": 1.0, "m³": 1.0,  # 가스의 경우 통상 Nm³과 동치
}
_ENERGY_TO_KWH = {
    "kwh": 1.0, "킬로와트시": 1.0, "kilowatt hour": 1.0,
    "mwh": 1000.0, "메가와트시": 1000.0,
    "gwh": 1e6, "기가와트시": 1000000.0,
    "gj": 277.778, "기가줄": 277.778,
    "mj": 0.277778, "메가줄": 0.277778,
    "kcal": 0.001163, "kj": 0.000277778,
}



# ══════════════════════════════════════════════════════════════
# 에너지원별 단위 변환 (C3/C8/C13/C14 공통)
# 사용자가 어떤 단위로 입력해도 배출계수 기준단위로 자동 환산
# 출처: 에너지법 시행규칙 8차 순발열량 기준
# ══════════════════════════════════════════════════════════════

# 에너지원별 (기준단위, 순발열량 MJ/기준단위)
ENERGY_BASE_UNIT = {
    "도시가스(LNG)":     ("Nm³", 38.5),
    "천연가스(LNG)":     ("kg",  49.4),
    "CNG(차량)":         ("kg",  49.4),
    "LNG(차량)":         ("kg",  49.4),
    "도시가스(LPG)":     ("Nm³", 58.3),
    "액화석유가스(LPG)": ("kg",  46.2),
    "LPG(차량)":         ("kg",  46.2),
    "경유":              ("L",   35.3),
    "보일러 등유":       ("L",   34.1),
    "실내 등유":         ("L",   34.1),
    "등유":              ("L",   34.1),
    "항공유":            ("L",   34.0),
    "휘발유":            ("L",   30.1),
    "전력":              ("kWh", None),   # 전력: 열량 환산 불필요
    "전기":              ("kWh", None),
    "열(스팀)":          ("GJ",  None),
    "스팀":              ("GJ",  None),
}

# 키워드로 에너지원 감지
ENERGY_NAME_ALIAS = {
    "경유": ["경유","디젤","diesel","gas oil"],
    "휘발유": ["휘발유","가솔린","gasoline","petrol"],
    "보일러 등유": ["보일러등유","보일러 등유","b등유","kerosene"],
    "실내 등유": ["실내등유","실내 등유","등유"],
    "항공유": ["항공유","제트유","jet fuel","aviation fuel"],
    "도시가스(LNG)": ["도시가스","도시가스(lng)","city gas","lng(도시)"],
    "천연가스(LNG)": ["천연가스","lng","natural gas"],
    "도시가스(LPG)": ["도시가스(lpg)","lpg도시"],
    "액화석유가스(LPG)": ["lpg","액화석유가스","프로판","부탄","propane","butane"],
    "전력": ["전력","전기","electricity","electric","kwh","power"],
    "열(스팀)": ["열","스팀","steam","district heat","지역난방"],
}

# MJ 기반 공통 환산 상수
_MJ_PER_KWH = 3.6          # 1 kWh = 3.6 MJ
_MJ_PER_MWH = 3600.0       # 1 MWh = 3,600 MJ
_MJ_PER_GJ  = 1000.0       # 1 GJ = 1,000 MJ
_MJ_PER_TJ  = 1_000_000.0  # 1 TJ = 1,000,000 MJ
_MJ_PER_TOE = 41_868.0     # 1 toe = 41,868 MJ (국내 기준)
_MJ_PER_GCAL= 4_186.8      # 1 Gcal = 4,186.8 MJ

def _detect_energy_name(raw: str) -> str:
    """
    입력된 에너지원명 → ENERGY_BASE_UNIT 키로 변환.

    PATCH: 기존에는 "별칭이 입력에 포함(a in q)" 또는 "입력이 별칭에 포함(q in a)"
    둘 다를 곧바로 매칭으로 인정해서, 예를 들어 짧은 입력 "LPG"(q)가
    "도시가스(LPG)"의 별칭 "도시가스(lpg)"(a) 안에 부분 문자열로 포함된다는 이유만으로
    (q in a) "액화석유가스(LPG)"(kg 기준, 올바른 항목)보다 먼저 "도시가스(LPG)"
    (Nm³ 기준, 실제로는 다른 종류의 가스)로 잘못 매칭되는 문제가 있었다.
    이로 인해 불필요한 kg→Nm³ 환산이 시도되고, 그 환산은 밀도 정보 없이는
    신뢰성 있게 산정하기 어려워 결과적으로 "환산 재시도 필요" 메시지가 남았다.

    이를 막기 위해 아래 순서로 매칭한다 (부분일치보다 정확일치를 항상 우선):
      1) 별칭과 정확히 일치 (대소문자/공백/괄호 무시)
      2) 표준명(std_name) 자체와 정확히 일치
      3) 그래도 없으면 "별칭이 입력 문자열 안에 포함된 경우(a in q)"만 부분일치로 인정
         (반대 방향 q in a는 인정하지 않음 — 짧은 입력이 다른 항목의 긴 별칭 속
          부분 문자열이 되어 오매칭되는 것을 방지하기 위함)
    """
    q = str(raw).strip().lower().replace(" ", "").replace("(", "").replace(")", "")

    # 1) 별칭 정확 일치
    for std_name, aliases in ENERGY_NAME_ALIAS.items():
        for a in aliases:
            if a.replace(" ", "") == q:
                return std_name

    # 2) 표준명 정확 일치
    for std_name in ENERGY_BASE_UNIT:
        if std_name.lower().replace(" ", "") == q:
            return std_name

    # 3) 부분 일치 (별칭 ⊂ 입력, 한 방향만 허용)
    for std_name, aliases in ENERGY_NAME_ALIAS.items():
        for a in aliases:
            a_norm = a.replace(" ", "")
            if a_norm and a_norm in q:
                return std_name

    return raw  # 알 수 없으면 원본 반환


@lru_cache(maxsize=512)
def _ai_unit_conversion_factor(unit_raw: str, base_unit: str, context: str = ""):
    """
    rule 테이블에 없는 단위 → base_unit 환산 배수를 AI가 추정.
    규칙에 없는 단위라도 "무조건" 기준단위로 환산해야 하므로, 절대 조용히 포기하지 않는다:
      1) JSON 형식으로 배수를 요청 (최대 3회 재시도는 _call_claude_json 내부에서 처리)
      2) 그래도 유효한 factor를 못 받으면, 숫자만 답하라는 훨씬 단순한 프롬프트로 다시 시도
    두 단계 모두 실패하는 경우(예: API 완전 중단)에만 None을 반환한다.
    """
    prompt = f"""단위 환산 배수를 추정하라.
입력 단위: {unit_raw}
목표(기준) 단위: {base_unit}
{f"맥락: {context}" if context else ""}

"1 {unit_raw}"가 "{base_unit}" 몇 개에 해당하는지 배수(factor)를 계산하라.
화학/물리적으로 정확한 값을 모르면, 온실가스 배출량 산정에 통상적으로 쓰이는
표준 환산값(예: 표준상태 기준 밀도, 발열량 등)을 근거로 최선의 추정값을 제시하라.
JSON으로만 답하라: {{"factor": <숫자>}}"""
    parsed = _call_claude_json(prompt, max_tokens=80)
    factor = to_float(parsed.get("factor")) if parsed else None
    if factor is not None and factor > 0:
        return factor

    # 1차 시도 실패 → 훨씬 단순한 프롬프트 + 자유 텍스트에서 숫자만 추출 (최후 재시도)
    simple_prompt = (
        f'"1 {unit_raw}"는 "{base_unit}" 단위로 몇인가? '
        f'{f"(맥락: {context}) " if context else ""}'
        f"설명 없이 숫자(배수)만 답하라."
    )
    text = _call_claude_text(simple_prompt, max_tokens=30, retries=2)
    if text:
        m = re.search(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", text.replace(",", ""))
        if m:
            f2 = to_float(m.group())
            if f2 is not None and f2 > 0:
                return f2
    return None

def convert_energy_unit(value: float, input_unit: str, energy_name: str) -> tuple:
    """
    에너지 사용량의 입력 단위 → 해당 에너지원의 배출계수 기준단위로 환산.

    반환: (환산값, 기준단위, 환산비고)

    예시:
      경유 100 kWh → L   : 100 × (3.6/35.3) = 10.2 L
      도시가스 1 toe → Nm³: 1 × (41868/38.5) = 1087.5 Nm³
      전력 1 MWh → kWh   : 1,000 kWh
      열 1 Gcal → GJ     : 4.1868 GJ
    """
    if value is None:
        return None, input_unit, ""

    u_in  = str(input_unit).strip().lower().replace(" ","")
    e_std = _detect_energy_name(energy_name)
    info  = ENERGY_BASE_UNIT.get(e_std)

    if info is None:
        return value, input_unit, ""

    base_unit, lhv_mj = info

    # 기준단위 정규화
    base_norm = base_unit.lower().replace(" ","").replace("³","3").replace("²","2")
    u_in_norm = u_in.replace("³","3").replace("²","2")

    # 이미 기준단위와 동일하면 그대로
    if u_in_norm == base_norm:
        return value, base_unit, ""

    # ── 에너지(MJ) 공통 경로 ──────────────────────────────────
    # 1단계: 입력단위 → MJ 환산
    mj = None

    # 전력·열은 직접 환산
    if base_norm == "kwh":
        kwh_factors = {
            "kwh": 1.0, "wh": 0.001, "mwh": 1000.0, "gwh": 1e6,
            "kj": 1/3.6, "mj": 1/_MJ_PER_KWH, "gj": _MJ_PER_GJ/_MJ_PER_KWH,
            "tj": _MJ_PER_TJ/_MJ_PER_KWH,
            "kcal": 1/860, "mcal": 1000/860, "gcal": _MJ_PER_GCAL/_MJ_PER_KWH,
            "toe": _MJ_PER_TOE/_MJ_PER_KWH,
        }
        factor = kwh_factors.get(u_in_norm)
        if factor:
            converted = value * factor
            note = f"{input_unit}→kWh (×{factor:.6g})"
            return converted, "kWh", note
        ai_factor = _ai_unit_conversion_factor(input_unit, "kWh", context=energy_name)
        if ai_factor:
            return value * ai_factor, "kWh", f"{input_unit}→kWh (AI 판단: ×{ai_factor:.6g})"
        # 규칙에도 없고 AI 판단도 실패한 극히 예외적인 경우: 값은 유지하되
        # "미지원"으로 단정하지 않고, 재확인이 필요하다는 중립적 안내만 남긴다.
        return value, input_unit, f"{input_unit}→kWh (AI 환산 재시도 필요 - 값 미변환, 수동 확인 권장)"

    if base_norm in ("gj", "gj"):
        gj_factors = {
            "gj": 1.0, "mj": 0.001, "tj": 1000.0, "kj": 1e-6,
            "kwh": _MJ_PER_KWH/1000, "mwh": _MJ_PER_MWH/1000,
            "kcal": 4.1868e-6*1000, "mcal": 4.1868/1000, "gcal": 4.1868,
            "toe": _MJ_PER_TOE/1000,
        }
        factor = gj_factors.get(u_in_norm)
        if factor:
            converted = value * factor
            note = f"{input_unit}→GJ (×{factor:.6g})"
            return converted, "GJ", note
        ai_factor = _ai_unit_conversion_factor(input_unit, "GJ", context=energy_name)
        if ai_factor:
            return value * ai_factor, "GJ", f"{input_unit}→GJ (AI 판단: ×{ai_factor:.6g})"
        return value, input_unit, f"{input_unit}→GJ (AI 환산 재시도 필요 - 값 미변환, 수동 확인 권장)"

    # 연료 (기준: L 또는 Nm³ 또는 kg) — MJ 경유 환산
    if lhv_mj is None or lhv_mj <= 0:
        return value, input_unit, ""

    # 입력단위 → MJ
    mj_per_input = {
        # 직접 에너지 단위
        "mj": 1.0,
        "kj": 0.001,
        "gj": _MJ_PER_GJ,
        "tj": _MJ_PER_TJ,
        "kwh": _MJ_PER_KWH,
        "mwh": _MJ_PER_MWH,
        "kcal": 4.1868e-3,
        "mcal": 4.1868,
        "gcal": _MJ_PER_GCAL,
        "toe": _MJ_PER_TOE,
        "toe/y": _MJ_PER_TOE,
    }
    # 기준단위 자체 (L, Nm³, kg)도 같은 단위면 위에서 처리됨
    # 동일 계열 질량 단위 (kg ↔ g ↔ ton)
    weight_mj = None
    if base_norm == "kg":
        kg_factors = {"kg":1.0,"g":0.001,"ton":1000.0,"t":1000.0,"lb":0.453592,"mt":1000.0}
        f = kg_factors.get(u_in_norm)
        if f:
            converted = value * f
            note = f"{input_unit}→kg (×{f})"
            return converted, "kg", note

    # 부피 단위 (L ↔ mL ↔ m³)
    if base_norm == "l":
        vol_factors = {
            "l":1.0, "ml":0.001, "kl":1000.0,
            "m3":1000.0, "m³":1000.0,
            "gal":3.78541, "gallon":3.78541,
        }
        f = vol_factors.get(u_in_norm)
        if f:
            converted = value * f
            note = f"{input_unit}→L (×{f})"
            return converted, "L", note

    # 가스 부피 (Nm³ ↔ m³ ↔ L)
    if base_norm in ("nm3", "nm³"):
        gas_factors = {"nm3":1.0,"nm³":1.0,"m3":1.0,"m³":1.0,"l":0.001,"ml":1e-6}
        f = gas_factors.get(u_in_norm)
        if f:
            converted = value * f
            note = f"{input_unit}→Nm³ (×{f})"
            return converted, "Nm³", note

    # 에너지 단위 → 기준단위 (MJ 경유)
    mj_factor = mj_per_input.get(u_in_norm)
    if mj_factor:
        mj_total = value * mj_factor
        # MJ → 기준단위 (기준단위 당 MJ로 나눔)
        converted = mj_total / lhv_mj
        factor_display = mj_factor / lhv_mj
        note = f"{input_unit}→{base_unit} (발열량 환산: ×{factor_display:.6g})"
        return converted, base_unit, note

    # 규칙 테이블(중량/부피/에너지 환산표)이나 발열량(LHV) 경유 환산으로 처리되지 않는
    # 단위 조합(예: kg → Nm³ 등 물성 정보가 필요한 경우)은 AI가 판단하여 환산한다.
    # 규칙에 없다고 해서 "미지원"으로 처리하지 않고, 반드시 AI 판단을 거치도록 한다.
    ai_factor = _ai_unit_conversion_factor(input_unit, base_unit, context=energy_name)
    if ai_factor:
        return value * ai_factor, base_unit, f"{input_unit}→{base_unit} (AI 판단: ×{ai_factor:.6g})"
    # 재시도까지 모두 실패한 극히 예외적인 경우에만 값을 그대로 두되,
    # "미지원 단위"라는 확정적 표현 대신 재확인이 필요하다는 안내만 남긴다.
    return value, input_unit, f"{input_unit}→{base_unit} (AI 환산 재시도 필요 - 값 미변환, 수동 확인 권장)"

def normalize_unit(unit_raw: str) -> str:
    """단위 표기를 표준화 (소문자 + 공백 제거)"""
    return str(unit_raw).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def convert_to_base_unit(value: float, unit_raw: str, base_type: str = "auto") -> tuple:
    """
    value와 unit을 받아 기준 단위로 환산.
    base_type: "currency"(금액), "weight"(중량), "volume"(부피),
               "energy_nm3"(가스), "energy_kwh"(전력), "auto"(자동감지)
    반환: (환산값, 기준단위, 환산비율, 원본단위, 판단근거)
      판단근거: "rule"(rule 테이블 매칭) / "ai"(rule에 없어 AI가 판단) / "none"(둘 다 실패, 1:1 통과)
    """
    if value is None:
        return None, unit_raw, 1.0, unit_raw, "none"

    u = normalize_unit(unit_raw or "")

    # auto 모드: 단위 타입 자동 감지
    if base_type == "auto":
        if u in _WEIGHT_TO_KG:
            base_type = "weight"
        elif u in _VOLUME_TO_L:
            base_type = "volume"
        elif u in _ENERGY_TO_NM3:
            base_type = "energy_nm3"
        elif u in _ENERGY_TO_KWH:
            base_type = "energy_kwh"
        else:
            # 변환 불필요
            return value, unit_raw, 1.0, unit_raw, "none"

    base_unit_map = {"weight": "kg", "volume": "L", "energy_nm3": "Nm³", "energy_kwh": "kWh"}
    rule_map = {
        "weight": _WEIGHT_TO_KG, "volume": _VOLUME_TO_L,
        "energy_nm3": _ENERGY_TO_NM3, "energy_kwh": _ENERGY_TO_KWH,
    }

    if base_type in rule_map:
        base_unit = base_unit_map[base_type]
        ratio = rule_map[base_type].get(u)
        if ratio is not None:
            return value * ratio, base_unit, ratio, unit_raw, "rule"
        ai_ratio = _ai_unit_conversion_factor(unit_raw, base_unit)
        if ai_ratio is not None:
            return value * ai_ratio, base_unit, ai_ratio, unit_raw, "ai"
        return value, unit_raw, 1.0, unit_raw, "none"

    return value, unit_raw, 1.0, unit_raw, "none"


def normalize_purchase_unit(row) -> dict:
    """
    C1/C2 구매금액 단위 정규화.
    - KRW → 그대로 사용
    - USD → BOK 환율 KRW 환산
    - 기타 외화 → USD 먼저 환산 후 KRW 환산 (향후 확장)
    반환: {"구매금액_KRW": float, "적용단위": str, "단위환산비고": str}
    """
    amount = to_float(row.get("구매금액"))
    unit_raw = str(row.get("단위") or row.get("통화") or "").strip()
    u = normalize_unit(unit_raw)
    std_currency = _CURRENCY_ALIAS.get(u, u.upper() if u else "KRW")

    result = {"원본단위": unit_raw or "KRW", "적용단위": std_currency}

    if amount is None:
        result["구매금액_KRW"] = None
        result["단위환산비고"] = "금액 없음"
        return result

    if std_currency == "KRW":
        result["구매금액_KRW"] = amount
        result["단위환산비고"] = "KRW 그대로"
    elif std_currency == "USD":
        fx = get_bok_usd_krw_annual_avg()
        result["구매금액_KRW"] = amount * fx
        result["단위환산비고"] = f"USD → KRW ({fx:,.0f} 환율 적용)"
    elif std_currency == "EUR":
        # 유로는 USD 경유 환산 (EUR/USD ≈ 1.09 고정, 추후 실시간 반영 가능)
        eur_usd = 1.09
        fx = get_bok_usd_krw_annual_avg()
        result["구매금액_KRW"] = amount * eur_usd * fx
        result["단위환산비고"] = f"EUR → USD ({eur_usd}) → KRW ({fx:,.0f})"
    elif std_currency == "JPY":
        # 100엔 ≈ 890원 (고정 근사, 추후 BOK JPY 환율 연동 가능)
        jpy_krw = 8.90
        result["구매금액_KRW"] = amount * jpy_krw
        result["단위환산비고"] = f"JPY → KRW ({jpy_krw}/100 적용)"
    elif std_currency == "CNY":
        # 1위안 ≈ 200원 근사
        cny_krw = 200.0
        result["구매금액_KRW"] = amount * cny_krw
        result["단위환산비고"] = f"CNY → KRW ({cny_krw} 적용)"
    elif std_currency == "VND":
        # 1동 ≈ 0.057원
        vnd_krw = 0.057
        result["구매금액_KRW"] = amount * vnd_krw
        result["단위환산비고"] = f"VND → KRW ({vnd_krw} 적용)"
    else:
        # 알 수 없는 통화 → KRW로 간주
        result["구매금액_KRW"] = amount
        result["적용단위"] = "KRW(추정)"
        result["단위환산비고"] = f"미지원 통화({unit_raw}) → KRW 간주"

    return result

def normalize_purchase_spend_for_usepa(row, year=FX_REFERENCE_YEAR):
    """
    구매금액 단위 환산 → USD 계산.
    normalize_purchase_unit()으로 다양한 통화(KRW/USD/EUR/JPY/CNY/VND 등) 지원.
    월 정보 없으면 연평균 환율 자동 적용.
    """
    purchase_month = _get_purchase_month(row)
    fx_rate = get_bok_usd_krw_monthly_avg(purchase_month, year=year)
    fx_label = f"{purchase_month}월" if purchase_month else f"{year}년 연평균"

    # 단위 환산 (다통화 지원)
    unit_info = normalize_purchase_unit(row)
    amount_krw = unit_info.get("구매금액_KRW")
    input_currency = unit_info.get("적용단위", "KRW")
    unit_note = unit_info.get("단위환산비고", "")

    if amount_krw is None:
        return {
            "입력통화":        input_currency or None,
            "단위환산비고":    unit_note,
            "환율기준연도":    year,
            "환율적용월":      fx_label,
            "적용환율(원/USD)": fx_rate,
            "원화환산금액":    None,
            "EF적용금액(USD)": None,
            "환율적용여부":    "N",
        }

    # KRW → USD
    spend_usd = amount_krw / fx_rate if fx_rate else None

    return {
        "입력통화":        input_currency,
        "단위환산비고":    unit_note,
        "환율기준연도":    year,
        "환율적용월":      fx_label,
        "적용환율(원/USD)": fx_rate,
        "원화환산금액":    amount_krw,
        "EF적용금액(USD)": spend_usd,
        "환율적용여부":    "Y",
    }

def _format_spend_formula_usd(spend_usd, ef, emission):
    spend_usd = to_float(spend_usd)
    ef = to_float(ef)
    emission = to_float(emission)
    if spend_usd is None or ef is None or emission is None:
        return ""
    return f"배출량 = EF적용금액(USD) × 배출계수 = {spend_usd:g} × {ef:g} = {emission:g}"

PURCHASE_GROUP_KO_BY_FAMILY = {
    "office_supplies_paper": "사무용품",
    "office_supplies_toner": "토너/잉크",
    "office_supplies_general": "사무용품",
    "office_furniture": "책상/의자/파티션/옷장",
    "software_it": "소프트웨어",
    "computer_equipment": "전산장비",
    "maintenance_repair": "전산설비수선비",
    "water_beverage": "주방용품",
    "coffee_tea": "주방용품",
    "packaging_material": "포장재",
}

def resolve_purchase_group_ko(product, desc, vendor, family, ef_db_name):
    family = _safe_text(family)
    text = f"{_safe_text(product).lower()} {_safe_text(desc).lower()} {_safe_text(vendor).lower()}"

    if family in PURCHASE_GROUP_KO_BY_FAMILY:
        return PURCHASE_GROUP_KO_BY_FAMILY[family]

    if any(k in text for k in ["a4용지", "복사지", "복사용지", "용지", "라벨지", "코팅지"]):
        return "사무용품"
    if any(k in text for k in ["토너", "잉크", "카트리지", "드럼"]):
        return "토너/잉크"
    if any(k in text for k in ["의자", "책상", "보조책상", "파티션", "옷장", "캐비닛"]):
        return "책상/의자/파티션/옷장"
    if any(k in text for k in ["소프트웨어", "라이선스"]):
        return "소프트웨어"
    if any(k in text for k in ["노트북", "데스크탑", "모니터", "pc", "프린터", "복합기", "서버"]):
        return "전산장비"
    if any(k in text for k in ["수리", "교체", "파쇄", "유지보수", "repair", "maintenance"]):
        return "전산설비수선비"
    if any(k in text for k in ["커피", "차", "생수", "물", "다과", "종이컵", "음료"]):
        return "주방용품"

    return family if family else (_safe_text(ef_db_name) or "기타구매")

def ensure_spend_candidates(primary_name=None, primary_ef=None, product="", desc="", vendor="", family=None, top_n=4):
    def _norm_key(title):
        return _safe_text(title).strip().lower()

    recs = []
    seen = set()

    def _push(title, ef, source):
        key = _norm_key(title)
        if not key or key in seen:
            return
        seen.add(key)
        recs.append({
            "title": _safe_text(title),
            "ef": to_float(ef),
            "source": source,
        })

    # 1) 대표 후보 먼저 고정
    _push(primary_name, primary_ef, "primary")

    # 2) family / rules 기반 추천
    try:
        family_recs, detected_family = get_spend_recommendations(
            product=product, desc=desc, vendor=vendor, top_k=20
        )
    except Exception:
        family_recs, detected_family = [], family

    for r in family_recs:
        _push(r.get("title"), r.get("ef"), "family")
        if len(recs) >= top_n:
            break

    # 3) 그래도 부족하면 spend_db 전체 fallback
    if len(recs) < top_n:
        fallback_df = spend_db.copy()

        name_col = None
        ef_col = None
        for c in ["name", "title", "매개변수"]:
            if c in fallback_df.columns:
                name_col = c
                break
        for c in ["ef", "EF", "배출계수"]:
            if c in fallback_df.columns:
                ef_col = c
                break

        if name_col is not None and ef_col is not None:
            fallback_df = (
                fallback_df.dropna(subset=[name_col])
                .drop_duplicates(subset=[name_col])
            )

            for _, row in fallback_df.iterrows():
                _push(row.get(name_col), row.get(ef_col), "global_fallback")
                if len(recs) >= top_n:
                    break

    # 4) 끝까지 4개 보장
    while len(recs) < top_n:
        filler_title = f"Fallback EF {len(recs)+1}"
        _push(filler_title, None, "hard_fallback")

    return recs[:top_n]


def _norm_ef_title(x):
    return _safe_text(x).strip().lower()

def build_purchase_group_summary(detail_rows):
    def _norm_ef_title(x):
        return _safe_text(x).strip().lower()

    df = pd.DataFrame([
        r.to_dict() if isinstance(r, pd.Series) else dict(r)
        for r in detail_rows
    ])
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # 숫자형 정리
    for col in ["구매 월", "구매금액", "배출계수", "배출량"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["구매금액"] = df.get("구매금액", 0).fillna(0)
    df["배출량"] = df.get("배출량", 0).fillna(0)

    key_cols = ["품목/자산명", "자동카테고리"]

    # 1) 월별 구매금액 합산
    month_df = (
        df.pivot_table(
            index=key_cols,
            columns="구매 월",
            values="구매금액",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    renamed_cols = []
    for c in month_df.columns:
        if isinstance(c, (int, float)) and not pd.isna(c):
            renamed_cols.append(f"{int(c)}월")
        else:
            renamed_cols.append(c)
    month_df.columns = renamed_cols

    for m in range(1, 13):
        col = f"{m}월"
        if col not in month_df.columns:
            month_df[col] = 0

    # 2) 대표 EF 선정
    # 같은 품목/자산명 그룹 안에서 구매금액 합이 가장 큰 EF를 대표로 선택
    ef_vote = (
        df.groupby(
            key_cols + ["매개변수", "EF_DB", "추천Family", "배출계수", "EF_DB_추천사유"],
            dropna=False
        )["구매금액"]
        .sum()
        .reset_index()
        .sort_values(key_cols + ["구매금액"], ascending=[True, True, False])
    )

    primary = ef_vote.groupby(key_cols, dropna=False).head(1).copy()

    # 3) 그룹 총합
    agg_df = (
        df.groupby(key_cols, dropna=False)
        .agg(
            구매금액=("구매금액", "sum"),
            실가스배출=("배출량", "sum"),
            입력건수=("구매금액", "size"),
            입력통화=("입력통화", "first"),
            환율기준연도=("환율기준연도", "first"),
            그룹판단근거=("그룹판단근거", lambda s: " | ".join(pd.Series(s).dropna().astype(str).unique()[:5])),
        )
        .reset_index()
    )

    out = (
        month_df
        .merge(agg_df, on=key_cols, how="left")
        .merge(
            primary[key_cols + ["매개변수", "EF_DB", "추천Family", "배출계수", "EF_DB_추천사유"]],
            on=key_cols,
            how="left"
        )
    )

    # 대표 EF 맵
    primary_map = {}
    for _, row in out[key_cols + ["매개변수", "배출계수", "추천Family", "EF_DB"]].drop_duplicates().iterrows():
        primary_map[(row["품목/자산명"], row["자동카테고리"])] = {
            "매개변수": row["매개변수"],
            "배출계수": row["배출계수"],
            "추천Family": row["추천Family"],
            "EF_DB": row["EF_DB"],
        }

    # 4) 추천 EF 4개 강제 생성
    # 대표와 중복되지 않도록 제거하고, 추천끼리도 중복 제거
    recommend_n = 4
    rec_rows = []
    for group_vals, g in df.groupby(key_cols, dropna=False):
        g = g.reset_index(drop=True)

        if isinstance(group_vals, tuple):
            group_name, group_cat = group_vals
        else:
            group_name, group_cat = group_vals, None

        rep_info = primary_map.get((group_name, group_cat), {})
        rep_title = rep_info.get("매개변수")
        rep_ef = rep_info.get("배출계수")
        rep_family = rep_info.get("추천Family")

        sample_product = _safe_text(g.get("원본품목/자산명", pd.Series([""])).iloc[0]) if "원본품목/자산명" in g.columns else _safe_text(g["품목/자산명"].iloc[0])
        sample_desc = _safe_text(g.get("설명/적요", pd.Series([""])).iloc[0]) if "설명/적요" in g.columns else ""
        sample_vendor = _safe_text(g.get("거래처명", pd.Series([""])).iloc[0]) if "거래처명" in g.columns else ""
        sample_family = _safe_text(g.get("추천Family", pd.Series([rep_family])).iloc[0]) if "추천Family" in g.columns else _safe_text(rep_family)

        # 넉넉하게 후보를 받아온 뒤, 대표와 중복 제거
        full_candidates = ensure_spend_candidates(
            primary_name=rep_title,
            primary_ef=rep_ef,
            product=sample_product,
            desc=sample_desc,
            vendor=sample_vendor,
            family=sample_family,
            top_n=10
        )

        seen = set()
        rep_key = _norm_ef_title(rep_title)
        if rep_key:
            seen.add(rep_key)

        picked = []
        for cand in full_candidates:
            cand_title = cand.get("title")
            cand_ef = cand.get("ef")
            cand_key = _norm_ef_title(cand_title)

            if not cand_key:
                continue
            if cand_key in seen:
                continue

            seen.add(cand_key)
            picked.append({
                "title": cand_title,
                "ef": cand_ef,
            })

            if len(picked) >= recommend_n:
                break

        # 그래도 부족하면 현재 데이터프레임 전체의 EF_DB에서 추가 충원
        if len(picked) < recommend_n:
            fallback_df = (
                df[["EF_DB", "배출계수"]]
                .dropna(subset=["EF_DB"])
                .drop_duplicates(subset=["EF_DB"])
                .copy()
            )

            for _, r in fallback_df.iterrows():
                cand_title = r.get("EF_DB")
                cand_ef = r.get("배출계수")
                cand_key = _norm_ef_title(cand_title)

                if not cand_key:
                    continue
                if cand_key in seen:
                    continue

                seen.add(cand_key)
                picked.append({
                    "title": cand_title,
                    "ef": cand_ef,
                })

                if len(picked) >= recommend_n:
                    break

        base = {
            "품목/자산명": group_name,
            "자동카테고리": group_cat,
        }

        for i in range(1, recommend_n + 1):
            if i <= len(picked):
                base[f"추천EF{i}"] = picked[i-1]["title"]
                base[f"추천EF{i}_배출계수"] = picked[i-1]["ef"]
            else:
                base[f"추천EF{i}"] = None
                base[f"추천EF{i}_배출계수"] = None

        rec_rows.append(base)

    rec_df = pd.DataFrame(rec_rows)
    out = out.merge(rec_df, on=key_cols, how="left")

    ordered = [
        "품목/자산명", "매개변수", "추천Family", "자동카테고리",
        "1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월",
        "구매금액", "배출계수", "실가스배출", "입력건수",
        "입력통화", "환율기준연도",
        "그룹판단근거", "EF_DB_추천사유",
        "추천EF1", "추천EF1_배출계수",
        "추천EF2", "추천EF2_배출계수",
        "추천EF3", "추천EF3_배출계수",
        "추천EF4", "추천EF4_배출계수",
        "EF_DB",
    ]

    for col in ordered:
        if col not in out.columns:
            out[col] = None

    remain = [c for c in out.columns if c not in ordered]
    return out[ordered + remain]

# ── [원본 셀 44] ─────────────────────────────────────────────
# ============================================================
# 안전망: 이 셀이 단독 실행되거나 Cell45보다 먼저 참조될 때를 대비
# Cell45에서 정의되지만, 표준화 함수들이 직접 참조하므로 여기서도 보장
# ============================================================
import pandas as pd

if "MONTH_NAME_TO_NUM" not in dir():
    MONTH_NAME_TO_NUM = {f"{i}월": i for i in range(1, 13)}

if "PURCHASE_WIDE_MONTH_COLS" not in dir():
    PURCHASE_WIDE_MONTH_COLS = list(MONTH_NAME_TO_NUM.keys())

if "FX_REFERENCE_YEAR" not in dir():
    FX_REFERENCE_YEAR = 2025

# _safe_text, to_float 등 핵심 헬퍼가 없을 경우 최소 fallback
if "_safe_text" not in dir():
    def _safe_text(x):
        if x is None: return ""
        try:
            if pd.isna(x): return ""
        except Exception: pass
        return str(x).strip()

if "to_float" not in dir():
    def to_float(x):
        if x is None: return None
        try:
            s = str(x).replace(",", "").strip()
            return float(s) if s else None
        except Exception: return None

if "_coerce_month_value" not in dir():
    def _coerce_month_value(value):
        if pd.isnull(value): return None
        s = str(value).strip()
        if s in MONTH_NAME_TO_NUM: return MONTH_NAME_TO_NUM[s]
        try: return int(float(s))
        except Exception: return None

if "_normalize_currency_unit" not in dir():
    def _normalize_currency_unit(value):
        s = _safe_text(value).upper()
        if s in ["원", "KRW", "₩"]: return "KRW"
        if s in ["달러", "USD", "$"]: return "USD"
        return s if s else "KRW"

if "_normalize_purchase_classification" not in dir():
    def _normalize_purchase_classification(value):
        s = _safe_text(value).lower()
        if not s: return None
        if "2" in s or "자본" in s or "capital" in s: return "Category2"
        if "1" in s or "구매" in s or "소모" in s: return "Category1"
        return None

if "_reshape_purchase_wide_to_long" not in dir():
    def _reshape_purchase_wide_to_long(df):
        month_cols = [c for c in df.columns if str(c).strip() in PURCHASE_WIDE_MONTH_COLS]
        id_cols = [c for c in df.columns if c not in month_cols]
        try:
            long_df = df.melt(id_vars=id_cols, value_vars=month_cols,
                              var_name="구매 월", value_name="구매금액")
            long_df["구매 월"] = long_df["구매 월"].apply(_coerce_month_value)
            return long_df.dropna(subset=["구매금액"]).reset_index(drop=True)
        except Exception:
            return df

# ============================================================


def _purchase_text(row) -> str:
    """그룹핑 키워드 매칭용 텍스트 생성 — 새 v1 템플릿 컬럼 포함"""
    parts = [
        str(row.get("상세 품목명/서비스명", "") or ""),
        str(row.get("상세 자산명", "") or ""),
        str(row.get("품목명", "") or ""),
        str(row.get("자산명", "") or ""),
        str(row.get("품목/자산명", "") or ""),
        str(row.get("설명/적요", "") or ""),
        str(row.get("거래처명(선택)", "") or ""),
        str(row.get("거래처명", "") or ""),
    ]
    return " ".join(parts).lower()


def _infer_family_from_user_group(group_text: str) -> str:
    """사용자 입력 그룹명 텍스트로 family 추정"""
    t = str(group_text).strip().lower()
    mapping = [
        (["원재료", "원자재", "부자재", "철강", "알루미늄", "수지", "화학"],   "raw_material"),
        (["포장재", "포장지", "박스", "골판지", "비닐"],                        "packaging"),
        (["a4", "복사지", "용지", "복사용지"],                                   "paper"),
        (["토너", "잉크", "카트리지", "드럼"],                                   "printer_supplies"),
        (["사무용품", "문구", "소모품"],                                          "office_supplies"),
        (["커피", "원두", "차", "음료", "생수", "물", "탕비", "다과"],           "pantry_supplies"),
        (["의자", "책상", "파티션", "캐비닛", "옷장", "가구", "비품"],           "office_furniture"),
        (["노트북", "데스크탑", "컴퓨터", "모니터", "서버", "pc", "it"],         "computer_equipment"),
        (["프린터", "복합기", "스캐너"],                                          "printer_device"),
        (["소프트웨어", "라이선스", "software"],                                  "software_it"),
        (["외주", "용역", "서비스", "컨설팅", "청소", "경비"],                   "service_outsourcing"),
        (["생산설비", "기계장치", "설비"],                                         "manufacturing_equipment"),
        (["차량", "트럭", "승용차"],                                               "vehicle"),
        (["건물", "구축물", "창고"],                                               "building"),
    ]
    for keywords, family in mapping:
        if any(k in t for k in keywords):
            return family
    return "misc_purchase"


def _infer_group_name_llm(product_text: str) -> str:
    """
    상세 품목명을 입력받아 LLM으로 대표 그룹명 추론.
    예) 'Samsung 노트북 15인치' → 'IT장비'
    """
    try:
        prompt = f"""다음 구매 품목명을 간결한 그룹명으로 묶어라.

품목명: {product_text}

규칙:
- 5~15자 이내 한글 그룹명만 출력
- 설명/부연 없이 그룹명만 출력
- 예: 노트북, 데스크탑 → IT장비 / 사무의자, 회전의자 → 의자 / 원재료 A → 원재료

그룹명:"""
        res = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_response_text(res).strip()
    except Exception:
        return "Unknown"


def classify_purchase_group(row) -> dict:
    """
    그룹핑 우선순위:
      1순위: 사용자가 직접 입력한 그룹명
             - 새 v1 템플릿: '품목군'(C1) / '자산분류'(C2) 컬럼
             - 구 템플릿: '분류' 컬럼
      2순위: 키워드 룰 기반 자동 그룹핑 (PURCHASE_GROUP_RULES)
      3순위: LLM 기반 그룹핑 (상세 품목명으로 그룹명 추론)
      4순위: '기타구매' fallback
    """
    # ── 1순위: 사용자 직접 입력 그룹명 ──────────────────────────
    user_group = (
        _safe_text(row.get("품목군"))
        or _safe_text(row.get("자산분류"))
        or _safe_text(row.get("분류"))
    )
    manual_category = _normalize_purchase_classification(
        row.get("자동카테고리") or row.get("분류")
    )

    if user_group:
        category = manual_category if manual_category else "Category1"
        family = _infer_family_from_user_group(user_group)
        return {
            "group_key": f"user_{user_group[:20].replace(' ', '_')}",
            "품목/자산명": user_group,
            "추천Family": family,
            "자동카테고리": category,
            "그룹판단근거": f"사용자입력: {user_group}",
        }

    # ── 2순위: 키워드 룰 기반 자동 그룹핑 ──────────────────────
    text = _purchase_text(row)
    for rule in PURCHASE_GROUP_RULES:
        matched = [kw for kw in rule["keywords"] if kw.lower() in text]
        if matched:
            category = manual_category if manual_category else rule["category"]
            return {
                "group_key": rule["group_key"],
                "품목/자산명": rule["group_ko"],
                "추천Family": rule["family"],
                "자동카테고리": category,
                "그룹판단근거": "키워드매칭: " + ", ".join(matched[:3]),
            }

    # ── 3순위: LLM 그룹핑 시도 ──────────────────────────────────
    product_text = (
        _safe_text(row.get("상세 품목명/서비스명"))
        or _safe_text(row.get("상세 자산명"))
        or _safe_text(row.get("품목/자산명"))
        or _safe_text(row.get("품목명"))
        or _safe_text(row.get("자산명"))
    )
    if product_text:
        try:
            llm_group = _infer_group_name_llm(product_text)
            if llm_group and llm_group != "Unknown":
                category = manual_category if manual_category else "Category1"
                family = _infer_family_from_user_group(llm_group)
                return {
                    "group_key": f"llm_{llm_group[:20].replace(' ', '_')}",
                    "품목/자산명": llm_group,
                    "추천Family": family,
                    "자동카테고리": category,
                    "그룹판단근거": f"AI그룹핑: {product_text[:30]}",
                }
        except Exception:
            pass

    # ── 4순위: fallback ──────────────────────────────────────────
    category = manual_category if manual_category else "Category1"
    fallback_name = product_text or "기타구매"
    return {
        "group_key": "misc_purchase",
        "품목/자산명": fallback_name,
        "추천Family": "misc_purchase",
        "자동카테고리": category,
        "그룹판단근거": "rule_unmatched",
    }



# ==========================================
# Theme router - 수정본 템플릿(v1) 기준
# 시트명: C1_구매품서비스, C2_자본재, C3_연료에너지,
#         C4-1_운송거리, C4-2_운송금액, C5_사업장폐기물,
#         C6-1_출장거리, C6-2_출장비, C7_임직원통근,
#         C8-1/2/3_임차자산, C9-1/2_운송,
#         C11_판매제품사용, C12_판매제품폐기,
#         C13-1/2/3_임대자산, C14_프랜차이즈, C15_투자
# ==========================================

from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────
# 출력 시트명 매핑
# ─────────────────────────────────────────
THEME_OUTPUT_SHEETS = {
    "purchase": {
        "Category1": "Category1_구매품서비스",
        "Category2": "Category2_자본재",
    },
    "energy": {
        "Category3": "Category3_연료에너지",
    },
    "transport": {
        "Category4": "Category4_업스트림운송",
        "Category9": "Category9_다운스트림운송",
    },
    "waste": {
        "Category5": "Category5_사업장폐기물",
    },
    "travel": {
        "Category6": "Category6_출장",
        "Category7": "Category7_통근",
    },
    "leased_asset": {
        "Category8": "Category8_업스트림임차자산",
    },
    "downstream_leased": {
        "Category13": "Category13_다운스트림임대자산",
    },
    "product_use": {
        "Category11": "Category11_판매제품사용",
    },
    "product_eol": {
        "Category12": "Category12_판매제품폐기",
    },
    "franchise": {
        "Category14": "Category14_프랜차이즈",
    },
    "investment": {
        "Category15": "Category15_투자",
    },
}

THEME_AI_ENGINES = {
    "purchase":          {"engine_name": "purchase_scope3_ai",       "processor": "process_purchase_theme"},
    "energy":            {"engine_name": "energy_scope3_ai",          "processor": "process_energy_theme"},
    "transport":         {"engine_name": "transport_scope3_ai",       "processor": "process_transport_theme"},
    "waste":             {"engine_name": "waste_scope3_ai",           "processor": "process_waste_theme"},
    "travel":            {"engine_name": "travel_scope3_ai",          "processor": "process_travel_theme"},
    "leased_asset":      {"engine_name": "leased_asset_scope3_ai",    "processor": "process_leased_asset_theme"},
    "downstream_leased": {"engine_name": "downstream_leased_scope3_ai","processor": "process_downstream_leased_theme"},
    "product_use":       {"engine_name": "product_use_scope3_ai",     "processor": "process_product_use_theme"},
    "product_eol":       {"engine_name": "product_eol_scope3_ai",     "processor": "process_product_eol_theme"},
    "franchise":         {"engine_name": "franchise_scope3_ai",       "processor": "process_franchise_theme"},
    "investment":        {"engine_name": "investment_scope3_ai",      "processor": "process_investment_theme"},
}

# ─────────────────────────────────────────
# 시트명 키워드 → 테마 감지
# ─────────────────────────────────────────
# 각 시트명에 포함되는 키워드로 테마 판별
SHEET_THEME_MAP = [
    # (키워드 리스트, 테마)  — 우선순위 순서대로
    (["c1_", "c1 ", "구매품", "구매품서비스"],              "purchase_c1"),
    (["c2_", "c2 ", "자본재"],                             "purchase_c2"),
    (["c3_", "c3 ", "연료에너지", "연료", "에너지관련"],   "energy"),
    (["c4-1", "c4_1", "c4 1", "업스트림 운송거리",
      "업스트림운송거리", "c4운송거리"],                     "transport_c4_dist"),
    (["c4-2", "c4_2", "c4 2", "업스트림 운송금액",
      "업스트림운송금액", "c4운송금액"],                     "transport_c4_spend"),
    (["c5_", "c5 ", "사업장폐기물", "폐기물"],             "waste"),
    (["c6-1", "c6_1", "c6 1", "출장거리", "출장 거리"],    "travel_c6_dist"),
    (["c6-2", "c6_2", "c6 2", "출장비", "출장 비"],        "travel_c6_spend"),
    (["c7_", "c7 ", "임직원통근", "통근"],                  "travel_c7"),
    (["c8-1", "c8_1", "c8 1", "임차자산", "에너지사용량"],  "leased_c8_energy"),
    (["c8-2", "c8_2", "c8 2", "임차자산", "면적"],          "leased_c8_area"),
    (["c8-3", "c8_3", "c8 3", "임차자산", "인원"],          "leased_c8_person"),
    (["c9-1", "c9_1", "c9 1", "다운스트림 운송거리",
      "다운스트림운송거리", "c9운송거리"],                    "transport_c9_dist"),
    (["c9-2", "c9_2", "c9 2", "다운스트림 운송금액",
      "다운스트림운송금액", "c9운송금액"],                    "transport_c9_spend"),
    (["c11_", "c11 ", "판매제품사용", "제품사용"],          "product_use"),
    (["c12_", "c12 ", "판매제품폐기", "제품폐기"],          "product_eol"),
    (["c13-1", "c13_1", "임대자산", "에너지사용"],          "downstream_leased_c13_energy"),
    (["c13-2", "c13_2", "임대자산", "면적"],                "downstream_leased_c13_area"),
    (["c13-3", "c13_3", "임대자산", "인원"],                "downstream_leased_c13_person"),
    (["c14_", "c14 ", "프랜차이즈"],                        "franchise"),
    (["c15_", "c15 ", "투자"],                              "investment"),
]

def _sheet_sub_theme(sheet_name: str) -> Optional[str]:
    """시트명 → sub-theme 문자열 반환"""
    n = str(sheet_name).strip().lower()
    for keywords, sub in SHEET_THEME_MAP:
        if any(k in n for k in keywords):
            return sub
    return None

# 구 버전 호환 - 구매/운송/출장통근 테마 시트 후보
PURCHASE_SHEET_CANDIDATES = {
    "C1_구매품서비스", "C2_자본재",
    "01_구매_Template", "구매", "purchase",
}
TRANSPORT_INBOUND_KEYWORDS  = ["c4", "업스트림", "입고", "inbound", "upstream", "category4"]
TRANSPORT_OUTBOUND_KEYWORDS = ["c9", "다운스트림", "출고", "outbound", "downstream", "category9"]
TRAVEL_BUSINESS_KEYWORDS    = ["c6", "출장", "business", "travel", "category6"]
TRAVEL_COMMUTE_KEYWORDS     = ["c7", "통근", "commute", "category7"]

# 새 템플릿 필수 컬럼 집합
PURCHASE_C1_REQUIRED = {"*품목군", "품목군", "*상세 품목명/서비스명", "상세 품목명/서비스명"}
PURCHASE_C2_REQUIRED = {"*자산분류", "자산분류", "*상세 자산명", "상세 자산명"}
ENERGY_C3_REQUIRED   = {"*연료/에너지명", "연료/에너지명", "*단위", "단위"}
TRANSPORT_DIST_REQUIRED = {"*출발지", "출발지", "*도착지", "도착지", "*중량단위", "중량단위"}
TRANSPORT_SPEND_REQUIRED = {"*자재명/품목명", "자재명/품목명"}
WASTE_REQUIRED       = {"*폐기물 종류", "폐기물 종류", "*처리방법", "처리방법"}
TRAVEL_DIST_REQUIRED = {"*교통수단", "교통수단", "*인원수", "인원수"}
TRAVEL_SPEND_REQUIRED = {"*출장항목", "출장항목"}
COMMUTE_REQUIRED     = {"*교통수단", "교통수단", "*인원수", "인원수"}

# ─────────────────────────────────────────
# 새 템플릿 컬럼 정규화 헬퍼
# ─────────────────────────────────────────
def _strip_asterisk(col: str) -> str:
    """헤더 앞 * 제거"""
    return str(col).strip().lstrip("*").strip()

def normalize_template_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명 앞 * 제거 + 공백 제거 + Unnamed 처리"""
    df = df.copy()
    new_cols = []
    for c in df.columns:
        s = str(c).strip()
        if s.startswith("Unnamed") or s == "nan":
            new_cols.append(s)  # 그대로 유지 (이후 필요시 드롭)
        else:
            new_cols.append(_strip_asterisk(s))
    df.columns = new_cols
    return df

# ─────────────────────────────────────────
# 새 템플릿 시트 표준화 함수들
# ─────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# 월별/연간 melt 공통 헬퍼 — 중복 방지
# ──────────────────────────────────────────────────────────────
def _melt_monthly_or_annual(df: pd.DataFrame,
                             month_var_name: str = "발생월",
                             value_name: str = "월간값",
                             annual_col_hint: str = None) -> pd.DataFrame:
    """
    read_template_sheet에서 분리된 입력방식(연간/월별)에 따라 안전하게 melt.
    - 월별 행: month_cols를 melt → long형
    - 연간 행: annual_col_hint 컬럼명 → value_name으로 rename만
    - 중복 없음
    """
    month_cols = [c for c in df.columns if str(c).strip() in PURCHASE_WIDE_MONTH_COLS]

    if not month_cols:
        # 월별 컬럼 자체가 없으면 annual rename만
        if annual_col_hint and annual_col_hint in df.columns:
            df = df.rename(columns={annual_col_hint: value_name})
        return df

    has_input_type = "입력방식" in df.columns

    if has_input_type:
        df_m = df[df["입력방식"] == "월별"].copy()
        df_a = df[df["입력방식"] != "월별"].copy()
    else:
        # 입력방식 컬럼 없으면: 월별값 있는 행 vs 없는 행 자동 구분
        def _has_monthly(r):
            return any(pd.notnull(r.get(m)) and str(r.get(m)).strip() not in ("","nan")
                       for m in month_cols)
        mask = df.apply(_has_monthly, axis=1)
        df_m = df[mask].copy()
        df_a = df[~mask].copy()

    results = []

    # 월별 행 melt
    if not df_m.empty:
        id_cols = [c for c in df_m.columns if c not in month_cols]
        try:
            long = df_m.melt(id_vars=id_cols, value_vars=month_cols,
                             var_name=month_var_name, value_name=value_name)
            long = long.dropna(subset=[value_name]).reset_index(drop=True)
            long = long[long[value_name].astype(str).str.strip() != ""]
            long[month_var_name] = long[month_var_name].apply(_coerce_month_value)
            results.append(long)
        except Exception:
            results.append(df_m)

    # 연간 행: annual_col → value_name rename
    if not df_a.empty:
        if annual_col_hint and annual_col_hint in df_a.columns and annual_col_hint != value_name:
            df_a = df_a.rename(columns={annual_col_hint: value_name})
        # 월별 컬럼 제거
        df_a = df_a.drop(columns=[c for c in month_cols if c in df_a.columns], errors="ignore")
        results.append(df_a)

    if results:
        return pd.concat(results, ignore_index=True)
    return df


def standardize_c1_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C1_구매품서비스 시트 표준화
    컬럼: 국가, 사업장, 품목군, 상세 품목명/서비스명, 계정과목(선택),
          거래처명(선택), 연간지출금액(원), 1월~12월, 비고(선택)
    """
    df = normalize_template_columns(df)
    rename = {}
    # * 제거 후 표준화
    if "*상세 품목명/서비스명" in df.columns:
        rename["*상세 품목명/서비스명"] = "상세 품목명/서비스명"
    if "*품목군" in df.columns:
        rename["*품목군"] = "품목군"
    if "*국가" in df.columns:
        rename["*국가"] = "국가"
    if "*사업장" in df.columns:
        rename["*사업장"] = "사업장"
    if rename:
        df = df.rename(columns=rename)
        rename = {}

    # 새 컬럼 → 구 코드 호환
    if "상세 품목명/서비스명" in df.columns and "품목/자산명" not in df.columns:
        rename["상세 품목명/서비스명"] = "품목/자산명"
    if "품목군" in df.columns and "분류" not in df.columns:
        rename["품목군"] = "분류"
    # 연간지출금액: read_template_sheet에서 이미 서브헤더로 처리됨
    if "연간지출금액(원)" in df.columns and "구매금액" not in df.columns:
        rename["연간지출금액(원)"] = "구매금액"
    if "계정과목(선택)" in df.columns:
        rename["계정과목(선택)"] = "계정과목"
    if "거래처명(선택)" in df.columns:
        rename["거래처명(선택)"] = "거래처명"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    # 월별 vs 연간 입력방식에 따라 분기 처리
    month_cols = [c for c in df.columns if str(c).strip() in PURCHASE_WIDE_MONTH_COLS]
    annual_col = next((c for c in ["구매금액", "연간금액", "연간지출금액(원)"]
                       if c in df.columns), None)

    if month_cols and "입력방식" in df.columns:
        # 월별 행만 melt
        df_m = df[df["입력방식"] == "월별"].copy()
        df_a = df[df["입력방식"] != "월별"].copy()

        if not df_m.empty:
            id_cols = [c for c in df_m.columns if c not in month_cols]
            try:
                long_m = df_m.melt(id_vars=id_cols, value_vars=month_cols,
                                   var_name="구매 월", value_name="구매금액")
                long_m = long_m.dropna(subset=["구매금액"]).reset_index(drop=True)
                long_m["구매 월"] = long_m["구매 월"].apply(_coerce_month_value)
                df_m = long_m
            except Exception:
                pass

        # 연간 행: 연간금액 → 구매금액
        if not df_a.empty and annual_col and annual_col != "구매금액":
            df_a = df_a.rename(columns={annual_col: "구매금액"})

        df = pd.concat([df_a, df_m], ignore_index=True)

    elif month_cols:
        # 입력방식 컬럼 없으면 기존 방식 (월별값 있는 행만 melt)
        id_cols = [c for c in df.columns if c not in month_cols
                   and c != annual_col]
        try:
            long_df = df.melt(id_vars=id_cols + ([annual_col] if annual_col else []),
                              value_vars=month_cols,
                              var_name="구매 월", value_name="구매금액")
            long_df = long_df.dropna(subset=["구매금액"]).reset_index(drop=True)
            long_df["구매 월"] = long_df["구매 월"].apply(_coerce_month_value)
            df = long_df
        except Exception:
            pass
    elif annual_col and annual_col != "구매금액":
        df = df.rename(columns={annual_col: "구매금액"})

    if "품목/자산명" in df.columns:
        if "품목명" not in df.columns:
            df["품목명"] = df["품목/자산명"]

    if "구매 월" in df.columns:
        df["구매 월"] = df["구매 월"].apply(_coerce_month_value)

    # 단위/통화 기본값
    if "단위" not in df.columns:
        df["단위"] = "KRW"
    df["단위"] = df["단위"].apply(_normalize_currency_unit)
    if "통화" not in df.columns:
        df["통화"] = df["단위"]

    return df


def standardize_c2_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C2_자본재 시트 표준화
    컬럼: 국가, 사업장, 자산분류, 상세 자산명, 자산번호(선택), 계정과목(선택),
          거래처명(선택), 연간취득금액(원), 1월~12월, 비고(선택)
    """
    df = normalize_template_columns(df)
    rename = {}
    # * 제거
    for col in ["*상세 자산명","*자산분류","*국가","*사업장"]:
        clean = col.lstrip("*")
        if col in df.columns:
            rename[col] = clean
    if rename:
        df = df.rename(columns=rename)
        rename = {}

    if "상세 자산명" in df.columns and "품목/자산명" not in df.columns:
        rename["상세 자산명"] = "품목/자산명"
    if "자산분류" in df.columns and "분류" not in df.columns:
        rename["자산분류"] = "분류"
    if "연간취득금액(원)" in df.columns and "구매금액" not in df.columns:
        rename["연간취득금액(원)"] = "구매금액"
    if "자산번호(선택)" in df.columns:
        rename["자산번호(선택)"] = "자산번호"
    if "계정과목(선택)" in df.columns:
        rename["계정과목(선택)"] = "계정과목"
    if "거래처명(선택)" in df.columns:
        rename["거래처명(선택)"] = "거래처명"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    month_cols = [c for c in df.columns if str(c).strip() in PURCHASE_WIDE_MONTH_COLS]
    if month_cols:
        annual_col = next((c for c in ["구매금액","연간취득금액(원)"] if c in df.columns), None)
        id_cols = [c for c in df.columns if c not in month_cols and c != annual_col]
        try:
            long_df = df.melt(id_vars=id_cols + ([annual_col] if annual_col else []),
                              value_vars=month_cols,
                              var_name="구매 월", value_name="구매금액")
            long_df = long_df.dropna(subset=["구매금액"]).reset_index(drop=True)
            long_df["구매 월"] = long_df["구매 월"].apply(_coerce_month_value)
            df = long_df
        except Exception:
            pass

    if "품목/자산명" in df.columns:
        if "자산명" not in df.columns:
            df["자산명"] = df["품목/자산명"]

    if "구매 월" in df.columns:
        df["구매 월"] = df["구매 월"].apply(_coerce_month_value)

    if "단위" not in df.columns:
        df["단위"] = "KRW"
    df["단위"] = df["단위"].apply(_normalize_currency_unit)
    if "통화" not in df.columns:
        df["통화"] = df["단위"]

    return df


def standardize_c3_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C3_연료에너지 시트 표준화
    컬럼: 국가, 사업장, 연료/에너지명, 단위, 연간사용량, 1월~12월, 비고(선택)
    산정식: Σ(연료/에너지 사용량 × 업스트림/손실 배출계수) — 환경성적표지(EPD)
    """
    df = normalize_template_columns(df)
    rename = {}
    if "연료/에너지명" in df.columns and "에너지원" not in df.columns:
        rename["연료/에너지명"] = "에너지원"
    if "연간사용량" in df.columns and "사용량" not in df.columns:
        rename["연간사용량"] = "사용량"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    df = _melt_monthly_or_annual(df, month_var_name="사용월",
                                    value_name="사용량",
                                    annual_col_hint="연간사용량")

    return df


def standardize_c4_dist_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C4-1_운송거리 시트 표준화
    컬럼: 국가, 사업장, 운송수단(선택), 자재명/품목명, 중량단위, 출발지, 도착지,
          운송거리(km)(선택), 연간운송량, 1월~12월, 비고(선택)
    산정식: Σ(화물중량 × 운송거리 × 운송수단별 배출계수) — EPD
    """
    df = normalize_template_columns(df)
    rename = {}
    # * 제거
    for col in list(df.columns):
        if str(col).startswith("*"):
            rename[col] = col.lstrip("*").strip()
    if rename:
        df = df.rename(columns=rename)
        rename = {}

    if "자재명/품목명" in df.columns and "자재명" not in df.columns:
        rename["자재명/품목명"] = "자재명"
    if "운송수단(선택)" in df.columns and "운송수단" not in df.columns:
        rename["운송수단(선택)"] = "운송수단"
    # C4-1: *출도착지 → 출발지, *도착지 처리
    # read_template_sheet의 서브헤더 처리로 출발지/도착지 컬럼이 분리되어 있으나
    # 주헤더에 *출도착지가 남아 있을 수 있으므로 추가 처리
    if "*출도착지" in df.columns and "출발지" not in df.columns:
        rename["*출도착지"] = "출발지"
    if "*도착지" in df.columns and "도착지" not in df.columns:
        rename["*도착지"] = "도착지"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    # 월별 가로형 → 장형
    if any(str(c).strip() in PURCHASE_WIDE_MONTH_COLS for c in df.columns):
        month_cols = [c for c in df.columns if str(c).strip() in PURCHASE_WIDE_MONTH_COLS]
        id_cols = [c for c in df.columns if c not in month_cols]
        try:
            df = df.melt(id_vars=id_cols, value_vars=month_cols,
                         var_name="발생월", value_name="월운송량")
            df["발생월"] = df["발생월"].apply(_coerce_month_value)
            df = df.dropna(subset=["월운송량"]).reset_index(drop=True)
        except Exception:
            pass

    # 화물중량 컬럼: 중량단위 + 월운송량 or 연간운송량
    if "화물중량(kg)" not in df.columns:
        weight_val = df.get("월운송량") if "월운송량" in df.columns else df.get("연간운송량")
        unit_col = df.get("중량단위") if "중량단위" in df.columns else None
        if weight_val is not None:
            # ton → kg 변환
            def _to_kg(val, unit):
                v = to_float(val)
                if v is None:
                    return None
                u = str(unit).strip().lower() if pd.notnull(unit) else "ton"
                if u in ["ton", "t"]:
                    return v * 1000
                return v  # already kg
            if unit_col is not None:
                df["화물중량(kg)"] = [_to_kg(w, u) for w, u in zip(weight_val, unit_col)]
            else:
                df["화물중량(kg)"] = weight_val.apply(lambda x: to_float(x))

    df["운송구분"] = "입고"
    return df


def standardize_c4_spend_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C4-2_운송금액 시트 표준화
    컬럼: 국가, 사업장, 운송수단(선택), 자재명/품목명, 거래처명(선택),
          연간운송금액(원), 1월~12월, 비고(선택)
    산정식: Σ(운송지출금액 × 지출기반 배출계수) — US EPA
    """
    df = normalize_template_columns(df)
    rename = {}
    if "자재명/품목명" in df.columns and "자재명" not in df.columns:
        rename["자재명/품목명"] = "자재명"
    if "운송수단(선택)" in df.columns and "운송수단" not in df.columns:
        rename["운송수단(선택)"] = "운송수단"
    if "연간운송금액(원)" in df.columns and "운송금액" not in df.columns:
        rename["연간운송금액(원)"] = "운송금액"
    if "거래처명(선택)" in df.columns:
        rename["거래처명(선택)"] = "거래처명"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    df = _melt_monthly_or_annual(df, month_var_name="발생월",
                                    value_name="운송금액",
                                    annual_col_hint="연간운송금액(원)")

    df["운송구분"] = "입고"
    df["계산방식"] = "Spend"
    return df


def standardize_c5_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C5_사업장폐기물 시트 표준화
    컬럼: 국가, 사업장, 폐기물 종류, 상세 폐기물명(선택), 단위, 처리방법,
          연간처리량, 1월~12월, 비고(선택)
    산정식: Σ(폐기물 발생량 × 처리방법별 배출계수) — EPD
    """
    df = normalize_template_columns(df)
    rename = {}
    if "상세 폐기물명(선택)" in df.columns:
        rename["상세 폐기물명(선택)"] = "상세폐기물명"
    if "연간처리량" in df.columns and "처리량" not in df.columns:
        rename["연간처리량"] = "처리량"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    df = _melt_monthly_or_annual(df, month_var_name="발생월",
                                    value_name="처리량",
                                    annual_col_hint="연간처리량")

    return df


def standardize_c6_dist_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C6-1_출장거리 시트 표준화
    컬럼: 국가, 출장주체, 교통수단, 인원수, 출발지, 도착지,
          편도거리(km)(선택), 연간이동횟수, 1월~12월, 비고(선택)
    산정식: Σ(왕복 이동거리 × 교통수단별 배출계수 × 인원수 × 이동횟수) — UK defra
    """
    df = normalize_template_columns(df)
    rename = {}
    # * 제거
    for col in list(df.columns):
        if str(col).startswith("*"):
            rename[col] = col.lstrip("*").strip()
    if rename:
        df = df.rename(columns=rename)
        rename = {}

    if "교통수단" in df.columns and "이동수단" not in df.columns:
        rename["교통수단"] = "이동수단"
    if "출장주체" in df.columns and "사업장" not in df.columns:
        rename["출장주체"] = "사업장"
    # 서브헤더에서 "편도거리(km)(선택)"이 직접 컬럼으로 오는 경우
    if "편도거리(km)(선택)" in df.columns and "이동거리(km)" not in df.columns:
        rename["편도거리(km)(선택)"] = "이동거리(km)"
    if "연간이동횟수" in df.columns and "이동횟수" not in df.columns:
        rename["연간이동횟수"] = "이동횟수"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    # *출도착지 병합셀 → 서브헤더에서 "출발지","도착지" 로 분리됨 (이미 처리)
    # 혹시 남아있는 경우 추가 처리
    if "*출도착지" in df.columns and "출발지" not in df.columns:
        df = df.rename(columns={"*출도착지": "출발지"})

    if "출장수단" not in df.columns and "이동수단" in df.columns:
        df["출장수단"] = df["이동수단"]
    if "출장인원" not in df.columns and "인원수" in df.columns:
        df["출장인원"] = df["인원수"]

    df = _melt_monthly_or_annual(df, month_var_name="발생월",
                                    value_name="이동횟수",
                                    annual_col_hint="연간이동횟수")

    df["구분"] = "출장"
    df["계산방식"] = "Distance"
    return df


def standardize_c6_spend_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C6-2_출장비 시트 표준화
    컬럼: 국가, 출장주체, 출장항목, 인원수, 연간지출금액(원), 1월~12월, 비고(선택)
    산정식: Σ(출장지출금액 × 지출기반 배출계수) — US EPA
    """
    df = normalize_template_columns(df)
    rename = {}
    if "출장주체" in df.columns and "사업장" not in df.columns:
        rename["출장주체"] = "사업장"
    if "출장항목" in df.columns and "이동수단" not in df.columns:
        rename["출장항목"] = "이동수단"
    if "연간지출금액(원)" in df.columns and "출장비용" not in df.columns:
        rename["연간지출금액(원)"] = "출장비용"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    if "출장인원" not in df.columns and "인원수" in df.columns:
        df["출장인원"] = df["인원수"]
    if "출장수단" not in df.columns and "이동수단" in df.columns:
        df["출장수단"] = df["이동수단"]

    df = _melt_monthly_or_annual(df, month_var_name="발생월",
                                    value_name="출장비용",
                                    annual_col_hint="연간지출금액(원)")

    df["구분"] = "출장"
    df["계산방식"] = "Spend"
    return df


def standardize_c7_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C7_임직원통근 시트 표준화
    컬럼: 국가, 사업장, 교통수단, 인원수, 출발지, 도착지,
          편도거리(km)(선택), 연간근무일수, 1월~12월, 비고(선택)
    산정식: Σ(왕복 이동거리 × 연간 근무일수 × 교통수단별 배출계수 × 인원수) — UK defra
    """
    df = normalize_template_columns(df)
    rename = {}
    for col in list(df.columns):
        if str(col).startswith("*"):
            rename[col] = col.lstrip("*").strip()
    if rename:
        df = df.rename(columns=rename)
        rename = {}

    if "교통수단" in df.columns and "이동수단" not in df.columns:
        rename["교통수단"] = "이동수단"
    if "편도거리(km)(선택)" in df.columns and "이동거리(km)" not in df.columns:
        rename["편도거리(km)(선택)"] = "이동거리(km)"
    if "연간근무일수" in df.columns and "근무일수" not in df.columns:
        rename["연간근무일수"] = "근무일수"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    # *출도착지 병합셀 처리
    if "*출도착지" in df.columns and "출발지" not in df.columns:
        df = df.rename(columns={"*출도착지": "출발지"})

    if "교통수단" not in df.columns and "이동수단" in df.columns:
        df["교통수단"] = df["이동수단"]
    if "직원수" not in df.columns and "인원수" in df.columns:
        df["직원수"] = df["인원수"]
    if "평균통근거리(km)" not in df.columns and "이동거리(km)" in df.columns:
        df["평균통근거리(km)"] = df["이동거리(km)"]
    elif "평균통근거리(km)" not in df.columns and "편도거리(km)(선택)" in df.columns:
        df["평균통근거리(km)"] = df["편도거리(km)(선택)"].apply(
            lambda x: None if pd.isnull(x) else float(str(x).replace(",",""))
            if str(x).strip() not in ["","nan"] else None
        )

    df = _melt_monthly_or_annual(df, month_var_name="발생월",
                                    value_name="근무일수",
                                    annual_col_hint="연간근무일수")

    df["구분"] = "통근"
    return df


def standardize_c8_energy_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C8-1 업스트림 임차자산 - 에너지사용량
    컬럼: 국가, 임차자산명, 에너지종류, 단위, 연간사용량, 1월~12월, 비고(선택)
    산정식: Σ(에너지 사용량 × 에너지원별 배출계수) — EPD/IPCC
    """
    df = normalize_template_columns(df)
    rename = {}
    if "에너지종류" in df.columns and "에너지원" not in df.columns:
        rename["에너지종류"] = "에너지원"
    if "연간사용량" in df.columns and "사용량" not in df.columns:
        rename["연간사용량"] = "사용량"
    if "임차자산명" in df.columns and "자산명" not in df.columns:
        rename["임차자산명"] = "자산명"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)
    df["임차자산여부"] = True
    df["산정방식"] = "에너지사용량"
    return df


def standardize_c8_area_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C8-2 업스트림 임차자산 - 면적배분
    컬럼: 국가, 임차자산명, 건물전체 면적, 사용 면적, 에너지종류, 단위, 연간사용량 등
    산정식: Σ((사용면적/건물전체면적) × 건물전체 에너지 사용량 × 에너지원별 배출계수)
    """
    df = normalize_template_columns(df)
    rename = {}
    if "에너지종류" in df.columns and "에너지원" not in df.columns:
        rename["에너지종류"] = "에너지원"
    if "임차자산명" in df.columns and "자산명" not in df.columns:
        rename["임차자산명"] = "자산명"
    if "건물전체 연간사용량" in df.columns and "건물전체사용량" not in df.columns:
        rename["건물전체 연간사용량"] = "건물전체사용량"
    if "건물전체 면적" in df.columns:
        rename["건물전체 면적"] = "건물전체면적"
    if "사용 면적" in df.columns:
        rename["사용 면적"] = "사용면적"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    # 면적 비율로 사용량 계산
    if "건물전체사용량" in df.columns and "건물전체면적" in df.columns and "사용면적" in df.columns:
        df["면적비율"] = df.apply(
            lambda r: (to_float(r.get("사용면적")) / to_float(r.get("건물전체면적")))
                      if to_float(r.get("건물전체면적")) else None, axis=1)
        df["사용량"] = df.apply(
            lambda r: (to_float(r.get("건물전체사용량")) * r.get("면적비율"))
                      if pd.notnull(r.get("면적비율")) else None, axis=1)

    df["임차자산여부"] = True
    df["산정방식"] = "면적배분"
    return df


def standardize_c8_person_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C8-3 업스트림 임차자산 - 인원배분
    산정식: Σ((자사인원수/건물전체인원수) × 건물전체 에너지 사용량 × 에너지원별 배출계수)
    """
    df = normalize_template_columns(df)
    rename = {}
    if "에너지종류" in df.columns and "에너지원" not in df.columns:
        rename["에너지종류"] = "에너지원"
    if "임차자산명" in df.columns and "자산명" not in df.columns:
        rename["임차자산명"] = "자산명"
    if "건물전체 연간사용량" in df.columns:
        rename["건물전체 연간사용량"] = "건물전체사용량"
    if "건물전체 인원수" in df.columns:
        rename["건물전체 인원수"] = "건물전체인원수"
    if "자사 인원수" in df.columns:
        rename["자사 인원수"] = "자사인원수"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    if "건물전체사용량" in df.columns and "건물전체인원수" in df.columns and "자사인원수" in df.columns:
        df["인원비율"] = df.apply(
            lambda r: (to_float(r.get("자사인원수")) / to_float(r.get("건물전체인원수")))
                      if to_float(r.get("건물전체인원수")) else None, axis=1)
        df["사용량"] = df.apply(
            lambda r: (to_float(r.get("건물전체사용량")) * r.get("인원비율"))
                      if pd.notnull(r.get("인원비율")) else None, axis=1)

    df["임차자산여부"] = True
    df["산정방식"] = "인원배분"
    return df


def standardize_c9_dist_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C9-1_운송거리 (다운스트림)
    컬럼: 국가, 사업장, 운송수단, 제품명, 중량단위, 출발지, 도착지,
          운송거리(km)(선택), 연간운송량, 1월~12월, 비고(선택)
    산정식: C4-1과 동일, EPD 배출계수
    """
    df = standardize_c4_dist_sheet(df)  # 구조 동일
    df["운송구분"] = "출고"
    return df


def standardize_c9_spend_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C9-2_운송금액 (다운스트림)
    컬럼: 국가, 사업장, 운송수단, 제품명, 거래처명(선택),
          연간운송금액(원), 1월~12월, 비고(선택)
    """
    df = standardize_c4_spend_sheet(df)
    df["운송구분"] = "출고"
    return df


def standardize_c11_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C11_판매제품사용 시트 표준화
    컬럼: 국가, 그룹(선택), 보고연도, 제품명, 제품수량, 에너지명,
          제품당 연간 에너지 사용량, 단위, 수명(개월)/평균사용기간(개월), 설명/적요(선택)
    산정식: Σ(판매수량 × 제품당 연간 에너지 사용량 × 평균 사용기간 × 에너지원별 배출계수) — EPD
    """
    df = normalize_template_columns(df)
    rename = {}
    if "그룹(선택)" in df.columns:
        rename["그룹(선택)"] = "그룹"
    if "설명/적요(선택)" in df.columns:
        rename["설명/적요(선택)"] = "설명/적요"
    if "수명(개월) 혹은 평균 사용기간(개월)" in df.columns:
        rename["수명(개월) 혹은 평균 사용기간(개월)"] = "수명(개월)"
    if rename:
        df = df.rename(columns=rename)
    return df


def standardize_c12_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C12_판매제품폐기 시트 표준화
    컬럼: 국가, 폐기물 대분류(선택), 폐기물 중분류(선택), 폐기물 소분류(선택),
          제품명, 판매량, 제품중량, 제품중량 단위,
          재활용 비중(%)(선택), 소각 비중(%)(선택), 매립 비중(%)(선택), 기타 비중(%)(선택), 설명/적요(선택)
    산정식: Σ(판매수량 × 제품중량 × 폐기방식 비율 × 폐기방식별 배출계수) — EPD
    """
    df = normalize_template_columns(df)
    rename = {}
    for col in ["재활용 비중(%)(선택)", "소각 비중(%)(선택)", "매립 비중(%)(선택)", "기타 비중(%)(선택)"]:
        clean = col.replace("(선택)", "").strip()
        if col in df.columns:
            rename[col] = clean
    if "설명/적요(선택)" in df.columns:
        rename["설명/적요(선택)"] = "설명/적요"
    if rename:
        df = df.rename(columns=rename)
    return df


def standardize_c13_energy_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C13-1 다운스트림 임대자산 - 에너지사용량
    C8-1과 동일 구조, 임대자산 대상
    """
    df = normalize_template_columns(df)
    rename = {}
    if "에너지종류" in df.columns and "에너지원" not in df.columns:
        rename["에너지종류"] = "에너지원"
    if "연간사용량" in df.columns and "사용량" not in df.columns:
        rename["연간사용량"] = "사용량"
    if "임대자산명" in df.columns and "자산명" not in df.columns:
        rename["임대자산명"] = "자산명"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)
    df["임대자산여부"] = True
    df["산정방식"] = "에너지사용량"
    return df


def standardize_c13_area_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C13-2 다운스트림 임대자산 - 면적배분"""
    df = normalize_template_columns(df)
    rename = {}
    if "에너지종류" in df.columns and "에너지원" not in df.columns:
        rename["에너지종류"] = "에너지원"
    if "임대자산명" in df.columns and "자산명" not in df.columns:
        rename["임대자산명"] = "자산명"
    if "건물전체 연간사용량" in df.columns:
        rename["건물전체 연간사용량"] = "건물전체사용량"
    if "건물전체 면적" in df.columns:
        rename["건물전체 면적"] = "건물전체면적"
    if "사용 면적" in df.columns:
        rename["사용 면적"] = "사용면적"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    if "건물전체사용량" in df.columns and "건물전체면적" in df.columns and "사용면적" in df.columns:
        df["면적비율"] = df.apply(
            lambda r: (to_float(r.get("사용면적")) / to_float(r.get("건물전체면적")))
                      if to_float(r.get("건물전체면적")) else None, axis=1)
        df["사용량"] = df.apply(
            lambda r: (to_float(r.get("건물전체사용량")) * r.get("면적비율"))
                      if pd.notnull(r.get("인원비율")) else None, axis=1)

    df["임대자산여부"] = True
    df["산정방식"] = "면적배분"
    return df


def standardize_c13_person_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C13-3 다운스트림 임대자산 - 인원배분"""
    df = normalize_template_columns(df)
    rename = {}
    if "에너지종류" in df.columns and "에너지원" not in df.columns:
        rename["에너지종류"] = "에너지원"
    if "임대자산명" in df.columns and "자산명" not in df.columns:
        rename["임대자산명"] = "자산명"
    if "건물전체 연간사용량" in df.columns:
        rename["건물전체 연간사용량"] = "건물전체사용량"
    if "건물전체 인원수" in df.columns:
        rename["건물전체 인원수"] = "건물전체인원수"
    if "자사 인원수" in df.columns:
        rename["자사 인원수"] = "자사인원수"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    if "건물전체사용량" in df.columns and "건물전체인원수" in df.columns and "자사인원수" in df.columns:
        df["인원비율"] = df.apply(
            lambda r: (to_float(r.get("자사인원수")) / to_float(r.get("건물전체인원수")))
                      if to_float(r.get("건물전체인원수")) else None, axis=1)
        df["사용량"] = df.apply(
            lambda r: (to_float(r.get("건물전체사용량")) * r.get("인원비율"))
                      if pd.notnull(r.get("인원비율")) else None, axis=1)

    df["임대자산여부"] = True
    df["산정방식"] = "인원배분"
    return df


def standardize_c14_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C14_프랜차이즈 시트 표준화
    컬럼: 국가, 프랜차이즈명, 가맹점 수, 연료 종류, 연료 사용량, 연료 단위
    산정식: Σ(가맹점 활동 유형별 사용량 × 에너지원별 배출계수) — 국내 승인 / IPCC
    """
    df = normalize_template_columns(df)
    rename = {}
    if "연료 종류" in df.columns and "에너지원" not in df.columns:
        rename["연료 종류"] = "에너지원"
    if "연료 사용량" in df.columns and "사용량" not in df.columns:
        rename["연료 사용량"] = "사용량"
    if "연료 단위" in df.columns and "단위" not in df.columns:
        rename["연료 단위"] = "단위"
    if "가맹점 수" in df.columns and "가맹점수" not in df.columns:
        rename["가맹점 수"] = "가맹점수"
    if rename:
        df = df.rename(columns=rename)
    return df


def standardize_c15_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C15_투자 시트 표준화
    컬럼: 피투자기업명, 지분율(%), 피투자기업 Scope1배출량, 피투자기업 Scope2배출량
    산정식: Σ(투자 대상의 Scope 1&2 배출량 × 당사의 지분 비율)
    """
    df = normalize_template_columns(df)
    rename = {}
    if "피투자기업명 (필수)" in df.columns:
        rename["피투자기업명 (필수)"] = "피투자기업명"
    if "지분율(%) (선택)" in df.columns:
        rename["지분율(%) (선택)"] = "지분율(%)"
    if "피투자기업 Scope1배출량" in df.columns:
        rename["피투자기업 Scope1배출량"] = "Scope1배출량"
    if "피투자기업 Scope2배출량" in df.columns:
        rename["피투자기업 Scope2배출량"] = "Scope2배출량"
    if rename:
        df = df.rename(columns=rename)

    # 산정 배출량 = (Scope1 + Scope2) × 지분율
    if "Scope1배출량" in df.columns and "Scope2배출량" in df.columns and "지분율(%)" in df.columns:
        df["배출량"] = df.apply(
            lambda r: ((to_float(r.get("Scope1배출량")) or 0) + (to_float(r.get("Scope2배출량")) or 0))
                      * ((to_float(r.get("지분율(%)")) or 0) / 100), axis=1)

    return df


# ─────────────────────────────────────────
# 새 템플릿 시트 감지 → 처리 라우팅
# ─────────────────────────────────────────
# 시트명 → (표준화함수, 카테고리, 테마) 매핑
SHEET_PROCESSOR_MAP = {
    "c1_구매품서비스":      (standardize_c1_sheet,          "Category1",  "purchase"),
    "c2_자본재":           (standardize_c2_sheet,          "Category2",  "purchase"),
    "c3_연료에너지":        (standardize_c3_sheet,          "Category3",  "energy"),
    "c4-1_운송거리":       (standardize_c4_dist_sheet,     "Category4",  "transport"),
    "c4-2_운송금액":       (standardize_c4_spend_sheet,    "Category4",  "transport"),
    "c5_사업장폐기물":      (standardize_c5_sheet,          "Category5",  "waste"),
    "c6-1_출장거리":       (standardize_c6_dist_sheet,     "Category6",  "travel"),
    "c6-2_출장비":         (standardize_c6_spend_sheet,    "Category6",  "travel"),
    "c7_임직원통근":        (standardize_c7_sheet,          "Category7",  "travel"),
    "c8-1_에너지사용량 아는 경우":  (standardize_c8_energy_sheet, "Category8",  "leased_asset"),
    "c8-2_면적":           (standardize_c8_area_sheet,     "Category8",  "leased_asset"),
    "c8-3_인원":           (standardize_c8_person_sheet,   "Category8",  "leased_asset"),
    "c9-1_운송거리":       (standardize_c9_dist_sheet,     "Category9",  "transport"),
    "c9-2_운송금액":       (standardize_c9_spend_sheet,    "Category9",  "transport"),
    "c11_판매제품사용":     (standardize_c11_sheet,         "Category11", "product_use"),
    "c12_판매제품폐기":     (standardize_c12_sheet,         "Category12", "product_eol"),
    "c13-1_에너지사용량 아는경우": (standardize_c13_energy_sheet,"Category13","downstream_leased"),
    "c13-2_면적":          (standardize_c13_area_sheet,    "Category13", "downstream_leased"),
    "c13-3_인원":          (standardize_c13_person_sheet,  "Category13", "downstream_leased"),
    "c14_프랜차이즈":       (standardize_c14_sheet,         "Category14", "franchise"),
    "c15_투자":             (standardize_c15_sheet,         "Category15", "investment"),
}

def _match_sheet_processor(sheet_name: str):
    """시트명 → (standardize_fn, category, theme) 반환. 없으면 None."""
    n = str(sheet_name).strip().lower()
    # exact match
    if n in SHEET_PROCESSOR_MAP:
        return SHEET_PROCESSOR_MAP[n]
    # prefix/contains match
    for key, val in SHEET_PROCESSOR_MAP.items():
        if key in n or n.startswith(key[:4]):
            return val
    return None


# ─────────────────────────────────────────
# 새 구매 템플릿 KNOWN_TEMPLATE_HEADERS 확장
# ─────────────────────────────────────────
KNOWN_TEMPLATE_HEADERS = KNOWN_TEMPLATE_HEADERS | {
    # C1/C2
    "국가", "사업장", "품목군", "자산분류", "상세 품목명/서비스명", "상세 자산명",
    "계정과목(선택)", "거래처명(선택)", "연간지출금액(원)", "연간취득금액(원)",
    "자산번호(선택)", "비고(선택)",
    # C3
    "연료/에너지명", "연간사용량", "월사용량",
    # C4/C9
    "운송수단(선택)", "자재명/품목명", "중량단위", "출발지", "도착지",
    "운송거리(km)(선택)", "연간운송량", "연간운송금액(원)", "월운송량",
    # C5
    "폐기물 종류", "상세 폐기물명(선택)", "처리방법", "연간처리량",
    # C6/C7
    "출장주체", "교통수단", "편도거리(km)(선택)",
    "연간이동횟수", "연간근무일수", "출장항목", "연간지출금액(원)",
    # C8/C13
    "임차자산명", "임대자산명", "에너지종류",
    "건물전체 면적", "사용 면적", "건물전체 인원수", "자사 인원수",
    "건물전체 연간사용량",
    # C11
    "보고연도", "제품명", "제품수량", "에너지명",
    "제품당 연간 에너지 사용량", "수명(개월) 혹은 평균 사용기간(개월)",
    # C12
    "폐기물 대분류(선택)", "폐기물 중분류(선택)", "폐기물 소분류(선택)",
    "판매량", "제품중량", "제품중량 단위",
    "재활용 비중(%)(선택)", "소각 비중(%)(선택)", "매립 비중(%)(선택)", "기타 비중(%)(선택)",
    # C14
    "프랜차이즈명", "가맹점 수", "연료 종류", "연료 사용량", "연료 단위",
    # C15
    "피투자기업명 (필수)", "지분율(%) (선택)",
    "피투자기업 Scope1배출량", "피투자기업 Scope2배출량",
}


# ─────────────────────────────────────────
# 구매 그룹 규칙 (C1: 구매품서비스 / C2: 자본재 기준으로 확장)
# ─────────────────────────────────────────
PURCHASE_GROUP_RULES = [
    # ── Category1: 구매품 및 서비스 (원재료/부자재/포장재/소모품/사무용품/외주/서비스) ──
    {
        "group_key": "raw_material",
        "group_ko": "원재료/부자재",
        "category": "Category1",
        "family": "raw_material",
        "keywords": ["원재료", "부자재", "원자재", "철강", "철판", "알루미늄", "플라스틱", "수지", "원료"],
    },
    {
        "group_key": "packaging",
        "group_ko": "포장재",
        "category": "Category1",
        "family": "packaging",
        "keywords": ["포장재", "포장지", "골판지", "박스", "비닐", "테이프", "완충재", "에어캡"],
    },
    {
        "group_key": "paper_a4",
        "group_ko": "A4/복사지",
        "category": "Category1",
        "family": "office_sup_paper",
        "keywords": ["a4용지", "a3용지", "복사지", "복사용지", "용지", "코팅지", "라벨지"],
    },
    {
        "group_key": "toner_ink",
        "group_ko": "토너/잉크",
        "category": "Category1",
        "family": "office_sup_printer",
        "keywords": ["토너", "잉크", "카트리지", "드럼", "복사기토너", "프린트 토너", "프린터토너"],
    },
    {
        "group_key": "office_stationery",
        "group_ko": "문구/일반사무용품",
        "category": "Category1",
        "family": "office_sup_stationery",
        "keywords": [
            "사무용품", "문구", "소모품", "스테플러", "가위", "서류봉투", "고무인",
            "도장", "결재판", "자석테이프", "책꽂이", "화이트보드", "명찰",
        ],
    },
    {
        "group_key": "pantry",
        "group_ko": "주방/탕비용품",
        "category": "Category1",
        "family": "pantry_supplies",
        "keywords": ["커피", "차", "생수", "물", "종이컵", "음료", "다과", "원두"],
    },
    {
        "group_key": "outsourcing_service",
        "group_ko": "외주가공/서비스용역",
        "category": "Category1",
        "family": "service_outsourcing",
        "keywords": [
            "외주", "외주가공", "가공비", "용역", "서비스", "컨설팅", "청소",
            "경비", "시설관리", "유지보수", "수리비",
        ],
    },
    # ── Category2: 자본재 (설비/장비/차량/건물 등 자산으로 회계처리된 항목) ──
    {
        "group_key": "manufacturing_equipment",
        "group_ko": "생산설비/기계장치",
        "category": "Category2",
        "family": "manufacturing_equipment",
        "keywords": ["생산설비", "기계장치", "프레스", "CNC", "선반", "용접기", "컨베이어"],
    },
    {
        "group_key": "vehicle",
        "group_ko": "차량",
        "category": "Category2",
        "family": "vehicle",
        "keywords": ["차량", "트럭", "승용차", "화물차", "지게차", "포크리프트"],
    },
    {
        "group_key": "building_construction",
        "group_ko": "건물/구축물",
        "category": "Category2",
        "family": "building",
        "keywords": ["건물", "구축물", "창고", "건설", "공장"],
    },
    {
        "group_key": "desk_table",
        "group_ko": "책상/보조책상",
        "category": "Category2",
        "family": "office_furniture_desk",
        "keywords": ["책상", "보조책상", "테이블"],
    },
    {
        "group_key": "chair",
        "group_ko": "의자",
        "category": "Category2",
        "family": "office_furniture_chair",
        "keywords": ["의자"],
    },
    {
        "group_key": "partition",
        "group_ko": "파티션",
        "category": "Category2",
        "family": "office_furniture_partition",
        "keywords": ["파티션"],
    },
    {
        "group_key": "cabinet_storage",
        "group_ko": "옷장/캐비닛/수납장",
        "category": "Category2",
        "family": "office_furniture_storage",
        "keywords": ["옷장", "캐비닛", "수납장"],
    },
    {
        "group_key": "computer_equipment",
        "group_ko": "컴퓨터/IT장비",
        "category": "Category2",
        "family": "computer_equipment",
        "keywords": ["노트북", "데스크탑", "컴퓨터", "pc", "모니터", "서버", "태블릿", "스마트폰"],
    },
    {
        "group_key": "printer_device",
        "group_ko": "프린터/복합기",
        "category": "Category2",
        "family": "printer_device",
        "keywords": ["프린터", "복합기", "스캐너"],
    },
    {
        "group_key": "software_license",
        "group_ko": "소프트웨어",
        "category": "Category2",
        "family": "software_it",
        "keywords": ["소프트웨어", "라이선스", "license", "software"],
    },
]


# ─────────────────────────────────────────
# 새 템플릿 기반 테마 감지 (detect_upload_theme 교체)
# ─────────────────────────────────────────
def detect_upload_theme(input_file: str) -> str:
    """새 템플릿 시트명 우선 감지. 구 템플릿이면 fallback."""
    xls = pd.ExcelFile(input_file)
    sheet_names = list(xls.sheet_names)
    lowered = [str(s).strip().lower() for s in sheet_names]

    # 새 템플릿 시트 우선 감지
    for n in lowered:
        match = _match_sheet_processor(n)
        if match:
            _, _, theme = match
            # transport는 purchase보다 먼저 감지될 수 있으므로 파일 전체 스캔
            break

    # 새 템플릿 전체 스캔으로 최우선 테마 결정
    detected_themes = set()
    for n in lowered:
        m = _match_sheet_processor(n)
        if m:
            _, _, th = m
            detected_themes.add(th)

    # 우선순위: purchase > transport > travel > 나머지
    if "purchase" in detected_themes:
        return "purchase"
    if "transport" in detected_themes:
        return "transport"
    if "travel" in detected_themes:
        return "travel"
    if detected_themes:
        return list(detected_themes)[0]

    # fallback: 구 시트명 기반
    if any(_contains_any(s, TRAVEL_BUSINESS_KEYWORDS) for s in lowered) or        any(_contains_any(s, TRAVEL_COMMUTE_KEYWORDS) for s in lowered):
        return "travel"
    if any(_contains_any(s, TRANSPORT_INBOUND_KEYWORDS) for s in lowered) or        any(_contains_any(s, TRANSPORT_OUTBOUND_KEYWORDS) for s in lowered):
        return "transport"
    if any(s in {x.lower() for x in PURCHASE_SHEET_CANDIDATES} or "구매" in s for s in lowered):
        return "purchase"

    raise ValueError(f"업로드 테마를 판별하지 못했습니다. 시트명: {sheet_names}")


# ─────────────────────────────────────────
# 새 템플릿 전용 처리 함수 - 모든 시트 일괄 처리
# ─────────────────────────────────────────
def process_new_template(input_file: str, output_file: str, report_year: int = None):
    """
    새 v1 템플릿의 모든 시트를 자동 감지하여 처리 후 단일 Excel 출력.
    report_year: 보고연도별 기계산 KRW EF 컬럼 선택 기준. None이면 FX_REFERENCE_YEAR 사용.
    """
    global FX_REFERENCE_YEAR
    if report_year is not None:
        _year = int(report_year)
    else:
        _year = FX_REFERENCE_YEAR

    FX_REFERENCE_YEAR = _year

    # C1/C2 및 Spend 방식은 통합 DB의 기계산 KRW EF 컬럼을 직접 사용
    try:
        _reload_spend_ef_dbs_for_report_year(_year)
        _krw_col = globals().get("USEPA_SPEND_EF_COL")
        print(f"보고연도: {_year}년  |  적용 배출계수: 통합 DB 기계산 KRW EF 컬럼 직접 사용")
        if _krw_col:
            print(f"- C1/C2 적용 EF 컬럼: {_krw_col}")
    except Exception as e:
        print(f"[경고] 기계산 KRW EF 재로딩 실패: {e}")
        print("       KRW EF 컬럼이 없으면 기존 USD EF fallback이 사용될 수 있습니다.")

    xls = pd.ExcelFile(input_file)
    results: Dict[str, list] = {}

    # ── 월별 합산 헬퍼 ──────────────────────────────────────────
    MONTH_COLS = [f"{i}월" for i in range(1, 13)]

    def _sum_monthly(df: pd.DataFrame, value_col: str, month_cols_in_df: list) -> pd.DataFrame:
        """
        월별 입력 행: 1월~12월 컬럼 합산 → value_col에 저장 후 월 컬럼 제거.
        연간 입력 행: value_col 그대로 유지.
        """
        df = df.copy()
        if not month_cols_in_df:
            return df
        monthly_mask = df.get("입력방식", pd.Series("연간", index=df.index)) == "월별"
        if monthly_mask.any():
            monthly_sum = df.loc[monthly_mask, month_cols_in_df].apply(
                pd.to_numeric, errors="coerce").sum(axis=1)
            # value_col이 이미 있으면 덮어쓰기, 없으면 생성
            if value_col not in df.columns:
                df[value_col] = None
            df.loc[monthly_mask, value_col] = monthly_sum.where(monthly_sum > 0, other=None)
        return df

    def _sum_monthly_spend(df: pd.DataFrame, spend_col: str) -> pd.DataFrame:
        """C1/C2: 월별 구매금액 합산 → spend_col에 저장"""
        month_cols = [c for c in df.columns if c in set(MONTH_COLS)]
        if not month_cols:
            return df
        df = df.copy()
        monthly_mask = df.get("입력방식", pd.Series("연간", index=df.index)) == "월별"
        if monthly_mask.any():
            s = df.loc[monthly_mask, month_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            if spend_col not in df.columns:
                df[spend_col] = None
            df.loc[monthly_mask, spend_col] = s.where(s > 0, other=None)
        return df
    # ─────────────────────────────────────────────────────────────

    for sheet_name in xls.sheet_names:
        match = _match_sheet_processor(sheet_name)
        if match is None:
            print(f"[skip] {sheet_name} - 처리 대상 시트 아님")
            continue

        std_fn, category, theme = match
        print(f"[processing] {sheet_name} → {category}")

        try:
            df = read_template_sheet(input_file, sheet_name)
            if df.empty:
                print(f"  [empty] {sheet_name}")
                continue
            df = std_fn(df)
            df["자동카테고리"] = category
            df["출처시트"] = sheet_name

            # 카테고리별 추가 처리
            if category in ("Category1", "Category2"):
                # 월별 구매금액 합산
                df = _sum_monthly_spend(df, "구매금액")
                rows_out = []
                for _, row in df.iterrows():
                    detail = calc_category1_2_detail(row, category)
                    row_out = _theme_merge_row(row, detail)
                    rows_out.append(row_out)
                df = pd.DataFrame(rows_out)

            elif category in ("Category4", "Category9"):
                rows_out = []
                for _, row in df.iterrows():
                    row2 = row.copy()
                    if str(row2.get("계산방식","")).strip() == "Spend":
                        # 지출기반 산정 (US EPA NAICS)
                        result = calc_transport_spend_emission(row2)
                        for k, v in result.items():
                            row2[k] = v
                    else:
                        # 거리기반 산정
                        if pd.isnull(row2.get("운송거리(km)")) and                            pd.notnull(row2.get("출발지")) and pd.notnull(row2.get("도착지")):
                            dist = get_distance_km(row2.get("출발지"), row2.get("도착지"))
                            row2["운송거리(km)"] = dist
                        ef, emission = calc_transport(row2)
                        row2["배출계수(거리기반)"] = ef
                        row2["배출량(tCO2e)"]    = emission
                    rows_out.append(row2)
                df = pd.DataFrame(rows_out)

            elif category == "Category6":
                # 월별 이동횟수 합산 (C6-1 거리기반)
                month_cols_c6 = [c for c in df.columns if c in set(MONTH_COLS)]
                if month_cols_c6:
                    df = _sum_monthly(df, "이동횟수", month_cols_c6)
                    # Spend 방식의 경우 출장비용 합산
                    df = _sum_monthly_spend(df, "출장비용")
                rows_out = []
                for _, row in df.iterrows():
                    if row.get("계산방식") == "Spend":
                        # 출장 지출기반: US EPA NAICS 기반
                        row2 = row.copy()
                        result = calc_travel_spend_emission(row2)
                        for k, v in result.items():
                            row2[k] = v
                    else:
                        dist, ef_val, emission = calc_business_travel(row)
                        row2 = row.copy()
                        if pd.isnull(row.get("이동거리(km)")) and dist is not None:
                            row2["이동거리(km)"] = dist
                        # 월별 이동횟수 반영: 합산된 이동횟수로 배출량 재계산
                        n_trips = to_float(row2.get("이동횟수"))
                        n_people = to_float(row2.get("출장인원") or row2.get("인원수")) or 1
                        dist_km = to_float(row2.get("이동거리(km)")) or dist
                        if ef_val and dist_km and n_trips:
                            emission = dist_km * 2 * ef_val * n_people * n_trips / 1000
                        row2["배출계수"] = ef_val
                        row2["배출량"] = emission
                    row2["정규화이동수단"] = normalize_travel_mode(row.get("출장수단") or row.get("이동수단"))
                    rows_out.append(row2)
                df = pd.DataFrame(rows_out)

            elif category == "Category7":
                # 월별 근무일수 합산
                month_cols_c7 = [c for c in df.columns if c in set(MONTH_COLS)]
                df = _sum_monthly(df, "근무일수", month_cols_c7)
                rows_out = []
                for _, row in df.iterrows():
                    dist, ef_val, emission = calc_commute(row)
                    row2 = row.copy()
                    if pd.isnull(row.get("평균통근거리(km)")) and dist is not None:
                        row2["평균통근거리(km)"] = dist
                    row2["배출계수"] = ef_val
                    row2["배출량"] = emission
                    row2["정규화이동수단"] = normalize_commute_mode(row.get("교통수단") or row.get("이동수단"))
                    rows_out.append(row2)
                df = pd.DataFrame(rows_out)

            elif category == "Category3":
                # C3: 연료/에너지 사용량 × 통합DB 에너지 배출계수
                # 산정식: Σ(사용량 × EF(kgCO2eq/단위)) / 1000 → tCO2e
                month_cols_c3 = [c for c in df.columns if c in set(MONTH_COLS)]
                df = _sum_monthly(df, "사용량", month_cols_c3)
                rows_out = []
                for _, row in df.iterrows():
                    row2 = row.copy()
                    energy_name = _safe_text(row2.get("에너지원") or row2.get("연료/에너지명"))
                    usage_raw = to_float(row2.get("사용량") or row2.get("월사용량"))
                    unit_raw  = _safe_text(row2.get("단위"))
                    # 통합DB 에너지 시트에서 EF 조회
                    std_name, ef_val, ef_unit = lookup_energy_ef_c3(energy_name)
                    ef_val = to_float(ef_val)
                    # 단위 환산: 입력단위 → EF 기준단위
                    # 단위 자동 환산 (경유+kWh, 도시가스+toe 등 모든 조합 지원)
                    usage, std_unit, unit_note = convert_energy_unit(
                        usage_raw, unit_raw, energy_name)
                    if std_unit and std_unit != ef_unit:
                        # 환산 후 단위가 EF 기준단위와 다르면 한 번 더 시도
                        usage2, std_unit2, note2 = convert_energy_unit(usage, std_unit, energy_name)
                        if std_unit2 == ef_unit:
                            usage, std_unit, unit_note = usage2, std_unit2, (unit_note + " → " + note2).strip(" → ")
                        elif std_name and std_name != energy_name:
                            # PATCH: DB에서 원래 연료(예: 도시가스(LNG))가 삭제되어 다른 연료의
                            # EF로 대체 매칭된 경우(예: 천연가스(LNG)), 두 연료의 발열량(LHV)을
                            # 거쳐 물리적으로 타당한 단위 환산(예: 도시가스 Nm³ → 천연가스 kg)을 시도한다.
                            bridged, bridged_unit, bridge_note = _bridge_convert_via_lhv(
                                usage, std_unit, energy_name, std_name, ef_unit)
                            if bridged_unit == ef_unit:
                                usage, std_unit, unit_note = bridged, bridged_unit, (unit_note + " → " + bridge_note).strip(" → ")
                    emission = (usage * ef_val / 1000) if (usage is not None and ef_val) else None
                    row2["배출계수(kgCO2eq/단위)"] = ef_val
                    row2["EF단위"] = ef_unit
                    row2["EF매핑명"] = std_name
                    if unit_note:
                        row2["단위환산비고"] = unit_note
                    row2["배출량(tCO2e)"] = emission
                    rows_out.append(row2)
                df = pd.DataFrame(rows_out)

            elif category == "Category5":
                # C5: 폐기물 처리량 × 환경부 폐기물 배출계수 (처리방법별)
                # 산정식: Σ(처리량(kg) × 처리방법별 배출계수(kgCO2eq/kg)) / 1000 → tCO2e
                month_cols_c5 = [c for c in df.columns if c in set(MONTH_COLS)]
                df = _sum_monthly(df, "처리량", month_cols_c5)
                rows_out = []
                for _, row in df.iterrows():
                    row2 = row.copy()
                    waste_name = _safe_text(row2.get("폐기물 종류") or row2.get("상세폐기물명"))
                    treatment  = _safe_text(row2.get("처리방법"))
                    usage_raw  = to_float(row2.get("처리량") or row2.get("월처리량"))
                    unit_str   = _safe_text(row2.get("단위") or "kg")
                    # 단위 환산 → kg (g, ton, lb 등 모든 무게 단위 지원)
                    if usage_raw is not None:
                        conv_val, std_u, ratio, _, conv_src = convert_to_base_unit(usage_raw, unit_str, "weight")
                        usage_kg = conv_val
                        if ratio != 1.0:
                            note_prefix = "AI 판단: " if conv_src == "ai" else ""
                            row2["단위환산비고"] = f"{note_prefix}{unit_str}→kg (×{ratio})"
                    else:
                        usage_kg = None

                    ef_val = lookup_waste_ef(waste_name, treatment) if waste_name else None
                    emission = (usage_kg * ef_val / 1000) if (usage_kg and ef_val) else None
                    row2["배출계수(kgCO2eq/kg)"] = ef_val
                    row2["처리방법_정규화"] = _normalize_treatment(treatment)
                    row2["배출량(tCO2e)"] = emission
                    rows_out.append(row2)
                df = pd.DataFrame(rows_out)

            elif category in ("Category8", "Category13"):
                # C8/C13: 에너지 사용량 × 통합DB 에너지 배출계수
                # 면적/인원 배분은 standardize_c8/c13 함수에서 이미 사용량 계산됨
                month_cols_c8 = [c for c in df.columns if c in set(MONTH_COLS)]
                df = _sum_monthly(df, "사용량", month_cols_c8)
                rows_out = []
                for _, row in df.iterrows():
                    row2 = row.copy()
                    energy_name = _safe_text(row2.get("에너지원") or row2.get("에너지종류"))
                    usage_raw   = to_float(row2.get("사용량"))
                    unit_raw    = _safe_text(row2.get("단위"))
                    std_name, ef_val, ef_unit = lookup_energy_ef(energy_name)
                    ef_val = to_float(ef_val)
                    usage, std_unit, unit_note = convert_energy_unit(
                        usage_raw, unit_raw, energy_name)
                    emission = (usage * ef_val / 1000) if (usage is not None and ef_val) else None
                    row2["배출계수(kgCO2eq/단위)"] = ef_val
                    row2["EF단위"] = ef_unit
                    if unit_note:
                        row2["단위환산비고"] = unit_note
                    row2["배출량(tCO2e)"] = emission
                    rows_out.append(row2)
                df = pd.DataFrame(rows_out)

            elif category == "Category12":
                # C12: 판매제품 폐기 배출계수·배출량
                # 배출계수_통합_DB.xlsx > 폐기물_C12 시트를 소분류>중분류>대분류 코드 순으로 매칭해
                # 처리방법별(매립/소각/재활용) EF를 조회하고 처리비중(%)으로 가중평균합니다.
                rows_out = []
                for _, row in df.iterrows():
                    row2 = row.copy()
                    _c12_fn = globals().get("calc_c12_ef_and_emission")
                    if _c12_fn is not None:
                        _c12_result = _c12_fn(row2)
                        row2["배출계수"] = _c12_result.get("배출계수")
                        row2["배출량(tCO2e)"] = _c12_result.get("배출량(tCO2e)")
                    else:
                        row2["배출량(tCO2e)"] = calc_c12_emission(row2)
                    rows_out.append(row2)
                df = pd.DataFrame(rows_out)

            elif category == "Category15":
                pass  # 이미 표준화 함수에서 배출량 계산됨

            if category not in results:
                results[category] = []
            results[category].append(df)

        except Exception as e:
            import traceback
            print(f"  [error] {sheet_name}: {e}")
            # 디버그 필요 시 아래 주석 해제
            # traceback.print_exc()

    # 카테고리별 병합 후 출력
    out_sheet_name_map = {
        "Category1":  "C1_구매품서비스",
        "Category2":  "C2_자본재",
        "Category3":  "C3_연료에너지",
        "Category4":  "C4_업스트림운송",
        "Category5":  "C5_사업장폐기물",
        "Category6":  "C6_출장",
        "Category7":  "C7_통근",
        "Category8":  "C8_임차자산",
        "Category9":  "C9_다운스트림운송",
        "Category11": "C11_판매제품사용",
        "Category12": "C12_판매제품폐기",
        "Category13": "C13_임대자산",
        "Category14": "C14_프랜차이즈",
        "Category15": "C15_투자",
    }

    # ── 출력에서 제거할 중간 컬럼 ──────────────────────────────
    COLS_TO_DROP = {
        # 환율 중간값 컬럼
        "입력통화", "환율기준연도", "환율적용월", "적용환율(원/USD)",
        "원화환산금액", "EF적용금액(USD)", "환율적용여부",
        # Spend 중간값
        "EF(kgCO2e/USD)", "NAICS", "NAICS_Title", "Spend방식",
        "적용환율(KRW/USD)", "USD환산금액",
        # 내부 처리용
        "자동카테고리", "출처시트",
        # 월별 컬럼 (합산 후 불필요)
        *[f"{i}월" for i in range(1, 13)],
    }

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for cat, dfs in results.items():
            merged = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            # 불필요 컬럼 제거
            drop_cols = [c for c in merged.columns if c in COLS_TO_DROP]
            merged = merged.drop(columns=drop_cols, errors="ignore")
            # 빈 컬럼 정리
            merged = merged.loc[:, merged.notna().any()]
            sname = out_sheet_name_map.get(cat, cat)[:31]
            merged.to_excel(writer, sheet_name=sname, index=False)
            print(f"  → 출력: {sname} ({len(merged)}행, {len(merged.columns)}열)")

    print(f"\n완료: {output_file}")


# ─────────────────────────────────────────
# 기존 함수 교체 패치 (하위호환)
# ─────────────────────────────────────────
def _theme_merge_row(base_row, extra):
    if isinstance(base_row, pd.Series):
        merged = base_row.to_dict()
    else:
        merged = dict(base_row)
    if isinstance(extra, pd.Series):
        extra = extra.to_dict()
    elif extra is None:
        extra = {}
    for k, v in extra.items():
        merged[k] = v
    return pd.Series(merged)

def _theme_sheet_name(sheet_name: str) -> str:
    return str(sheet_name).strip().lower()

def get_theme_engine(theme: str) -> Dict[str, str]:
    theme = str(theme).strip().lower()
    if theme not in THEME_AI_ENGINES:
        # 구 테마 fallback
        fallback = {"purchase": "purchase", "transport": "transport", "travel": "travel"}
        theme = fallback.get(theme, theme)
    if theme not in THEME_AI_ENGINES:
        raise ValueError(f"지원하지 않는 테마입니다: {theme}")
    return THEME_AI_ENGINES[theme]


# 구버전 호환: standardize_purchase_theme_sheet
def standardize_purchase_theme_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    n = _theme_sheet_name(sheet_name)
    if "c2" in n or "자본재" in n:
        return standardize_c2_sheet(df)
    return standardize_c1_sheet(df)

# 구버전 호환: standardize_transport_theme_sheet
def standardize_transport_theme_sheet(df: pd.DataFrame, fixed_category=None) -> pd.DataFrame:
    df = normalize_template_columns(df)
    rename = {}
    if "금액" in df.columns and "운송금액" not in df.columns:
        rename["금액"] = "운송금액"
    if rename:
        df = df.rename(columns=rename)
    if "발생월" in df.columns:
        df["발생월"] = df["발생월"].apply(_coerce_month_value)
    if fixed_category == "Category4":
        df["운송구분"] = "입고"
    elif fixed_category == "Category9":
        df["운송구분"] = "출고"
    elif "운송구분" not in df.columns:
        df["운송구분"] = None
    return df

# 구버전 호환: standardize_travel_business_sheet
def standardize_travel_business_sheet(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_c6_dist_sheet(df)

# 구버전 호환: standardize_travel_commute_sheet
def standardize_travel_commute_sheet(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_c7_sheet(df)


# ─────────────────────────────────────────
# 메인 엔트리포인트 (새 + 구 템플릿 모두 지원)
# ─────────────────────────────────────────
def process_template_inventory(input_file: str, output_file: str,
                               forced_theme: Optional[str] = None,
                               report_year: int = None):
    """
    새 v1 템플릿(C1~C15 시트명)이면 process_new_template 사용,
    구 템플릿이면 기존 테마 라우터 사용.
    report_year: 보고연도 (환율 기준). 예) 2024
    """
    xls = pd.ExcelFile(input_file)
    sheet_names = list(xls.sheet_names)
    lowered = [str(s).strip().lower() for s in sheet_names]

    # 새 v1 템플릿 감지: C로 시작하는 숫자 시트가 있으면 새 템플릿
    is_new_template = any(
        s.startswith("c") and len(s) > 1 and s[1].isdigit()
        for s in lowered
    )

    if is_new_template and forced_theme is None:
        print(f"새 v1 템플릿 감지 → process_new_template 실행")
        return process_new_template(input_file, output_file, report_year=report_year)

    # 구 테마 기반 처리
    theme = (forced_theme or detect_upload_theme(input_file)).strip().lower()
    engine = get_theme_engine(theme)
    processor_name = engine["processor"]
    processor = globals().get(processor_name)
    if processor is None:
        raise ValueError(f"테마 처리 함수를 찾지 못했습니다: {processor_name}")
    return processor(input_file, output_file)

def process_scope3_upload(input_file: str, output_file: str, forced_theme: Optional[str] = None, report_year: int = None):
    return process_template_inventory(input_file, output_file, forced_theme, report_year=report_year)

print("[OK] Theme router (v1 템플릿 대응) 로드 완료")
print("   새 템플릿(C1~C15 시트): process_new_template() 자동 실행")
print("   구 템플릿: 기존 테마 라우터 유지")
print("   사용법: process_template_inventory(input_file, output_file)")

# ── [원본 셀 45] ─────────────────────────────────────────────
# ============================================================
# PATCH 2026-06-11
# 사용자 검증 결과 반영
# 1) C4/C9 거리기반: 월별 운송량 합산 후 중량단위(ton/kg 등) → 화물중량(kg) 환산
# 2) C4-2/C9-2 금액기반: 시트 전체를 무조건 Spend 방식으로 처리하고 월별 금액 합산
# 3) C6-2 출장비: 월별 출장비 합산, 환율 호출 인수 순서 고정, 배출량 컬럼 일원화
# 4) C12: 드롭다운 보조목록이 데이터로 들어오는 문제 방지(헤더/빈 행 감지 강화)
# 5) C8/C13/C11: 에너지 기반 산정식 추가 및 fallback EF 적용
# ============================================================
import re
import pandas as pd
import numpy as np

# ---------- 공통 안전 헬퍼 ----------
def _patch_safe_text(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def _patch_to_float(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).replace(",", "").strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None

def _patch_month_cols(df):
    return [c for c in df.columns if str(c).strip() in {f"{i}월" for i in range(1, 13)}]

def _patch_sum_values(row, cols):
    vals = [_patch_to_float(row.get(c)) for c in cols]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals)

def _patch_pick_first(row, cols):
    for c in cols:
        if c in row.index:
            v = row.get(c)
            if _patch_safe_text(v) != "":
                return v
    return None

_WEIGHT_TO_KG_PATCH = {
    "kg": 1.0, "킬로그램": 1.0, "kilogram": 1.0, "kilograms": 1.0,
    "g": 0.001, "gram": 0.001, "grams": 0.001, "그램": 0.001,
    "ton": 1000.0, "tons": 1000.0, "t": 1000.0, "톤": 1000.0, "mt": 1000.0, "metric ton": 1000.0,
    "lb": 0.45359237, "lbs": 0.45359237, "pound": 0.45359237,
}

def _patch_weight_to_kg(value, unit):
    v = _patch_to_float(value)
    if v is None:
        return None
    u = _patch_safe_text(unit).lower().replace(" ", "")
    if u in ["", "nan"]:
        u = "kg"
    # 공백 제거로 놓치는 영문 단위 보정
    if u == "metricton":
        u = "metric ton"
    factor = _WEIGHT_TO_KG_PATCH.get(u)
    if factor is None:
        # unit에 문자열이 섞인 경우 부분 키워드로 보정
        if "ton" in u or "톤" in u:
            factor = 1000.0
        elif "kg" in u or "킬로" in u:
            factor = 1.0
        elif u == "g" or "gram" in u or "그램" in u:
            factor = 0.001
        else:
            factor = _ai_unit_conversion_factor(unit, "kg") or 1.0
    return v * factor

def _patch_collapse_monthly_to_value(df, value_col, annual_cols=None, drop_month_cols=True):
    """월별 1~12월 값을 합산하여 value_col에 채우고, 연간행은 annual_cols 값을 value_col로 이동."""
    df = df.copy()
    annual_cols = annual_cols or []
    month_cols = _patch_month_cols(df)
    if value_col not in df.columns:
        df[value_col] = None

    def _row_value(r):
        monthly_sum = _patch_sum_values(r, month_cols)
        if monthly_sum is not None and monthly_sum != 0:
            return monthly_sum
        cur = _patch_to_float(r.get(value_col))
        if cur is not None:
            return cur
        for c in annual_cols:
            v = _patch_to_float(r.get(c))
            if v is not None:
                return v
        return None

    df[value_col] = df.apply(_row_value, axis=1)
    if drop_month_cols and month_cols:
        df = df.drop(columns=month_cols, errors="ignore")
    # annual 컬럼이 value_col과 다르면 제거
    for c in annual_cols:
        if c in df.columns and c != value_col:
            df = df.drop(columns=[c], errors="ignore")
    return df

# ---------- C12 보조목록 유입 방지용 read_template_sheet 재정의 ----------
_ORIGINAL_read_template_sheet = globals().get("read_template_sheet")

def read_template_sheet(input_file, sheet_name):
    """
    v1 템플릿 시트 읽기 패치.
    - _col 계열 보조/드롭다운 목록 컬럼을 무조건 제거
    - 헤더 뒤의 helper list가 데이터로 들어오는 문제 방지
    """
    if _ORIGINAL_read_template_sheet is None:
        return pd.DataFrame()
    df = _ORIGINAL_read_template_sheet(input_file, sheet_name)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # 숨은 드롭다운 목록/보조열 제거: 헤더가 없는 _col*은 데이터가 있어도 실제 입력 컬럼이 아님
    helper_cols = [c for c in df.columns if str(c).startswith("_col")]
    if helper_cols:
        df = df.drop(columns=helper_cols, errors="ignore")
    # 완전히 빈 행 제거
    df = df.dropna(how="all").reset_index(drop=True)
    return df

# ---------- 운송 EF/운송수단 보강 ----------
DEFAULT_TRANSPORT_EF_MAP_PATCH = {
    # kgCO2e / ton-km 기준 fallback
    "road": 0.120,
    "parcel": 0.180,
    "rail": 0.030,
    "sea": 0.015,
    "air": 0.600,
}

# 기존 키워드 사전에 빠진 한국어를 보강
try:
    ROAD_KEYWORDS = list(set(list(ROAD_KEYWORDS) + ["도로", "육로", "화물", "차량", "운송차량", "트레일러"]))
    PARCEL_KEYWORDS = list(set(list(PARCEL_KEYWORDS) + ["택배", "소포", "특송", "퀵배송"]))
    RAIL_KEYWORDS = list(set(list(RAIL_KEYWORDS) + ["철도", "철송", "KTX", "SRT"]))
    SEA_KEYWORDS = list(set(list(SEA_KEYWORDS) + ["해상", "선박", "해운", "포워더", "수출", "수입", "항만"]))
    AIR_KEYWORDS = list(set(list(AIR_KEYWORDS) + ["항공", "항공화물", "항공특송", "공항"]))
except Exception:
    pass

def _patch_normalize_transport_code(code):
    s = _patch_safe_text(code).lower()
    if not s:
        return None
    if any(k in s for k in ["road", "truck", "도로", "육로", "화물", "차량"]):
        return "road"
    if any(k in s for k in ["parcel", "courier", "택배", "소포", "특송"]):
        return "parcel"
    if any(k in s for k in ["rail", "train", "철도", "철송", "ktx", "srt"]):
        return "rail"
    if any(k in s for k in ["sea", "ocean", "ship", "vessel", "해상", "선박", "해운"]):
        return "sea"
    if any(k in s for k in ["air", "flight", "항공", "공항"]):
        return "air"
    return s

# 기존 TRANSPORT_EF_MAP을 표준코드 기준으로 재정규화 + fallback 병합
try:
    _old_map = dict(globals().get("TRANSPORT_EF_MAP", {}) or {})
    _new_map = dict(DEFAULT_TRANSPORT_EF_MAP_PATCH)
    for k, v in _old_map.items():
        code = _patch_normalize_transport_code(k)
        fv = _patch_to_float(v)
        if code and fv is not None:
            _new_map[code] = fv
    TRANSPORT_EF_MAP = _new_map
except Exception:
    TRANSPORT_EF_MAP = dict(DEFAULT_TRANSPORT_EF_MAP_PATCH)

def refresh_transport_ef_map():
    """통합 DB 재로드 후 fallback/alias까지 병합."""
    global TRANSPORT_EF_MAP
    base = dict(DEFAULT_TRANSPORT_EF_MAP_PATCH)
    try:
        raw = _build_transport_ef_map_from_db()
        for k, v in (raw or {}).items():
            code = _patch_normalize_transport_code(k)
            fv = _patch_to_float(v)
            if code and fv is not None:
                base[code] = fv
    except Exception:
        pass
    TRANSPORT_EF_MAP = base
    print(f"[OK] TRANSPORT_EF_MAP 갱신: {TRANSPORT_EF_MAP}")
    return TRANSPORT_EF_MAP

def calc_transport(row):
    """
    C4/C9 거리기반 배출량.
    산식: 운송거리(km) × 화물중량(ton) × EF(kgCO2e/ton-km) / 1000 = tCO2e
    """
    mode, mode_source, mode_confidence = infer_transport_mode(row)
    mode = _patch_normalize_transport_code(mode) or "road"

    distance_km = _patch_to_float(_patch_pick_first(row, ["운송거리(km)", "운송거리(km)(선택)", "거리(km)"]))
    if distance_km is None:
        distance_km, _distance_source = infer_distance(row, mode)

    weight_kg = _patch_to_float(row.get("화물중량(kg)"))
    if weight_kg is None:
        raw_weight = _patch_pick_first(row, ["월운송량", "연간운송량", "운송량", "화물중량"])
        weight_kg = _patch_weight_to_kg(raw_weight, row.get("중량단위"))

    if distance_km is None or weight_kg is None:
        return None, None

    weight_ton = weight_kg / 1000.0
    ef = TRANSPORT_EF_MAP.get(mode)
    if ef is None:
        return None, None

    emission_t = distance_km * weight_ton * ef / 1000.0
    return ef, emission_t

# ---------- 시트 표준화 함수 재정의 ----------
def standardize_c4_dist_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C4-1 거리기반: 월별 운송량 합산 + ton/kg → 화물중량(kg) 환산."""
    df = normalize_template_columns(df)
    rename = {}
    for col in list(df.columns):
        if str(col).startswith("*"):
            rename[col] = col.lstrip("*").strip()
    if rename:
        df = df.rename(columns=rename)
        rename = {}
    if "자재명/품목명" in df.columns and "자재명" not in df.columns:
        rename["자재명/품목명"] = "자재명"
    if "제품명/품목명" in df.columns and "제품명" not in df.columns:
        rename["제품명/품목명"] = "제품명"
    if "운송수단(선택)" in df.columns and "운송수단" not in df.columns:
        rename["운송수단(선택)"] = "운송수단"
    if "출도착지" in df.columns and "출발지" not in df.columns:
        rename["출도착지"] = "출발지"
    if "운송거리(km)(선택)" in df.columns and "운송거리(km)" not in df.columns:
        rename["운송거리(km)(선택)"] = "운송거리(km)"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)

    # 월별 운송량을 연간 합계로 접어 월운송량에 저장. 연간운송량만 있으면 그대로 사용.
    df = _patch_collapse_monthly_to_value(df, "월운송량", annual_cols=["연간운송량"], drop_month_cols=True)
    if "화물중량(kg)" not in df.columns:
        df["화물중량(kg)"] = None
    df["화물중량(kg)"] = df.apply(
        lambda r: _patch_to_float(r.get("화물중량(kg)")) or _patch_weight_to_kg(
            _patch_pick_first(r, ["월운송량", "연간운송량", "운송량"]), r.get("중량단위")
        ), axis=1
    )
    df["운송구분"] = "입고"
    df["계산방식"] = df.get("계산방식", "Distance")
    df["계산방식"] = df["계산방식"].fillna("Distance") if hasattr(df["계산방식"], "fillna") else "Distance"
    return df

def standardize_c9_dist_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_c4_dist_sheet(df)
    df["운송구분"] = "출고"
    return df

def standardize_c4_spend_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C4-2 운송금액: 전체 행을 Spend로 고정하고 월별 운송금액은 연간 합계로 산정."""
    df = normalize_template_columns(df)
    rename = {}
    if "자재명/품목명" in df.columns and "자재명" not in df.columns:
        rename["자재명/품목명"] = "자재명"
    if "제품명/품목명" in df.columns and "제품명" not in df.columns:
        rename["제품명/품목명"] = "제품명"
    if "운송수단(선택)" in df.columns and "운송수단" not in df.columns:
        rename["운송수단(선택)"] = "운송수단"
    if "거래처명(선택)" in df.columns:
        rename["거래처명(선택)"] = "거래처명"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)
    df = _patch_collapse_monthly_to_value(df, "운송금액", annual_cols=["연간운송금액(원)"], drop_month_cols=True)
    df["운송구분"] = "입고"
    df["계산방식"] = "Spend"
    return df

def standardize_c9_spend_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_c4_spend_sheet(df)
    df["운송구분"] = "출고"
    df["계산방식"] = "Spend"
    return df

def standardize_c6_spend_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C6-2 출장비: 출장비 기반 시트는 무조건 Spend, 월별 지출금액 합산."""
    df = normalize_template_columns(df)
    rename = {}
    if "출장주체" in df.columns and "사업장" not in df.columns:
        rename["출장주체"] = "사업장"
    if "출장항목" in df.columns and "이동수단" not in df.columns:
        rename["출장항목"] = "이동수단"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)
    df = _patch_collapse_monthly_to_value(df, "출장비용", annual_cols=["연간지출금액(원)"], drop_month_cols=True)
    if "출장인원" not in df.columns and "인원수" in df.columns:
        df["출장인원"] = df["인원수"]
    if "출장수단" not in df.columns and "이동수단" in df.columns:
        df["출장수단"] = df["이동수단"]
    df["구분"] = "출장"
    df["계산방식"] = "Spend"
    return df

# ---------- Spend 계산 함수 보강 ----------
def _patch_get_fx(month=None, year=None):
    # 인수 순서 혼동 방지: 반드시 month, year keyword로 호출
    y = int(year) if year is not None else int(globals().get("FX_REFERENCE_YEAR", 2025))
    m = _coerce_month_value(month) if "_coerce_month_value" in globals() else None
    try:
        return get_bok_usd_krw_monthly_avg(month=m, year=y)
    except TypeError:
        return get_bok_usd_krw_monthly_avg(m, y)
    except Exception:
        return 1400.0

def calc_transport_spend_emission(row) -> dict:
    """C4/C9 운송 지출기반: 운송금액(KRW) / 환율 × EF / 1000."""
    spend_krw = _patch_to_float(_patch_pick_first(row, ["운송금액", "월운송금액", "연간운송금액(원)", "금액"]))
    mode = _patch_safe_text(_patch_pick_first(row, ["운송수단", "자재명", "제품명", "거래처명"]))
    if spend_krw is None:
        return {"배출량(tCO2e)": None, "배출량": None, "EF(kgCO2e/USD)": None, "Spend방식": "운송_Spend"}
    fx = _patch_get_fx(month=row.get("발생월"), year=globals().get("FX_REFERENCE_YEAR", 2025))
    spend_usd = spend_krw / fx if fx else spend_krw / 1400.0
    title, ef, naics = lookup_transport_spend_ef(mode)
    emission = (spend_usd * ef / 1000.0) if (ef is not None and spend_usd is not None) else None
    return {
        "배출량(tCO2e)": emission,
        "배출량": emission,
        "EF(kgCO2e/USD)": ef,
        "NAICS": naics,
        "NAICS_Title": title,
        "Spend방식": "운송_Spend",
        "적용환율(KRW/USD)": fx,
        "USD환산금액": spend_usd,
    }

def calc_travel_spend_emission(row) -> dict:
    """C6 출장 지출기반: 출장비용(KRW) / 환율 × EF / 1000."""
    spend_krw = _patch_to_float(_patch_pick_first(row, ["출장비용", "월출장비", "연간지출금액(원)", "금액"]))
    item = _patch_safe_text(_patch_pick_first(row, ["출장항목", "이동수단", "출장수단", "비고"]))
    if spend_krw is None:
        return {"배출량(tCO2e)": None, "배출량": None, "EF(kgCO2e/USD)": None, "Spend방식": "출장_Spend"}
    fx = _patch_get_fx(month=row.get("발생월"), year=globals().get("FX_REFERENCE_YEAR", 2025))
    spend_usd = spend_krw / fx if fx else spend_krw / 1400.0
    title, ef, naics = lookup_travel_spend_ef(item)
    emission = (spend_usd * ef / 1000.0) if (ef is not None and spend_usd is not None) else None
    return {
        "배출량(tCO2e)": emission,
        "배출량": emission,
        "EF(kgCO2e/USD)": ef,
        "NAICS": naics,
        "NAICS_Title": title,
        "Spend방식": "출장_Spend",
        "적용환율(KRW/USD)": fx,
        "USD환산금액": spend_usd,
    }

# ---------- 에너지 EF fallback + C8/C13/C11 계산 보강 ----------
_ORIGINAL_lookup_energy_ef = globals().get("lookup_energy_ef")
ENERGY_FALLBACK_EF_PATCH = {
    # kgCO2e / 단위 기준. 통합DB 매칭 실패 시 데모/안전 fallback으로만 사용.
    "전력": ("전력", 0.45941, "kWh"),
    "전기": ("전력", 0.45941, "kWh"),
    "electricity": ("전력", 0.45941, "kWh"),
    "도시가스": ("도시가스(LNG)", 2.23, "Nm³"),
    "도시가스(lng)": ("도시가스(LNG)", 2.23, "Nm³"),
    "천연가스": ("도시가스(LNG)", 2.23, "Nm³"),
    "lng": ("도시가스(LNG)", 2.23, "Nm³"),
    "경유": ("경유", 2.65, "L"),
    "휘발유": ("휘발유", 2.30, "L"),
    "스팀": ("열(스팀)", 56.10, "GJ"),
    "열(스팀)": ("열(스팀)", 56.10, "GJ"),
    "상수도": ("상수도", 0.332, "m³"),
    "수도": ("상수도", 0.332, "m³"),
}

def lookup_energy_ef(energy_name: str, unit_hint: str = None):
    """기존 통합DB 조회 우선, 실패 시 common energy fallback 반환."""
    if _ORIGINAL_lookup_energy_ef is not None:
        try:
            name, ef, unit = _ORIGINAL_lookup_energy_ef(energy_name, unit_hint=unit_hint)
            if ef is not None:
                return name, ef, unit
        except TypeError:
            try:
                name, ef, unit = _ORIGINAL_lookup_energy_ef(energy_name)
                if ef is not None:
                    return name, ef, unit
            except Exception:
                pass
        except Exception:
            pass
    q = _patch_safe_text(energy_name).lower()
    for key, val in ENERGY_FALLBACK_EF_PATCH.items():
        if key in q or q in key:
            return val
    return None, None, None

def _patch_convert_energy_to_ef_unit(value, input_unit, energy_name, ef_unit):
    v = _patch_to_float(value)
    if v is None:
        return None, _patch_safe_text(input_unit), ""
    in_u = _patch_safe_text(input_unit)
    target_u = _patch_safe_text(ef_unit) or in_u
    # 기존 convert_energy_unit을 먼저 사용
    try:
        conv, std_u, note = convert_energy_unit(v, in_u, energy_name)
        if not target_u or _patch_safe_text(std_u).lower() == target_u.lower():
            return conv, std_u, note
        # 기본 단위가 이미 동일하면 그대로
        if in_u.lower() == target_u.lower():
            return v, in_u, ""
        return conv, std_u, note
    except Exception:
        pass
    # 최소 fallback: 같은 단위면 그대로
    if not target_u or in_u.lower() == target_u.lower():
        return v, in_u, ""
    return v, in_u, f"단위확인 필요({in_u}→{target_u})"

def _patch_calc_energy_row(row, qty_col="사용량"):
    energy_name = _patch_safe_text(_patch_pick_first(row, ["에너지원", "에너지종류", "에너지명", "연료/에너지명"]))
    usage_raw = _patch_to_float(row.get(qty_col))
    unit_raw = _patch_safe_text(row.get("단위"))
    std_name, ef_val, ef_unit = lookup_energy_ef(energy_name, unit_hint=unit_raw)
    ef_val = _patch_to_float(ef_val)
    usage, std_unit, unit_note = _patch_convert_energy_to_ef_unit(usage_raw, unit_raw, energy_name, ef_unit)
    emission = (usage * ef_val / 1000.0) if (usage is not None and ef_val is not None) else None
    return usage, std_unit, unit_note, std_name, ef_val, ef_unit, emission

def _patch_prepare_leased_usage(df, asset_col, is_downstream=False):
    df = normalize_template_columns(df)
    rename = {}
    if asset_col in df.columns and "자산명" not in df.columns:
        rename[asset_col] = "자산명"
    if "에너지종류" in df.columns and "에너지원" not in df.columns:
        rename["에너지종류"] = "에너지원"
    if "연간사용량" in df.columns and "사용량" not in df.columns:
        rename["연간사용량"] = "사용량"
    if "건물전체 연간사용량" in df.columns and "건물전체사용량" not in df.columns:
        rename["건물전체 연간사용량"] = "건물전체사용량"
    if "건물전체 면적" in df.columns and "건물전체면적" not in df.columns:
        rename["건물전체 면적"] = "건물전체면적"
    if "사용 면적" in df.columns and "사용면적" not in df.columns:
        rename["사용 면적"] = "사용면적"
    if "건물전체 인원수" in df.columns and "건물전체인원수" not in df.columns:
        rename["건물전체 인원수"] = "건물전체인원수"
    if "자사 인원수" in df.columns and "자사인원수" not in df.columns:
        rename["자사 인원수"] = "자사인원수"
    if "비고(선택)" in df.columns:
        rename["비고(선택)"] = "비고"
    if rename:
        df = df.rename(columns=rename)
    return df

def standardize_c8_energy_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = _patch_prepare_leased_usage(df, "임차자산명")
    df = _patch_collapse_monthly_to_value(df, "사용량", annual_cols=["연간사용량"], drop_month_cols=True)
    df["임차자산여부"] = True
    df["산정방식"] = "에너지사용량"
    return df

def standardize_c13_energy_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = _patch_prepare_leased_usage(df, "임대자산명", is_downstream=True)
    df = _patch_collapse_monthly_to_value(df, "사용량", annual_cols=["연간사용량"], drop_month_cols=True)
    df["임대자산여부"] = True
    df["산정방식"] = "에너지사용량"
    return df

def _patch_standardize_area(df, asset_col, flag_col, flag_value=True):
    df = _patch_prepare_leased_usage(df, asset_col)
    df = _patch_collapse_monthly_to_value(df, "건물전체사용량", annual_cols=["건물전체사용량", "건물전체 연간사용량"], drop_month_cols=True)
    def _ratio(r):
        total = _patch_to_float(r.get("건물전체면적"))
        used = _patch_to_float(r.get("사용면적"))
        return (used / total) if total else None
    df["면적비율"] = df.apply(_ratio, axis=1)
    df["사용량"] = df.apply(
        lambda r: (_patch_to_float(r.get("건물전체사용량")) * r.get("면적비율"))
                  if (_patch_to_float(r.get("건물전체사용량")) is not None and pd.notnull(r.get("면적비율"))) else None,
        axis=1
    )
    df[flag_col] = flag_value
    df["산정방식"] = "면적배분"
    return df

def _patch_standardize_person(df, asset_col, flag_col, flag_value=True):
    df = _patch_prepare_leased_usage(df, asset_col)
    df = _patch_collapse_monthly_to_value(df, "건물전체사용량", annual_cols=["건물전체사용량", "건물전체 연간사용량"], drop_month_cols=True)
    def _ratio(r):
        total = _patch_to_float(r.get("건물전체인원수"))
        own = _patch_to_float(r.get("자사인원수"))
        return (own / total) if total else None
    df["인원비율"] = df.apply(_ratio, axis=1)
    df["사용량"] = df.apply(
        lambda r: (_patch_to_float(r.get("건물전체사용량")) * r.get("인원비율"))
                  if (_patch_to_float(r.get("건물전체사용량")) is not None and pd.notnull(r.get("인원비율"))) else None,
        axis=1
    )
    df[flag_col] = flag_value
    df["산정방식"] = "인원배분"
    return df

def standardize_c8_area_sheet(df: pd.DataFrame) -> pd.DataFrame:
    return _patch_standardize_area(df, "임차자산명", "임차자산여부", True)

def standardize_c8_person_sheet(df: pd.DataFrame) -> pd.DataFrame:
    return _patch_standardize_person(df, "임차자산명", "임차자산여부", True)

def standardize_c13_area_sheet(df: pd.DataFrame) -> pd.DataFrame:
    return _patch_standardize_area(df, "임대자산명", "임대자산여부", True)

def standardize_c13_person_sheet(df: pd.DataFrame) -> pd.DataFrame:
    return _patch_standardize_person(df, "임대자산명", "임대자산여부", True)

def standardize_c11_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C11 판매제품 사용 산정식 추가.
    산식: 제품수량 × 제품당 연간 에너지 사용량 × (수명개월/12) × EF / 1000
    """
    df = normalize_template_columns(df)
    rename = {}
    if "그룹(선택)" in df.columns:
        rename["그룹(선택)"] = "그룹"
    if "설명/적요(선택)" in df.columns:
        rename["설명/적요(선택)"] = "설명/적요"
    if "수명(개월) 혹은 평균 사용기간(개월)" in df.columns:
        rename["수명(개월) 혹은 평균 사용기간(개월)"] = "수명(개월)"
    if rename:
        df = df.rename(columns=rename)

    rows = []
    for _, row in df.iterrows():
        row2 = row.copy()
        qty = _patch_to_float(row2.get("제품수량") or row2.get("판매수량") or row2.get("판매량"))
        annual_use = _patch_to_float(row2.get("제품당 연간 에너지 사용량"))
        months = _patch_to_float(row2.get("수명(개월)"))
        years = (months / 12.0) if months is not None else 1.0
        energy_name = _patch_safe_text(row2.get("에너지명") or row2.get("에너지원"))
        unit_raw = _patch_safe_text(row2.get("단위"))
        std_name, ef_val, ef_unit = lookup_energy_ef(energy_name, unit_hint=unit_raw)
        ef_val = _patch_to_float(ef_val)
        use_per_product, std_unit, unit_note = _patch_convert_energy_to_ef_unit(annual_use, unit_raw, energy_name, ef_unit)
        total_usage = qty * use_per_product * years if (qty is not None and use_per_product is not None) else None
        emission = (total_usage * ef_val / 1000.0) if (total_usage is not None and ef_val is not None) else None
        row2["사용기간(년)"] = years
        row2["총사용량"] = total_usage
        row2["배출계수(kgCO2eq/단위)"] = ef_val
        row2["EF단위"] = ef_unit
        if unit_note:
            row2["단위환산비고"] = unit_note
        row2["배출량(tCO2e)"] = emission
        rows.append(row2)
    return pd.DataFrame(rows)

def standardize_c12_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """C12 헤더/드롭다운 보조목록 행 제거 강화."""
    df = normalize_template_columns(df)
    rename = {}
    for col in ["재활용 비중(%)(선택)", "소각 비중(%)(선택)", "매립 비중(%)(선택)", "기타 비중(%)(선택)"]:
        clean = col.replace("(선택)", "").strip()
        if col in df.columns:
            rename[col] = clean
    if "설명/적요(선택)" in df.columns:
        rename["설명/적요(선택)"] = "설명/적요"
    if rename:
        df = df.rename(columns=rename)

    # C12 시트 우측의 드롭다운 보조목록이 헤더처럼 읽힌 경우 제거
    allowed = {
        "국가", "폐기물 대분류(선택)", "폐기물 중분류(선택)", "폐기물 소분류(선택)",
        "제품명", "판매량", "제품중량", "제품중량 단위",
        "재활용 비중(%)", "소각 비중(%)", "매립 비중(%)", "기타 비중(%)",
        "설명/적요", "입력방식", "비고", "배출량(tCO2e)", "배출량"
    }
    helper_cols = []
    for c in df.columns:
        cs = _patch_safe_text(c).strip()
        if cs in allowed:
            continue
        # 예: "사업장일반 51 | 사업장일반폐기물", "51-11 | 폐토사류", "51-01-03 | 분뇨처리오니"
        if "|" in cs or re.match(r"^\d{2}(?:-\d{2}){0,2}\b", cs):
            helper_cols.append(c)
    if helper_cols:
        df = df.drop(columns=helper_cols, errors="ignore")

    # helper/dropdown rows 제거: 실제 산정 필수값이 없는 행은 제외
    required_any = ["제품명", "판매량", "제품중량"]
    present = [c for c in required_any if c in df.columns]
    if present:
        mask = df[present].notna().any(axis=1)
        # 제품명 없이 숫자만 있는 helper 행 방지: 제품명/판매량/제품중량 모두 있어야 산정 가능
        if all(c in df.columns for c in required_any):
            mask = df[required_any].notna().all(axis=1)
        df = df[mask].reset_index(drop=True)
    return df

# ---------- C12 계산식도 보강: '선택안함' 하위코드는 상위코드로 fallback ----------
_ORIGINAL_calc_c12_emission = globals().get("calc_c12_emission")

def _patch_clean_waste_code(x):
    s = _patch_safe_text(x)
    if not s or s == "-" or "선택안함" in s:
        return ""
    return s.split("|")[0].strip() if "|" in s else s

def calc_c12_emission(row) -> float:
    try:
        # 기존 함수가 동작하면 우선 사용
        if _ORIGINAL_calc_c12_emission is not None:
            val = _ORIGINAL_calc_c12_emission(row)
            if val is not None and not pd.isna(val):
                return val
    except Exception:
        pass
    qty = _patch_to_float(row.get("판매량"))
    weight = _patch_to_float(row.get("제품중량"))
    unit = _patch_safe_text(row.get("제품중량 단위") or "kg").lower()
    if qty is None or weight is None:
        return None
    if unit in ["g", "gram", "그램"]:
        weight = weight / 1000.0
    elif unit in ["ton", "t", "톤"]:
        weight = weight * 1000.0
    total_weight_kg = qty * weight

    pct_lf = _patch_to_float(row.get("매립 비중(%)") or row.get("매립비중(%)"))
    pct_inc = _patch_to_float(row.get("소각 비중(%)") or row.get("소각비중(%)"))
    pct_rec = _patch_to_float(row.get("재활용 비중(%)") or row.get("재활용비중(%)"))
    def _pct(v):
        if v is None:
            return None
        return v / 100.0 if v > 1 else v
    pct_lf, pct_inc, pct_rec = _pct(pct_lf), _pct(pct_inc), _pct(pct_rec)

    big = _patch_clean_waste_code(row.get("폐기물 대분류(선택)"))
    mid = _patch_clean_waste_code(row.get("폐기물 중분류(선택)"))
    sub = _patch_clean_waste_code(row.get("폐기물 소분류(선택)"))
    product = _patch_safe_text(row.get("제품명"))
    def _ef(treatment):
        if sub:
            return lookup_waste_ef_by_code(소분류코드=sub, treatment=treatment)
        if mid:
            return lookup_waste_ef_by_code(중분류코드=mid, treatment=treatment)
        if big:
            return lookup_waste_ef_by_code(대분류코드=big, treatment=treatment)
        if product:
            return lookup_waste_ef(product, treatment)
        return WASTE_AVG_EF.get(_normalize_treatment(treatment))
    ef_lf, ef_inc, ef_rec = _ef("매립"), _ef("소각"), _ef("재활용")
    if pct_lf is None and pct_inc is None and pct_rec is None:
        ef_avg = (WASTE_AVG_EF["매립"] + WASTE_AVG_EF["소각"] + WASTE_AVG_EF["재활용"]) / 3
        return total_weight_kg * ef_avg / 1000.0
    factor = 0.0
    if pct_lf is not None and ef_lf is not None:
        factor += ef_lf * pct_lf
    if pct_inc is not None and ef_inc is not None:
        factor += ef_inc * pct_inc
    if pct_rec is not None and ef_rec is not None:
        factor += ef_rec * pct_rec
    return total_weight_kg * factor / 1000.0

# ---------- SHEET_PROCESSOR_MAP이 기존 함수 객체를 들고 있으므로 새 함수로 재등록 ----------
try:
    SHEET_PROCESSOR_MAP.update({
        "c4-1_운송거리":       (standardize_c4_dist_sheet,     "Category4",  "transport"),
        "c4-2_운송금액":       (standardize_c4_spend_sheet,    "Category4",  "transport"),
        "c6-2_출장비":         (standardize_c6_spend_sheet,    "Category6",  "travel"),
        "c8-1_에너지사용량 아는 경우": (standardize_c8_energy_sheet, "Category8",  "energy"),
        "c8-2_면적":           (standardize_c8_area_sheet,     "Category8",  "energy"),
        "c8-3_인원":           (standardize_c8_person_sheet,   "Category8",  "energy"),
        "c9-1_운송거리":       (standardize_c9_dist_sheet,     "Category9",  "transport"),
        "c9-2_운송금액":       (standardize_c9_spend_sheet,    "Category9",  "transport"),
        "c11_판매제품사용":     (standardize_c11_sheet,         "Category11", "product_use"),
        "c12_판매제품폐기":     (standardize_c12_sheet,         "Category12", "product_eol"),
        "c13-1_에너지사용량 아는경우": (standardize_c13_energy_sheet, "Category13", "energy"),
        "c13-2_면적":          (standardize_c13_area_sheet,    "Category13", "energy"),
        "c13-3_인원":          (standardize_c13_person_sheet,  "Category13", "energy"),
    })
except Exception:
    pass

print("[OK] 사용자 검증 반영 패치 로드 완료")
print("   - C4/C9 거리·금액, C6 Spend, C12 헤더/보조목록, C8/C13/C11 계산식 보강")

# ── [원본 셀 47] ─────────────────────────────────────────────
# ============================================================
# PATCH 2026-06-11B
# 템플릿 표시명 ↔ 배출계수 DB raw명 매핑 보강
# ------------------------------------------------------------
# 목적
# - 템플릿 드롭다운/입력값은 사용자 친화명(전기, LNG, LPG, 외항선 등)으로 유지
# - 배출계수 DB 조회 시 raw명/표준코드(전력, 도시가스(LNG), 액화석유가스(LPG), sea 등)로 자동 변환
# - 매핑 실패 시 exact → alias → 부분일치 → fuzzy 순서로 후보를 찾고,
#   출력에 EF매핑명/EF매핑방식/EF매핑점수를 남겨 검증 가능하게 함
# ============================================================
import re
import unicodedata
from difflib import SequenceMatcher
import pandas as pd


# ----------------------------------------------------------------
# 0) 통합 DB 헤더 행 자동 보정
# ----------------------------------------------------------------
def _ef_auto_header_row(xlsx_path, sheet_name, must_contain=None, scan_rows=10):
    """시트 상단 제목/설명 행 때문에 header_row가 틀어지는 것을 방지."""
    must_contain = must_contain or []
    try:
        preview = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, nrows=scan_rows)
        best_row, best_score = 0, -1
        for i in range(len(preview)):
            vals = [str(v).strip() for v in preview.iloc[i].tolist() if pd.notna(v) and str(v).strip()]
            joined = " ".join(vals)
            score = len(vals)
            for kw in must_contain:
                if kw in joined:
                    score += 20
            if score > best_score:
                best_row, best_score = i, score
        return best_row
    except Exception:
        return None


def _ef_reload_sheet_auto(sheet_name, must_contain=None, str_cols=None):
    """_EF_DB_FILE에서 시트별 실제 헤더 행을 찾아 재로드."""
    path = globals().get("_EF_DB_FILE")
    if not path:
        try:
            path = os.path.join(base_path, "배출계수_통합_DB.xlsx")
        except Exception:
            path = None
    if not path:
        return pd.DataFrame()
    header_row = _ef_auto_header_row(path, sheet_name, must_contain=must_contain)
    if header_row is None:
        return pd.DataFrame()
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        for col in df.columns:
            if str_cols and col in str_cols:
                continue
            df[col] = _to_numeric_ignore(df[col])
        if "_fix_cached_krw_ef_columns" in globals():
            df = _fix_cached_krw_ef_columns(df)
        return df.dropna(how="all").reset_index(drop=True)
    except Exception as e:
        print(f"[경고] {sheet_name} 자동 재로드 실패: {e}")
        return pd.DataFrame()


def refresh_integrated_ef_db_mapping_patch():
    """현재 통합 DB를 헤더 자동감지 방식으로 다시 읽어 매핑 누락을 줄임."""
    global energy_ef_db, transport_ef_db, travel_ef_db, waste_c5_db, waste_c12_db, transport_spend_db, travel_spend_db
    # 에너지 시트는 제목/설명/분류 행이 있어 기존 header_row=2로 읽으면 컬럼이 틀어질 수 있음
    e = _ef_reload_sheet_auto("에너지", must_contain=["활동유형", "CO2eq", "기준단위"])
    if not e.empty:
        energy_ef_db = e
    # 나머지 시트도 안전하게 헤더 자동감지로 갱신 가능하면 갱신
    t = _ef_reload_sheet_auto("운송", must_contain=["표준코드", "배출계수"])
    if not t.empty:
        transport_ef_db = t
    tr = _ef_reload_sheet_auto("출장_통근", must_contain=["표준코드", "EF"])
    if not tr.empty:
        travel_ef_db = tr
    w5 = _ef_reload_sheet_auto("폐기물_C5", must_contain=["처리방법", "배출계수"])
    if not w5.empty:
        waste_c5_db = w5
    w12 = _ef_reload_sheet_auto("폐기물_C12", must_contain=["폐기물 명칭", "매립 EF"], str_cols=["대분류\n코드","중분류\n코드","소분류\n코드"])
    if not w12.empty:
        waste_c12_db = w12
    ts = _ef_reload_sheet_auto("운송_Spend", must_contain=["운송 구분", "NAICS", "EF"])
    if not ts.empty:
        transport_spend_db = ts
    tvs = _ef_reload_sheet_auto("출장_Spend", must_contain=["출장 항목", "NAICS", "EF"])
    if not tvs.empty:
        travel_spend_db = tvs
    return {
        "energy_ef_db": len(energy_ef_db) if "energy_ef_db" in globals() else 0,
        "transport_ef_db": len(transport_ef_db) if "transport_ef_db" in globals() else 0,
        "travel_ef_db": len(travel_ef_db) if "travel_ef_db" in globals() else 0,
        "waste_c5_db": len(waste_c5_db) if "waste_c5_db" in globals() else 0,
        "waste_c12_db": len(waste_c12_db) if "waste_c12_db" in globals() else 0,
        "transport_spend_db": len(transport_spend_db) if "transport_spend_db" in globals() else 0,
        "travel_spend_db": len(travel_spend_db) if "travel_spend_db" in globals() else 0,
    }

try:
    _reload_summary = refresh_integrated_ef_db_mapping_patch()
    print(f"[OK] 통합 DB 헤더 자동 보정/재로드 완료: {_reload_summary}")
except Exception as e:
    print(f"[경고] 통합 DB 자동 재로드 스킵: {e}")

if "_LAST_EF_MATCH_INFO" not in globals():
    _LAST_EF_MATCH_INFO = {}

# 재실행 안전장치: 이 셀을 여러 번 실행해도 원본 함수 참조가 재귀로 꼬이지 않도록 최초 1회만 보관
if "_LOOKUP_ENERGY_EF_BEFORE_ALIAS_PATCH" not in globals():
    _LOOKUP_ENERGY_EF_BEFORE_ALIAS_PATCH = globals().get("lookup_energy_ef")
if "_INFER_TRANSPORT_MODE_BEFORE_ALIAS_PATCH" not in globals():
    _INFER_TRANSPORT_MODE_BEFORE_ALIAS_PATCH = globals().get("infer_transport_mode")
if "_PATCH_NORMALIZE_TRANSPORT_CODE_BEFORE_ALIAS_PATCH" not in globals():
    _PATCH_NORMALIZE_TRANSPORT_CODE_BEFORE_ALIAS_PATCH = globals().get("_patch_normalize_transport_code")
if "_LOOKUP_TRANSPORT_SPEND_EF_BEFORE_ALIAS_PATCH" not in globals():
    _LOOKUP_TRANSPORT_SPEND_EF_BEFORE_ALIAS_PATCH = globals().get("lookup_transport_spend_ef")
if "_STANDARDIZE_C11_BEFORE_ALIAS_PATCH" not in globals():
    _STANDARDIZE_C11_BEFORE_ALIAS_PATCH = globals().get("standardize_c11_sheet")
if "_STANDARDIZE_C14_BEFORE_ALIAS_PATCH" not in globals():
    _STANDARDIZE_C14_BEFORE_ALIAS_PATCH = globals().get("standardize_c14_sheet")


# ----------------------------------------------------------------
# 1) 공통 정규화
# ----------------------------------------------------------------
def _ef_safe_text(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def _ef_norm_text(v):
    """비교용 문자열 정규화: 대소문자/공백/기호/괄호 차이를 제거."""
    s = _ef_safe_text(v)
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("㎥", "m3").replace("m³", "m3").replace("nm³", "nm3")
    s = s.replace("co₂", "co2").replace("co₂e", "co2e")
    # 괄호는 내부 텍스트는 살리고 괄호 기호만 제거
    s = re.sub(r"[\(\)\[\]\{\}]", "", s)
    s = re.sub(r"[\s_\-\/,·・|]+", "", s)
    return s


def _ef_norm_unit(unit):
    s = _ef_norm_text(unit)
    if not s:
        return ""
    unit_map = {
        "kwh": "kwh",
        "kwhr": "kwh",
        "kw": "kw",
        "wh": "wh",
        "gj": "gj",
        "mj": "mj",
        "tj": "tj",
        "l": "l",
        "liter": "l",
        "litre": "l",
        "리터": "l",
        "kg": "kg",
        "킬로그램": "kg",
        "ton": "ton",
        "t": "ton",
        "톤": "ton",
        "m3": "m3",
        "nm3": "nm3",
        "n㎥": "nm3",
        "n루베": "nm3",
        "루베": "m3",
        "㎥": "m3",
    }
    return unit_map.get(s, s)


def _ef_find_col(df, predicates):
    for c in getattr(df, "columns", []):
        cs = str(c)
        if all(p in cs for p in predicates):
            return c
    return None


def _ef_first_existing_col(df, candidate_keywords):
    for keywords in candidate_keywords:
        col = _ef_find_col(df, keywords if isinstance(keywords, (list, tuple)) else [keywords])
        if col:
            return col
    return None


def _ef_similarity(a, b):
    a, b = _ef_norm_text(a), _ef_norm_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    return SequenceMatcher(None, a, b).ratio()


# ----------------------------------------------------------------
# 2) 에너지: 템플릿 표시명 → DB raw 활동유형명
# ----------------------------------------------------------------
def _energy_db_cols():
    db = globals().get("energy_ef_db", pd.DataFrame())
    if db is None or getattr(db, "empty", True):
        return db, None, None, None
    name_col = next((c for c in db.columns if "활동유형" in str(c) or "에너지원" in str(c)), None)
    ef_col = (
        next((c for c in db.columns if "CO2eq" in str(c) and "★" in str(c)), None)
        or next((c for c in db.columns if "CO2eq" in str(c)), None)
    )
    unit_col = next((c for c in db.columns if "기준단위" in str(c) or ("단위" in str(c) and "발열량" not in str(c))), None)
    return db, name_col, ef_col, unit_col


def _energy_alias_candidates(energy_name, unit_hint=None):
    """
    사용자 친화명/약어를 DB raw명 후보로 확장.
    예: 전기 → 전력, LNG+Nm³ → 도시가스(LNG), LNG+kg → 천연가스(LNG)
    """
    raw = _ef_safe_text(energy_name)
    q = _ef_norm_text(raw)
    u = _ef_norm_unit(unit_hint)
    candidates = []

    def add(*vals):
        for v in vals:
            if v and v not in candidates:
                candidates.append(v)

    add(raw)

    # 전력
    if any(k in q for k in ["전기", "전력", "electricity", "electric", "power", "gridpower"]):
        add("전력")

    # 열/스팀
    if any(k in q for k in ["스팀", "steam", "열스팀", "열"]):
        add("열(스팀)")

    # LNG / 천연가스 / 도시가스
    if "cng" in q or "압축천연가스" in q:
        add("CNG(차량)")
    if any(k in q for k in ["lng", "천연가스", "도시가스", "액화천연가스", "naturalgas"]):
        if any(k in q for k in ["차량", "자동차", "vehicle", "truck", "bus"]):
            add("LNG(차량)")
        elif u == "kg" or "천연가스" in q:
            add("천연가스(LNG)", "도시가스(LNG)")
        else:
            add("도시가스(LNG)", "천연가스(LNG)")

    # LPG
    if any(k in q for k in ["lpg", "액화석유가스", "프로판", "부탄"]):
        if any(k in q for k in ["차량", "자동차", "vehicle", "truck", "bus"]):
            add("LPG(차량)")
        elif "도시가스" in q or u in {"nm3", "m3"}:
            add("도시가스(LPG)", "액화석유가스(LPG)")
        else:
            add("액화석유가스(LPG)", "도시가스(LPG)", "LPG(차량)")

    # 액체연료
    if any(k in q for k in ["경유", "diesel", "dieseloil"]):
        add("경유")
    if any(k in q for k in ["휘발유", "가솔린", "gasoline", "petrol"]):
        add("휘발유")
    if any(k in q for k in ["등유", "kerosene"]):
        # 용도 미상일 때는 보일러 등유를 우선 후보로 두되 실내 등유도 후보로 둠
        add("보일러 등유", "실내 등유")
    if any(k in q for k in ["항공유", "jetfuel", "jet", "aviationfuel"]):
        add("항공유")

    return candidates


def _unit_match_score(row_unit, unit_hint):
    ru = _ef_norm_unit(row_unit)
    uh = _ef_norm_unit(unit_hint)
    if not uh or not ru:
        return 0.0
    if ru == uh:
        return 0.08
    # Nm³/m³는 사용자 입력에서 혼용될 수 있어 약한 보너스
    if {ru, uh} <= {"nm3", "m3"}:
        return 0.04
    return -0.05


def resolve_energy_ef_row(energy_name, unit_hint=None):
    """
    반환: (row, match_type, score)
    match_type:
      - exact_raw
      - alias_exact
      - alias_contains
      - contains
      - fuzzy
    """
    db, name_col, ef_col, unit_col = _energy_db_cols()
    if db is None or getattr(db, "empty", True) or not name_col or not ef_col:
        return None, "db_empty_or_columns_missing", 0.0

    work = db.dropna(subset=[name_col]).copy()
    if work.empty:
        return None, "db_empty", 0.0

    q = _ef_norm_text(energy_name)
    candidates = _energy_alias_candidates(energy_name, unit_hint=unit_hint)

    def _sort_subset(subset, base_score):
        if subset.empty:
            return None, 0.0
        tmp = subset.copy()
        if unit_col:
            tmp["_unit_bonus"] = tmp[unit_col].apply(lambda x: _unit_match_score(x, unit_hint))
        else:
            tmp["_unit_bonus"] = 0.0
        tmp["_score"] = base_score + tmp["_unit_bonus"]
        tmp = tmp.sort_values("_score", ascending=False)
        return tmp.iloc[0], float(tmp.iloc[0]["_score"])

    # 1) raw exact / alias exact
    for idx, cand in enumerate(candidates):
        cn = _ef_norm_text(cand)
        subset = work[work[name_col].apply(lambda x: _ef_norm_text(x) == cn)]
        row, score = _sort_subset(subset, 1.0 if idx == 0 else 0.97)
        if row is not None:
            return row, "exact_raw" if idx == 0 else "alias_exact", min(score, 1.0)

    # 2) alias contains
    for cand in candidates:
        cn = _ef_norm_text(cand)
        if not cn:
            continue
        subset = work[work[name_col].apply(lambda x: cn in _ef_norm_text(x) or _ef_norm_text(x) in cn)]
        row, score = _sort_subset(subset, 0.90)
        if row is not None:
            return row, "alias_contains", min(score, 0.98)

    # 3) query contains
    if q:
        subset = work[work[name_col].apply(lambda x: q in _ef_norm_text(x) or _ef_norm_text(x) in q)]
        row, score = _sort_subset(subset, 0.86)
        if row is not None:
            return row, "contains", min(score, 0.95)

    # 4) fuzzy fallback
    scored = []
    for _, r in work.iterrows():
        base = _ef_similarity(energy_name, r.get(name_col))
        score = base + (_unit_match_score(r.get(unit_col), unit_hint) if unit_col else 0)
        scored.append((score, r))
    if scored:
        score, row = max(scored, key=lambda x: x[0])
        if score >= 0.72:
            return row, "fuzzy", min(float(score), 0.93)

    return None, "not_found", 0.0


def lookup_energy_ef(energy_name: str, unit_hint: str = None):
    """
    에너지원 EF 조회 강화 버전.
    템플릿 표시명을 DB raw명으로 먼저 해석한 뒤, 실패하면 기존 lookup/fallback을 사용.
    """
    db, name_col, ef_col, unit_col = _energy_db_cols()

    row, match_type, score = resolve_energy_ef_row(energy_name, unit_hint=unit_hint)
    if row is not None and ef_col:
        ef = pd.to_numeric(row.get(ef_col), errors="coerce")
        raw_name = row.get(name_col) if name_col else None
        raw_unit = row.get(unit_col) if unit_col else None

        # PATCH: DB "투입량 기준단위"가 TJ인 항목(예: 열(스팀) = 59,685 kgCO2eq/TJ)은
        # ENERGY_BASE_UNIT/convert_energy_unit이 전부 GJ 기준으로 사용량을 환산하므로,
        # EF를 여기서 GJ 기준으로 정규화하지 않으면 사용량(GJ) × EF(/TJ)가 1000배 부풀려진다.
        if pd.notna(ef) and raw_unit and str(raw_unit).strip().upper() == "TJ":
            ef = ef / 1000.0
            raw_unit = "GJ"

        _LAST_EF_MATCH_INFO["energy"] = {
            "input": _ef_safe_text(energy_name),
            "unit_hint": _ef_safe_text(unit_hint),
            "raw_name": raw_name,
            "raw_unit": raw_unit,
            "match_type": match_type,
            "score": round(float(score), 4),
        }
        if pd.notna(ef):
            return raw_name, float(ef), raw_unit

    # 기존 함수 fallback
    original = globals().get("_LOOKUP_ENERGY_EF_BEFORE_ALIAS_PATCH")
    if original is not None:
        try:
            name, ef, unit = original(energy_name, unit_hint=unit_hint)
        except TypeError:
            name, ef, unit = original(energy_name)
        except Exception:
            name, ef, unit = None, None, None

        if ef is not None:
            _LAST_EF_MATCH_INFO["energy"] = {
                "input": _ef_safe_text(energy_name),
                "unit_hint": _ef_safe_text(unit_hint),
                "raw_name": name,
                "raw_unit": unit,
                "match_type": "legacy_lookup",
                "score": None,
            }
            return name, ef, unit

    _LAST_EF_MATCH_INFO["energy"] = {
        "input": _ef_safe_text(energy_name),
        "unit_hint": _ef_safe_text(unit_hint),
        "raw_name": None,
        "raw_unit": None,
        "match_type": match_type,
        "score": round(float(score), 4) if score is not None else None,
    }
    return None, None, None


def _append_last_energy_mapping_columns(row_like):
    """최근 lookup_energy_ef 결과를 출력 row에 기록."""
    info = (_LAST_EF_MATCH_INFO or {}).get("energy", {}) or {}
    try:
        row_like["EF매핑명"] = info.get("raw_name")
        row_like["EF매핑방식"] = info.get("match_type")
        row_like["EF매핑점수"] = info.get("score")
    except Exception:
        pass
    return row_like


def annotate_energy_mapping_columns(df, energy_cols=("에너지원", "에너지종류", "에너지명", "연료/에너지명", "연료 종류"), unit_col="단위"):
    """이미 계산된 DataFrame에도 EF매핑명/방식/점수를 일괄 추가."""
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    if "EF매핑명" not in out.columns:
        out["EF매핑명"] = None
    if "EF매핑방식" not in out.columns:
        out["EF매핑방식"] = None
    if "EF매핑점수" not in out.columns:
        out["EF매핑점수"] = None

    for idx, row in out.iterrows():
        energy = ""
        for c in energy_cols:
            if c in out.columns and _ef_safe_text(row.get(c)):
                energy = _ef_safe_text(row.get(c))
                break
        if not energy:
            continue
        unit = row.get(unit_col) if unit_col in out.columns else None
        lookup_energy_ef(energy, unit_hint=unit)
        info = (_LAST_EF_MATCH_INFO or {}).get("energy", {}) or {}
        out.at[idx, "EF매핑명"] = info.get("raw_name")
        out.at[idx, "EF매핑방식"] = info.get("match_type")
        out.at[idx, "EF매핑점수"] = info.get("score")
    return out


# C11은 standardize 단계에서 계산되므로, 기존 함수 결과에 매핑 검증 컬럼을 보강
def standardize_c11_sheet(df: pd.DataFrame) -> pd.DataFrame:
    original = globals().get("_STANDARDIZE_C11_BEFORE_ALIAS_PATCH")
    if original is not None:
        out = original(df)
    else:
        out = normalize_template_columns(df)
    return annotate_energy_mapping_columns(out, energy_cols=("에너지명", "에너지원"), unit_col="단위")


# C14는 사용자 입력 컬럼명이 "연료 종류/연료 단위"라서 에너지원/단위로 표준화한 뒤 매핑 컬럼 보강
def standardize_c14_sheet(df: pd.DataFrame) -> pd.DataFrame:
    original = globals().get("_STANDARDIZE_C14_BEFORE_ALIAS_PATCH")
    if original is not None:
        out = original(df)
    else:
        out = normalize_template_columns(df)
        rename = {}
        if "연료 종류" in out.columns and "에너지원" not in out.columns:
            rename["연료 종류"] = "에너지원"
        if "연료 사용량" in out.columns and "사용량" not in out.columns:
            rename["연료 사용량"] = "사용량"
        if "연료 단위" in out.columns and "단위" not in out.columns:
            rename["연료 단위"] = "단위"
        if rename:
            out = out.rename(columns=rename)
    return annotate_energy_mapping_columns(out, energy_cols=("에너지원", "연료 종류"), unit_col="단위")


# ----------------------------------------------------------------
# 3) 운송: 템플릿 표시명 → 운송 표준코드
# ----------------------------------------------------------------
def _template_transport_code(text):
    s = _ef_norm_text(text)
    if not s:
        return None

    # 더 구체적인 항목을 먼저 판정
    if any(k in s for k in ["택배", "소화물", "특송", "courier", "parcel", "express"]):
        return "parcel"
    if any(k in s for k in ["벌크", "bulk", "drybulk", "bulkcarrier"]):
        return "sea_bulk"
    if any(k in s for k in ["내륙수로", "inlandwater", "riverfreight", "barge"]):
        return "inland_water"
    if any(k in s for k in ["외항선", "컨테이너선", "해상", "선박", "해운", "배", "항만", "ocean", "sea", "vessel", "ship", "marine", "container"]):
        return "sea"
    if any(k in s for k in ["항공화물", "항공", "공항", "aircargo", "airfreight", "air", "flight", "awb"]):
        return "air"
    if any(k in s for k in ["철도", "철송", "기차", "열차", "rail", "railway", "train", "ktx", "srt"]):
        return "rail"
    if any(k in s for k in ["도로", "육상", "육로", "트럭", "화물차", "윙바디", "탑차", "차량", "truck", "lorry", "road", "van"]):
        return "road"
    return None


def _patch_normalize_transport_code(code):
    mapped = _template_transport_code(code)
    if mapped:
        return mapped

    original = globals().get("_PATCH_NORMALIZE_TRANSPORT_CODE_BEFORE_ALIAS_PATCH")
    if original is not None:
        try:
            return original(code)
        except Exception:
            pass

    s = _ef_safe_text(code).strip()
    return s or None


@lru_cache(maxsize=512)
def _ai_infer_transport_mode(text: str):
    """
    rule/keyword 테이블로 운송수단을 판별하지 못했을 때(비어있지 않은 텍스트가 있는 경우) AI로 판단.
    실패하거나 유효하지 않은 값이면 None.
    """
    prompt = f"""다음은 화물 운송 관련 텍스트(운송수단/적요/거래처명 등)이며, 규칙 기반 키워드로는 운송수단을 판별하지 못했다.
텍스트: {text}

아래 표준 코드 중 가장 적절한 것 하나를 선택하라: road, parcel, rail, sea, air, sea_bulk, inland_water
JSON으로만 답하라: {{"mode": "<code>"}}"""
    parsed = _call_claude_json(prompt, max_tokens=40)
    mode = _ef_safe_text(parsed.get("mode")) if parsed else ""
    valid = {"road", "parcel", "rail", "sea", "air", "sea_bulk", "inland_water"}
    return mode if mode in valid else None

def infer_transport_mode(row):
    """
    기존 infer_transport_mode 보강:
    템플릿의 운송수단/운송방법/설명/자재명 등에서 사용자 친화 운송명을 먼저 표준코드로 매핑.
    """
    fields = [
        "운송수단", "운송방법", "교통수단", "운송구분", "운송수단(선택)",
        "설명/적요", "비고", "자재명", "품목명", "제품명"
    ]
    parts = []
    for f in fields:
        try:
            v = row.get(f)
        except Exception:
            v = None
        if _ef_safe_text(v):
            parts.append(_ef_safe_text(v))
    joined = " ".join(parts)

    code = _template_transport_code(joined)
    if code:
        return code, "template_alias", 0.95

    original = globals().get("_INFER_TRANSPORT_MODE_BEFORE_ALIAS_PATCH")
    if original is not None:
        try:
            mode, src, conf = original(row)
            mode2 = _patch_normalize_transport_code(mode)
            mode2 = mode2 if mode2 and mode2 != "unknown" else None
            if mode2:
                return mode2, src, conf
        except Exception:
            pass

    if joined:
        ai_mode = _ai_infer_transport_mode(joined)
        if ai_mode:
            return ai_mode, "ai", 0.70

    return "road", "fallback", 0.50


# 운송 지출기반도 표시명을 raw 키워드로 확장해서 조회
_TRANSPORT_CODE_TO_SPEND_HINT = {
    "road": "도로 트럭 화물 truck freight",
    "parcel": "택배 소화물 courier parcel express",
    "rail": "철도 기차 rail train",
    "sea": "해상 선박 ocean freight sea freight vessel",
    "sea_bulk": "벌크선 bulk ocean freight sea freight",
    "inland_water": "내륙수로 river freight inland water",
    "air": "항공화물 항공 air freight air cargo flight",
}

def lookup_transport_spend_ef(mode_or_item: str) -> tuple:
    original = globals().get("_LOOKUP_TRANSPORT_SPEND_EF_BEFORE_ALIAS_PATCH")
    q = _ef_safe_text(mode_or_item)
    code = _template_transport_code(q)
    expanded = q
    if code:
        expanded = f"{q} {_TRANSPORT_CODE_TO_SPEND_HINT.get(code, '')}".strip()

    if original is not None:
        try:
            title, ef, naics = original(expanded)
            if ef is not None:
                return title, ef, naics
        except Exception:
            pass
        # 원문으로 한 번 더
        try:
            title, ef, naics = original(q)
            if ef is not None:
                return title, ef, naics
        except Exception:
            pass

    return None, None, None


# TRANSPORT_EF_MAP을 새 alias 로직 기준으로 재정규화
try:
    refresh_transport_ef_map()
except Exception:
    pass

# SHEET_PROCESSOR_MAP이 이전 함수 객체를 들고 있을 수 있으므로 C11/C14 재등록
try:
    SHEET_PROCESSOR_MAP.update({
        "c11_판매제품사용": (standardize_c11_sheet, "Category11", "sold_product_use"),
        "c14_프랜차이즈": (standardize_c14_sheet, "Category14", "franchise"),
    })
except Exception:
    pass

print("[OK] 템플릿 표시명 ↔ 배출계수 raw명 매핑 패치 적용 완료")
print("   - 에너지: 전기/전력/LNG/LPG/스팀/등유 등 사용자 친화명 → DB raw명 자동 매핑")
print("   - 운송: 외항선/컨테이너선/벌크/택배/항공화물 등 → 표준코드 자동 매핑")
print("   - 출력 검증 컬럼: EF매핑명, EF매핑방식, EF매핑점수")

# ── [원본 셀 49] ─────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════
# PATCH 2026-06-11C
# 기계산 KRW 배출계수 직접 매핑
# - C1/C2 구매_USEPA: EF(kgCO2e/KRW) {보고연도} 컬럼 직접 사용
# - C4/C6 Spend: 운송_Spend/출장_Spend에 KRW EF 컬럼이 있으면 우선 사용
# - 기존 방식(KRW → USD 환산 → USD EF 적용) 제거
# ══════════════════════════════════════════════════════════════
import re
import os
import pandas as pd


def _header_norm_for_ef(col) -> str:
    """엑셀 헤더 비교용 정규화: 공백/줄바꿈 제거 + 소문자."""
    return re.sub(r"\s+", "", str(col)).lower()


def _is_precomputed_krw_ef_col(col) -> bool:
    """'EF (kgCO2e/KRW) 2025년말 기준 ...' 유형 컬럼 판별."""
    t = _header_norm_for_ef(col)
    return (
        "ef" in t
        and (
            "kgco2e/krw" in t
            or "kgco2eq/krw" in t
            or "kgco₂e/krw" in t
            or "kgco₂eq/krw" in t
        )
    )


def _extract_years_from_header(col):
    years = re.findall(r"20\d{2}", str(col))
    return [int(y) for y in years]


def _find_precomputed_krw_ef_col(df: pd.DataFrame, report_year=None):
    """
    보고연도에 맞는 기계산 KRW EF 컬럼 선택.
    예: EF (kgCO2e/KRW)\n2025년말 기준\n(... KRW/USD)
    """
    if df is None or df.empty:
        return None

    candidates = [c for c in df.columns if _is_precomputed_krw_ef_col(c)]
    if not candidates:
        return None

    if report_year is not None:
        y = str(int(report_year))
        year_matches = [c for c in candidates if y in str(c)]
        if year_matches:
            return year_matches[0]

    # 보고연도 컬럼이 없으면 가장 최신 연도 컬럼 사용
    def _latest_year(c):
        ys = _extract_years_from_header(c)
        return max(ys) if ys else -1

    return sorted(candidates, key=_latest_year, reverse=True)[0]


def _find_usd_ef_col(df: pd.DataFrame):
    """fallback용 USD EF 컬럼."""
    if df is None or df.empty:
        return None
    for c in df.columns:
        t = _header_norm_for_ef(c)
        if "ef" in t and ("kgco2e/usd" in t or "kgco2eq/usd" in t):
            return c
    return next((c for c in df.columns if "EF" in str(c) and "USD" in str(c) and "KRW" not in str(c)), None)


def _read_integrated_db_sheet(sheet_name: str, header_row: int = 2) -> pd.DataFrame:
    """통합 DB 시트 로드. Colab Drive의 base_path 기준."""
    path = globals().get("_EF_DB_FILE")
    if not path:
        path = os.path.join(base_path, "배출계수_통합_DB.xlsx")
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    if "_fix_cached_krw_ef_columns" in globals():
        df = _fix_cached_krw_ef_columns(df)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _load_purchase_usepa_as_krw_spend_db(report_year=None) -> pd.DataFrame:
    """
    구매_USEPA 시트를 spend_db 형식으로 변환하되,
    ef는 USD가 아니라 보고연도별 기계산 KRW EF(kgCO2e/KRW)를 사용.
    """
    year = int(report_year or globals().get("FX_REFERENCE_YEAR", 2025))
    raw = _read_integrated_db_sheet("구매_USEPA", header_row=2)

    title_col = next((c for c in raw.columns if "NAICS Title" in c or "Title" in c), None)
    code_col  = next((c for c in raw.columns if "NAICS 코드" in c or ("NAICS" in c and "코드" in c)), None)
    cat_col   = next((c for c in raw.columns if "품목군" in c), None)
    sub_col   = next((c for c in raw.columns if "상세 카테고리" in c or "카테고리" in c), None)
    kw_col    = next((c for c in raw.columns if "키워드" in c), None)
    krw_col   = _find_precomputed_krw_ef_col(raw, year)
    usd_col   = _find_usd_ef_col(raw)

    if not title_col:
        raise ValueError("구매_USEPA 시트에서 NAICS Title 컬럼을 찾지 못했습니다.")
    if not krw_col:
        raise ValueError("구매_USEPA 시트에서 EF (kgCO2e/KRW) 기계산 컬럼을 찾지 못했습니다.")

    out = pd.DataFrame({
        "name": raw[title_col].astype(str).str.strip(),
        "ef": pd.to_numeric(raw[krw_col], errors="coerce"),
        "ef_unit": "kgCO2e/KRW",
        "ef_source_col": krw_col,
    })
    if usd_col:
        out["ef_usd"] = pd.to_numeric(raw[usd_col], errors="coerce")
    if code_col:
        out["naics_code"] = raw[code_col]
    if cat_col:
        out["품목군"] = raw[cat_col]
    if sub_col:
        out["상세카테고리"] = raw[sub_col]
    if kw_col:
        out["한국어매핑키워드"] = raw[kw_col]

    out = out.dropna(subset=["name", "ef"]).drop_duplicates(subset=["name"]).reset_index(drop=True)
    globals()["USEPA_SPEND_EF_MODE"] = "PRECOMPUTED_KRW"
    globals()["USEPA_SPEND_EF_YEAR"] = year
    globals()["USEPA_SPEND_EF_COL"] = krw_col
    return out


def _reload_spend_ef_dbs_for_report_year(report_year=None):
    """
    보고연도별 기계산 KRW EF 컬럼을 기준으로 spend_db를 재구성하고,
    검색/추천 캐시를 초기화한다.
    """
    global spend_db, SPEND_DB_NAMES, SPEND_DB_NAME_SET, CURATED_SPEND_CANDIDATES

    year = int(report_year or globals().get("FX_REFERENCE_YEAR", 2025))
    globals()["FX_REFERENCE_YEAR"] = year

    # C1/C2 구매·자산 spend_db를 통합 DB의 기계산 KRW EF로 재로딩
    spend_db = _load_purchase_usepa_as_krw_spend_db(year)
    SPEND_DB_NAMES = (
        spend_db["name"].dropna().astype(str).str.strip().drop_duplicates().tolist()
    )
    SPEND_DB_NAME_SET = set(SPEND_DB_NAMES)

    # 기존 curated 후보가 있으면 현재 DB에 존재하는 title만 유지
    if "CURATED_SPEND_CANDIDATES" in globals():
        CURATED_SPEND_CANDIDATES = [x for x in CURATED_SPEND_CANDIDATES if x in SPEND_DB_NAME_SET]

    # lru_cache가 이전 USD EF 기준 결과를 들고 있지 않도록 초기화
    for fn_name in [
        "search_spend_ef",
        "translate_product_for_search_info",
        "translate_product_for_search",
        "normalize_product_to_naics",
    ]:
        fn = globals().get(fn_name)
        if hasattr(fn, "cache_clear"):
            try:
                fn.cache_clear()
            except Exception:
                pass

    print(f"[OK] 구매_USEPA spend_db 재로딩: {len(spend_db)}행")
    print(f"   - 보고연도: {year}년")
    print(f"   - 적용 EF 컬럼: {globals().get('USEPA_SPEND_EF_COL')}")
    print("   - 계산 방식: 구매금액(KRW) × EF(kgCO2e/KRW)")
    return spend_db


# 운송/출장 Spend DB도 KRW EF 컬럼이 있으면 그것을 우선 선택하도록 override
# lookup_transport_spend_ef / lookup_travel_spend_ef는 실행 시점에 이 함수를 참조함.
def _get_spend_ef_col(db: pd.DataFrame):
    year = globals().get("FX_REFERENCE_YEAR", None)
    krw_col = _find_precomputed_krw_ef_col(db, year)
    if krw_col:
        return krw_col
    return _find_usd_ef_col(db)


def _current_spend_ef_unit_from_db(db: pd.DataFrame) -> str:
    col = _get_spend_ef_col(db)
    if col and _is_precomputed_krw_ef_col(col):
        return "kgCO2e/KRW"
    return "kgCO2e/USD"


def normalize_purchase_spend_for_usepa(row, year=None):
    """
    C1/C2 구매금액을 KRW 기준으로 정규화.
    EF는 이미 kgCO2e/KRW로 계산된 DB 컬럼을 사용하므로 USD 환산은 하지 않음.
    """
    year = int(year or globals().get("FX_REFERENCE_YEAR", 2025))
    unit_info = normalize_purchase_unit(row)
    amount_krw = unit_info.get("구매금액_KRW")
    input_currency = unit_info.get("적용단위", "KRW")
    unit_note = unit_info.get("단위환산비고", "")

    return {
        "입력통화": input_currency,
        "단위환산비고": unit_note,
        "배출계수기준연도": year,
        "배출계수기준컬럼": globals().get("USEPA_SPEND_EF_COL"),
        "배출계수단위": "kgCO2e/KRW",
        "원화환산금액": amount_krw,
        "EF적용금액(KRW)": amount_krw,
        # 구버전 출력/집계 호환용. 더 이상 사용하지 않음.
        "환율기준연도": None,
        "환율적용월": None,
        "적용환율(원/USD)": None,
        "EF적용금액(USD)": None,
        "환율적용여부": "N - 기계산 KRW EF 직접 적용",
    }


def _format_spend_formula_krw(spend_krw, ef, emission):
    spend_krw = to_float(spend_krw)
    ef = to_float(ef)
    emission = to_float(emission)
    if spend_krw is None or ef is None or emission is None:
        return ""
    return f"배출량 = 구매금액(KRW) × 기계산 KRW 배출계수 = {spend_krw:g} × {ef:g} = {emission:g}"


# C4/C9 운송 Spend override: KRW EF 컬럼이 있으면 직접 적용
def calc_transport_spend_emission(row) -> dict:
    """
    C4/C9 운송 지출기반 배출량 계산.
    DB에 기계산 KRW EF가 있으면: 운송금액(KRW) × EF(kgCO2e/KRW) / 1000 → tCO2e
    """
    spend_krw = to_float(row.get("운송금액") or row.get("월운송금액") or row.get("연간운송금액(원)"))
    mode = _safe_text(row.get("운송수단") or row.get("자재명") or "")

    if spend_krw is None:
        return {"배출량(tCO2e)": None, "Spend방식": "운송"}

    title, ef, naics = lookup_transport_spend_ef(mode)
    unit = _current_spend_ef_unit_from_db(transport_spend_db)

    if unit == "kgCO2e/KRW":
        emission = (spend_krw * ef / 1000) if (ef is not None and spend_krw is not None) else None
        return {
            "배출량(tCO2e)": emission,
            "EF(kgCO2e/KRW)": ef,
            "배출계수단위": unit,
            "NAICS": naics,
            "NAICS_Title": title,
            "Spend방식": "운송_Spend_KRW_EF",
            "EF적용금액(KRW)": spend_krw,
        }

    # fallback: KRW EF 컬럼이 없을 때만 기존 USD EF 방식 사용
    월 = to_float(row.get("발생월"))
    fx = get_bok_usd_krw_monthly_avg(int(월) if 월 else None, year=FX_REFERENCE_YEAR)
    spend_usd = spend_krw / fx if fx else spend_krw / 1400
    emission = (spend_usd * ef / 1000) if (ef is not None and spend_usd is not None) else None
    return {
        "배출량(tCO2e)": emission,
        "EF(kgCO2e/USD)": ef,
        "배출계수단위": unit,
        "NAICS": naics,
        "NAICS_Title": title,
        "Spend방식": "운송_Spend_USD_EF_fallback",
        "적용환율(KRW/USD)": fx,
        "USD환산금액": spend_usd,
    }


# C6 출장 Spend override: KRW EF 컬럼이 있으면 직접 적용
def calc_travel_spend_emission(row) -> dict:
    """
    C6 출장 지출기반 배출량 계산.
    DB에 기계산 KRW EF가 있으면: 출장비용(KRW) × EF(kgCO2e/KRW) / 1000 → tCO2e
    """
    spend_krw = to_float(row.get("출장비용") or row.get("월출장비") or row.get("연간지출금액(원)"))
    item = _safe_text(row.get("출장항목") or row.get("이동수단") or row.get("출장수단") or "")

    if spend_krw is None:
        return {"배출량(tCO2e)": None, "Spend방식": "출장"}

    title, ef, naics = lookup_travel_spend_ef(item)
    unit = _current_spend_ef_unit_from_db(travel_spend_db)

    if unit == "kgCO2e/KRW":
        emission = (spend_krw * ef / 1000) if (ef is not None and spend_krw is not None) else None
        return {
            "배출량(tCO2e)": emission,
            "EF(kgCO2e/KRW)": ef,
            "배출계수단위": unit,
            "NAICS": naics,
            "NAICS_Title": title,
            "Spend방식": "출장_Spend_KRW_EF",
            "EF적용금액(KRW)": spend_krw,
        }

    # fallback: KRW EF 컬럼이 없을 때만 기존 USD EF 방식 사용
    월 = to_float(row.get("발생월"))
    fx = get_bok_usd_krw_monthly_avg(int(월) if 월 else None, year=FX_REFERENCE_YEAR)
    spend_usd = spend_krw / fx if fx else spend_krw / 1400
    emission = (spend_usd * ef / 1000) if (ef is not None and spend_usd is not None) else None
    return {
        "배출량(tCO2e)": emission,
        "EF(kgCO2e/USD)": ef,
        "배출계수단위": unit,
        "NAICS": naics,
        "NAICS_Title": title,
        "Spend방식": "출장_Spend_USD_EF_fallback",
        "적용환율(KRW/USD)": fx,
        "USD환산금액": spend_usd,
    }


# 현재 기본 기준연도에 맞춰 즉시 1회 재로딩
try:
    _reload_spend_ef_dbs_for_report_year(globals().get("FX_REFERENCE_YEAR", 2025))
    print("[OK] 기계산 KRW EF 직접 매핑 패치 적용 완료")
except Exception as e:
    print(f"[경고] 기계산 KRW EF 패치 초기화 실패: {e}")
    print("       통합 DB 파일 경로 또는 구매_USEPA 시트의 EF (kgCO2e/KRW) 컬럼을 확인하세요.")

# ── [원본 셀 51] ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
# PATCH 2026-06-11D
# 결과 검토 이슈 전체 보정
# 1) C1/C2: 기계산 KRW EF는 kgCO2e/KRW이므로 /1000 처리해 tCO2e 통일
# 2) C1/C2: 구매_USEPA의 한국어 키워드/품목군 기반 EF 후보 랭킹 강화
# 3) C6: Distance/Spend 최종 배출량 컬럼을 배출량(tCO2e)로 통일, 출장비용 오입력 제거
# 4) C7: 편도거리 ×2 × 근무일수 × 인원수 산식 적용
# 5) C11: 에너지명+단위 기준 EF 재매핑. 천연가스+Nm³는 도시가스(LNG) EF 우선
# 6) C14: 프랜차이즈 에너지 사용량 배출량 계산 추가
# 7) C13 면적배분 등 기존 표준화 보강 유지
# 8) 환율 문구 제거: 출력 기준 컬럼은 '기계산 KRW EF 직접 적용'으로 표기
# ══════════════════════════════════════════════════════════════
import os
import re
import pandas as pd


def _fix_text(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def _fix_num(x):
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = to_float(x)
        if f is not None:
            return f
    except Exception:
        pass
    try:
        s = _fix_text(x).replace(",", "")
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _fix_norm(x):
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", _fix_text(x).lower())


def _fix_unit_norm(x):
    s = _fix_text(x).lower().replace(" ", "").replace("㎥", "m3").replace("m³", "m3").replace("nm³", "nm3")
    s = s.replace("co₂", "co2")
    return s


def _fix_contains_roundtrip(row):
    txt = " ".join(_fix_text(row.get(c)) for c in ["비고", "설명/적요", "단위환산비고"] if hasattr(row, "get"))
    t = txt.lower()
    return any(k in t for k in ["왕복", "round trip", "roundtrip", "return trip", "왕복기준"])


# ----------------------------------------------------------------
# A. 에너지 EF: 동일 에너지원 후보가 여러 개일 때 단위 호환 우선
# ----------------------------------------------------------------
def _fix_energy_db_cols():
    try:
        return _energy_db_cols()
    except Exception:
        db = globals().get("energy_ef_db")
        if db is None or getattr(db, "empty", True):
            return None, None, None, None
        name_col = next((c for c in db.columns if "에너지원" in str(c) or "활동유형" in str(c)), None)
        ef_col = next((c for c in db.columns if "CO2eq" in str(c) and "단위" in str(c)), None)
        unit_col = next((c for c in db.columns if "기준단위" in str(c) or "투입량" in str(c)), None)
        return db, name_col, ef_col, unit_col


def _fix_lookup_energy_title_exact(title):
    db, name_col, ef_col, unit_col = _fix_energy_db_cols()
    if db is None or getattr(db, "empty", True) or not name_col or not ef_col:
        return None, None, None
    subset = db[db[name_col].astype(str).map(_fix_norm) == _fix_norm(title)]
    if subset.empty:
        return None, None, None
    r = subset.iloc[0]
    ef = pd.to_numeric(r.get(ef_col), errors="coerce")
    if pd.isna(ef):
        return None, None, None
    return r.get(name_col), float(ef), r.get(unit_col) if unit_col else None


_ORIGINAL_lookup_energy_ef_FULL_FIX = globals().get("lookup_energy_ef")

def lookup_energy_ef(energy_name: str, unit_hint: str = None):
    q = _fix_norm(energy_name)
    u = _fix_unit_norm(unit_hint)

    # 천연가스/LNG/도시가스는 DB에 Nm³ 기준 도시가스(LNG)와 kg 기준 천연가스(LNG)가 함께 있으므로 단위를 우선한다.
    is_gas = any(k in q for k in ["천연가스", "도시가스", "lng", "naturalgas"])
    if is_gas and u in {"nm3", "m3"}:
        name, ef, unit = _fix_lookup_energy_title_exact("도시가스(LNG)")
        if ef is not None:
            try:
                _LAST_EF_MATCH_INFO["energy"] = {"input": _fix_text(energy_name), "unit_hint": _fix_text(unit_hint), "raw_name": name, "raw_unit": unit, "match_type": "unit_priority_alias", "score": 1.0}
            except Exception:
                pass
            return name, ef, unit
    if is_gas and u == "kg":
        name, ef, unit = _fix_lookup_energy_title_exact("천연가스(LNG)")
        if ef is not None:
            try:
                _LAST_EF_MATCH_INFO["energy"] = {"input": _fix_text(energy_name), "unit_hint": _fix_text(unit_hint), "raw_name": name, "raw_unit": unit, "match_type": "unit_priority_alias", "score": 1.0}
            except Exception:
                pass
            return name, ef, unit

    if _ORIGINAL_lookup_energy_ef_FULL_FIX is not None:
        return _ORIGINAL_lookup_energy_ef_FULL_FIX(energy_name, unit_hint=unit_hint)
    return None, None, None


def _fix_convert_to_ef_unit(value, input_unit, energy_name, ef_unit):
    v = _fix_num(value)
    if v is None:
        return None, _fix_text(input_unit), ""
    in_u = _fix_text(input_unit)
    target_u = _fix_text(ef_unit) or in_u
    try:
        conv, std_u, note = convert_energy_unit(v, in_u, energy_name)
        if _fix_unit_norm(std_u) == _fix_unit_norm(target_u):
            return conv, std_u, note
        if _fix_unit_norm(in_u) == _fix_unit_norm(target_u):
            return v, in_u, ""
        # 단위가 다르면 계산 보류. 잘못된 EF를 곱하지 않도록 함.
        return None, std_u, f"[단위 불일치로 산정 보류: {in_u}→{target_u}]"
    except Exception:
        if _fix_unit_norm(in_u) == _fix_unit_norm(target_u):
            return v, in_u, ""
        return None, in_u, f"[단위 확인 필요: {in_u}→{target_u}]"


def _fix_calc_energy_usage_row(row, usage_col="사용량", energy_cols=("에너지원", "에너지종류", "에너지명", "연료/에너지명", "연료 종류")):
    energy_name = ""
    for c in energy_cols:
        if c in row.index and _fix_text(row.get(c)):
            energy_name = _fix_text(row.get(c))
            break
    usage_raw = _fix_num(row.get(usage_col))
    unit_raw = _fix_text(row.get("단위"))
    std_name, ef_val, ef_unit = lookup_energy_ef(energy_name, unit_hint=unit_raw)
    ef_val = _fix_num(ef_val)
    usage, std_unit, unit_note = _fix_convert_to_ef_unit(usage_raw, unit_raw, std_name or energy_name, ef_unit)
    emission_t = (usage * ef_val / 1000.0) if (usage is not None and ef_val is not None) else None
    return usage, std_unit, unit_note, std_name, ef_val, ef_unit, emission_t


# ----------------------------------------------------------------
# B. C1/C2 구매_USEPA EF 후보 랭킹 강화
# ----------------------------------------------------------------
_MANUAL_PURCHASE_TITLE_RULES = [
    (["플라스틱수지", "수지", "레진", "resin", "polymer", "pe", "pp", "pet", "abs", "pvc"], ["Plastics Material and Resin Manufacturing", "Plastics Packaging Film and Sheet Manufacturing"]),
    (["골판지", "포장박스", "박스", "carton", "corrugated"], ["Corrugated and Solid Fiber Box Manufacturing", "Folding Paperboard Box Manufacturing"]),
    (["erp", "시스템유지보수", "소프트웨어유지보수", "it서비스", "개발용역", "si", "시스템통합", "소프트웨어개발"], ["Custom Computer Programming Services", "Data Processing, Hosting, and Related Services", "Software Publishers"]),
    (["전문서비스", "컨설팅", "자문", "경영자문", "consulting"], ["Management Consulting Services", "Professional and Management Development Training"]),
    (["창고랙", "랙설비", "랙", "선반", "shelving", "locker", "storage rack"], ["Showcase, Partition, Shelving, and Locker Manufacturing", "Lessors of Miniwarehouses and Self-Storage Units"]),
    (["지게차", "포크리프트", "forklift", "스태커"], ["Industrial Truck, Tractor, Trailer, and Stacker Machinery Manufacturing", "Light Truck and Utility Vehicle Manufacturing"]),
    (["자동포장", "포장라인", "생산설비", "기계장치", "제조설비", "기계", "설비", "machine", "machinery", "equipment"], ["Other Industrial Machinery Manufacturing", "Conveyor and Conveying Equipment Manufacturing", "Special Industry Machinery Manufacturing"]),
    (["차량운반구", "승용차", "자동차", "차량", "suv", "vehicle", "car"], ["Automobile Manufacturing", "Light Truck and Utility Vehicle Manufacturing", "Automotive Parts and Accessories Stores"]),
    (["노트북", "데스크탑", "컴퓨터", "pc", "서버", "laptop"], ["Electronic Computer Manufacturing", "Computer and Computer Peripheral Equipment and Software Merchant Wholesalers"]),
]


def _fix_spend_row_by_title(title):
    db = globals().get("spend_db")
    if db is None or getattr(db, "empty", True) or "name" not in db.columns:
        return None
    subset = db[db["name"].astype(str).str.strip() == _fix_text(title)]
    if subset.empty:
        return None
    r = subset.iloc[0]
    return {
        "title": _fix_text(r.get("name")),
        "ef": _fix_num(r.get("ef")),
        "품목군": _fix_text(r.get("품목군")),
        "상세카테고리": _fix_text(r.get("상세카테고리")),
        "source": "manual_title",
        "score": 999,
    }


def _fix_split_keywords(s):
    raw = _fix_text(s)
    if not raw:
        return []
    return [x.strip() for x in re.split(r"[,;/|]+", raw) if x.strip()]


def _fix_purchase_query_from_row(row):
    fields = [
        "상세 품목명/서비스명", "상세 자산명", "품목명", "자산명", "품목/자산명",
        "분류", "품목군", "자산분류", "계정과목", "계정과목(선택)", "거래처명", "거래처명(선택)", "설명/적요", "비고"
    ]
    return " ".join(_fix_text(row.get(c)) for c in fields if hasattr(row, "get") and _fix_text(row.get(c)))


def _fix_rank_spend_candidates(query_text, group_hint="", top_n=5):
    db = globals().get("spend_db")
    if db is None or getattr(db, "empty", True) or "name" not in db.columns:
        return []
    text = _fix_text(query_text)
    text_norm = _fix_norm(text)
    group_norm = _fix_norm(group_hint)

    picked = []
    seen = set()

    # 1) 수작업 룰: 검토 결과에서 문제였던 전문서비스/플라스틱/설비/차량/랙을 우선 보정
    for keys, titles in _MANUAL_PURCHASE_TITLE_RULES:
        if any(_fix_norm(k) and _fix_norm(k) in text_norm for k in keys):
            for title in titles:
                rec = _fix_spend_row_by_title(title)
                if rec and rec["title"] not in seen:
                    rec = dict(rec)
                    rec["source"] = "manual_keyword"
                    picked.append(rec)
                    seen.add(rec["title"])

    # 2) 구매_USEPA DB의 한국어 매핑 키워드/품목군/상세카테고리 기반 점수화
    rows = []
    for _, r in db.iterrows():
        title = _fix_text(r.get("name"))
        if not title or title in seen:
            continue
        score = 0
        cat = _fix_text(r.get("품목군"))
        sub = _fix_text(r.get("상세카테고리"))
        kw_list = _fix_split_keywords(r.get("한국어매핑키워드"))
        for kw in kw_list:
            kn = _fix_norm(kw)
            if not kn:
                continue
            if kn in text_norm:
                score += 80 + min(len(kn), 20)
            elif len(kn) >= 3 and any(part and part in text_norm for part in re.split(r"[·\-_/ ]+", kw)):
                score += 15
        if group_norm:
            if group_norm and group_norm in _fix_norm(cat + sub):
                score += 35
            # 템플릿 사용자 친화 분류 ↔ DB 품목군 보정
            group_alias = {
                "전문서비스": "외주서비스", "서비스": "외주서비스", "원재료": "원재료부자재",
                "플라스틱": "원재료부자재", "기계장치": "생산설비", "차량": "차량운송장비",
                "건물구축물": "건설자재", "it장비": "it장비", "종이인쇄물": "사무용품포장재",
            }
            aliased = group_alias.get(group_norm)
            if aliased and aliased in _fix_norm(cat + sub):
                score += 30
        if score > 0:
            rows.append({"title": title, "ef": _fix_num(r.get("ef")), "품목군": cat, "상세카테고리": sub, "source": "db_keyword", "score": score})

    rows = sorted(rows, key=lambda x: x.get("score", 0), reverse=True)
    for rec in rows:
        if rec["title"] not in seen:
            picked.append(rec)
            seen.add(rec["title"])
        if len(picked) >= top_n:
            break

    # 3) 그래도 부족하면 기존 추천 로직으로 채우되 중복 제거
    if len(picked) < top_n:
        try:
            old_recs, _family = _ORIGINAL_get_spend_recommendations_FULL_FIX(query_text, "", "", top_k=top_n * 2)
        except Exception:
            old_recs = []
        for r in old_recs or []:
            title = _fix_text(r.get("title"))
            if title and title not in seen:
                picked.append({"title": title, "ef": _fix_num(r.get("ef")), "source": "legacy_fallback", "score": 0})
                seen.add(title)
            if len(picked) >= top_n:
                break

    # 4) 최종 fallback: 아무것도 없을 때만 사무용품류 사용
    if not picked:
        for title in ["Other Miscellaneous Durable Goods Merchant Wholesalers", "Office Supplies and Stationery Stores"]:
            rec = _fix_spend_row_by_title(title)
            if rec:
                picked.append(rec)
                break
    return picked[:top_n]


_ORIGINAL_get_spend_recommendations_FULL_FIX = globals().get("get_spend_recommendations")
_ORIGINAL_select_primary_spend_candidate_FULL_FIX = globals().get("select_primary_spend_candidate")
_ORIGINAL_search_spend_ef_FULL_FIX = globals().get("search_spend_ef")


def get_spend_recommendations(product="", desc="", vendor="", top_k=8):
    q = " ".join([_fix_text(product), _fix_text(desc), _fix_text(vendor)]).strip()
    recs = _fix_rank_spend_candidates(q, group_hint="", top_n=top_k)
    family = detect_spend_family(product, desc, vendor) if "detect_spend_family" in globals() else "purchase"
    return [{"title": r["title"], "ef": r["ef"]} for r in recs], family


def select_primary_spend_candidate(product="", desc="", vendor=""):
    recs, family = get_spend_recommendations(product=product, desc=desc, vendor=vendor, top_k=8)
    if recs:
        return {"name": recs[0]["title"], "ef": _fix_num(recs[0]["ef"]), "family": family, "pick_source": "keyword_ranked", "recommendations": recs}
    if _ORIGINAL_select_primary_spend_candidate_FULL_FIX is not None:
        return _ORIGINAL_select_primary_spend_candidate_FULL_FIX(product=product, desc=desc, vendor=vendor)
    return {"name": None, "ef": None, "family": family, "pick_source": "none", "recommendations": []}


def search_spend_ef(product):
    recs = _fix_rank_spend_candidates(product, group_hint="", top_n=1)
    if recs:
        return recs[0]["title"], recs[0]["ef"]
    if _ORIGINAL_search_spend_ef_FULL_FIX is not None:
        return _ORIGINAL_search_spend_ef_FULL_FIX(product)
    return None, None


def _fix_spend_formula_krw_t(spend_krw, ef, emission_t):
    if spend_krw is None or ef is None or emission_t is None:
        return ""
    return f"배출량(tCO2e) = 구매금액(KRW) × EF(kgCO2e/KRW) / 1000 = {spend_krw:g} × {ef:g} / 1000 = {emission_t:g}"


def calc_category1_2_detail(row, category):
    row = row.copy()
    if "품목/자산명" in row.index:
        if pd.isnull(row.get("품목명")):
            row["품목명"] = row.get("품목/자산명")
        if pd.isnull(row.get("자산명")):
            row["자산명"] = row.get("품목/자산명")

    spend_meta = normalize_purchase_spend_for_usepa(row, year=globals().get("FX_REFERENCE_YEAR", 2025))
    spend_krw = _fix_num(spend_meta.get("EF적용금액(KRW)"))
    group_info = classify_purchase_group(row) if "classify_purchase_group" in globals() else {"group_key": None, "품목/자산명": None, "그룹판단근거": None, "추천Family": None, "자동카테고리": category}
    group_hint = _fix_text(row.get("품목군") or row.get("자산분류") or row.get("분류") or group_info.get("품목/자산명"))
    query = _fix_purchase_query_from_row(row)

    candidates = _fix_rank_spend_candidates(query, group_hint=group_hint, top_n=5)
    primary = candidates[0] if candidates else {"title": None, "ef": None, "source": "none"}
    alt_candidates = candidates[1:5]
    rep_name = primary.get("title")
    rep_ef = _fix_num(primary.get("ef"))
    emission_kg = (spend_krw * rep_ef) if (spend_krw is not None and rep_ef is not None) else None
    emission_t = (emission_kg / 1000.0) if emission_kg is not None else None
    year = int(globals().get("FX_REFERENCE_YEAR", 2025))
    product_display = _fix_text(row.get("상세 품목명/서비스명") or row.get("상세 자산명") or row.get("품목명") or row.get("자산명") or row.get("품목/자산명"))

    result = {
        "구매 월": _get_purchase_month(row) if "_get_purchase_month" in globals() else None,
        "단위환산비고": spend_meta.get("단위환산비고") or "KRW 그대로",
        "배출계수기준연도": year,
        "배출계수기준컬럼": f"기계산 KRW EF 직접 적용 ({year}년, kgCO2e/KRW)",
        "배출계수단위": "kgCO2e/KRW",
        "EF적용금액(KRW)": spend_krw,
        "group_key": group_info.get("group_key"),
        "품목/자산명": group_info.get("품목/자산명") or product_display,
        "그룹판단근거": group_info.get("그룹판단근거"),
        "추천Family": group_info.get("추천Family"),
        "자동카테고리": group_info.get("자동카테고리", category),
        "매개변수": rep_name,
        "EF_DB": rep_name,
        "배출계수": rep_ef,
        "배출량(kgCO2e)": emission_kg,
        "배출량": emission_t,
        "배출량(tCO2e)": emission_t,
        "EF_DB_추천사유": f"템플릿 입력값/한국어 매핑 키워드 기반 후보 랭킹 적용({primary.get('source')}).",
        "계산방식": "Spend",
        "계산식": _fix_spend_formula_krw_t(spend_krw, rep_ef, emission_t),
    }
    for i in range(1, 5):
        cand = alt_candidates[i - 1] if i - 1 < len(alt_candidates) else {"title": None, "ef": None}
        ef_i = _fix_num(cand.get("ef"))
        result[f"추천EF{i}"] = cand.get("title")
        result[f"추천EF{i}_배출계수"] = ef_i
        result[f"추천EF{i}_예상배출량(tCO2e)"] = (spend_krw * ef_i / 1000.0) if (spend_krw is not None and ef_i is not None) else None
    return pd.Series(result)


# ----------------------------------------------------------------
# C. 출장/통근 산식 및 출장 Spend EF 매핑 보정
# ----------------------------------------------------------------
@lru_cache(maxsize=512)
def _ai_normalize_travel_mode(text: str):
    """rule/keyword로 출장·통근 이동수단을 판별하지 못했을 때 AI로 판단. 실패 시 None."""
    prompt = f"""다음 출장/통근 이동수단을 아래 6개 중 하나로 변환하라.
입력값: {text}
가능한 값: flight, train, subway, taxi, bus, car
JSON으로만 답하라: {{"mode": "<value>"}}"""
    parsed = _call_claude_json(prompt, max_tokens=40)
    mode = _fix_norm(parsed.get("mode")) if parsed else ""
    valid = {"flight", "train", "subway", "taxi", "bus", "car"}
    return mode if mode in valid else None

def normalize_travel_mode(mode):
    s = _fix_norm(mode)
    if any(k in s for k in ["숙박", "호텔", "모텔", "lodging", "hotel", "accommodation"]):
        return "lodging"
    if any(k in s for k in ["렌터카", "렌트카", "rentalcar", "carrental"]):
        return "rental_car"
    if any(k in s for k in ["철도", "기차", "ktx", "srt", "rail", "train"]):
        return "train"
    if any(k in s for k in ["항공", "비행", "flight", "air"]):
        return "flight"
    if any(k in s for k in ["택시", "taxi", "cab"]):
        return "taxi"
    if any(k in s for k in ["버스", "bus", "coach"]):
        return "bus"
    if any(k in s for k in ["지하철", "전철", "subway", "metro"]):
        return "subway"
    if any(k in s for k in ["승용차", "자가용", "차량", "car"]):
        return "car"

    try:
        rule_mode = detect_travel_mode_rule(mode)
    except Exception:
        rule_mode = "unknown"

    if rule_mode and rule_mode != "unknown":
        return rule_mode

    if s:
        ai_mode = _ai_normalize_travel_mode(s)
        if ai_mode:
            return ai_mode

    return "car"


def normalize_commute_mode(mode):
    m = normalize_travel_mode(mode)
    if m == "rental_car":
        return "car"
    return m


def _fix_lookup_travel_spend_candidate(text):
    db = globals().get("travel_spend_db")
    if db is None or getattr(db, "empty", True):
        return None
    year = int(globals().get("FX_REFERENCE_YEAR", 2025))
    try:
        ef_col = _find_precomputed_krw_ef_col(db, year)
    except Exception:
        ef_col = None
    if not ef_col:
        try:
            ef_col = _get_spend_ef_col(db)
        except Exception:
            ef_col = next((c for c in db.columns if "EF" in str(c)), None)
    item_col = next((c for c in db.columns if "출장 항목" in str(c)), None)
    title_col = next((c for c in db.columns if "NAICS Title" in str(c) or "Title" in str(c)), None)
    kw_col = next((c for c in db.columns if "키워드" in str(c)), None)
    code_col = next((c for c in db.columns if "NAICS 코드" in str(c) or "NAICS" in str(c)), None)
    qn = _fix_norm(text)
    manual = [
        (["항공", "비행", "airfare", "flight", "항공권"], "Scheduled Passenger Air Transportation"),
        (["철도", "기차", "ktx", "srt", "rail", "train"], "Line-Haul Railroads"),
        (["숙박", "호텔", "모텔", "hotel", "lodging", "accommodation"], "Hotels (except Casino Hotels) and Motels"),
        (["렌터카", "렌트카", "rentalcar", "carrental"], "Passenger Cars, Rental"),
        (["택시", "taxi", "cab"], "Taxi Service"),
        (["버스", "bus", "coach"], "Interurban and Rural Bus Transportation"),
    ]
    for keys, title in manual:
        if any(_fix_norm(k) in qn for k in keys):
            subset = db[db[title_col].astype(str).str.strip() == title] if title_col else pd.DataFrame()
            if not subset.empty:
                r = subset.iloc[0]
                return _fix_text(r.get(title_col)), _fix_num(r.get(ef_col)), _fix_text(r.get(code_col))
    scored = []
    for _, r in db.iterrows():
        score = 0
        for kw in _fix_split_keywords(r.get(kw_col)) if kw_col else []:
            kn = _fix_norm(kw)
            if kn and kn in qn:
                score += 100 + len(kn)
        if item_col and _fix_norm(r.get(item_col)) in qn:
            score += 30
        if title_col and _fix_norm(r.get(title_col)) in qn:
            score += 20
        if score > 0:
            scored.append((score, r))
    if scored:
        _, r = max(scored, key=lambda x: x[0])
        return _fix_text(r.get(title_col)), _fix_num(r.get(ef_col)), _fix_text(r.get(code_col))
    return None


def lookup_travel_spend_ef(item: str) -> tuple:
    cand = _fix_lookup_travel_spend_candidate(item)
    if cand:
        return cand
    original = globals().get("_LOOKUP_TRAVEL_SPEND_EF_BEFORE_FULL_FIX") or globals().get("_LOOKUP_TRAVEL_SPEND_EF_BEFORE_ALIAS_PATCH")
    if original is not None:
        try:
            return original(item)
        except Exception:
            pass
    return None, None, None


def calc_travel_spend_emission(row) -> dict:
    spend_krw = _fix_num(row.get("출장비용") or row.get("월출장비용") or row.get("연간출장비용(원)"))
    item = _fix_text(row.get("출장항목") or row.get("출장수단") or row.get("이동수단") or row.get("비고") or row.get("설명/적요"))
    if spend_krw is None:
        return {"배출량(tCO2e)": None, "Spend방식": "출장"}
    title, ef, naics = lookup_travel_spend_ef(item)
    ef = _fix_num(ef)
    emission_t = (spend_krw * ef / 1000.0) if ef is not None else None
    return {
        "배출량(tCO2e)": emission_t,
        "배출량": emission_t,
        "Spend방식": "출장",
        "EF_DB": title,
        "NAICS": naics,
        "배출계수": ef,
        "배출계수단위": "kgCO2e/KRW",
        "계산식": f"배출량(tCO2e) = 출장비용(KRW) × EF(kgCO2e/KRW) / 1000 = {spend_krw:g} × {ef:g} / 1000 = {emission_t:g}" if emission_t is not None else "",
    }


def calc_business_travel(row):
    distance = _fix_num(row.get("이동거리(km)"))
    people = _fix_num(row.get("출장인원") or row.get("인원수")) or 1
    trips = _fix_num(row.get("이동횟수")) or 1
    mode = normalize_travel_mode(row.get("출장수단") or row.get("이동수단"))
    if distance is None and _fix_text(row.get("출발지")) and _fix_text(row.get("도착지")):
        try:
            distance = get_distance_km(row.get("출발지"), row.get("도착지"), mode="air" if mode == "flight" else "rail" if mode == "train" else "road")
        except Exception:
            distance = None
    ef = BUSINESS_TRAVEL_EF.get(mode) or BUSINESS_TRAVEL_EF.get("car")
    if distance is not None and ef is not None:
        factor = 1 if _fix_contains_roundtrip(row) else 2
        emission_t = distance * factor * people * trips * ef / 1000.0
        return distance, ef, emission_t
    return distance, ef, None


def calc_commute(row):
    employees = _fix_num(row.get("직원수") or row.get("인원수")) or 1
    distance = _fix_num(row.get("평균통근거리(km)") or row.get("이동거리(km)"))
    workdays = _fix_num(row.get("근무일수")) or 245
    mode = normalize_commute_mode(row.get("교통수단") or row.get("이동수단"))
    if distance is None and _fix_text(row.get("출발지")) and _fix_text(row.get("도착지")):
        geo_mode = "rail" if mode in ["train", "subway"] else "road"
        try:
            distance = get_distance_km(row.get("출발지"), row.get("도착지"), mode=geo_mode)
            # PATCH: 통근은 매일 반복되는 이동이라 편도 100km를 넘는 경우가 드물다.
            # 지오코딩이 주소에 시/도 정보가 빠져있는 등의 이유로 동명이지(同名異地)를
            # 잘못 매칭해 비현실적으로 큰 거리가 나오는 사례가 다수 발견되어,
            # 100km를 초과하면 AI에게 별도로 재추정을 요청해 교차검증한다.
            if distance is not None and distance > 100:
                ai_straight = _ai_estimate_straight_km(
                    row.get("출발지"), row.get("도착지"), country_hint="KR",
                )
                if ai_straight is not None:
                    ai_distance = validate_distance_km(
                        ai_straight * _DISTANCE_MODE_FACTOR.get(geo_mode, 1.0),
                        mode=geo_mode, country_hint="KR",
                    )
                    if ai_distance is not None:
                        distance = ai_distance
        except Exception:
            distance = None
    mode_text = _fix_text(row.get("교통수단") or row.get("이동수단")).lower()
    if ("버스" in mode_text or "bus" in mode_text) and ("지하철" in mode_text or "subway" in mode_text or "전철" in mode_text):
        ef = BUS_SUBWAY_AVG_EF
    else:
        ef = COMMUTE_EF.get(mode)
    if distance is not None and ef is not None:
        factor = 1 if _fix_contains_roundtrip(row) else 2
        emission_t = employees * distance * factor * workdays * ef / 1000.0
        return distance, ef, emission_t
    return distance, ef, None


# ----------------------------------------------------------------
# D. C11/C14 표준화 단계 계산 보강
# ----------------------------------------------------------------
def standardize_c11_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_template_columns(df)
    rename = {}
    if "그룹(선택)" in df.columns:
        rename["그룹(선택)"] = "그룹"
    if "설명/적요(선택)" in df.columns:
        rename["설명/적요(선택)"] = "설명/적요"
    if "수명(개월) 혹은 평균 사용기간(개월)" in df.columns:
        rename["수명(개월) 혹은 평균 사용기간(개월)"] = "수명(개월)"
    if rename:
        df = df.rename(columns=rename)
    rows = []
    for _, row in df.iterrows():
        row2 = row.copy()
        qty = _fix_num(row2.get("제품수량") or row2.get("판매수량") or row2.get("판매량"))
        annual_use = _fix_num(row2.get("제품당 연간 에너지 사용량"))
        months = _fix_num(row2.get("수명(개월)"))
        years = (months / 12.0) if months is not None else 1.0
        energy_name = _fix_text(row2.get("에너지명") or row2.get("에너지원"))
        unit_raw = _fix_text(row2.get("단위"))
        std_name, ef_val, ef_unit = lookup_energy_ef(energy_name, unit_hint=unit_raw)
        ef_val = _fix_num(ef_val)
        use_per_product, std_unit, unit_note = _fix_convert_to_ef_unit(annual_use, unit_raw, std_name or energy_name, ef_unit)
        total_usage = qty * use_per_product * years if (qty is not None and use_per_product is not None) else None
        emission_t = (total_usage * ef_val / 1000.0) if (total_usage is not None and ef_val is not None) else None
        row2["사용기간(년)"] = years
        row2["총사용량"] = total_usage
        row2["배출계수(kgCO2eq/단위)"] = ef_val
        row2["EF단위"] = ef_unit
        row2["EF매핑명"] = std_name
        if unit_note:
            row2["단위환산비고"] = unit_note
        row2["배출량(tCO2e)"] = emission_t
        rows.append(row2)
    return pd.DataFrame(rows)


def standardize_c14_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_template_columns(df)
    rename = {}
    if "연료 종류" in df.columns and "에너지원" not in df.columns:
        rename["연료 종류"] = "에너지원"
    if "연료 사용량" in df.columns and "사용량" not in df.columns:
        rename["연료 사용량"] = "사용량"
    if "전력사용량" in df.columns and "사용량" not in df.columns:
        rename["전력사용량"] = "사용량"
    if "연료 단위" in df.columns and "단위" not in df.columns:
        rename["연료 단위"] = "단위"
    if "전력사용량 단위" in df.columns and "단위" not in df.columns:
        rename["전력사용량 단위"] = "단위"
    if "가맹점 수" in df.columns and "가맹점수" not in df.columns:
        rename["가맹점 수"] = "가맹점수"
    if rename:
        df = df.rename(columns=rename)
    try:
        df = _patch_collapse_monthly_to_value(df, "사용량", annual_cols=["사용량", "연간사용량", "연료 사용량", "전력사용량"], drop_month_cols=True)
    except Exception:
        pass
    rows = []
    for _, row in df.iterrows():
        row2 = row.copy()
        usage, std_unit, unit_note, std_name, ef_val, ef_unit, emission_t = _fix_calc_energy_usage_row(row2, usage_col="사용량")
        row2["사용량"] = usage if usage is not None else row2.get("사용량")
        row2["배출계수(kgCO2eq/단위)"] = ef_val
        row2["EF단위"] = ef_unit
        row2["EF매핑명"] = std_name
        row2["EF매핑방식"] = (_LAST_EF_MATCH_INFO or {}).get("energy", {}).get("match_type") if "_LAST_EF_MATCH_INFO" in globals() else None
        row2["EF매핑점수"] = (_LAST_EF_MATCH_INFO or {}).get("energy", {}).get("score") if "_LAST_EF_MATCH_INFO" in globals() else None
        if unit_note:
            row2["단위환산비고"] = unit_note
        row2["배출량(tCO2e)"] = emission_t
        row2["계산식"] = f"배출량(tCO2e) = 사용량 × EF / 1000 = {usage:g} × {ef_val:g} / 1000 = {emission_t:g}" if emission_t is not None else ""
        rows.append(row2)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------
# E. 최종 결과 파일 후처리: 컬럼명/단위/계산 누락 재검증
# ----------------------------------------------------------------
def _fix_result_c1_c2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "구매금액" in out.columns and "배출계수" in out.columns:
        spend = pd.to_numeric(out["구매금액"], errors="coerce")
        ef = pd.to_numeric(out["배출계수"], errors="coerce")
        out["배출량(kgCO2e)"] = spend * ef
        out["배출량(tCO2e)"] = out["배출량(kgCO2e)"] / 1000.0
        out["배출량"] = out["배출량(tCO2e)"]
        year = int(globals().get("FX_REFERENCE_YEAR", 2025))
        out["배출계수기준컬럼"] = f"기계산 KRW EF 직접 적용 ({year}년, kgCO2e/KRW)"
        out["배출계수단위"] = "kgCO2e/KRW"
        out["계산식"] = [
            f"배출량(tCO2e) = 구매금액(KRW) × EF(kgCO2e/KRW) / 1000 = {s:g} × {e:g} / 1000 = {v:g}" if pd.notna(s) and pd.notna(e) and pd.notna(v) else ""
            for s, e, v in zip(spend, ef, out["배출량(tCO2e)"])
        ]
        for i in range(1, 5):
            c = f"추천EF{i}_배출계수"
            if c in out.columns:
                out[f"추천EF{i}_예상배출량(tCO2e)"] = spend * pd.to_numeric(out[c], errors="coerce") / 1000.0
    return out


def _fix_result_c6(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        method = _fix_text(row.get("계산방식"))
        if method == "Distance":
            dist = _fix_num(row.get("이동거리(km)"))
            ef = _fix_num(row.get("배출계수"))
            people = _fix_num(row.get("출장인원") or row.get("인원수")) or 1
            trips = _fix_num(row.get("이동횟수")) or 1
            factor = 1 if _fix_contains_roundtrip(row) else 2
            emission_t = dist * factor * people * trips * ef / 1000.0 if None not in (dist, ef) else None
            out.at[idx, "배출량(tCO2e)"] = emission_t
            out.at[idx, "배출량"] = emission_t
            out.at[idx, "출장비용"] = None
            out.at[idx, "거리왕복계수"] = factor
            out.at[idx, "계산식"] = f"배출량(tCO2e) = 이동거리 × 왕복계수 × 인원수 × 이동횟수 × EF / 1000 = {dist:g} × {factor:g} × {people:g} × {trips:g} × {ef:g} / 1000 = {emission_t:g}" if emission_t is not None else ""
        elif method == "Spend":
            res = calc_travel_spend_emission(row)
            for k, v in res.items():
                out.at[idx, k] = v
        out.at[idx, "정규화이동수단"] = normalize_travel_mode(row.get("출장수단") or row.get("이동수단"))
    return out


def _fix_result_c7(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        dist, ef, emission_t = calc_commute(row)
        out.at[idx, "평균통근거리(km)"] = dist if dist is not None else row.get("평균통근거리(km)")
        out.at[idx, "배출계수"] = ef
        out.at[idx, "배출량(tCO2e)"] = emission_t
        out.at[idx, "배출량"] = emission_t
        employees = _fix_num(row.get("직원수") or row.get("인원수")) or 1
        workdays = _fix_num(row.get("근무일수")) or 245
        factor = 1 if _fix_contains_roundtrip(row) else 2
        out.at[idx, "거리왕복계수"] = factor
        out.at[idx, "정규화이동수단"] = normalize_commute_mode(row.get("교통수단") or row.get("이동수단"))
        out.at[idx, "계산식"] = f"배출량(tCO2e) = 평균통근거리 × 왕복계수 × 근무일수 × 직원수 × EF / 1000 = {dist:g} × {factor:g} × {workdays:g} × {employees:g} × {ef:g} / 1000 = {emission_t:g}" if emission_t is not None else ""
    return out


def _fix_result_energy_sheet(df: pd.DataFrame, usage_col="사용량") -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        usage, std_unit, unit_note, std_name, ef_val, ef_unit, emission_t = _fix_calc_energy_usage_row(row, usage_col=usage_col)
        if usage is not None:
            out.at[idx, usage_col] = usage
        out.at[idx, "배출계수(kgCO2eq/단위)"] = ef_val
        out.at[idx, "EF단위"] = ef_unit
        out.at[idx, "EF매핑명"] = std_name
        if unit_note:
            out.at[idx, "단위환산비고"] = unit_note
        out.at[idx, "배출량(tCO2e)"] = emission_t
        out.at[idx, "계산식"] = f"배출량(tCO2e) = 사용량 × EF / 1000 = {usage:g} × {ef_val:g} / 1000 = {emission_t:g}" if emission_t is not None else ""
    return out


def _postprocess_scope3_result_file(output_file):
    if not output_file or not os.path.exists(output_file):
        return
    try:
        sheets = pd.read_excel(output_file, sheet_name=None)
    except Exception as e:
        print(f"[경고] 결과 후처리 스킵: {e}")
        return
    fixed = {}
    for sname, df in sheets.items():
        name = _fix_text(sname)
        try:
            if name.startswith("C1_") or name.startswith("C2_"):
                df = _fix_result_c1_c2(df)
            elif name.startswith("C6_"):
                df = _fix_result_c6(df)
            elif name.startswith("C7_"):
                df = _fix_result_c7(df)
            elif name.startswith("C11_") or name.startswith("C14_"):
                df = _fix_result_energy_sheet(df, usage_col="총사용량" if "총사용량" in df.columns and name.startswith("C11_") else "사용량") if not name.startswith("C11_") else df
                if name.startswith("C11_"):
                    # C11은 제품수량/연간사용량/사용기간 기준으로 다시 계산
                    df = standardize_c11_sheet(df)
            # C6/C7에서 생긴 모호한 기존 배출량 컬럼은 tCO2e와 동일하게 맞춤
            fixed[sname] = df
        except Exception as e:
            print(f"[경고] {sname} 후처리 실패: {e}")
            fixed[sname] = df
    tmp = output_file.replace(".xlsx", "__fixed_tmp.xlsx")
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        for sname, df in fixed.items():
            df.to_excel(writer, sheet_name=sname[:31], index=False)
    os.replace(tmp, output_file)
    print("[OK] 결과 후처리 완료: C1/C2 단위, C6/C7 산식, C11/C14 에너지 계산 보정")


_ORIGINAL_process_new_template_FULL_FIX = globals().get("process_new_template")

def process_new_template(input_file: str, output_file: str, report_year: int = None):
    if _ORIGINAL_process_new_template_FULL_FIX is None:
        raise RuntimeError("원본 process_new_template 함수를 찾지 못했습니다.")
    result = _ORIGINAL_process_new_template_FULL_FIX(input_file, output_file, report_year=report_year)
    _postprocess_scope3_result_file(output_file)
    return result

print("[OK] Full fix patch loaded: C1/C2 tCO2e, EF keyword ranking, C6/C7 formulas, C11/C14 energy calculation, unit-safe gas mapping")

# ── [원본 셀 52] ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
# PATCH: C12(판매제품 폐기) 배출계수 계산 버그 수정 + 배출계수 컬럼 추가
# ------------------------------------------------------------------
# 기존 calc_c12_emission()은 "폐기물 소분류(선택)" 등의 값이
# "01-01-02 | 폐폴리프로필렌" 처럼 코드+명칭이 합쳐진 문자열인데,
# 배출계수_통합_DB.xlsx > 폐기물_C12 시트의 코드("01-01-02")와 정확히 일치하는지만
# 비교하다 보니 거의 항상 매칭에 실패하고, 전체 평균 EF(WASTE_AVG_EF)로만 계산되고 있었습니다.
# 아래 함수는 "|" 앞부분만 코드로 추출해 정상적으로 폐기물_C12 DB와 매칭하고,
# 처리방법별(매립/소각/재활용) EF를 처리비중(%)으로 가중평균한 배출계수도 함께 반환합니다.
# ══════════════════════════════════════════════════════════════

def _c12_clean_code(x):
    """'01-01-02 | 폐폴리프로필렌' -> '01-01-02' (파이프 앞부분 코드만 추출)"""
    s = str(x if x is not None else "").strip()
    if not s or s == "-" or "선택안함" in s:
        return ""
    return s.split("|")[0].strip() if "|" in s else s


def calc_c12_ef_and_emission(row) -> dict:
    """
    C12 판매제품 폐기 배출계수·배출량 계산 (배출계수_통합_DB.xlsx > 폐기물_C12 시트 참조).
    - 폐기물 소분류 > 중분류 > 대분류 코드 순으로 매칭해 처리방법별(매립/소각/재활용) EF를 조회하고,
      입력된 처리비중(%)으로 가중평균한 배출계수를 계산합니다.
    - 코드가 없거나 매칭 실패 시 제품명으로 조회하거나, 그마저 없으면 폐기물_C12 DB 전체
      평균 EF(WASTE_AVG_EF)로 대체합니다 (기존 calc_c12_emission()과 동일한 fallback 규칙).
    반환: {"배출계수": 가중평균 EF(kgCO2eq/kg) 또는 None, "배출량(tCO2e)": 배출량 또는 None}
    """
    qty    = to_float(row.get("판매량"))
    weight = to_float(row.get("제품중량"))
    unit   = str(row.get("제품중량 단위") or "kg").strip().lower()
    if qty is None or weight is None:
        return {"배출계수": None, "배출량(tCO2e)": None}

    if unit in ["g", "gram", "그램"]:
        weight = weight / 1000.0
    elif unit in ["ton", "t", "톤"]:
        weight = weight * 1000.0
    total_weight_kg = qty * weight

    def _pct(v):
        if v is None:
            return None
        return v / 100.0 if v > 1 else v

    pct_lf  = _pct(to_float(row.get("매립 비중(%)") or row.get("매립비중(%)")))
    pct_inc = _pct(to_float(row.get("소각 비중(%)") or row.get("소각비중(%)")))
    pct_rec = _pct(to_float(row.get("재활용 비중(%)") or row.get("재활용비중(%)")))

    sub = _c12_clean_code(row.get("폐기물 소분류(선택)"))
    mid = _c12_clean_code(row.get("폐기물 중분류(선택)"))
    big = _c12_clean_code(row.get("폐기물 대분류(선택)"))
    product = str(row.get("제품명") or "").strip()

    def _ef(treatment):
        if sub:
            return lookup_waste_ef_by_code(소분류코드=sub, treatment=treatment)
        if mid:
            return lookup_waste_ef_by_code(중분류코드=mid, treatment=treatment)
        if big:
            return lookup_waste_ef_by_code(대분류코드=big, treatment=treatment)
        if product:
            return lookup_waste_ef(product, treatment)
        return WASTE_AVG_EF.get(_normalize_treatment(treatment))

    ef_lf, ef_inc, ef_rec = _ef("매립"), _ef("소각"), _ef("재활용")

    if pct_lf is None and pct_inc is None and pct_rec is None:
        weighted_ef = (WASTE_AVG_EF["매립"] + WASTE_AVG_EF["소각"] + WASTE_AVG_EF["재활용"]) / 3
    else:
        weighted_ef = 0.0
        if pct_lf  is not None and ef_lf  is not None:
            weighted_ef += ef_lf  * pct_lf
        if pct_inc is not None and ef_inc is not None:
            weighted_ef += ef_inc * pct_inc
        if pct_rec is not None and ef_rec is not None:
            weighted_ef += ef_rec * pct_rec

    emission = total_weight_kg * weighted_ef / 1000.0
    return {"배출계수": weighted_ef, "배출량(tCO2e)": emission}


print("[OK] calc_c12_ef_and_emission() 로드 완료 - C12 배출계수를 폐기물_C12 DB 코드 매칭으로 계산합니다.")

# ── [원본 셀 53] ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
# PATCH: C3(연료 및 에너지 관련 활동) 전용 배출계수 신설
# ------------------------------------------------------------------
# C3는 정의상 "Scope1·Scope2에 포함되지 않는 연료/에너지의 업스트림(채굴·정제·수송)
# 배출량과 송배전 손실분"만 계산해야 합니다. 그런데 지금까지는 C8/C13과 같은
# '에너지' 시트(직접연소·Scope2용, 예: 전력 0.4173 kgCO2e/kWh)를 그대로 썼기 때문에
# C3 배출량이 실제보다 훨씬 크게(자릿수 단위로) 계산되고 있었습니다.
#
# 아래는 배출계수_통합_DB.xlsx에 새로 추가한 "에너지_C3" 시트를 그대로 불러와 사용합니다.
# (경유/휘발유/등유는 kg 기준 입력을 전제로 합니다 — L로 입력하는 경우 밀도 환산이 필요합니다.)
# ══════════════════════════════════════════════════════════════
energy_c3_ef_db = _load_ef_sheet("에너지_C3", header_row=2)
energy_c3_ef_db.columns = [str(c).strip() for c in energy_c3_ef_db.columns]
print(f"[OK] energy_c3_ef_db (배출계수_통합_DB.xlsx > 에너지_C3): {len(energy_c3_ef_db)}행")


def _c3_norm(x):
    return _ef_norm_text(x) if "_ef_norm_text" in globals() else str(x or "").strip().lower().replace(" ", "")


def _c3_norm_unit(x):
    return _ef_norm_unit(x) if "_ef_norm_unit" in globals() else str(x or "").strip().lower().replace(" ", "")


def _c3_energy_cols():
    db = energy_c3_ef_db
    name_col = next((c for c in db.columns if "활동유형" in str(c) or "에너지원" in str(c)), None)
    unit_col = next((c for c in db.columns if "기준단위" in str(c)), None)
    ef_col = next((c for c in db.columns if "EF" in str(c) and "kgCO2" in str(c)), None)
    return db, name_col, unit_col, ef_col


def _c3_energy_alias_candidates(energy_name, unit_hint=None):
    """C3 조회용 별칭 후보 (기존 lookup_energy_ef의 별칭 규칙과 동일하게 유지)."""
    raw = str(energy_name or "").strip()
    q = _c3_norm(raw)
    u = _c3_norm_unit(unit_hint)
    candidates = []

    def add(*vals):
        for v in vals:
            if v and v not in candidates:
                candidates.append(v)

    add(raw)
    if any(k in q for k in ["전기", "전력", "electricity", "electric", "power"]):
        add("전력")
    if any(k in q for k in ["스팀", "steam", "열"]):
        if "열병합" in q or "chp" in q:
            add("스팀(열병합발전)", "스팀")
        else:
            add("스팀")
    if any(k in q for k in ["lng", "천연가스", "도시가스", "액화천연가스", "naturalgas"]):
        if u == "kg":
            add("천연가스(LNG)", "도시가스(LNG)")
        else:
            add("도시가스(LNG)", "천연가스(LNG)")
    if any(k in q for k in ["lpg", "액화석유가스", "프로판", "부탄"]):
        add("액화석유가스(LPG)")
    if any(k in q for k in ["경유", "diesel"]):
        add("경유")
    if any(k in q for k in ["휘발유", "가솔린", "gasoline", "petrol"]):
        add("휘발유")
    if any(k in q for k in ["등유", "kerosene"]):
        add("등유")
    if any(k in q for k in ["무연탄"]):
        add("무연탄")
    if any(k in q for k in ["벙커c", "bunkerc", "벙커씨"]):
        add("벙커C유")
    if any(k in q for k in ["중유", "heavyoil"]):
        add("중유")
    if any(k in q for k in ["석탄", "유연탄", "coal"]):
        add("석탄")
    return candidates


def lookup_energy_ef_c3(energy_name: str, unit_hint: str = None):
    """
    C3(연료 및 에너지 관련 활동) 전용 EF 조회.
    배출계수_통합_DB.xlsx > 에너지_C3 시트(업스트림·송배전손실 기준)에서만 찾는다.
    반환: (raw_name, ef_kgCO2eq_per_unit, ef_unit) — 못 찾으면 (None, None, None)
    """
    db, name_col, unit_col, ef_col = _c3_energy_cols()
    if db is None or db.empty or not name_col or not ef_col:
        return None, None, None

    candidates = _c3_energy_alias_candidates(energy_name, unit_hint=unit_hint)
    for cand in candidates:
        cn = _c3_norm(cand)
        subset = db[db[name_col].apply(lambda x: _c3_norm(x) == cn)]
        if not subset.empty:
            r = subset.iloc[0]
            ef = pd.to_numeric(r[ef_col], errors="coerce")
            if pd.notna(ef):
                return r[name_col], float(ef), (r[unit_col] if unit_col else None)

    # contains 매칭 폴백
    q = _c3_norm(energy_name)
    if q:
        subset = db[db[name_col].apply(lambda x: q in _c3_norm(x) or _c3_norm(x) in q)]
        if not subset.empty:
            r = subset.iloc[0]
            ef = pd.to_numeric(r[ef_col], errors="coerce")
            if pd.notna(ef):
                return r[name_col], float(ef), (r[unit_col] if unit_col else None)

    return None, None, None


def _bridge_convert_via_lhv(value, value_unit, from_energy_name, to_energy_name, to_unit):
    """
    DB에서 원래 연료(from_energy_name, 예: 도시가스(LNG))의 EF가 삭제되어
    다른 연료(to_energy_name, 예: 천연가스(LNG))의 EF로 대체 매칭됐을 때,
    두 연료 각각의 발열량(LHV, MJ/단위)을 거쳐 value_unit(예: Nm³) → to_unit(예: kg)으로
    물리적으로 타당하게 환산한다. (동일한 열량을 갖는다고 가정)
    조건이 안 맞으면(LHV 정보 없음/단위 불일치) 환산하지 않고 원본 그대로 반환한다.
    """
    def _u_norm(u):
        return str(u or "").strip().lower().replace(" ", "").replace("³", "3").replace("²", "2")

    from_info = ENERGY_BASE_UNIT.get(_detect_energy_name(from_energy_name))
    to_info = ENERGY_BASE_UNIT.get(_detect_energy_name(to_energy_name))
    if not from_info or not to_info:
        return value, value_unit, ""

    from_unit, from_lhv = from_info
    to_unit_expected, to_lhv = to_info
    if from_lhv is None or to_lhv is None or to_lhv <= 0:
        return value, value_unit, ""
    if _u_norm(value_unit) != _u_norm(from_unit) or _u_norm(to_unit) != _u_norm(to_unit_expected):
        return value, value_unit, ""

    converted = value * from_lhv / to_lhv
    note = f"{value_unit}→{to_unit} (발열량 환산: {from_energy_name}→{to_energy_name}, ×{(from_lhv / to_lhv):.6g})"
    return converted, to_unit, note


print("[OK] lookup_energy_ef_c3() 로드 완료 - C3 계산에서 이 함수를 사용하도록 연결됩니다.")

# ── [원본 셀 58] ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
# 추가 기능: "원본 템플릿 형식 유지 + 회색 칸 채우기" 결과 파일 생성
# ------------------------------------------------------------------
# - 위 단계에서 만든 진단용 결과(diagnostic_output_file, 예: *_result.xlsx)의
#   배출계수 / 배출량 값을 그대로 가져와서,
#   원본 업로드 템플릿(input_file)의 회색 칸("AI 추천" / "AI 자동산정")에 채워 넣습니다.
# - 원본 템플릿의 안내문구, 서식, 드롭다운, 색상 등은 전혀 건드리지 않습니다.
# - 행 매칭은 "위치(순서)"가 아니라 "비고/설명 텍스트" 기준으로 합니다.
#   → 일부 시트(standardize_c1_sheet, _melt_monthly_or_annual 등)는 연간입력 행과
#     월별입력 행을 내부적으로 분리했다가 다시 합치면서 원본 파일의 행 순서와
#     달라집니다. 위치 기준 매칭은 이 경우 엉뚱한 행의 값을 가져오게 되므로,
#     대신 각 행의 "비고(선택)"/"설명/적요(선택)" 텍스트로 정확히 짝을 맞춥니다.
#     (비고 컬럼이 없는 시트는 위치 기준 매칭으로 폴백합니다.)
# ══════════════════════════════════════════════════════════════
import openpyxl
from openpyxl.cell.cell import MergedCell
import pandas as pd
import os


def _set_cell_value(ws, row, col, value):
    """병합된 셀이면 병합 범위의 좌상단 셀에 값을 쓴다(그 외 셀은 읽기전용이라 직접 쓰면 에러 발생)."""
    if value is None:
        return
    cell = ws.cell(row=row, column=col)
    if isinstance(cell, MergedCell):
        for merged_range in ws.merged_cells.ranges:
            if cell.coordinate in merged_range:
                top_left_row, top_left_col = merged_range.min_row, merged_range.min_col
                ws.cell(row=top_left_row, column=top_left_col).value = value
                return
        return  # 해당 병합범위를 못 찾으면 조용히 스킵
    cell.value = value


def _find_ef_em_columns(ws, max_scan_row=30):
    """
    시트에서 '배출계수' 헤더와 '배출량' 헤더가 같은 행에 있는
    (헤더 마지막행, EF열, 배출량열, EF헤더텍스트)를 찾는다.
    헤더가 2행에 걸쳐 병합되어 있는 시트(대부분의 시트)를 고려해, 병합범위의 마지막 행까지 반영한다.
    """
    max_row = min(max_scan_row, ws.max_row)
    for r in range(1, max_row + 1):
        ef_col = em_col = None
        ef_header_text = None
        for c in range(1, ws.max_column + 1):
            v = ws.cell(row=r, column=c).value
            if isinstance(v, str):
                if "배출계수" in v:
                    ef_col = c
                    ef_header_text = v
                elif "배출량" in v:
                    em_col = c
        if ef_col and em_col:
            header_end = r
            for col in (ef_col, em_col):
                hc = ws.cell(row=r, column=col)
                for mc in ws.merged_cells.ranges:
                    if hc.coordinate in mc:
                        header_end = max(header_end, mc.max_row)
            return header_end, ef_col, em_col, ef_header_text
    return None, None, None, None


def _ef_unit_scale_factor(header_text):
    """
    템플릿의 배출계수 헤더 텍스트를 보고, 배출계수_통합_DB의 kgCO2e 기준 값을
    그 헤더가 표시하는 단위(주로 tCO2e)로 맞추기 위한 배율을 반환한다.
    - 배출계수_통합_DB.xlsx의 모든 물리량 기반 배출계수(에너지/운송/출장/폐기물 등)는
      kgCO2e(kg 단위계)로 관리되는데, 원본 템플릿 헤더 중 상당수가 "tCO2e"라고
      표기되어 있어(예: '배출계수(tCO2e/단위)') 그대로 채우면 단위가 1000배 어긋난다.
    - 헤더에 kgCO2e/kgCO2eq가 명시된 경우(C1/C2/C4-2/C6-2/C9-2의 구매·운송·출장
      지출기반 EF)는 이미 kg 기준이 맞으므로 그대로 둔다.
    """
    t = str(header_text or "")
    t_compact = t.replace(" ", "")
    if "kgco2e" in t_compact.lower() or "kgco2eq" in t_compact.lower():
        return 1.0
    if "tco2e" in t_compact.lower() or "tco2eq" in t_compact.lower():
        return 1.0 / 1000.0
    return 1.0


def _find_note_column(ws, max_scan_row=30):
    """원본 시트에서 '비고' 또는 '설명/적요' 헤더 열을 찾는다 (행 매칭용 키).
    헤더 텍스트는 항상 짧은 라벨이므로, 안내문(예: '...분류방법 설명서...' 같은 긴 문장)에
    우연히 '설명'/'적요'/'비고'가 섞여 들어가 있어 오탐지되는 것을 길이 제한으로 막는다."""
    max_row = min(max_scan_row, ws.max_row)
    for r in range(1, max_row + 1):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(row=r, column=c).value
            if isinstance(v, str) and len(v.strip()) <= 15 and ("비고" in v or "설명" in v or "적요" in v):
                return c
    return None


def _find_data_rows(ws, header_end_row, n_needed, scan_max_col):
    """
    헤더 마지막행 다음부터, 완전히 빈 행은 건너뛰고 실제 데이터가 있는 행 번호를 순서대로 n_needed개 찾는다.
    scan_max_col까지만 검사해서, 시트 오른쪽 끝의 드롭다운 보조목록 열(예: 대분류목록 등)이
    데이터 행으로 오인되는 것을 방지한다.
    """
    rows = []
    r = header_end_row + 1
    while len(rows) < n_needed and r <= ws.max_row:
        vals = [ws.cell(row=r, column=c).value for c in range(1, scan_max_col + 1)]
        if any(v is not None and str(v).strip() != "" for v in vals):
            rows.append(r)
        r += 1
    return rows


_EF_META_COLS = {"배출계수기준연도", "배출계수기준컬럼", "배출계수단위", "EF단위", "EF매핑명", "EF매핑방식", "EF매핑점수"}


def _extract_ef_em(row: pd.Series):
    """계산 결과 한 행(row)에서 대표 배출계수/배출량(tCO2e) 숫자값을 뽑아낸다."""
    ef_val = None
    for c in row.index:
        cs = str(c)
        if cs in _EF_META_COLS:
            continue
        if "배출계수" in cs or cs == "EF(kgCO2e/KRW)":
            v = row.get(c)
            if isinstance(v, (int, float)) and pd.notnull(v) and not isinstance(v, bool):
                ef_val = v
                break

    em_val = None
    priority_cols = [c for c in row.index if str(c) == "배출량(tCO2e)"]
    priority_cols += [c for c in row.index if str(c) == "배출량"]
    priority_cols += [
        c for c in row.index
        if "배출량" in str(c)
        and "kgCO2e" not in str(c)
        and "예상" not in str(c)
        and "추천" not in str(c)
        and str(c) not in ("배출량(tCO2e)", "배출량")
    ]
    for c in priority_cols:
        v = row.get(c)
        if isinstance(v, (int, float)) and pd.notnull(v) and not isinstance(v, bool):
            em_val = v
            break
    return ef_val, em_val


def _find_result_note_column(df: pd.DataFrame):
    """진단용 결과 시트에서 '비고' 또는 '설명/적요' 컬럼을 찾는다."""
    for c in df.columns:
        cs = str(c)
        if "비고" in cs or "설명" in cs or "적요" in cs:
            return c
    return None


_OUT_SHEET_NAME_MAP = {
    "Category1":  "C1_구매품서비스",
    "Category2":  "C2_자본재",
    "Category3":  "C3_연료에너지",
    "Category4":  "C4_업스트림운송",
    "Category5":  "C5_사업장폐기물",
    "Category6":  "C6_출장",
    "Category7":  "C7_통근",
    "Category8":  "C8_임차자산",
    "Category9":  "C9_다운스트림운송",
    "Category11": "C11_판매제품사용",
    "Category12": "C12_판매제품폐기",
    "Category13": "C13_임대자산",
    "Category14": "C14_프랜차이즈",
    "Category15": "C15_투자",
}


def build_filled_template(input_file: str, diagnostic_output_file: str, filled_output_file: str):
    """
    diagnostic_output_file(이미 생성된 *_result.xlsx)의 계산값을 이용해,
    input_file(원본 템플릿)의 회색 칸을 채운 filled_output_file을 생성한다.
    """
    result_sheets = pd.read_excel(diagnostic_output_file, sheet_name=None)
    xls = pd.ExcelFile(input_file)

    # 카테고리별 결과 커서: 같은 카테고리 결과 시트(예: C4_업스트림운송)에
    # 여러 원본 시트(C4-1, C4-2)가 순서대로 이어붙여져 있으므로 순서대로 잘라서 매칭한다.
    cursors = {cat: 0 for cat in _OUT_SHEET_NAME_MAP}

    wb = openpyxl.load_workbook(input_file)

    for sheet_name in xls.sheet_names:
        match = _match_sheet_processor(sheet_name)
        if match is None:
            continue
        std_fn, category, theme = match
        out_name = _OUT_SHEET_NAME_MAP.get(category)
        if out_name is None or out_name not in result_sheets:
            continue

        try:
            df0 = read_template_sheet(input_file, sheet_name)
            if df0.empty:
                continue
            df0 = std_fn(df0)
        except Exception as e:
            print(f"  [filled-template][skip] {sheet_name}: 표준화 실패 ({e})")
            continue

        n = len(df0)
        start = cursors.get(category, 0)
        end = start + n
        matched = result_sheets[out_name].iloc[start:end].copy()
        cursors[category] = end

        if sheet_name not in wb.sheetnames or matched.empty:
            continue

        ws = wb[sheet_name]
        header_end, ef_col, em_col, ef_header_text = _find_ef_em_columns(ws)
        if header_end is None:
            print(f"  [filled-template][skip] {sheet_name}: 배출계수/배출량 칸을 찾지 못함")
            continue
        ef_scale = _ef_unit_scale_factor(ef_header_text)

        data_rows = _find_data_rows(ws, header_end, max(len(matched), 500), em_col)
        # 원본 파일에 실제로 있는 데이터 행 수만큼만 사용 (n행 기준)
        data_rows = data_rows[:n] if len(data_rows) > n else data_rows

        note_col_ws = _find_note_column(ws, header_end)
        note_col_df = _find_result_note_column(matched)

        filled_count = 0

        if note_col_ws and note_col_df:
            # ── 비고/설명 텍스트 기준 매칭 (행 순서에 의존하지 않음) ──
            used_idx = set()
            unmatched_rows = []  # 비고가 비어있거나 텍스트로 못 찾은 ws 행 번호(순서 보존)
            matched_notes = matched[note_col_df].astype(str).str.strip()
            for r in data_rows:
                note_val = ws.cell(row=r, column=note_col_ws).value
                note_key = str(note_val).strip() if note_val is not None else ""
                match_row = None
                if note_key:
                    cand_idx = [i for i in matched.index[matched_notes == note_key] if i not in used_idx]
                    if cand_idx:
                        match_row = matched.loc[cand_idx[0]]
                        used_idx.add(cand_idx[0])
                if match_row is None:
                    unmatched_rows.append(r)
                    continue
                ef_val, em_val = _extract_ef_em(match_row)
                if ef_val is not None:
                    _set_cell_value(ws, r, ef_col, ef_val * ef_scale)
                    filled_count += 1
                if em_val is not None:
                    _set_cell_value(ws, r, em_col, em_val)
                    filled_count += 1

            # ── 텍스트로 못 짝지은 행들(비고 공란/중복)은 남은 결과 행과 순서대로 짝지어 채움 ──
            leftover_idx = [i for i in matched.index if i not in used_idx]
            if unmatched_rows and leftover_idx:
                for r, i in zip(unmatched_rows, leftover_idx):
                    match_row = matched.loc[i]
                    used_idx.add(i)
                    ef_val, em_val = _extract_ef_em(match_row)
                    if ef_val is not None:
                        _set_cell_value(ws, r, ef_col, ef_val * ef_scale)
                        filled_count += 1
                    if em_val is not None:
                        _set_cell_value(ws, r, em_col, em_val)
                        filled_count += 1

            if len(used_idx) < len(matched):
                print(f"  [filled-template][주의] {sheet_name}: 비고 텍스트로 매칭 안 된 결과 행 "
                      f"{len(matched) - len(used_idx)}개 존재 (비고가 비어있거나 중복된 경우일 수 있음)")
        else:
            # ── 비고 컬럼이 없는 시트(C14 등)는 위치 기준 매칭으로 폴백 ──
            for i, r in enumerate(data_rows):
                if i >= len(matched):
                    break
                ef_val, em_val = _extract_ef_em(matched.iloc[i])
                if ef_val is not None:
                    _set_cell_value(ws, r, ef_col, ef_val * ef_scale)
                    filled_count += 1
                if em_val is not None:
                    _set_cell_value(ws, r, em_col, em_val)
                    filled_count += 1

        print(f"  [filled-template] {sheet_name}: {filled_count}칸 채움")

    wb.save(filled_output_file)
    print(f"\n완료: {filled_output_file}")


# ══════════════════════════════════════════════════════════════
# 웹/서버용 진입점
# ══════════════════════════════════════════════════════════════
def run_scope3_pipeline(input_file: str, output_dir: str = "result",
                         report_year_value: int = 2025, theme_value=None) -> dict:
    """
    Scope3 템플릿(xlsx) 하나를 받아 결과 파일 2개를 생성합니다.

    Args:
        input_file: 입력 템플릿 엑셀 파일 경로
        output_dir: 결과 파일을 저장할 폴더 (없으면 자동 생성)
        report_year_value: 보고연도 (배출계수_통합_DB의 기계산 KRW EF 컬럼 선택 기준)
        theme_value: 특정 테마로 강제 지정하고 싶을 때만 값을 넣음 (기본 None)

    Returns:
        {
            "result_file": "<진단용 상세 결과 파일 경로>",
            "filled_template_file": "<원본 템플릿 형식 + 회색 칸 채운 결과 파일 경로>",
        }
    """
    global report_year, theme
    report_year = report_year_value
    theme = theme_value

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_result.xlsx")
    filled_output_file = os.path.join(output_dir, f"{base_name}_filled_template.xlsx")

    process_template_inventory(
        input_file, output_file,
        forced_theme=theme,
        report_year=report_year,
    )
    print("생성 파일 (1/2, 진단용 상세 결과):", output_file)

    build_filled_template(input_file, output_file, filled_output_file)
    print("생성 파일 (2/2, 원본 템플릿 형식):", filled_output_file)

    return {"result_file": output_file, "filled_template_file": filled_output_file}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scope3 AI 자동 산정 파이프라인")
    parser.add_argument("input_file", help="입력 템플릿 엑셀 파일 경로")
    parser.add_argument("--output-dir", default="result", help="결과 파일 저장 폴더 (기본: result 폴더)")
    parser.add_argument("--report-year", type=int, default=2025, help="보고연도 (기본: 2025)")
    parser.add_argument("--theme", default=None, help="테마 강제 지정 (기본: 자동 판별)")
    args = parser.parse_args()

    result = run_scope3_pipeline(
        args.input_file,
        output_dir=args.output_dir,
        report_year_value=args.report_year,
        theme_value=args.theme,
    )
    print(result)
