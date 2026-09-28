# Scope3 AI 자동 산정 파이프라인

Scope3 배출량 산정 템플릿(엑셀)을 입력받아, 카테고리 1~15의 온실가스 배출량을 자동으로 계산하는 파이프라인입니다.
품목 분류·도시명 변환·이동수단 분류·거리 추정 등에 LLM(Claude 또는 OpenAI)을 사용합니다.

## 폴더 구조

```
data/배출계수_통합_DB.xlsx     배출계수 DB (카테고리별 EF, 항상 이 위치에서 읽음)
result/                        파이프라인 실행 결과가 저장되는 폴더
scope3_pipeline_claude.py      파이프라인 본체 (Claude Sonnet 버전)
scope3_pipeline_OPENAI.py      파이프라인 본체 (OpenAI 버전, 계산 로직은 동일)
Scope3_AI_Claude_Colab.ipynb   Google Colab용 노트북 (Claude 버전)
Scope3_AI_OpenAI_Colab.ipynb   Google Colab용 노트북 (OpenAI 버전)
Scope3_템플릿_수정본_v2.2.xlsx  회사별 커스터마이징 없는 기본(마스터) 템플릿
```

두 `.py` 파일과 두 `.ipynb` 파일은 계산 로직이 완전히 동일하고, 품목 분류·거리 추정 등에 쓰는 LLM만 다릅니다.
Colab 노트북은 각 `.py` 파일의 핵심 로직을 그대로 셀 단위로 옮겨 둔 것이라, 둘 중 하나만 실제로 수정하면
다른 쪽은 동기화가 어긋납니다(아래 "코드 동기화" 참고).

## 1. 사전 준비

```powershell
pip install -r requirements.txt
```

`requirements.txt` 맨 아래 주석대로, OpenAI 버전을 쓸 경우 `anthropic` 대신 `openai`만 있으면 됩니다(둘 다 설치해도 무방).

API 키는 `.env` 파일에 넣어두면 터미널 실행 시 자동으로 읽힙니다(`.env`는 git에 올라가지 않음):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

`배출계수_통합_DB.xlsx`는 기본적으로 `data/` 폴더에서 찾습니다. 다른 위치를 쓰려면 환경변수로 지정하세요.

```powershell
$env:SCOPE3_BASE_PATH = "다른경로"
```

## 2. 터미널에서 실행하기

```powershell
# Claude 버전
python scope3_pipeline_claude.py "입력템플릿.xlsx"

# OpenAI 버전
python scope3_pipeline_OPENAI.py "입력템플릿.xlsx"
```

옵션:

```powershell
python scope3_pipeline_claude.py "입력템플릿.xlsx" --output-dir result --report-year 2025
```

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--output-dir` | 결과 파일 저장 폴더 | `result` |
| `--report-year` | 보고연도(배출계수_통합_DB의 연도별 KRW 환산 EF 컬럼 선택 기준) | `2025` |
| `--theme` | 특정 테마로 강제 지정(보통 지정 안 해도 됨) | 자동 판별 |

## 3. Google Colab에서 실행하기

1. `Scope3_AI_Claude_Colab.ipynb`(또는 `Scope3_AI_OpenAI_Colab.ipynb`)를 Colab에서 엽니다.
2. 왼쪽 열쇠 아이콘(보안 비밀/Secrets)에서 `ANTHROPIC_API_KEY`(또는 `OPENAI_API_KEY`)를 등록하고 "노트북 액세스"를 켭니다.
3. 셀을 위에서부터 순서대로 실행합니다.
   - Google Drive를 마운트하고, `배출계수_통합_DB.xlsx`가 들어있는 Drive 폴더 경로(`base_path`)를 확인/수정합니다.
   - 파일 업로드 셀에서 산정할 Scope3 템플릿(xlsx)을 선택합니다.
4. 마지막 실행 셀까지 가면 결과 파일 2개가 자동으로 다운로드됩니다.

## 4. 결과 파일

실행하면 `<입력파일명>_result.xlsx`, `<입력파일명>_filled_template.xlsx` 두 개가 생성됩니다.

- **`_result.xlsx`** (진단용 상세 결과): 카테고리별로 계산에 쓰인 중간값(배출계수, 매칭 근거 등)까지 전부 보여줍니다.
  카테고리 5/12에는 `화석기원 CO2e(tCO2e)` / `생물기원 CO2e(tCO2e)` 컬럼이 별도로 있는데, `배출량(tCO2e)`(Scope 3
  집계에 쓰이는 값)에는 화석기원분만 포함되고 생물기원 CO2는 제외되어 있습니다.
- **`_filled_template.xlsx`**: 원본 템플릿의 서식·안내문·드롭다운은 그대로 두고, 회색 칸(배출계수/배출량)만 채운 버전입니다.

## 5. 회사별 자동매칭을 쓰는 템플릿이라면

`기업정보`/`회사별_폐기물매핑`/`회사별_교통수단매핑` 시트가 있는 템플릿(예: 회사별 커스터마이징된 버전)은,
엑셀의 자동매칭 수식이 재계산되지 않은 상태로 업로드돼도 파이프라인이 매핑 시트를 직접 읽어 계산합니다.
다만 엑셀에서 그 파일을 열어봤을 때 "폐기물 종류"/"교통수단" 칸이 비어 보이는 것 자체는 정상입니다
(계산 결과에는 영향 없음 — 보고 싶으면 엑셀에서 한 번 열었다 저장하면 화면에도 값이 채워집니다).

## 6. 생물 기원 수치
BIOGENIC_WASTE_FCF = {"음식물류": 0.0, "폐지류": 0.01, "폐목재류": 0.0}
값의 의미: 소각했을 때 배출량 중 몇 %가 "화석기원"이냐는 비율입니다.
출처 : IPCC 2006 Vol.5 Table 2.4 

## 코드 동기화 (유지보수용)

`scope3_pipeline_*.py` 파일 안에는 `# ── [원본 셀 N] ─` 형태의 주석이 남아있고, 대응하는 Colab 노트북의
코드 셀 순서와 1:1로 맞춰져 있습니다. `.py` 파일의 계산 로직을 고친 뒤에는, 그 노트북의 같은 번호 셀에
동일한 내용을 옮겨 적어야 노트북도 최신 상태가 됩니다(Colab 전용 설정 셀·마지막 실행/다운로드 셀은 건드리지 않음).
