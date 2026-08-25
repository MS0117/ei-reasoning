# Toy-cliff 실험 플레이북 (transient LoRA / bridge / staged)

한 사이클(rollout → partition → anchor → improve → filters)만 돌려 **개선 연산자끼리 짝지어 비교**하는
실험의 실행·분석·확장 방법. train/eval은 하지 않으므로 지표는 **conversion rate**(cliff 문제 중 정답
샘플을 1개 이상 만들어낸 비율)와 후보 품질 지표들이다. 풀 EI 루프는 `configs/methods/*.yaml` + `scripts/run.sh`.

작성 2026-08-22. 결과 스냅샷은 `docs/results/toy_cliff/`, 코드 진입점은 `data/toy_cliff.py`.

---

## 1. 데이터: 무엇 위에서 도는가

| 항목 | 값 |
|---|---|
| 문제 집합 | `data/cliff_sets/openr1_qwen3-4b-2507_n2000_with_gold.jsonl` (137문제, gold 포함) |
| 정책 모델 | `Qwen/Qwen3-4B-Instruct-2507` |
| cliff 정의 | rollout 8개가 **전부 오답**(`partition.cliff_max_correct: 0`) → 137개 중 **107개** |
| 공유 rollout | `runs/toy_cliff/default_LSPO_20260813_155520` (모든 arm이 재사용) |
| cliff qid 목록 | `docs/results/toy_cliff/shared_rollout_cliff_qids.jsonl` (107줄, GPU 없이 분석용) |

**gold 해답(y\*)은 옵트인**이다. 위 `_with_gold` 파일은 `scripts/backfill_gold_solutions.py`로 OpenR1에서
정답 풀이를 붙여둔 것이고, y\*는 `questions/train.jsonl`의 meta에만 존재한다. y\*는 **bridge 생성 프롬프트와
privileged 스코어링에만** 들어가고 rollout/eval 프롬프트나 훈련 텍스트에는 절대 들어가지 않는다.

### 왜 rollout을 재사용하나 (`--reuse-rollout`)

cliff 집합은 "K=8 샘플이 전부 틀렸다"는 **한 번의 측정**으로 정의된다. 다시 뽑으면 pass rate가 낮지만 0이
아닌 문제들이 들락날락한다(실측: 137 → 107로 변동). rollout을 재사용하면 (1) cliff 집합이 고정되고
(2) anchor가 자르는 실패 궤적까지 동일해져서, arm 간 차이가 **개선 연산자 차이만** 반영하는 paired 비교가
된다. 덤으로 4B rollout 비용(~1 GPU-h)도 아낀다.

`data/toy_cliff.py:reuse_rollout`이 rollout을 결정하는 config 필드(`ROLLOUT_INPUT_FIELDS`: seed, model,
data, rollout, partition)를 **정확히 대조**해서 하나라도 다르면 거부한다. 즉 improve/filter/anchor만 바꾼
arm은 안전하게 재사용할 수 있고, 모델이나 샘플 수를 바꾸면 자동으로 막힌다.

> **다른 머신에서 시작한다면**: `runs/` 는 gitignore이고 공유 rollout 디렉토리는 218MB라 저장소에 없다.
> srv04에서는 위 경로를 그대로 쓰면 되고, 그 외 환경에서는 `-c data/configs/LSPO.yaml`을
> `--reuse-rollout` 없이 한 번 돌려 새 기준 rollout을 만든 뒤(=~1 GPU-h) 그 디렉토리를 계속 재사용하면 된다.

---

## 2. 실행

```bash
# 항상 프로젝트 venv (시스템 python 금지)
bash data/run_toy_cliff.sh -c data/configs/STAGED.yaml -b \
    -- --reuse-rollout runs/toy_cliff/default_LSPO_20260813_155520
```

- `-b` : 백그라운드 실행. 로그는 `logs/toy_cliff_<run>.log`, 결과는 `runs/toy_cliff/<slug>_<타임스탬프>/`
  (**매 실행마다 새 디렉토리** — 덮어쓰기 없음).
- `-r <run_dir>` : 크래시한 런 이어서(스테이지별 `.done` 마커 + content-addressed adapter 캐시로 스킵).
- `-- --override a.b.c=값` : 임시 변형. 다만 **재현 가능한 arm은 YAML을 직접 고치는 쪽**이 낫다
  (frozen config가 곧 실험 기록이 된다).
- 소요: 107 cliff 기준 **약 2.5시간**(A100 2장). anchor ~7분, bridge 생성 ~40분, fit 수십 분, rollout 나머지.

### GPU 에티켓 (srv04)

GPU는 공유다. **`a100` tmux 세션 안에서** 실행할 것 — 세션 밖에서는 `nvidia-smi`가 다른 장치를 보여준다
(cgroup 격리). 세션 안에서는 `-g` 플래그 없이 돌리면 보이는 GPU를 전부 쓴다. PCIe 머신이라 NVLink가
없어서 스크립트가 `NCCL_P2P_DISABLE=1`을 자동으로 켠다.

**런을 죽일 때는 자식 vLLM 프로세스까지** 죽여야 한다. 드라이버만 죽이면 `VLLM::EngineCore`가 살아남아
GPU 메모리 78GB를 잡고 있고, 다음 런이 "Free memory ... less than desired" 로 실패한다:

```bash
pkill -u "$USER" -f "toy_cliff.py --config data/configs/STAGED.yaml"
pkill -u "$USER" -f "VLLM::EngineCore"          # 반드시 같이
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # 0 MiB 확인
```

### 배관만 빠르게 확인하고 싶을 때 (수 분, 0.6B)

```bash
bash scripts/smoke.sh <GPU> configs/methods/smoke_staged.yaml
```
`smoke_staged.yaml`은 regen_bridge + full_pool + self-wash + DPO를 한 번에 통과하도록 맞춰져 있다.
smoke.sh는 `--override`를 받지 않으므로 다른 조합을 보려면 그 파일을 직접 고친다.

---

## 3. YAML: 어디에 무엇이

| 파일 | 용도 |
|---|---|
| `data/configs/LSPO.yaml` | toy: gold (x→y\*) LoRA — `lora_sft` 연산자 |
| `data/configs/BRIDGE.yaml` | toy: 자기 생성 bridge LoRA — `bridge_sft` |
| `data/configs/STAGED.yaml` | toy: 다단계 bridge — `staged_bridge_sft` |
| `data/configs/CONTROL.yaml` | toy: 연산자 없음(`self_resample`) |
| `configs/methods/staged_bridge_sft.yaml` | **풀 EI 루프** 메인 arm (train/eval 포함) |
| `configs/methods/bridge_sft.yaml` | 풀 루프 단일-fit 통제 |
| `configs/methods/gold_lora_sft.yaml` | 풀 루프 gold 기준선 |
| `configs/methods/self_resample.yaml` | 풀 루프 무연산자 통제 |

**어떤 연산자를 쓸지는 `improve.operator` 한 줄**이 정한다(`lora_sft | bridge_sft | staged_bridge_sft |
self_resample`). 알 수 없는 YAML 키는 로드 시 하드 에러이므로 오타가 조용히 무시되지 않는다.

### staged 연산자 knob (`improve.lora_sft.staged`)

| knob | 기본 | 의미 |
|---|---|---|
| `rollout_n` | 8 | stage-1 어댑터에서 문제당 샘플 수 (중간 rollout에도 동일 적용) |
| `final_rollout_n` / `final_rollout_scope` | 16 / `unsolved` | 마지막 rollout의 샘플 수와 대상 |
| `num_stages` | 1 | stage-2 (fit→rollout) 사이클 반복 횟수 |
| `stage2_steps` | 2 | stage-2 fit의 gradient step (stage-1은 `fit.steps`) |
| `chain_adapter` | true | stage-2를 직전 어댑터에서 warm-start |
| `unsolved_targets` | `reuse_bridge` | 못 푼 문제의 fit 타깃: 기존 bridge 재사용 / `regen_bridge`(현 어댑터로 재생성) / `add_bridge`(둘을 병합) |
| `stage_bridge_n`, `stage_max_keep` | null | 재생성 시 샘플 수와 병합 후 문제당 상한 |
| `train_scope` / `solved_targets` | `unsolved_only` / `self_wash_min_c` | 이미 푼 문제도 fit에 넣을지, 넣는다면 어떤 궤적으로 |
| `stage1_chunk_size` / `stage2_chunk_size` | 0 / 0 | 샤딩(0=pooled). stage-2는 stage-1 샤드의 **하위 샤드**라 `stage2 ≤ stage1` |
| `stage2_objective` + `dpo.*` | `sft` | stage-2를 SFT 대신 DPO로 (chosen=bridge, rejected=직전 rollout의 자기 실패) |
| `emit` | `all` | filters로 내보낼 rollout 라운드 범위 |

관련 공용 knob: `improve.lora_sft.fit.*`(LoRA 하이퍼파라미터, `steps`=stage-1 스텝),
`improve.lora_sft.bridge.*`(bridge 생성 수·채택 규칙 `keep_selection`),
`filter.selection.method`(문제당 최종 선택: shortest / c_score / random), `filter.max_per_question`.

주의: **`improve.n`은 staged에서 쓰이지 않는다**(rollout 수는 `staged.*`가 정함). 단
`improve.temperature/top_p/max_tokens`는 bridge·rollout 생성에 그대로 쓰인다.

---

## 4. 코드: 어디를 고쳐야 하나

```
src/expert_iter/
  improve.py            개선 연산자 스테이지 진입점 + 레지스트리 (self_resample, teacher)
  lora_sft.py           transient LoRA 연산자의 뼈대: propose / _build_targets / _sample /
                        _collect / _fit_adapter  ← 다른 연산자들이 전부 재사용
  bridge_sft.py         LoraSftOperator 상속. bridge 생성·검증·페어 구성(_build_targets)
  staged_bridge_sft.py  BridgeSftOperator 상속. propose를 오버라이드해 다단계 스케줄 구현
  lora_fit.py           GPU 서브프로세스: 순수 torch AdamW 루프(SFT/DPO), DDP, 어댑터 저장
  filters.py            학습가능성 게이트 + C(y) 스코어링/선택
  config.py             전 knob의 단일 소스 (dataclass ↔ YAML, 알 수 없는 키는 에러)
data/toy_cliff.py       toy 드라이버(스테이지 in-process 실행 + 퍼널/알파/C(y) 분석)
data/rank_toy_runs.py   완료된 toy 런 랭킹·품질 표
```

### 새 변형을 추가하는 전형적 경로

1. **knob 추가**: `config.py`의 해당 dataclass에 필드 + `validate()`에 범위/enum 검사.
   (필드 없이 YAML에만 쓰면 로드 에러가 난다.)
2. **동작 구현**: 대개 `staged_bridge_sft.py`의 `propose()` 또는 헬퍼(`_dpo_pairs`, `_merge_pairs`,
   `_assign_chunks`)에 분기 추가. bridge 생성 자체를 바꿔야 하면 `bridge_sft.py`의
   `_build_targets`를 건드리는데, **기존 동작은 기본값에서 byte-identical 유지**할 것
   (기존 테스트가 회귀 가드다).
3. **프롬프트를 바꾼다면** `templates.py`에만 문자열/렌더러를 둔다. 텍스트가 토큰 id가 되는 곳은 이 파일뿐.
4. **테스트**: `tests/test_staged_bridge_sft_op.py` 패턴 — `_ls.run_pool`과 `_ls._fit_adapter`를
   monkeypatch해서 GPU 없이 "어떤 fit이 어떤 페어로 몇 스텝 돌았는지"를 검증한다. 새 분기마다 1개씩.
5. **프리셋 반영**: `data/configs/STAGED.yaml`(toy)과 `configs/methods/staged_bridge_sft.yaml`(메인)에
   knob과 주석 추가.

### 반드시 지켜야 하는 규칙

- **토큰 id가 진실**이다. 프롬프트·훈련 입력은 id 리스트 concat으로 만든다. 디코딩된 텍스트를 다시
  토큰화하면 BPE 병합이 경계를 흔든다(`templates.bridge_pair_ids` 등을 그대로 쓸 것).
- 연산자 모듈은 **`expert_iter.improve`를 import하면 안 된다**(모듈-as-main 재실행 → 중복 등록).
  공용 인프라는 `from . import lora_sft as _ls`로 접근한다. 회귀 테스트가 있다.
- 연산자가 질문 외에 무엇을 보여줬다면 `ImprovedCandidate.external_context`에 기록해야 한다
  (필터가 걸러낸다). LoRA 연산자들은 y\*가 **가중치 채널**로만 흐르므로 None을 유지한다.
- 후보의 `op_meta["lora_path"]`를 계속 채울 것 — C(y)의 D_tail이 그걸 쓴다.
- 새 fit을 추가하면 `params`에 실린 값이 그대로 fit 캐시 키에 들어간다(재현/resume이 여기 달려 있다).

---

## 5. 지금까지의 결과 (107 cliff, 전부 동일 rollout)

`.venv/bin/python data/rank_toy_runs.py` (커밋된 스냅샷: `--runs-dir docs/results/toy_cliff`)

| conv | 연산자 | 스텝 | 샘플 | 비고 |
|---|---|---|---|---|
| **0.551** | bridge_sft | 4 | 24 | chunk 25 |
| **0.551** | staged | 2+2 | 8+16 | 기본(reuse, unsolved_only) ← s2lift 최고 |
| 0.551 | staged | 2+2 | 8+16 | full_pool + self-wash |
| 0.533 | staged | 2+2 | 8+16 | stage-2 DPO (w=1 / w=0 동률) |
| 0.514 | bridge_sft | 4 | 24 | pooled |
| 0.514 | staged | 1+3 | 8+16 | add_bridge |
| 0.458–0.495 | staged | 2+2 | 8+16 | chunk25 / add_bridge |
| 0.421 | bridge_sft | 3 | 16 | 기준선 |
| 0.308 | lora_sft (gold) | 3 | 16 | LSPO |
| 0.243 | self_resample | – | 16 | 무연산자 |

**결론 요약**
- **가장 큰 단일 효과는 bridge > gold**(0.421 vs 0.308, p≈0.06)와 **예산**(3→4스텝: +9pp; 최종 rollout 8→16: +10pp).
- 같은 4스텝·24샘플 예산 안에서는 **스케줄·샤딩·목적함수·데이터 구성 어느 축도 0.55를 못 넘었다**
  (모든 arm 0.46~0.55, paired p 전부 > 0.3).
- **anchor는 켜지 말 것**: 유일한 anchored 런이 0.252로 같은 예산 대비 최악이었다.
- 못 푼 19문제 해부: **10개는 bridge 자체가 안 만들어지고**(base가 y\*를 봐도 못 씀), 9개는 bridge가 있어도
  전이 실패. 스케줄 조정으로는 안 움직이는 구조적 잔여다.
- DPO는 conversion 중립이지만 **훈련으로 넘어가는 후보의 C(y)가 가장 낮다**(2.91 vs SFT 3.63) — "더 많이
  푸는" 레버가 아니라 "더 배우기 쉬운 데이터" 레버일 가능성. 판정은 풀 루프에서만 가능하다.

### 해석 시 반드시 주의할 것

1. **stage-1 전환 수는 동일 설정에서도 13~30으로 흔들린다**(vLLM 배칭 비결정성). 최종 conversion만 보면
   운 좋은 stage-1이 만든 착시에 속는다. → `rank_toy_runs.py`의 **`s2lift`**(stage-1 이후 미해결 대비
   stage-2 전환율) 또는 두 런의 **공통 미해결 집합** 기준으로 비교할 것.
2. **±6문제 이하 차이는 노이즈**로 취급(N=107, 단일 시드). McNemar가 대부분 p>0.3이었다.
3. 품질 표의 `mean p̂`/`p̂≥.25`는 **staged에 불리한 편향**이 있다: stage-1에서 풀린 문제는 약한 2스텝
   어댑터로 8개만 뽑고 끝이라 분모가 다르다. 공정 비교는 "최종 어댑터로 뽑은 문제 집합"으로 제한할 것.
4. conversion은 "1개라도 맞췄나"라 분포의 질을 못 본다. p̂(정답 비율), 정답 길이, C(y)를 같이 볼 것.

---

## 6. 분석 도구

```bash
.venv/bin/python data/rank_toy_runs.py                       # 실제 런 디렉토리
.venv/bin/python data/rank_toy_runs.py --runs-dir docs/results/toy_cliff   # 커밋된 스냅샷
```

두 arm의 paired 비교(McNemar)는 이 패턴을 쓴다:

```python
import json
from math import comb
def solved(run):
    return {json.loads(l)["qid"]
            for l in open(f"runs/toy_cliff/{run}/iter_0/improve/improved.jsonl")
            if json.loads(l).get("correct")}
A, B = solved("default_STAGED_..."), solved("default_BRIDGE_...")
x, y = len(A - B), len(B - A); n, k = x + y, min(x, y)
p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
print(f"A-only {x}  B-only {y}  p≈{p:.2f}")
```

런 디렉토리에서 자주 보는 파일:
- `metrics.json` — 퍼널·conversion·C(y) 요약 (커밋 스냅샷에도 있음)
- `iter_0/improve/stats.json` — bridge 수율, 스테이지별 전환, 샤드/DPO 카운터
- `iter_0/improve/adapters/<name>/<hash>/fit_meta.json` — **fit별 loss 곡선**(DPO는 `reward_margin_per_step`,
  `pref_acc_per_step`). 언더피팅·likelihood 붕괴 진단은 여기서 시작한다.
- `iter_0/improve/bridge/bridges.jsonl` — bridge 샘플별 채점/채택 기록
- `iter_0/improve/improved.jsonl` — 후보 전체(정답 여부 prefilled, `op_meta.stage`로 라운드 구분)

---

## 7. 다음 실험 큐

**풀 EI 루프 (우선순위 높음)** — toy의 conversion 축은 소진됐고, 남은 질문은 "그 데이터로 학생이 실제로
좋아지는가"다. arm 4개는 이미 프리셋으로 정렬돼 있다(동일 데이터·예산, operator만 다름):
`staged_bridge_sft.yaml` / `bridge_sft.yaml` / `gold_lora_sft.yaml` / `self_resample.yaml`,
그리고 P1으로 staged + `--override improve.lora_sft.staged.stage2_objective=dpo`.

풀 루프 전에 정해야 할 설계 문제: **cliff 유래 예제가 iteration당 ~120/7,000(1.6%)** 이라 연산자 차이가
학생 eval에 안 드러날 수 있다. 후보는 `filter.max_per_question`↑ + `partition.solved_keep_max`↓ 로 개수를
맞추거나, 데이터셋을 키워 cliff 비율 자체를 40%로 만드는 것(이 경우 chunk 100 샤딩 필요).

**toy에서 남은 것 (선택)**
- failure-conditioned bridging: bridge 프롬프트에 y\*와 함께 **그 정책의 실패 궤적**을 넣어
  "같은 실수를 피해 다시 써라"로 유도. 실패를 손실 타깃이 아니라 **입력**으로 쓰므로 앵커(=분기점 검출)가
  필요 없다. 미구현.
- 500-cliff 파일럿: pooled vs chunk100 — "샤드당 250~320 pair" 규칙이 스케일에서 유효한지 확인.
- 이미 접은 것: anchor 계열, add_bridge, self-wash, per_problem refit, gold 폴백(전부 이득 없거나 억지).

---

## 8. 자주 밟는 함정

- **YAML을 IDE에서 열어둔 채 편집하면** 스크립트가 고친 내용이 나중 저장에 덮어써진다. 실행 전
  `Config.load`로 값을 한 번 확인할 것.
- 정규식으로 YAML 일괄 치환 금지(주석 블록을 삼켜 `rl:` 헤더가 사라진 사고가 있었다). 문자열 단위 편집.
- `improve.lora_sft.chunk_size`(전역)는 staged에서 **거부**된다 — 샤딩은 `staged.stage1/2_chunk_size`로.
- staged에서 `emit: final_only` + `final_rollout_scope: unsolved` 조합도 거부된다(일찍 풀린 문제가
  훈련 데이터에서 조용히 사라진다).
- `math_verify`의 `Timeout during comparison` 경고는 정상 노이즈다.
- 런이 죽어도 `-r`로 재개하면 bridge 생성과 fit은 캐시에서 스킵된다(수십 분 절약).
