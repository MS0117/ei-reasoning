# Toy-cliff LoRA 훈련 대장 + 후보 품질 측정

toy cliff(rollout → partition → anchor → improve → filters 한 사이클)에서 돌린 **transient LoRA fit**의
전수 목록과, 각 arm이 만들어낸 정답 후보의 **품질 지표**(C(y), 베이스 정책 NLL, 길이, 참조 언급률).
실행법·해석 규칙·다음 큐는 [toy_cliff_playbook.md](toy_cliff_playbook.md).

| 버전 | 날짜 | 내용 |
|---|---|---|
| **v2 (본문)** | 2026-08-31 | **250-cliff 세트, 5 arm 동일 예산(4스텝·24샘플·어댑터당 ≤100문제)**, 후보 품질 지표 추가 |
| v1 (부록 A) | 2026-08-29 | 107-cliff 세트, 20런 40 fit 대장. 원문 그대로 보존 |

### v1 → v2에서 달라진 것

| | v1 (부록 A) | v2 (본문) |
|---|---|---|
| 문제 집합 | `openr1_qwen3-4b-2507_n2000_with_gold.jsonl` 137문항 → cliff **107** (0/8 기준) | `openr1_default_cliff450_k16_with_gold.jsonl` 450후보 → cliff 384 → **시드 고정 부분집합 250** (0/16 기준, default config 전용) |
| 공유 rollout | `runs/toy_cliff/default_LSPO_20260813_155520` | `runs/toy_cliff_2/_subset250` (원본 `default_CONTROL_20260830_020008`) |
| 예산 | arm마다 달랐음 (CONTROL/LSPO 3스텝·16샘플, BRIDGE/STAGED 4스텝·24샘플) | **5 arm 전부 4스텝·24샘플**로 통일 |
| 어댑터 분할 | 거의 전부 pooled(107문제/어댑터) | `chunk_size 100` (3 어댑터) — pooled 조건을 문제 수 기준으로 재현 |
| 출력 위치 | `runs/toy_cliff/` | `runs/toy_cliff_2/` (`run_toy_cliff.sh -o`) — 분모가 다른 런이 한 랭킹표에 섞이지 않게 분리 |
| 지표 | conversion, fit loss | + **C(y)·s_mean·s_tail(베이스 NLL), 후보 길이, 참조 언급률, paired 검정, stage 분해, conversion@k** |
| 세트 빌드 | 수동 backfill | `data/make_toy_cliff_set.py`, `data/make_cliff_subset_bundle.py` (시드·규칙·qid를 manifest에 기록) |
| 실행 | 런마다 수동 | `data/run_toy_cliff_arms.sh` 순차 큐 |

v1의 숫자(conversion 0.55 등)는 **v2와 직접 비교하면 안 된다** — 분모(107 vs 250)와 cliff 정의(0/8 vs 0/16)가
다르다. v2 세트가 더 어렵다(CONTROL 0.243 → 0.260이지만 LSPO 0.308 → 0.324, BRIDGE 0.514 → 0.500).

---

## Part I — 라운드 1 @ 250 cliff (2026-08-30)

### 0. 한눈에

- 5 arm, 같은 250문항, 같은 실패 궤적, 같은 예산(4 gradient step, 문제당 최대 24샘플, 어댑터당 ≤100문제).
  총 **18 fit / 34 GPU-h(5 arm 합계 26.5h 벽시계, 2×A100)**.
- **conversion은 두 계층**으로 갈린다: bridge 계열 3개(0.49~0.51)가 gold(0.32)와 CONTROL(0.26)을
  p<10⁻⁵로 이기고, bridge 계열 셋끼리는 p>0.7로 완전 동률.
- **후보 품질은 conversion과 다른 순서**다. stage-2 DPO가 만든 정답 후보는 SFT 계열보다 베이스 정책에
  훨씬 가깝고(paired ΔC −0.56~−0.95, p<10⁻⁶), **CONTROL(베이스 자체 샘플)의 on-policy 수준에 근접**한다.
  conversion이 같은 세 arm 중 어느 데이터를 학생에게 먹일지는 이 축이 가른다.

### 1. conversion

| arm | operator | conversion | kept | 소요 | run |
|---|---|---|---|---|---|
| STAGED | staged_bridge_sft 2+2, 8+16 | **0.508** | 127/250 | 5h59m | `default_STAGED_20260830_175900` |
| BRIDGE | bridge_sft 4, 24 | **0.500** | 125/250 | 6h17m | `default_BRIDGE_20260830_114141` |
| STAGED_DPO | staged 2+2 (stage-2 DPO, w=0), 8+16 | **0.492** | 123/250 | 6h44m | `default_STAGED_DPO_20260830_235807` |
| LSPO | lora_sft (gold y\*) 4, 24 | 0.324 | 81/250 | 3h56m | `default_LSPO_20260830_074451` |
| CONTROL | self_resample, 24 | 0.260 | 65/250 | 3h32m | `default_CONTROL_20260830_041232` |

paired McNemar (b = A만 전환, c = B만 전환):

| A vs B | Δ | b / c | 불일치 | p |
|---|---|---|---|---|
| BRIDGE vs CONTROL | +24.0pp | 80 / 20 | 40.0% | 1.1e-9 |
| STAGED vs CONTROL | +24.8pp | 78 / 16 | 37.6% | 5.8e-11 |
| STAGED_DPO vs CONTROL | +23.2pp | 73 / 15 | 35.2% | 2.6e-10 |
| BRIDGE vs LSPO | +17.6pp | 68 / 24 | 36.8% | 4.9e-6 |
| STAGED vs LSPO | +18.4pp | 67 / 21 | 35.2% | 9.2e-7 |
| STAGED_DPO vs LSPO | +16.8pp | 65 / 23 | 35.2% | 8.5e-6 |
| LSPO vs CONTROL | +6.4pp | 40 / 24 | 25.6% | **0.060** |
| BRIDGE vs STAGED | −0.8pp | 27 / 29 | 22.4% | 0.89 |
| BRIDGE vs STAGED_DPO | +0.8pp | 32 / 30 | 24.8% | 0.90 |
| STAGED vs STAGED_DPO | +1.6pp | 34 / 30 | 25.6% | 0.71 |

- 107-set에서 경계였던 **bridge > gold(+11.2pp, p=0.058)가 +17.6pp, p<10⁻⁵로 확정**됐다.
- **gold는 250에서도 CONTROL과 유의하게 안 갈린다**(p=0.060). 107(p=0.248)에서 N을 2.3배 키워도 경계에
  머문다 — "gold 텍스트 조건화만으로는 부족하고 self-generated bridge가 필요하다"가 더 선명해졌다.
- bridge 계열 셋의 ±1.6pp는 어떤 N에서도 안 벌어지는 크기다(검정력표: 5pp도 N=400에서 0.48).
  스케줄·목적함수 축은 **conversion에서는 평평하다는 것이 적정 N에서 확인된 null**이다.
- 층화(continuity 45 = 기존 107 출신 / fresh 205): 순위 동일. 기존 107 출신이 더 어렵다
  (CONTROL 0.178 vs 0.278, BRIDGE 0.422 vs 0.517).

### 2. 후보 품질 — C(y), 베이스 NLL, 길이

**지표 정의** ([filters.py `_score_candidates`](../src/expert_iter/filters.py)). filters가 정답 후보 전부를
**베이스(학생) 정책 π_θ 아래에서** 스코어링한다(`always_score: true`):

- `s_mean` = continuation 토큰당 평균 NLL under π_θ. **낮을수록 학생이 이미 잘 예측하는 = on-policy한 후보.**
- `s_tail` = 최악 10% 토큰의 평균 NLL (CVaR). 학생이 "절대 못 냈을" 토큰이 얼마나 심한가.
- `C(y) = s_mean + 1.0·s_tail` (λ=1, γ=0). 학습가능성 점수. 낮을수록 좋음.
- `d_tail`(어댑터-베이스 괴리)은 `gamma_dtail=0`이라 **계산되지 않았고**, 생성 풀도 어댑터 아래 logprob를
  저장하지 않는다 → 이번 라운드에서 "어댑터가 얼마나 벗어났나"는 직접 측정 불가(§5).

#### 2-1. arm별 전체 (정답 후보 전부)

| arm | 정답 후보 | 전환 문제 | s_mean | s_tail | C 전체 | **C kept** | minC/문제 | 평균 길이 | 참조 언급 |
|---|---|---|---|---|---|---|---|---|---|
| CONTROL | 108 | 65 | **0.354** | **2.40** | **2.76** | 2.80 | 2.71 | 5,971 | 0.0% |
| LSPO | 144 | 81 | 0.440 | 3.00 | 3.44 | 3.52 | 3.42 | 5,585 | 0.7% |
| BRIDGE | 626 | 125 | 0.449 | 3.28 | 3.73 | **4.27** | 4.03 | 4,255 | 0.6% |
| STAGED | 380 | 127 | 0.443 | 3.19 | 3.63 | 3.84 | 3.64 | 4,658 | 0.5% |
| **STAGED_DPO** | 416 | 123 | **0.336** | **2.54** | **2.88** | **3.22** | **3.08** | 4,451 | 0.0% |

- **CONTROL이 바닥선**이다 — 베이스가 스스로 뽑은 정답이니 정의상 가장 on-policy(s_mean 0.354).
- **STAGED_DPO의 정답 후보는 그 바닥선 근처**다(s_mean 0.336, s_tail 2.54). SFT 계열(0.44~0.45, 3.2~3.3)과
  뚜렷이 갈린다. 107-set에서 본 "DPO는 conversion 중립이지만 C(y)가 가장 낮다(2.91 vs 3.63)"의 재현.
- "C kept"(filters가 문제당 1개 골라 훈련으로 넘기는 후보의 C)는 selection이 `random`이라 "C 전체"보다 높다
  — 지금 학생에게 가는 데이터는 최선의 후보가 아니다. `filter.selection.method: c_score`면 minC/문제 열이
  실제 훈련 데이터가 된다.
- 참조 풀이 언급률은 전 arm ≤0.7% — 가중치 채널로 흐른 y\*가 텍스트로 새지 않는다(0815 메모의 1.4%보다 낮음).

전체 평균은 **문제 구성이 다르다는 교란**이 있다(arm마다 전환한 문제가 다르고, 어려운 문제의 정답은 원래 NLL이
높다). 그래서 아래 paired가 본 판정이다.

#### 2-2. paired — 두 arm이 모두 전환한 문제에서 문제별 minC 차이

| A − B | 공통 n | ΔminC 평균 / 중앙 | A>B / A<B | sign p | Δs_mean |
|---|---|---|---|---|---|
| STAGED − STAGED_DPO | 93 | **+0.56 / +0.53** | 69 / 24 | 3.3e-6 | +0.082 |
| BRIDGE − STAGED_DPO | 93 | **+0.72 / +0.70** | 85 / 8 | 2.3e-17 | +0.094 |
| BRIDGE − STAGED | 98 | +0.20 / +0.13 | 60 / 38 | 0.033 | +0.017 |
| LSPO − STAGED_DPO | 58 | +0.10 / +0.09 | 34 / 24 | 0.24 | +0.024 |
| CONTROL − STAGED_DPO | 50 | **−0.47 / −0.47** | 4 / 46 | 4.5e-10 | −0.055 |
| CONTROL − STAGED | 49 | **−0.90 / −0.64** | 3 / 46 | 7.0e-11 | −0.122 |

같은 문제에서 재도 순서는 **CONTROL < STAGED_DPO ≈ LSPO < STAGED < BRIDGE** (낮을수록 on-policy).
DPO 후보는 gold-LoRA 후보와 구분되지 않고(p=0.24), CONTROL과의 격차(0.47)는 SFT-staged(0.90)의 절반이다.

#### 2-3. stage 분해 — DPO는 stage-2만 바꾼다

| arm | stage | 후보 | s_mean | s_tail | C | 길이 |
|---|---|---|---|---|---|---|
| STAGED | stage-1 (SFT 2스텝) | 71 | 0.367 | 2.56 | 2.92 | 5,734 |
| STAGED | stage-2 (SFT +2스텝) | 309 | 0.460 | 3.33 | 3.79 | 4,410 |
| STAGED_DPO | stage-1 (SFT 2스텝) | 60 | 0.388 | 2.71 | 3.10 | 5,101 |
| STAGED_DPO | **stage-2 (DPO +2스텝)** | 356 | **0.327** | **2.51** | **2.84** | 4,341 |

- SFT를 2스텝 더 하면 후보가 **베이스에서 멀어진다**(C 2.92 → 3.79). DPO를 2스텝 하면 **가까워진다**
  (3.10 → 2.84). 같은 chosen(bridge), 같은 warm-start 어댑터에서 목적함수만 다른 결과다.
- stage-2에서 양쪽 다 전환한 문제 43개 paired: **ΔminC(SFT−DPO) = +0.95, 42/1, p=1e-11**, Δs_mean +0.136.
- 대조군 확인 — stage-1은 두 런 모두 같은 SFT라 차이가 없어야 한다: n=17, ΔminC −0.22, 3/14, p=0.01.
  **차이가 있다** — 다만 방향이 반대다(DPO 런의 stage-1이 더 나쁨). 즉 stage-2의 DPO 우위는 "그 런이
  전반적으로 운이 좋아서"가 아니고, 오히려 불리한 stage-1을 이기고 나온 값이다. 동시에 이 n=17 결과는
  **런 간 노이즈가 minC 기준 ~0.2 수준**이라는 경고이기도 하다(bridge 생성이 vLLM 배칭 비결정성으로
  런마다 조금씩 다르다: 342 vs 340 pairs, stage-1 전환 51 vs 42). stage-2 효과(0.95)는 그 5배다.

#### 2-4. 해석과 한계

DPO의 stage-2 fit 진단은 107-set과 같다: loss 0.693 → 0.04, reward margin 7.6~12.6, pref_acc 0.996~1.000
— **2스텝에 완전 포화**. 그런데도 conversion은 SFT와 같고 후보만 on-policy해졌다. 기제 후보: DPO는
rejected(어댑터 자신의 실패)를 밀어내며 chosen을 **상대적으로만** 올리므로, SFT처럼 chosen의 절대
확률을 끌어올리며 베이스에서 멀어질 필요가 없다 — reference(=init 어댑터)에 묶인 채 순위만 바꾼다.
그 결과 어댑터가 뽑는 정답이 베이스 분포에서 덜 이탈한다.

**이것이 학생 훈련에서 이득인지는 toy가 답할 수 없다.** C(y)가 낮다 = 학생이 배우기 쉽다는 가설이지,
측정된 전이가 아니다. L3에서 S4-v0(사후 DPO)는 S3와 통계적으로 구분되지 않았다는 점을 기억할 것
([L3_results §4-4](L3_results_20260826.md)). 판정은 풀 루프(L5 `l5_staged_dpo_s3`)에서만 난다.

### 3. 생성 길이 / 절단 (풀 원본, 전 샘플)

| arm | 풀 | n | 중앙값 | p90 | 절단 |
|---|---|---|---|---|---|
| LSPO | main | 6,000 | 5,488 | 10,067 | 8.0% |
| BRIDGE | main | 6,000 | 6,154 | 10,372 | 3.3% |
| STAGED | stage-1 / stage-2 | 2,000 / 3,184 | 5,727 / 6,066 | 10,459 / 10,289 | 5.3% / 3.6% |
| STAGED_DPO | stage-1 / stage-2 | 2,000 / 3,328 | 5,831 / 5,898 | 10,516 / 10,084 | 5.8% / 4.3% |

- 어느 arm도 길이 병리 없음(107-set의 span-UL 16.2%와 대조). **DPO stage-2도 길이를 안 늘린다**(5,898,
  절단 4.3%).
- LSPO 절단 8.0%가 가장 높다 — gold 어댑터가 종결을 덜 배운다(gold pair는 평균 448토큰이라 EOS 신호가
  bridge 대비 1/10).

### 4. 라운드 1 fit 대장 (18 fit)

공통: `r=16, α=32, lr=1e-4, dropout 0, micro_batch 1(full batch), bf16, 7 target modules, world_size=2, sync_every 8`
— v1과 동일, 여전히 한 번도 변주 안 함.

| arm | fit | pairs | steps | obj | init | tok | 분 | loss (step별) | 비고 |
|---|---|---|---|---|---|---|---|---|---|
| LSPO | pooled_c0 | 100 | 4 | sft | cold | 0.05M | 1.6 | 1.290 / 1.221 / 1.098 / 0.982 | gold 1/문제 |
| LSPO | pooled_c1 | 100 | 4 | sft | cold | 0.05M | 1.7 | 1.176 / 1.115 / 1.005 / 0.901 | |
| LSPO | pooled_c2 | 50 | 4 | sft | cold | 0.03M | 0.9 | 1.267 / 1.198 / 1.075 / 0.967 | |
| BRIDGE | pooled_c0 | 285 | 4 | sft | cold | 1.47M | 20.2 | 0.547 / 0.538 / 0.522 / 0.509 | |
| BRIDGE | pooled_c1 | 315 | 4 | sft | cold | 1.51M | 20.4 | 0.486 / 0.476 / 0.460 / 0.445 | |
| BRIDGE | pooled_c2 | 142 | 4 | sft | cold | 0.79M | 14.3 | 0.596 / 0.587 / 0.570 / 0.556 | 50문제 잔여 청크 |
| STAGED | stage1_c0 | 340 | 2 | sft | cold | 1.77M | 12.2 | 0.538 / 0.530 | |
| STAGED | stage1_c1 | 364 | 2 | sft | cold | 1.77M | 12.0 | 0.500 / 0.491 | |
| STAGED | stage1_c2 | 41 | 2 | sft | cold | 0.25M | 2.0 | 0.577 / 0.566 | 12문제 잔여 청크 |
| STAGED | stage2_c0 | 264 | 2 | sft | warm | 1.36M | 9.3 | 0.467 / 0.452 | |
| STAGED | stage2_c1 | 272 | 2 | sft | warm | 1.27M | 8.6 | 0.478 / 0.461 | |
| STAGED | stage2_c2 | 37 | 2 | sft | warm | 0.24M | 1.9 | 0.561 / 0.541 | |
| STAGED_DPO | stage1_c0 | 342 | 2 | sft | cold | 1.78M | 12.1 | 0.538 / 0.530 | |
| STAGED_DPO | stage1_c1 | 358 | 2 | sft | cold | 1.77M | 12.3 | 0.494 / 0.485 | |
| STAGED_DPO | stage1_c2 | 44 | 2 | sft | cold | 0.26M | 2.0 | 0.582 / 0.571 | |
| STAGED_DPO | stage2_c0 | 296 | 2 | **dpo** | warm | 1.53M | 28.6 | 0.693 / 0.040 | margin 7.57, pref_acc 1.000 |
| STAGED_DPO | stage2_c1 | 274 | 2 | **dpo** | warm | 1.41M | 25.5 | 0.693 / 0.037 | margin 8.24, pref_acc 0.996 |
| STAGED_DPO | stage2_c2 | 37 | 2 | **dpo** | warm | 0.23M | 4.5 | 0.693 / 0.001 | margin 12.64, pref_acc 1.000 |

- **4스텝에서 loss가 아직 선형 하강 중**(BRIDGE 0.547→0.509). underfit 구간 — 라운드 2가 6스텝으로 가는 근거.
- `chunk_size 100`은 잔여 청크(c2)가 작게 남는다(50/12/10문제). 잔여 청크의 loss가 가장 높다 — 문제 수가
  적어 그 어댑터가 본 gradient가 적기 때문. 250을 정확히 3등분(84)하면 사라지는 인공물.
- gold fit은 bridge fit의 **1/30 토큰**(0.13M vs 3.77M)을 본다. v1의 경고 그대로: LSPO vs BRIDGE는 데이터
  출처 차이이자 훈련 신호량 차이다.
- DPO fit 시간 ≈ SFT의 2.5~3배(28.6 vs 9.3분) — rejected forward + reference pass.

### 5. 측정하지 못한 것 / 다음에 켤 것

| 항목 | 상태 | 켜려면 |
|---|---|---|
| 어댑터-베이스 괴리 `d_tail` | None (`gamma_dtail=0`) | `filter.selection.gamma_dtail > 0` — 스코어링 풀이 어댑터 아래 logprob를 한 번 더 뽑는다(비용 +1 score pass) |
| 생성 정책 아래 logprob | 생성 풀이 저장 안 함 | 엔진 옵션 필요(현재 미구현) |
| attractor 질량 (staged) | stage-2가 stage-1 최빈오답에 남긴 질량: SFT **0.476**, DPO **0.449** | v1 밴드(0.41~0.50) 안 — 목적함수로는 안 움직임 (`scripts/diag_stage_attractor.py`) |
| C 기반 선택의 효과 | selection=random이라 kept ≠ minC | `filter.selection.method: c_score`로 바꾸면 훈련 데이터의 C가 §2-1 minC 열이 된다 |

### 6. conversion@k와 라운드 2 예산

`improved.jsonl`의 attempt 순서로 "n을 줄였다면"을 재산출:

| | @8 | @12 | @16 | @20 | @24 |
|---|---|---|---|---|---|
| LSPO | .180 | .228 | .280 | .312 | .324 |
| BRIDGE | .352 | .388 | .448 | .480 | .500 |
| STAGED (8 + final@k) | final@8 .432 | final@12 .468 | final@16 .508 | | |

샘플 17–24는 절대값 +4~5pp를 사지만 **대조는 그대로**다: BRIDGE−LSPO @24 +17.6pp(p=4.9e-6) vs
@16 +16.8pp(p=1.4e-5), 불일치율 36.8% 동일. 반면 fit loss는 4스텝에서 미수렴. 단가(250 cliff): 샘플 1개
≈ 9분, fit 스텝 1개 ≈ 13분.

→ **라운드 2 예산: 6스텝 · 16샘플** (single: `fit.steps 6, improve.n 16`; staged: `3+3, 8+8`). 5개 config에
2026-08-31 반영. staged의 final-rollout 곡선이 더 가팔라(16→8 −7.6pp vs single −5.2pp) 이 예산은 staged에
약간 불리하다 — 해석 시 기록할 것. 라운드-1 런은 conversion@16으로 재채점하면 라운드 2와 짝비교가 된다
(같은 rollout, 같은 250문항, 유일한 변수 = 스텝).

### 7. 어디를 보나

```
runs/toy_cliff_2/
  _subset250/                                  공유 rollout 번들 (subset_manifest.json에 qid 250개)
  default_<ARM>_<ts>/
    config.yaml                                frozen config
    metrics.json                               퍼널·conversion·C(y)
    iter_0/improve/stats.json                  bridge 수율, stage별 전환, DPO pair 수
    iter_0/improve/adapters/<name>/<hash>/fit_meta.json
    iter_0/improve/improved.jsonl              후보 전체 (attempt_idx 순 → conversion@k 재산출)
    iter_0/filtered/candidate_scores.jsonl     정답 후보별 s_mean / s_tail / d_tail / c / kept  ← §2의 원천
    iter_0/filtered/kept.jsonl                 훈련으로 넘어간 후보 (qid 집합 = conversion 분자)
```

랭킹: `.venv/bin/python data/rank_toy_runs.py --runs-dir runs/toy_cliff_2`.

---
---

## 부록 A — v1 원문 (2026-08-29, 107-cliff 세트)

> 아래는 v1 문서 전문이다. 숫자·경로·결론 모두 **107-cliff 세트(`runs/toy_cliff/`) 기준**이며 본문(250)과
> 분모·cliff 정의가 다르다. UL 계열(D절)은 이 세트에서 결론이 났고 250에서는 재실행하지 않았다:
> uniform UL은 무효과+attractor 퇴행, span UL(`default_STAGED_UL_SPAN_20260829_204203`, conv 0.467,
> attractor 0.450, stage-2 절단 16.2%)은 퇴행은 고쳤으나 이득 없이 길이 비용만 남겼다.

# Toy-cliff에서 돌린 LoRA 훈련 전수 목록

`runs/toy_cliff/` 아래 **20개 런에서 실제로 실행된 40개의 LoRA fit** 대장. "무엇을 어떤 설정으로
훈련시켰나"만 다룬다 — 실험 해석·실행법·다음 큐는 [toy_cliff_playbook.md](toy_cliff_playbook.md).

작성 2026-08-29. 출처는 각 런의 `iter_0/improve/adapters/<name>/<hash>/fit_meta.json`(fit별 실측)과
`config.yaml`(frozen config). 랭킹 재생성은 `.venv/bin/python data/rank_toy_runs.py`.

---

## 0. 한눈에

- 훈련 대상은 **항상 transient LoRA**다. toy 드라이버는 rollout → partition → anchor → improve → filters만
  돌리므로, **학생 모델 SFT(`train` 스테이지)는 toy cliff에서 단 한 번도 돌지 않았다.** 여기의 모든
  "훈련"은 개선 연산자 안에서 후보를 뽑기 위해 붙였다 버리는 어댑터다.
- 베이스는 전부 `Qwen/Qwen3-4B-Instruct-2507`, 문제는 전부 같은 107 cliff(공유 rollout
  `default_LSPO_20260813_155520`).
- 합계: **40 fit / 399분(≈6.7 GPU-h, 대부분 2-GPU DDP) / response 토큰 41.7M**.
- 목적함수는 3종뿐: **SFT 36 / DPO 2 / UL 2**. 그 위에 어댑터 파라미터 GRPO RL이 별도로 1회 완주.

---

## 1. 모든 fit이 공유한 하이퍼파라미터 (한 번도 변주 안 함)

| 항목 | 값 |
|---|---|
| rank / alpha / dropout | `r=16`, `lora_alpha=32`, `dropout=0.0` |
| lr | `1.0e-4` (SFT·DPO·UL 전부. RL만 별도 lr) |
| target_modules | `q,k,v,o,gate,up,down_proj` (≈33.0M trainable = 4B의 1.6%) |
| 배치 | `micro_batch_size=1` + grad-accum으로 **매 스텝 full batch** |
| 기타 | `max_grad_norm=1.0`, `bf16`, gradient checkpointing on, `max_pair_tokens=16384` |
| 분산 | `world_size=2` DDP + `sync_every=8`. **2026-08-15 이전 두 fit만 단일 프로세스**(202912, LSPO 155520) |

즉 변주한 축은 하이퍼파라미터가 아니라 다음 다섯 가지다:
**(a) fit 데이터가 무엇인가 · (b) 몇 gradient step · (c) 목적함수 · (d) 어댑터를 몇 개로 쪼개나 ·
(e) 이전 어댑터에서 warm-start 하는가.**

---

## 2. 훈련 계열 5개

### A. gold SFT — `(x → y*)`, 연산자 `lora_sft`

| 런 | pairs | steps | tok/pair | 시간 | loss | conv |
|---|---|---|---|---|---|---|
| `default_LSPO_20260813_155520` | 107 (문제당 1) | 3 | **448** | 2.1분 | 1.243 → 1.176 → 1.053 | 0.308 |

toy cliff에서 gold를 직접 fit한 유일한 런이다. 두 가지가 눈에 띈다:

- **pair당 448 토큰** — bridge 계열(≈4,700)의 1/10.5. "3 스텝"이 같아도 gradient가 본 토큰은
  48K vs 1.47M(**30배**). LSPO vs BRIDGE 비교는 데이터 출처 차이인 동시에 **훈련 신호량 차이**이기도
  하다는 점을 해석에 반드시 반영할 것.
- **loss가 1.24에서 시작**한다. bridge fit은 전부 0.41~0.58에서 출발한다. 베이스 정책에게 terse gold
  풀이가 자기 분포에서 얼마나 먼지가 그대로 숫자로 나온다.

### B. bridge SFT — `(x → z*)`, 연산자 `bridge_sft` (단일 fit)

| 런 | 어댑터 | pairs | steps | 비고 | conv |
|---|---|---|---|---|---|
| `BRIDGE_20260813_202912` | 1 | 313 | 3 | **기준선**(단일 프로세스 fit, 26.4분) | 0.421 |
| `BRIDGE_20260816_020408` | 1 | 308 | 3 | `anchor: privileged_divergence` — 유일한 앵커 런 | 0.252 |
| `BRIDGE_20260820_034605` | 1 | **89** | 3 | `bridge.max_keep=1` (문제당 1 pair) | 0.364 |
| `BRIDGE_20260820_010217` | 1 | 316 | 4 | 예산 증액(4스텝 / 24샘플) | 0.514 |
| `BRIDGE_20260820_232256` | **5** | 57·81·72·75·19 | 4 각각 | `chunk_size=25` (샤딩 실측 유일) | 0.551 |

- **데이터 양이 직접 먹힌다**: max_keep을 1로 줄여 pairs 313→89가 되면 0.421 → 0.364.
- **샤딩은 손해가 아니다**: 어댑터 5개로 쪼갠 chunk25(0.551)가 같은 예산 pooled(0.514)와 동률 이상.
  현재 저장소에서 `chunk_size > 0`을 실제로 돌린 결과는 이 런 하나뿐이다.
- **앵커는 명확히 손해**: 같은 3스텝·16샘플에서 0.252로, 무연산자 control(0.243) 바로 위다.

### C. staged bridge SFT — stage-1 cold fit → stage-2 warm-start fit

`chain_adapter: true`이므로 stage-2는 항상 stage-1 어댑터에서 이어 학습한다(`init_adapter` 세팅).
stage-2의 fit 대상은 stage-1 rollout에서 **아직 못 푼 문제**들이다.

| 런 | s1 pairs/steps | s2 pairs/steps | 무엇을 바꿨나 | conv |
|---|---|---|---|---|
| `STAGED_20260819_121856` | 312 / 2 | 266 / 2 | 기본(reuse_bridge, unsolved_only) | **0.551** |
| `STAGED_20260819_160735` | 314 / 2 | 265 / 1 | final rollout 8 (예산 축소) | 0.336 |
| `STAGED_20260819_215536` | 312 / 2 | 253 / 1 | stage-2 1스텝 | 0.439 |
| `STAGED_20260820_054935` | 311 / 2 | **379** / 2 | `add_bridge`(구+신 bridge 병합) | 0.495 |
| `STAGED_20260820_122233` | 315 / 2 | 241 / 2 | `full_pool` + `self_wash_min_c` | 0.551 |
| `STAGED_20260820_172732` | 313 / **1** | **397** / 3 | 1+3 스텝 배분 + add_bridge | 0.514 |
| `STAGED_20260821_040009` | 314 / 2 | 89·90·81 / 2 | stage-2만 `chunk 25` (3 sub-shard) | 0.458 |
| `STAGED_20260821_105714` | 304 / **1** | 92·90·70 / 3 | 1+3 + stage-2 chunk 25 | 0.495 |

- stage-2 pairs는 대개 stage-1보다 적다(미해결만 남으므로). **`add_bridge`만 늘어난다**(379/397) —
  그리고 그때 stage-2 loss가 stage-1보다 **올라간다**(0.526·0.540 vs 0.46대). 병합으로 들어온 신규
  bridge가 더 어려운 데이터라는 뜻이고, conversion 이득으로는 이어지지 않았다.
- stage-2 샤딩(`stage2_chunk_size=25`)은 stage-1 샤드의 하위 샤드로 3개가 만들어졌다.

### D. stage-2 대안 목적함수 — DPO / UL

chosen은 네 런 모두 동일한 bridge 궤적이고, rejected만 다르다. **비교가 목적함수만 분리하도록 설계됨.**

| 런 | 목적 | pairs | steps | 하이퍼 | fit 진단 | conv |
|---|---|---|---|---|---|---|
| `STAGED_20260821_154423` | dpo | 249 | 2 | β=0.1, `sft_weight=1.0`, rejected=random 자기실패 | loss 1.129→0.474, margin 0→**6.43**, pref_acc **0.99** | 0.533 |
| `STAGED_20260821_193219` | dpo | 233 | 2 | β=0.1, `sft_weight=0.0` | loss 0.693→**0.057**, margin 0→5.47, pref_acc **1.00** | 0.533 |
| `STAGED_UL_20260828_042737` | ul | 215 | 2 | **μ=0.1**, δ=0.02, guard on, rejected=**modal** | ul 2.807→2.760, `guard_active_frac` 0.0/0.0 | 0.439 |
| `STAGED_UL_20260828_102141` | ul | 236 | 2 | **μ=1.0**, δ=0.02, guard on, rejected=modal | loss 3.244→3.172, `guard_active_frac` 0.0/**0.042** | 0.458 |

- **DPO는 2스텝 만에 포화한다.** pref_acc가 0.99~1.00, reward margin 5~6.4. 즉 "덜 학습돼서 효과가
  안 났다"가 아니라 **완벽히 분리하고도 conversion이 안 움직였다**(0.533 vs SFT 0.551, 노이즈 범위).
  스텝을 더 주는 것은 레버가 아니다.
- **UL은 μ를 10배 올려도 거의 안 켜진다.** μ=0.1에서 displacement guard는 한 번도 활성화되지 않았고,
  μ=1.0에서도 활성 비율 4.2%. conversion은 0.44~0.46으로 SFT/DPO 아래.
  UL 두 런은 아직 다른 어떤 문서에도 결과가 실려 있지 않다 — 이 표가 현재 유일한 기록이다.
  설계 근거(stage-1 어댑터가 base attractor를 **재배치**하므로 negative를 modal로 잡는다)는
  [data/configs/STAGED_UL.yaml](../data/configs/STAGED_UL.yaml) 헤더와 `runs/_diag/stage_attractor.json`.

### E. 어댑터 위 GRPO RL (fit 이후 단계)

`improve.rl`이 켜진 런들. LoRA 파라미터에만 GRPO를 돌린다(`lora_rl.py`, colocate vLLM).

| 런 | RL 설정 | 결과 |
|---|---|---|
| `BRIDGE_20260815_142020` | grpo, **lr 1e-5**, importance-sampling knob이 없던 시절 config | improve 도중 중단, stats 없음 |
| `BRIDGE_20260815_183305` | grpo, lr 1e-6, `token_truncate` | **DDP fit이 NCCL SIGABRT로 사망** — `static_graph` 사건(STAGED.yaml 주석 참조) |
| `BRIDGE_20260815_194535` | grpo, lr 1e-6, group_size 8, grad_accum 4, ε=0.2, kl_beta 0, vllm util 0.5 | **완주**: 107 step(프롬프트 107개 × 1 epoch), 2.9시간, reward 0.0 → 0.125, seam mismatch 0 |

완주한 194535의 SFT fit은 315 pairs / 3 steps로 기준선(202912, 313 pairs / 3 steps)과 같은 예산인데,
**conversion은 0.355(38/107)로 기준선 0.421보다 낮다**(`metrics.json`이 없어 `improved.jsonl`에서 직접
집계). 즉 **toy cliff에서 SFT fit 뒤 GRPO를 붙이는 것은 이득이 없었다.** 진단은
[memos/2026-08-15_cliff_rl_signal.md](memos/2026-08-15_cliff_rl_signal.md) — cliff는 정의상 정답이
정책 support 밖이라 zero-advantage 그룹이 76.5%다.

---

## 3. fit 40개 전수 표

`tok` = fit이 gradient를 흘린 response 토큰 총량. `init` = cold(베이스에서) / warm(직전 어댑터에서).

| 런 | fit | pairs | steps | obj | init | tok | 분 | loss(step별) |
|---|---|---|---|---|---|---|---|---|
| LSPO_20260813_155520 | pooled_c0 | 107 | 3 | sft | cold | 48.0K | 2.1 | 1.243 / 1.176 / 1.053 |
| BRIDGE_20260813_202912 | pooled_c0 | 313 | 3 | sft | cold | 1.47M | 26.4 | 0.483 / 0.475 / 0.460 |
| BRIDGE_20260815_142020 | pooled_c0 | 314 | 3 | sft | cold | 1.48M | 15.6 | 0.492 / 0.484 / 0.469 |
| BRIDGE_20260815_194535 | pooled_c0 | 315 | 3 | sft | cold | 1.46M | 15.0 | 0.464 / 0.455 / 0.441 |
| BRIDGE_20260816_020408 | pooled_c0 | 308 | 3 | sft | cold | 1.21M | 13.6 | 0.488 / 0.478 / 0.460 |
| BRIDGE_20260820_010217 | pooled_c0 | 316 | 4 | sft | cold | 1.48M | 20.8 | 0.487 / 0.479 / 0.465 / 0.453 |
| BRIDGE_20260820_034605 | pooled_c0 | 89 | 3 | sft | cold | 0.36M | 4.6 | 0.453 / 0.445 / 0.431 |
| BRIDGE_20260820_232256 | pooled_c0 | 57 | 4 | sft | cold | 0.27M | 4.0 | 0.466 / 0.458 / 0.442 / 0.428 |
| BRIDGE_20260820_232256 | pooled_c1 | 81 | 4 | sft | cold | 0.33M | 4.8 | 0.411 / 0.403 / 0.387 / 0.373 |
| BRIDGE_20260820_232256 | pooled_c2 | 72 | 4 | sft | cold | 0.32M | 4.4 | 0.488 / 0.480 / 0.464 / 0.450 |
| BRIDGE_20260820_232256 | pooled_c3 | 75 | 4 | sft | cold | 0.38M | 5.4 | 0.473 / 0.465 / 0.449 / 0.436 |
| BRIDGE_20260820_232256 | pooled_c4 | 19 | 4 | sft | cold | 0.09M | 1.6 | 0.442 / 0.432 / 0.415 / 0.398 |
| STAGED_20260819_121856 | stage1 | 312 | 2 | sft | cold | 1.47M | 10.5 | 0.460 / 0.453 |
| STAGED_20260819_121856 | stage2 | 266 | 2 | sft | warm | 1.26M | 9.0 | 0.426 / 0.413 |
| STAGED_20260819_160735 | stage1 | 314 | 2 | sft | cold | 1.47M | 10.4 | 0.470 / 0.462 |
| STAGED_20260819_160735 | stage2 | 265 | 1 | sft | warm | 1.29M | 4.6 | 0.456 |
| STAGED_20260819_215536 | stage1 | 312 | 2 | sft | cold | 1.44M | 10.2 | 0.460 / 0.452 |
| STAGED_20260819_215536 | stage2 | 253 | 1 | sft | warm | 1.18M | 4.4 | 0.429 |
| STAGED_20260820_054935 | stage1 | 311 | 2 | sft | cold | 1.45M | 10.0 | 0.461 / 0.453 |
| STAGED_20260820_054935 | stage2 | 379 | 2 | sft | warm | 1.74M | 12.2 | 0.526 / 0.510 |
| STAGED_20260820_122233 | stage1 | 315 | 2 | sft | cold | 1.48M | 10.4 | 0.581 / 0.573 |
| STAGED_20260820_122233 | stage2 | 241 | 2 | sft | warm | 1.20M | 8.9 | 0.435 / 0.424 |
| STAGED_20260820_172732 | stage1 | 313 | 1 | sft | cold | 1.46M | 5.2 | 0.462 |
| STAGED_20260820_172732 | stage2 | 397 | 3 | sft | warm | 1.80M | 18.5 | 0.540 / 0.524 / 0.510 |
| STAGED_20260821_040009 | stage1 | 314 | 2 | sft | cold | 1.46M | 10.3 | 0.464 / 0.456 |
| STAGED_20260821_040009 | stage2_c0 | 89 | 2 | sft | warm | 0.43M | 3.2 | 0.417 / 0.403 |
| STAGED_20260821_040009 | stage2_c1 | 90 | 2 | sft | warm | 0.39M | 2.9 | 0.446 / 0.432 |
| STAGED_20260821_040009 | stage2_c2 | 81 | 2 | sft | warm | 0.44M | 3.2 | 0.460 / 0.446 |
| STAGED_20260821_105714 | stage1 | 304 | 1 | sft | cold | 1.40M | 5.0 | 0.462 |
| STAGED_20260821_105714 | stage2_c0 | 92 | 3 | sft | warm | 0.42M | 4.4 | 0.464 / 0.449 / 0.434 |
| STAGED_20260821_105714 | stage2_c1 | 90 | 3 | sft | warm | 0.41M | 4.5 | 0.471 / 0.454 / 0.439 |
| STAGED_20260821_105714 | stage2_c2 | 70 | 3 | sft | warm | 0.37M | 4.3 | 0.482 / 0.466 / 0.451 |
| STAGED_20260821_154423 | stage1 | 313 | 2 | sft | cold | 1.46M | 10.3 | 0.462 / 0.454 |
| STAGED_20260821_154423 | stage2 | 249 | 2 | **dpo** | warm | 1.19M | 23.7 | 1.129 / 0.474 |
| STAGED_20260821_193219 | stage1 | 309 | 2 | sft | cold | 1.43M | 10.0 | 0.503 / 0.495 |
| STAGED_20260821_193219 | stage2 | 233 | 2 | **dpo** | warm | 1.12M | 22.1 | 0.693 / 0.057 |
| STAGED_UL_20260828_042737 | stage1 | 316 | 2 | sft | cold | 1.50M | 10.0 | 0.466 / 0.458 |
| STAGED_UL_20260828_042737 | stage2 | 215 | 2 | **ul** | warm | 1.03M | 20.3 | 0.746 / 0.728 |
| STAGED_UL_20260828_102141 | stage1 | 303 | 2 | sft | cold | 1.40M | 9.9 | 0.459 / 0.451 |
| STAGED_UL_20260828_102141 | stage2 | 236 | 2 | **ul** | warm | 1.11M | 22.7 | 3.244 / 3.172 |

DPO/UL fit이 같은 pairs·steps의 SFT보다 2배 느린 것은 정상이다 — chosen 외에 rejected 쪽 forward와
reference 로그확률이 추가로 든다. UL fit_meta에는 그 양이 `total_rej_tokens`로 따로 잡혀 있고
(chosen과 비슷한 규모: 1.03M vs 1.11M, 1.11M vs 1.34M), DPO fit_meta에는 이 필드가 없다.

---

## 4. fit이 한 번도 안 돈 것 (있다고 착각하기 쉬움)

| 기능 | 상태 |
|---|---|
| `project_back` α 스윕 | **전 런 `enabled: false`.** `alpha_curves.jsonl`은 어느 런이든 `{"1": p̂}` 한 점과 `alpha_star: 1.0`만 들어 있다 — toy cliff에서 α\*는 **측정된 적이 없다** |
| `fit.adaptive` (τ_E 조기 종료) | 전 런 false. 스텝 수는 항상 고정값 |
| `adapter_scope: per_problem` | 없음. 전부 `pooled` |
| `refit_budget` | 전 런 0 |
| `staged.num_stages > 1` | 없음. 전 런 1 (즉 stage-2는 항상 딱 한 번) |
| LoRA rank / lr / target_modules 스윕 | 없음. §1 값에서 한 번도 안 움직였다 |
| `stage1_chunk_size > 0` | 없음. stage-1은 항상 pooled 단일 어댑터 (샤딩은 `bridge_sft` 쪽 chunk25와 staged의 stage-2에서만) |

**끝까지 못 간 런**(fit 기록 없음, 참고용):
`toy_cliff_20260813_152910`(rollout까지), `toy_cliff_20260813_202413`·`STAGED_s1_20260819_160049`·
`STAGED_20260820_172529`·`STAGED_20260821_193019`(anchor에서 kill),
`BRIDGE_20260816_011310`·`014724`(vLLM 풀이 안 뜸), `LSPO_20260813_235152`(연산자 없는 control이라
fit 자체가 없음).

---

## 5. 어디를 보나

```
runs/toy_cliff/<run>/
  config.yaml                                       실행 당시 frozen config = 실험 기록
  metrics.json                                      퍼널·conversion·C(y)
  iter_0/improve/stats.json                         bridge 수율, 스테이지별 전환, RL 카운터
  iter_0/improve/adapters/<name>/<hash>/
      fit_meta.json                                 이 문서의 원천: pairs·steps·loss·시간·params
      adapter_model.safetensors, adapter_config.json
  iter_0/improve/alpha_curves.jsonl                 project-back(현재는 α=1 한 점만)
```

`<hash>`는 fit 캐시 키다 — `params`에 실린 값이 그대로 들어가므로, 하이퍼파라미터를 하나라도 바꾸면
새 fit이 돌고 안 바꾸면 재개 시 스킵된다. 같은 pairs·steps인데 loss 곡선이 완전히 동일한 fit이 두 런에
있다면(예: 172732와 154423의 stage-1 첫 스텝 0.462) 캐시 재사용이 아니라 **같은 시드·같은 데이터**가
만든 재현이다.

풀 EI 루프(L2/L3/L5 런)에도 같은 연산자 스택의 LoRA fit이 들어 있지만 이 문서 범위 밖이다 —
[L5_runbook.md](L5_runbook.md), [L3_results_20260826.md](L3_results_20260826.md) 참조.
