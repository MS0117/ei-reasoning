# 구제 궤적(rescued cliff trajectories)을 학생이 어떻게 학습할 것인가 — objective 결정 (2026-08-23)

근거: 메모리 노트 `scaffold-credit-direction`(M1–M7), `docs/lit_rescued_trajectory_objectives.md`(~180편),
오늘 추가로 잰 CPU 측정(§1.3), 그리고 5개 framing 분석 + 각 framing에 대한 적대적 비평 + 종합 + 완결성 비평(12 agent)의 결과를
제가 다시 검증·정리한 것. 기준 toy run은 STAGED SFT `runs/toy_cliff/default_STAGED_20260819_121856`와
DPO twin `runs/toy_cliff/default_STAGED_20260821_193219`.

---

## 0. TL;DR

1. **세 framing은 경쟁 관계가 아니라 하나의 질문으로 수렴합니다.**
   - *Off-policy correction*: 정확한 보정은 존재하지 않는다는 것을 **증명**하는 역할만 합니다. 구제 샘플의 exact importance weight
     π_ref(y)/q(y) = exp(Λ)는 Λ 중앙값 −42 nat(범위 −176…+2)라 사실상 0이고, Λ는 길이 통계(Spearman −0.82)이며
     정답 sibling보다 오답 sibling에서 더 큽니다(20/57, p=0.03). 즉 **cliff의 학습 질량은 추정 대상(estimand)이 아니라 설계 변수(ρ)** 입니다.
     LSPO의 per-token IS가 inert한 이유(M6)와 HDPO의 λ가 같은 knob이라는 것이 이 한 문장으로 설명됩니다.
   - *Credit assignment*: 토큰 단위는 닫혔고(M1–M5), 남는 것은 **verifier가 주는 sequence-level 부호** 뿐입니다. 그런데 오늘 잰 바로는
     cliff의 전형은 "**하나의 오답에 ~70% 질량이 몰린 confident wrong attractor**"이고, adapter 구제는 그 attractor를 건드리지 않고
     ~12% 정답 mode를 하나 얹을 뿐입니다(converted cliff의 70–74%에서 adapter의 최빈 오답 == base의 최빈 오답; 과반이 24개 중 1개 성공).
     따라서 "credit"은 *정답 궤적 올리기 + attractor 내리기*라는 sequence-level outcome contrast로 환원됩니다. 토큰 credit이 아니라 **부호**입니다.
   - *Safe absorption*: 위 둘의 **운영 형태**입니다 — 명시적 share ρ, 문제 단위 정규화, 매 step 공급, 안전 envelope(non-cliff holdout/AIME/암기 지표/길이 drift), 그리고 **전이(transfer) endpoint**.

2. **권고 objective** (2항 + 선택적 negative):

   L(θ) = (1−ρ)·L_S + ρ·[ L_C + μ·L_N ]

   L_S = solved 데이터의 토큰 정규화 SFT(현행), L_C = 구제 궤적의 **문제 단위** 정규화 SFT, L_N = base 자신의 실패 rollout(최빈 오답)에 대한 sequence-level negative
   (zero-code 버전: 기존 `train.objective: sft+dpo`; 다음 버전: NSR/unlikelihood + displacement guard). ρ는 매 optimizer step에서 stratified sampler로 실현.

3. **첫 학생 측 측정은 dose–response + 전이입니다.** 지금까지 학생 쪽은 아무것도 측정된 바 없습니다. 주 endpoint는 held-out cliff B에서의
   (a) avg@32 vs base re-roll, (b) **attractor mass** P(최빈 오답) 변화 — pass@32가 0에 머물러도 움직이는 연속 지표. 부 endpoint는 A 흡수, non-cliff 회귀, 암기 지표.

4. **novelty는 loss 식이 아니라 (i) estimand 논증(데이터 포함), (ii) cliff attractor 해부, (iii) 최초의 A→B cliff 전이 dose–response, (iv) 버려지던 구제 실패 궤적의 style-vs-content 진단 활용**입니다. 식 자체는 HDPO λ + NSR의 조합이라는 것을 정직하게 쓰는 게 맞습니다.

---

## 1. 현재 지점

### 1.1 닫힌 것 (재론 금지)
- adapter 대조 g_t 토큰 credit (M1–M3): 신호가 문체/포맷, 성공·실패 무구분, outcome-contrast 변형도 자기 falsification 통과 못함.
- privileged 대조 g′_t 토큰 credit (M4–M5): sequence-level로만 성공·실패를 가르고, 그 차이는 **마지막 decile(답 쓰는 구간)에만** 있음 = verifier가 이미 아는 것. teacher는 답을 알지 경로를 모름.
- per-token off-policy 보정 (M6): 92–96% 토큰이 clip 안 → inert.
- outcome-MC credit: 비용으로 보류.
- 모든 per-token likelihood-contrast weighting은 문헌에 이미 있음(SDFT/SDPO/OPPO/RLSD/CAST/…), 그리고 그 문헌 자체가 "format/epistemic 토큰에 실린다"고 보고.

### 1.2 operator 축도 사실상 끝
107-cliff toy에서 4 step/~24 sample 급 arm은 전부 0.46–0.55, McNemar 전부 p>0.3. BRIDGE/STAGED/chunk/DPO-fit은 **treatment variable**로 두고, 학생이 받는 것은 converted cliff당 ≤2개 구제 성공(~5.6k tok), ~12–16개 구제 실패, base 실패 8개, y*(학생에겐 절대 노출 금지).

### 1.3 오늘 추가한 측정 (CPU, 기준 run 두 개)
| 측정 | SFT 0819_121856 | DPO 0821_193219 |
|---|---|---|
| converted cliff 수 | 59/107 | 57/107 |
| 성공 수 = 1인 cliff | **32/59** | **36/57** |
| 성공 수 분포 (꼬리) | 13–16/24 성공인 cliff 6개 | 6개 |
| cliff당 오답-with-\boxed 실패 수 | 16.2 | 12.3 |
| truncation/무답 실패 | 0.4 | 0.4 |
| 구제 실패 중 최빈 오답 점유율 | 0.64 | 0.72 |
| base 8 rollout 중 최빈 오답 점유율 | 0.70 | 0.74 |
| adapter 최빈 오답 == base 최빈 오답 | **35/50 (70%)** | **35/47 (74%)** |
| Λ = Σ(log π_base − log q) 범위 / 중앙값 | — (priv dump) | −176…+2 / **−42 nat** |
| Spearman(Λ, 길이) | — | **−0.82** |
| Λ가 정답 sibling에서 더 큰 비율 | 20/59 (p=.018, priv) | **20/57 (p=.033)** |
| 정답 sibling 간 Λ 격차 중앙값 | 27 nat | 13 nat |

메인 2000문항 mix에서 cliff 예제의 **토큰 질량 share**: solved 7,121개 × 평균 1,798 tok = 12.8M, 구제 ~137 × ~5.6k = 0.77M → **~5.7%**
(예제 수로는 1.9%). 종합 agent는 전체 정답 샘플 평균(2.9k)으로 ~3%를 냈는데, 실제 선택은 shortest-4라 5.7%가 더 가깝습니다. 어느 쪽이든 frozen build_dataset에서 재계산해야 함.
또한 현행 trainer는 grad-accum window(32)마다 정규화하므로 cliff 항은 step의 절반 이상에서 아예 없고, 나머지에서 한 5k-token 시퀀스가 step을 지배하는 **bursty** 형태입니다.

### 1.4 해석
- cliff = "모른다"가 아니라 **"확신을 갖고 틀린다"**. 구제 궤적은 attractor를 안 건드립니다 (adapter 구제의 과반은 24번 중 1번의 운).
- 따라서 구제 성공만 SFT로 얹는 현행 objective는 *70% 질량의 mode를 5.7% share의 positive만으로 뒤집으려는* 구조입니다. 학생 측 null이 나와도 놀랍지 않습니다.
- 실패 궤적은 잡음이 아니라 **attractor의 표본**입니다. 부정 신호는 verifier가 공짜로 주며(최빈 오답 identity), likelihood contrast가 아니므로 "토큰 credit 금지" 제약에 걸리지 않습니다.

---

## 2. 세 framing에 대한 판정

| framing | 판정 | 근거 |
|---|---|---|
| Off-policy correction | **분석 도구로만 채택, objective로는 기각** | exact IS weight exp(Λ)≈0 → 구제 샘플을 전부 지움. tempered/self-normalized 버전은 길이 prior이거나 sibling 1–2개 사이의 argmax로 퇴화(격차 13–27 nat). 학생이 매 iteration base에서 재초기화되므로 "현재 policy 대비 ratio" 게이트도 k≥1에서 의미 없음. **결론: ρ는 추정할 수 없고 정해야 한다.** |
| Credit assignment | **토큰 단위 기각, sequence 부호만 채택** | 문제당 성공 1–2개라 within-question 가중도 무의미. 남는 localization은 verifier 부호(정답 / 최빈 오답)와 위치 decile 진단뿐. |
| Safe absorption | **채택 (운영 형태)** | 명시적 share, 문제 단위 정규화, stratified 공급, envelope, 전이 endpoint. HDPO λ·SRPO routed loss와 같은 계열임을 인정. |

---

## 3. 권고 objective

### 표기
iteration k, 학생 π_θ는 base π_0에서 재초기화. D_S = solved 예제 i(문제당 ≤4 shortest 정답 rollout), T_i = loss 토큰 수.
D_C = converted cliff 문제 q의 구제 성공 j(≤2개, shortest), n_q ∈ {1,2}, T_j. F_q = base의 자기 실패 rollout(`finish_reason == stop`, ≤8), 그중 최빈 오답을 가진 것 우선.
ce_t(θ) = −log π_θ(y_t | x, y_<t).

### 배치 구성 (load-bearing)
global batch G=32 중 **정확히 m_C=1개는 D_C에서**(순환), 나머지는 D_S. 그래야 ρ가 "매 step의 share"가 되고 bursty하지 않음. (L_N을 켜면 m_N=1개 F_q 예제 추가, 가능하면 같은 q.)

### 항
- L_S = Σ_{i∈B_S} Σ_t ce / Σ_{i∈B_S} T_i  (현행 그대로)
- L_C = Σ_{j∈B_C} (1/(n_q·T_j)) Σ_t ce  /  Σ_{j∈B_C} 1/n_q  (**문제 단위 정규화**: 길이·성공 수와 무관하게 cliff 하나 = 1 단위)
- L_N (옵션) — 두 구현:
  - **v0 (zero-code)**: `train.objective: sft+dpo`. 이미 `build_dataset._build_dpo_pairs`가 chosen=구제, rejected=base 실패 rollout을 만들어 둠. 바꿀 것은 rejected 선택을 `base_selection=min_mean_nll` 대신 **최빈 오답 샘플**로 두는 옵션 하나.
  - **v1**: L_N = Σ_{k∈B_N} (1/(n⁻_q·T_k)) Σ_t −log(1−π_θ(y_{k,t}))  /  Σ 1/n⁻_q  (bounded unlikelihood; logit gradient ∝ π_θ라 확신 오답 토큰에만 작용하고 자기 제한)
    + displacement guard L_G = Σ_j (1/n_q)·max(0, mean_t ce_j(θ) − mean_t ce_j(θ_0)) (구제 성공이 base보다 덜 likely해지는 것을 금지; ce(θ_0)는 C(y) scoring pass에서 이미 나옴)
- **L = (1−ρ)·L_S + ρ·(L_C + μ·L_N + L_G)**, ρ ∈ {legacy(≈0.03–0.06), 0.1, 0.3}, μ ∈ {0, 0.1, 0.3}.

구현: build_dataset에 n_q 기록(`source`는 이미 있음), collator 가중 1/(n_q T_j), `_get_num_items_in_batch`를 [N_S, N_C] 2-벡터로 gather, compute_loss에서 두 정규화 합성, `_get_train_sampler` override. ≈150줄 + 테스트(두 정규화의 accum/rank 불변성, window당 cliff 1개, legacy 재현). 주의: loss 크기는 ρ에 따라 변함(구제의 per-token NLL이 solved의 ~3배) → `max_grad_norm=1.0` clipping 활성률과 **group별 gradient-norm share**를 로그해야 "share"가 실현됐는지 알 수 있음.

### Ablation arms (각 항 하나씩 분리)
| arm | 내용 | 분리하는 것 |
|---|---|---|
| S0 | ρ=0 (cliff 데이터 없음) | 전이의 zero |
| S1 | 현행 loss 그대로 | 오늘의 파이프라인 |
| S1′ | ρ=legacy share, stratified | 방문 횟수·sampler 효과 (S1↔S3 사이의 confound) |
| S3(ρ) | ρ∈{0.1,0.3}, 문제 정규화, stratified | **dose** |
| S3-tok | S3(0.3), 토큰 정규화 | 정규화 단위 |
| S4(μ) | S3(ρ*) + negative (v0 → v1) | **attractor 억제** — L3에 포함 (완결성 비평 수용: 1.4의 예측상 positive만으로 B가 안 움직일 수 있으므로 L4로 미루면 안 됨) |
| S2/S2′ | S3(0.3)에서 D_C를 (a) solved 문제의 5–6번째 정답 rollout, (b) **frontier(1–2/8) 문제의 on-policy 정답 rollout**으로 교체, 토큰 매칭 | "cliff 내용" vs "긴 데이터 더 넣기" / "어려움" vs "off-policy" |
| S5 | S3(ρ*) + DPO-fit 구제 (`stage2_objective=dpo`) | operator 측 learnability lever(C(y)↓) |

### 측정 panel (모든 arm 공통)
- A(학습 cliff): avg@32/pass@32, **attractor mass** P(최빈 오답), D_C NLL(암기 지표)
- B(held-out cliff; improve는 돌리되 train에는 절대 미포함): avg@32/pass@32 vs **base 2회 re-roll**(noise floor + zero), attractor mass, converted/never-converted/base-pass@32>0 층화
- 기제 진단 G_x: B의 adapter 샘플(성공·실패)에 대해 Δ_θ(y) = mean_t[log π_θ − log π_0]를 decile별로; G_x = mean_{R=1}Δ − mean_{R=0}Δ. 성공·실패 모두 올라가고 G_x≈0이면 style 흡수, G_x>0이고 질량이 decile 1–9에 있으면 content 전이.
- 회귀: non-cliff holdout greedy pass@1 + avg@8, AIME24 avg@8, MATH500-hard avg@4, 길이/truncation drift
- verifier는 A/B/re-roll 모두 loop의 `math`로 통일 (canary만 `math_strict`)

---

## 4. 실험 사다리

| 단계 | 내용 | 비용 | gate |
|---|---|---|---|
| **L0** (CPU) | frozen build_dataset에서 N_S/N_C/ρ_legacy/step당 cliff 부재율 재계산; A/B split 도구(qid hash, converted·p̂_x 층화); attractor-mass 스크립트(기존 rollout/improved.jsonl로 지금 계산 가능); B 검정력 시뮬(N_B∈{35,70,200}, zero-inflated avg@32); 2-정규화 trainer + sampler + 테스트; qid-exclusion config 필드 | 0 GPU, ~1–2일 | 테스트 green, 검정력 표 |
| **L1** (toy pilot) | toy arm union의 converted 73 cliff를 A/B로 분할(never-converted 34는 B 층). base에서 A 구제(~60 seq)로 학습: {ρ=1 cliff-only, replay 혼합} × {negative 없음, v0 negative}. A NLL↓, A/B attractor mass, G_x decile, B avg@32, MATH500-hard canary | ~4–6 GPU-h | **흡수·기제 읽기 전용** — 낮은 ρ에 대한 추론 금지(ρ=1 cliff-only는 style 붕괴 가능). A attractor mass가 안 움직이면 lever 자체 점검 |
| **L2** | 메인 preset iter 0 freeze(rollout 2000×8 → STAGED improve 전체 cliff, `always_score`, `emit: all`) + 모든 cliff에 base 2회 re-roll(avg@32) | 하룻밤 ~8–10h (2×A100) | B ≥60 cliff + re-roll 저장 |
| **L3** (결정 실험) | S0, S1, S1′, S3(0.1), S3(0.3), S3-tok, S4(μ), S2′ (+ S0 재시드) — train+eval만 | arm당 4–6h → 2–3일 | (i) A↑, B>S0·S2′ 초과, envelope 유지 → 전이 존재, ρ* 확정 → L4 (ii) A↑, B 평탄(attractor mass 포함) → 이 규모에서 전이 없음 → 측정 논문 + loop 재진입 (iii) A 안 오름 → lever 고장, 결론 금지 |
| **L4** | S5, S4 v1, ρ=0.5 (envelope 여유 시); L2b: passrate sweep +4000문항 → 전부 B로(검정력) | 2–4 training + ~3h improve | B 또는 attractor mass를 **같은 흡수량에서** 움직이는 항만 생존 |
| **L5** (headline) | 4-iteration loop, B는 매 iteration train에서 제외(improve는 iter 0 것 재사용). S1 vs S3(ρ*)+생존 항 vs gold_lora_sft(LSPO) operator 동일 objective, 2 seed. B는 iteration 걸쳐 pooling | arm당 며칠 | L3 결정 후에만 |

---

## 5. 기각한 제안 (한 줄 이유)
- within-cliff tempered SNIS 가중: n_q≤2, Λ는 길이, 오답 선호 → 무의미.
- m_x=(k_x/n_x)^γ 질량: 학생 측 근거 없음; p̂_x는 B 층화 변수로만.
- 구제 **실패**를 negative로: base 실패(70%에서 같은 attractor)가 학생 init에 on-policy라 더 깨끗함; 구제 실패는 G_x 진단용으로 보존. (단, 완결성 비평대로 "토큰 profile이 같다"는 M2에서 유도된 추정이지 측정은 아님 — L4에서 구제-실패 negative arm 하나는 열어둘 가치 있음.)
- \boxed 구간 마스크 NSR, HSD식 sibling divergence mask, (s−H)+ 감쇠, C(y) within-question 가중, 2-phase 학습, never-converted를 전이 control로: 각각 verifier-중복 / HSD 그대로 / 토큰 credit 금지 / n_q≤2 / 20 step짜리 phase / rescuability confound.
- 학생 측 결과 전에 8–10k 문항 채굴: L3 결과에 gate.

## 6. Novelty와 한계 (정직한 서술)
- 식: L_S+λL_C 분리 정규화 = HDPO λ / SRPO; 문제 단위 stratified 공급 = hard-prompt replay; negative = NSR/W-REINFORCE/UFT + DPOP guard; sequence weight = iw-SFT. **식으로는 새롭지 않음.**
- 새로운 것: (i) "cliff 질량은 estimand가 아니다" — exact IS가 구제를 소멸시키고 Λ가 길이·오답 편향이라는 데이터 포함 논증(LSPO의 IS와 HDPO의 λ를 같은 미검증 knob으로 재해석); (ii) cliff = confident wrong attractor, 구제는 attractor를 안 건드린다는 해부; (iii) adapter-rescued·gold-free·sequence-level off-policy 샘플의 **A→held-out B 전이 dose–response** (HDPO는 학습분포 pass@k만, LSPO는 surpass rate만); (iv) attractor mass라는 0-inflated pass@k보다 민감한 endpoint와 구제 실패를 이용한 style-vs-content 진단.
- B가 어떤 안전한 ρ에서도 안 움직이면: 논문은 "전이 negative result + loop 재진입 + HDPO λ ablation". 그것도 결과이고, 사다리는 그 경우를 최소 비용으로 알도록 짜여 있음.

## 7. 열린 위험
- B 검정력: n=2000에서 held-out cliff ~68개; ≤2pp 효과는 run pair로 못 봄 → attractor mass·G_x(연속, paired)가 조기 통계, L2b로 B 확장, L5에서 iteration pooling.
- 평균 회귀: 0/8 cliff가 n=32에서 >0으로 re-roll(self_resample 0.243) → base re-roll zero와 층화 필수.
- loss 크기/clipping이 ρ에 따라 변함 → gradient-norm share 로그.
- `data.accumulate`로 D_C가 iteration마다 누적(이미 natively 풀리는 문제의 옛 구제 포함) — ρ·sampler는 매 iteration 재계산, stale 구제 유지 여부는 미검토 선택.
- config hash가 train.* 키까지 묶어 frozen improve 위의 train-only arm이 .done을 무효화 → stage-scoped hash 또는 run-dir 복사 workflow 필요.
- zero2 + `average_tokens_across_devices`에서 2-벡터 num_items pass-through GPU 검증 필요.

---

## 8. Addendum (2026-08-24): 3-judge + 3-critic 패널 결과 — 이 문서에 대한 수정사항

전체 기록: `docs/objective_panel_20260824.md` (judge 제안 3개 전문 + critic 판정 12개 + 종합 3개).
판정: **SESA(=본 문서 §3의 objective + guess gate) survive 3/3, OPAL(부정항 스팬 국소화) weaken 3/3 → S4 arm으로 흡수, TPAU(g′-gated 부정항) kill 3/3, session-draft는 L3 프로토콜로 병합.**

### 새로 발견된 선행 연구 (이번 세션 fetch 검증 — 인용 필수)
- **OXA (arXiv 2603.16206)**: offline math SFT 안에서 "high-confidence verifier-failed 궤적"에 bounded unlikelihood — **confident-wrong mass 재분배 framing이 이미 출판됨**. → §3의 L_N은 "attractor 억제 자체"가 아니라 **span 국소화가 기여**라고 주장해야 하며, S4에 (no negative / OXA식 uniform / boxed-span) 3-way ablation이 필요.
- **VSI/NSRSA (arXiv 2603.21558)**: STaR loop에서 lucky-guess filtering (Qwen3-4B, transfer 측정 포함) — **"최초의 guess gate" 주장 사망**. → gate는 "sympy가 무력한 free-form 5k-token CoT에서 model-internal·parser-free 도구 + singleton-stratum 검증"으로 재포지셔닝.
- 인접: AMR-SD 2605.18529 (thresholded negative contrast on failures), RG-OPD 2607.04037 (sequence-level teacher gate), HiLL 2604.00698 (privileged reliance bound), OGLS-SD 2605.12400 (위치 분리 failure 신호 — 단 prefix 쪽).

### L3 설계 수정 (critic 합의)
1. **Guess gate Γ_j는 loss에서 빼고 logged covariate로 강등** — L0 dispersion/validity가 **두 run 모두** Bonferroni-honest 기준으로 통과할 때까지. (사실 정정: correct 후보는 86/78개, "~130–150"이 아님. singleton 층은 32/59·36/57. soft gate만 허용 — n_q=1 문항의 유일한 구제를 0으로 만들면 안 됨.)
2. 1라운드 grid 축소: **ρ ∈ {legacy, 0.3} × μ=0** + S0/S1′/S2′ + **S4는 μ-v0 단일 arm**(zero-code sft+dpo, rejected=최빈 오답). ρ×μ 9-cell은 검정력상 낭비.
3. **gold-y\* SFT(LSPO-data) control 1개를 L5에서 L3로 당김** — "구제의 on-policy성인가, 아무 정답이나 되는가"를 bracket (선택: bridge-z\* 변형으로 HDPO bracket).
4. attractor panel에 **P(top-2 wrong)** 추가(질량이 차순위 오답으로 이동하는 것 탐지), 부정항 arm에는 boxed-compliance·hedging-rate·길이를 co-primary로 (answer-avoidance confound).
5. **TPAU식 per-iteration g′ scoring pass 취소** — M8상 (verifier 멤버십 + 위치)가 같은 마스크를 0 GPU로 제공. terminal-window 정의와 4-gram sibling-success 보호 마스크만 L4 "S4-span" 변형(L3 branch (i)일 때만)으로 회수.
6. 전이 판정은 **≥2 iteration pooling** 후에만 (dip-and-recovery); D_C NLL 암기 지표와 base 2회 re-roll floor는 필수 계측.

### 판정 논리 요약
- **circularity 경고**: A의 attractor mass는 L_N의 직접 최적화 대상이라 "내려감"은 증명이 아님 — **비순환 endpoint는 B의 attractor mass·G_x**. M8상 오답 커밋 span은 경로가 이미 끝난 지점(p≈0.99)이므로 부정항의 기대 결과는 (b) "attractor↓, conversion 평탄"이며 그것도 측정 결과로 취급.
- **novelty의 최종 형태**: loss 식이 아니라 ① regime(transient-adapter 구제 궤적이 EI/SFT 학생에 통제된 share로 들어가는 최초의 통제 연구 — LSPO/HDPO 및 인용 ~28편 모두 미실시), ② endpoint(A→held-out B 전이 + 연속 attractor-mass dose–response — LSPO/HDPO가 명시적으로 미룬 것, LSPO 인용 0건 2026-08-24 기준), ③ estimand 논증(Λ 데이터), ④ 부정 분석(경로 수준 token credit은 이 데이터의 모든 self-contained 신호로 식별 불가; LSPO의 clipped IS는 수치적으로 inert — 둘 다 unclaimed).
