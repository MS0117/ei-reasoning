# Cliff objective 손실 명세 — 완전 전개 (2026-08-25)

`docs/objective_decision_20260823.md` §3(+§8)의 요약 수식을 **실제 구현**(`src/expert_iter/train.py`,
`build_dataset.py`, `config.py`)과 1:1로 맞춰 모든 기호를 정의한 명세.
결정의 *근거*는 원문서에 있고, 수식 *표기*는 이 문서가 기준이다.
원문서 §3 스케치와 구현이 다른 지점은 §7 정오표에 모아 두었다 (특히 F1/F2 — 실험 arm 설계에 영향).
cliff 항이 **왜** 문항 단위인지(길이·NLL·$n_q$ 실측과 근거, L3/L5 차이)는 §9.

---

## 0. 전체 손실 (한 optimizer step)

$$
L(\theta) \;=\; (1-\rho)\, L_S(\theta) \;+\; \rho\,\bigl(\, L_C(\theta) \;+\; \mu\, L_N(\theta) \;+\; L_G(\theta) \,\bigr)
$$

- $L_S$: solved(자연 정답) 데이터의 토큰 정규화 SFT 손실 — 현행 legacy 손실과 동일한 형태.
- $L_C$: 구제(rescued cliff) 궤적의 SFT 손실 — 별도 정규화(§4.2).
- $L_N$: base 자신의 최빈 오답(attractor) 실패 rollout에 대한 bounded unlikelihood (v1일 때만; §4.3).
- $L_G$: 구제 궤적이 학습 중 기준 시점보다 *덜* likely해지는 것을 막는 displacement hinge (§4.4).
- 네 항 모두 **매 optimizer step(window)마다** 그 step에 들어온 예제들로만 계산되고,
  각자 **자기만의 분모**로 정규화된다. 어떤 항의 분모가 0이면(해당 slice가 window에 없음) 그 항은 0.
- `train.sft.cliff.enabled=false`이면 위 구조 전체가 꺼지고 legacy 단일 정규화 손실과 byte-identical
  (테스트로 보증).

---

## 1. 데이터 정의 — 세 slice가 어디서 오는가

모든 예제는 `build_dataset`이 쓰는 `SFTExample` 행(row)이고, `source` 필드가 slice를 결정한다
(collator의 slice id: solved=0, improved=1, negative=2).

### $\mathcal{D}_S$ — solved (source=`solved`)
rollout에서 verifier가 정답 처리한 문제의 자기 생성 정답. 문제당 최단 길이 순으로 최대 4개
(`data` 단계 기존 규칙). 학습 토큰은 response 영역(`solution` region).

### $\mathcal{D}_C$ — 구제 성공 (source=`improved`)
cliff 문제(기본: base 8/8 전부 오답, `partition.cliff_max_correct=0`)에 대해 improve operator
(현재 고정: staged 2+2, stage2 pure-DPO)가 만든 **verifier-정답** 후보 중 C(y) 선택으로 남긴 것.
문제당 최대 `filter.max_per_question = 2`개 → $n_q \in \{1,2\}$.

- $n_q$ = **병합된 학습셋 안에서** 문제 $q$의 improved 행 수. `build_dataset._stamp_n_q`가
  merge( `data.accumulate` 포함) 및 `data.exclude_train_qids`(B셋 제외) **이후** 매 iteration 다시 찍는다.
  accumulate 시 과거 iteration의 구제도 세어진다.
- 각 improved 행에는 $\mathrm{ref}_j$ = C(y) scoring pass의 `s_mean`이 join된다(§4.4).
  join 실패 행은 sentinel $-1$ (guard에서 제외, `guard/skipped_ref`로 카운트).
- completion 끝에 EOS를 보장(`ensure_eos`). 학습 토큰은 continuation 영역.

### $\mathcal{D}_N$ — attractor negative (source=`negative`, `negative.mode=v1`일 때만)
converted cliff 문제 $q$마다, **base policy의 자기 실패 rollout** 중
`finish_reason=="stop"`(완주)이고 추출된 오답이 그 문제의 **최빈 오답**(modal wrong answer)과
일치하는 것 (`_modal_wrong_failures`: 오답 문자열 exact-string 그룹핑, 동률은 (−빈도, 답) 순으로 결정,
`sample_idx` 순 정렬). 문제당 최대 `negative.max_per_question = 8`개 → $n^-_q \in \{1,\dots,8\}$.

- $n^-_q$ = 병합셋 안에서 $q$의 negative 행 수 (같은 `_stamp_n_q`).
- 이 행들은 CE(양의 log-likelihood) 학습을 **절대 받지 않는다**. slice id로만 $L_N$에 들어간다.
- **종결(EOS) 토큰도 $L_N$에 포함된다 — 의도된 상태이며, 코드 주석이 말하는 "보호"는 존재하지 않는다.**
  build_dataset은 negative 행에 `ensure_eos`를 부르지 않지만(주석: EOS unlikelihood가 종결을 억제해
  answer-avoidance/길이 drift를 만든다), **rollout이 이미 EOS를 포함**한다: 기준 run에서
  `finish_reason=="stop"` 샘플 997/997이 `<|im_end|>`(151645)로 끝난다. `_modal_wrong_failures`는
  그 `stop` 샘플만 고르므로 모든 negative 행의 마지막 토큰은 EOS이고, completion 영역(가중 1)이라
  unlikelihood를 그대로 받는다. 2026-08-26 측정([scripts/diag_eos_pressure.py](../scripts/diag_eos_pressure.py),
  결과 `runs/toy_cliff/_diag_summaries/eos_pressure.json`): 오답 궤적의 EOS에서 $p$ median 0.76–0.96,
  **60–72%가 clamp($p\le 1-\delta$) 아래** = gradient가 실제로 흐른다 (per-token 크기는 그 궤적의
  median 토큰 확률 ~0.85와 비슷하다; 특이한 건 크기가 아니라 모든 negative 행이 같은 위치의 같은
  토큰을 같은 방향으로 민다는 **방향의 일관성**).
- **그래서 EOS를 남기는 것이 현재 결정이다** (2026-08-26): ① attractor는 답 토큰이 아니라
  "확신하고 쓰고 멈추는 행동" 전체이고 종결이 곧 커밋이다; ② 같은 측정에서 base는 오답 뒤에도
  정답 뒤와 똑같이 자신 있게 종료하므로(0.956 vs 0.959 / 0.759 vs 0.813 / 0.890 vs 0.905),
  positive의 EOS CE(+)와 negative의 EOS unlikelihood(−)가 만드는 "맞을 때만 종료하라"는 대조는
  **없던 구분을 새로 가르치는** 개입이다; ③ OXA식 uniform unlikelihood는 실패 궤적 전체에 걸리므로
  EOS만 빼면 S4의 3-way(무-부정항 / uniform / span-국소화)에서 uniform leg의 통제군 자격이 흐려진다.
  **전제 조건**: 회피(hedging)로 착지할 위험이 사라진 게 아니라 계측으로 관리되는 것이므로,
  boxed-compliance·hedging-rate·길이·truncation을 부정항 arm의 co-primary로 사전 등록해야 한다.
  **L4 계획**: `negative.drop_terminal_eos` 플래그(기본 false = 현행)를 L3 종료 후 추가하고
  (스키마 변경은 config hash를 바꿔 모든 `.done`을 무효화하므로 실험 중에는 금지),
  S4-v1 × {keepEOS, dropEOS} 짝 arm으로 검정한다.

- **2026-08-27 — 위 "전제 조건"의 계측이 실제로 발화했다 (S4-v1 실측).**
  `runs/L3_S4v1_20260826_194709` (keepEOS = 위 결정 그대로, μ=0.1, negative 465행)의
  held-out cliff B(176문항 ×32샘플) 생성 통계:

  | | 평균 토큰 | 중앙값 | p90 | `length`(잘림) |
  |---|---|---|---|---|
  | base | 3,893 | 2,739 | 8,414 | 2.8% |
  | S3 (부정항 없음) | 3,971 | 2,877 | 8,370 | 2.4% |
  | **S4-v1 (keepEOS)** | **6,116 (+57%)** | 4,703 | **16,384 = `max_tokens` 상한** | **11.2%** |

  holdout에서도 truncation greedy 6.5%→9.5%, @8 8.2%→10.1%. B re-roll 소요는 S0 1h46m → 3h30m+.
  즉 ①의 논거("종결이 곧 커밋")는 유지되지만, ②가 예측한 "없던 구분을 새로 가르치는" 개입의
  **부작용 쪽이 먼저, 크게 관측**되었다: 모델은 "맞을 때만 종료"가 아니라 **덜 종료하는 쪽**으로 이동했다.
  같은 run의 오답 확산(§L3 결과 §4-1: 그외오답 25.1→38.7%, 오답 종류 6.7→8.9)도
  **길이 폭증과 완전히 분리되지 않는다** — 길어진 생성이 더 많이 헤맬 여지를 주므로,
  두 기제의 귀속은 dropEOS arm이 나와야 확정된다.
  **따라서 계획된 짝 arm(S4-v1 × {keepEOS, dropEOS})의 우선순위가 올라갔다.**

- **2026-08-27(2) — 짝 arm 완료: "EOS가 길이 폭증의 원인"은 반증되었다.**
  `runs/L3_S4v1_dropEOS_20260827_041300` (`negative.drop_terminal_eos=true`, 나머지 조건 동일):

  | | 평균 토큰 | 중앙값 | p90 | 잘림 | B Δattractor | B Δavg@32 | 그외오답 | 오답종류 |
  |---|---|---|---|---|---|---|---|---|
  | keepEOS | 6,116 | 4,703 | 16,384 | 11.2% | −23.1pp | +10.0pp | 38.7% | 8.9 |
  | **dropEOS** | **6,491** | 5,324 | 16,384 | **12.6%** | −26.4pp | **+8.0pp** | **44.1%** | **10.4** |

  EOS를 제거해도 **길이가 줄지 않았다(오히려 소폭 증가)**. 확산은 오히려 **심화**됐고
  정답 비율은 12.0%→10.0%로 떨어졌다. 따라서 길이 폭증·확산의 원인은 종결 토큰이 아니라
  **실패 궤적 본문 전체(~3.5k 토큰)에 걸린 uniform unlikelihood**다: 그 궤적들은
  "확신 있고 정돈되고 종결되는 수학 풀이"이므로, 전체 확률을 내리면 모델이 그런 글 자체를
  덜 쓰게 되어 산만해지고 길어진다. EOS를 빼면 그 압력이 답 토큰 쪽으로 더 몰려 확산이 커진다.

  **결론**: ① 위 결정(EOS 유지)의 세 논거는 **길이 측면에서 무해함이 확인**되어 유효하다;
  ② `drop_terminal_eos` 기본값은 **false 유지**가 맞다; ③ uniform unlikelihood(v1)는
  attractor를 확실히 무너뜨리지만 질량을 정답이 아니라 오답 꼬리로 보내므로,
  부정항을 계속 쓴다면 **span 국소화만 남은 선택지**다(패널 OPAL: 최빈-오답 boxed 구간).
  전체 비교표는 [L3_results §4-2·4-3](L3_results_20260826.md).

`negative.mode=v0`는 손실 항이 아니라 **SFT 뒤에 붙는 별도 DPO 단계**다 (전개는 §4.5).
v0에서는 negative 행 자체가 만들어지지 않으므로 $\mathcal{D}_N=\varnothing$이고, SFT 손실은
$(1-\rho)L_S + \rho(L_C + L_G)$가 된다.

---

## 2. 토큰 · 마스크 · 가중 표기

예제 $i$의 학습 입력은 토큰 id 열
$\;y_i = p_i \oplus a_i \oplus c_i\;$ (prompt ⊕ anchor ⊕ completion; anchor는 현재 항상 빈 열,
길이들은 collator에 `prompt_len / anchor_len / completion_len`으로 전달).
텍스트 재토크나이즈 없음 — id 연결이 유일한 조립 방법 (token-id splicing invariant).

**토큰별 cross-entropy** (기호 $\mathrm{ce}$를 완전히 전개하면):

$$
\mathrm{ce}_{i,t}(\theta) \;=\; -\log \pi_\theta\!\bigl(y_{i,t} \,\bigm|\, y_{i,<t}\bigr)
$$

즉 위치 $t$의 관측 토큰 $y_{i,t}$에 대한 모델의 next-token NLL (shifted: 위치 $t$의 예측은
$t{-}1$까지의 접두로부터). 구현은 fp32 `cross_entropy(reduction="none")` (train.py:286–290).

**Region 가중**: 각 토큰은 자기 영역의 가중을 받는다 (`train.sft.region_weights`, 기본값
prompt 0 / anchor 0 / continuation 1 / solution 1):

$$
w_{i,t} \;=\;
\begin{cases}
w_\text{prompt}=0 & t \in p_i\\
w_\text{anchor}=0 & t \in a_i\\
w_\text{solution}=1 & t \in c_i,\ i \in \mathcal{D}_S\\
w_\text{continuation}=1 & t \in c_i,\ i \in \mathcal{D}_C \cup \mathcal{D}_N
\end{cases}
$$

$w_{i,t}=0$이면 collator가 label을 $-100$으로 마스크 → 그 토큰은 loss·분모 어디에도 없다.

**유효 토큰 집합과 가중 질량**:

$$
V_i = \{\, t : w_{i,t} > 0 \,\}, \qquad
W_i = \sum_{t \in V_i} w_{i,t}, \qquad
T_i = |V_i|
$$

기본 0/1 가중에서는 $W_i = T_i$ = completion 토큰 수. 원문서/발표 자료의 $T$는 이
$W$의 0/1-가중 특수형이다. 아래 수식은 전부 일반형 $w, W$로 쓴다.

---

## 3. 배치 표기 — $B$는 "optimizer window"다

- **Window** = 한 optimizer step이 보는 전역 배치. 크기
  $G$ = `train.sft.global_batch_size` = **32** (micro_batch 1 × grad_accum × world_size로 실현).
- $B_S(u), B_C(u), B_N(u)$ = window $u$에 들어온 solved / improved / negative 행의 집합.
  **micro-batch나 rank 단위가 아니다** — 분모는 window 전역에서 한 번에 모은다(§5).
- `StratifiedWindowSampler`(train.py:128)가 **모든** window에 정확히
  $|B_C| = m_C$ = `cliff.m_per_batch` = 1개 improved 행과 (v1일 때)
  $|B_N| = m_N$ = `negative.m_per_batch` = 1개 negative 행을 넣고, 나머지 $G - m_C - m_N$개는 solved.
  이 보장이 있어야 ρ가 "매 step의 실현된 share"가 된다.
- Sampler 세부: epoch당 window 수 $n_\text{win} = \lfloor |\mathcal{D}_S^\text{rows}| / (G-m_C-m_N) \rfloor$
  (나머지 solved 행은 그 epoch에서 drop); improved 행들은 **행 단위로 셔플된 하나의 순환 순서**를
  epoch 내내 cycle (행 수 < $n_\text{win} m_C$이면 의도적 oversampling);
  negative는 그 window의 cliff 행과 **같은 qid에서 우선** 뽑고(없으면 전역 pool),
  window 내부는 셔플(cliff 행이 항상 rank 0에 가지 않도록); stream 길이는 $G$의 배수로 절단되어
  partial window가 없고, accelerate `BatchSamplerShard`의 round-robin 아래에서 연속 $G$-block =
  정확히 한 optimizer window가 된다. `(seed, epoch)`에서 결정적.
- 축퇴 처리: improved 행이 0이면 $m_C{=}0$으로 강등(경고, $L = (1-\rho)L_S$);
  negative 행이 0이면 $m_N{=}0$; solved가 window를 못 채우는 smoke급 데이터셋은
  cliff 순환으로 채우는 fallback (L_S는 zero-guard).

---

## 4. 항별 완전 전개

### 4.1 $L_S$ — solved SFT (legacy와 동일한 형태)

$$
L_S \;=\;
\frac{\displaystyle \sum_{i \in B_S} \sum_{t \in V_i} w_{i,t}\, \mathrm{ce}_{i,t}(\theta)}
     {\displaystyle \sum_{i \in B_S} W_i}
$$

분모 $D_S = \sum_{i\in B_S} W_i$는 window 전역(모든 rank·accum step) 가중 토큰 질량.
0/1 가중이면 "window 안 solved 토큰들의 평균 NLL". legacy 손실과의 차이는 분모에서
cliff/negative 행이 빠진다는 것뿐.

### 4.2 $L_C$ — 구제 궤적, 문제 단위 정규화

예제별 **가중 평균 NLL** (길이 정규화된 시퀀스 손실):V

$$
m_j(\theta) \;=\; \frac{1}{W_j} \sum_{t \in V_j} w_{j,t}\, \mathrm{ce}_{j,t}(\theta)
\qquad [\text{nat/token}]
$$

**per_question_norm = true (기본, S3):**

$$
L_C \;=\;
\frac{\displaystyle \sum_{j \in B_C} \frac{1}{n_{q(j)}}\, m_j(\theta)}
     {\displaystyle \sum_{j \in B_C} \frac{1}{n_{q(j)}}}
\;=\;
\frac{\displaystyle \sum_{j \in B_C} \frac{1}{n_{q(j)}\,W_j} \sum_{t\in V_j} w_{j,t}\,\mathrm{ce}_{j,t}(\theta)}
     {\displaystyle \sum_{j \in B_C} \frac{1}{n_{q(j)}}}
$$

즉 $1/n_q$을 가중치로 하는 $m_j$들의 **가중 평균**. 분자·분모 모두에 $1/n_q$이 있는
비율(ratio) 형태라는 점이 원문서 표기와 같고, 아래 F1이 여기서 나온다.

**단, 기본 설정($m_C=1$)에서 이 식은 그냥 $m_j$다.** sampler가 모든 window에 cliff 행을
정확히 1개만 넣으므로(§3) $|B_C|=1$이고, $1/n_q$이 분자·분모에서 상쇄된다:

$$
\boxed{\;L_C \;=\; \frac{(1/n_q)\,m_j(\theta)}{1/n_q} \;=\; m_j(\theta)
\;=\; \frac{1}{W_j}\sum_{t\in V_j} w_{j,t}\,\mathrm{ce}_{j,t}(\theta)\;}
$$

**매 step의 cliff 항 = 그 step에 뽑힌 구제 1개의 토큰 평균 NLL.** 아래 S3-tok 식도 같은 값이 되고
(★ 블록), $n_q$가 손실 전체에서 죽은 변수라는 함의는 F1에 있다.

**per_question_norm = false (S3-tok):**

$$
L_C^{\text{tok}} \;=\;
\frac{\displaystyle \sum_{j \in B_C} \sum_{t \in V_j} w_{j,t}\, \mathrm{ce}_{j,t}(\theta)}
     {\displaystyle \sum_{j \in B_C} W_j}
$$

(cliff slice 안에서 토큰 pooling — 긴 구제가 window 내 다른 구제보다 큰 비중.)

#### ★ 실제 운용 형태 ($m_C = 1$) — 이 절에서 가장 중요한 식

sampler가 **모든** window에 cliff 행을 정확히 1개만 넣으므로(§3), 위 두 식은 **같은 하나로 붕괴한다**:

$$
L_C \;=\; m_j(\theta) \;=\; \frac{1}{W_j}\sum_{t\in V_j} w_{j,t}\,\mathrm{ce}_{j,t}(\theta)
$$

즉 그 step에 뽑힌 **구제 1개의 토큰 평균 NLL**. $m_N=1$인 $L_N$과 $L_G$도 같은 이유로 단일 행 형태가
되므로, 매 step의 총 손실은

$$
L \;=\; (1-\rho)\,L_S \;+\; \rho\Bigl(\,m_j \;+\; \mu\,\bar u_k \;+\; \max\bigl(0,\ \overline{\mathrm{ce}}_j-\mathrm{ref}_j\bigr)\Bigr)
$$

L3의 cliff-on arm은 negative가 off이므로 실질적으로 $L = (1-\rho)L_S + \rho\,(m_j + \text{hinge})$이다.

**평범한 SFT와의 대비.** 같은 32행을 분모 하나로 pooling하면 ($\Sigma = \sum_{i\in B_S} W_i$):

$$
L^{\text{plain}} \;=\; \frac{\Sigma\,L_S + W_j\,m_j}{\Sigma + W_j}
\;=\; (1-\beta_j)\,L_S + \beta_j\,m_j,
\qquad \beta_j = \frac{W_j}{\Sigma + W_j}
$$

**구조가 완전히 동일하고 섞는 계수만 다르다.** 그리고 그 차이의 원천은 "sequence-level이냐
token-level이냐"가 **아니다** — $m_C=1$에서는 두 모드가 똑같이 그 행의 토큰 평균이다. 원천은
**구제 행이 무엇으로 나뉘는가**이다:

- 평범한 SFT: **배치 전체 길이** $\Sigma+W_j$로 나뉜다 → $W_j$가 분자에 가중치로 남아
  **길이가 dose를 정한다**.
- cliff objective: **자기 길이** $W_j$로 나뉜다 (`per_question_norm` 두 모드 모두 — true는 분자의
  $1/W_j$로, false는 분모의 $W_j$로) → $W_j$가 약분되어 **dose는 언제나 ρ**.

수치 (window = solved 31행 ≈1.8k tok + 구제 1행 ≈5k tok):

| | 구제의 몫 |
|---|---|
| 예제 개수 비율 (손실과 무관한 숫자) | $1/32 = 3.1\%$ |
| 평범한 SFT, 구제가 든 window | $\beta_j \approx 8.2\%$ (구제가 2k 토큰이면 3.5%로 출렁) |
| 평범한 SFT, epoch 평균 | $\approx 5.7\%$ — 대부분의 window엔 구제가 0개 (= legacy token-mass share, `scripts/rho_legacy.py`가 재는 값, S1′이 쓰는 ρ) |
| **S3 ($\rho=0.3$)** | **30%, 매 step 고정** |

따라서 이 objective가 하는 일은 새 손실을 발명한 것이 아니라, **데이터 길이가 우연히 정하던 계수
$\beta_j$를 떼어내고 그 자리에 통제 가능한 상수 $\rho$를 꽂은 것**이다. (위 표의 1.8k/5k는 원문서 추정치 — L3 실측 길이로 다시 계산한
$\beta_j$ 표와 정규화 단위의 근거는 §9.) 이때 **증량**(≈5.7% → 30%)과
**균일 전달**(대부분 0이던 것을 매 step)이 동시에 바뀌므로, 그 둘을 분리하는 것이 S1′ arm
(ρ=legacy, sampler만 on)이다.

> **F1 — $m_C = 1$이면 $1/n_q$은 상쇄되어 완전히 inert하다.**
> $|B_C|=1$일 때 $L_C = \frac{(1/n_q)\,m_j}{1/n_q} = m_j$, 그리고 $L_C^{\text{tok}} = m_j$.
> sampler는 모든 window에 cliff 행을 정확히 1개만 넣고 stream이 $G$의 배수로 절단되어 있으므로
> ($\S3$), **$n_q$는 어떤 step의 손실에도 나타나지 않는다** — 같은 이유로 $L_G$($D_G=\sum 1/n_q$)와
> $m_N{=}1$인 $L_N$에서도 상쇄된다. 즉 현재 설정에서 $n_q$는 손실 전체에서 죽은 변수다
> (build_dataset은 계속 스탬프하고 `n_q_hist`로 로깅하지만 수학적으로 inert).
>
> **상쇄되는 것은 $1/n_q$뿐이고 $1/W_j$(길이 정규화)는 두 경로 모두에서 살아 있다** —
> true는 분자에, false는 분모에 $W_j$가 들어와 결과가 같다. 그래서 "긴 구제가 step을 지배하지
> 않는다"는 성질은 지켜진다. 원문서 §3의 "성공 수·길이와 무관하게 cliff 하나 = 1 단위" 중
> (i) 길이 무관 ✓, (ii) 성공 수 무관 ✗.
>
> 문제 단위 질량은 손실이 아니라 **sampler의 방문 빈도**가 결정하는데, 순환이 행(row) 단위
> 균등이라 $n_q{=}2$인 문제가 $n_q{=}1$인 문제의 2배 epoch 질량을 받는다 ($1/n_q$의 의도와 반대
> 방향; $n_q\le2$라 최대 2:1로 유계이고, converted cliff의 과반이 $n_q{=}1$이라 실제 분포는 더 완만).
>
> **결정 (2026-08-26): sampler를 고치지 않는다.** 손실 코드는 자기가 말하는 것을 정확히 계산하고,
> S3가 실제로 투여하는 처치 — "매 step 구제 1개, 길이 정규화된 평균 NLL, 전체 손실에서 ρ의 몫" —
> 은 그 자체로 잘 정의되어 있으며 A→B 전이라는 L3의 질문을 훼손하지 않는다(A 내부 가중의 문제일 뿐).
> 문제 단위 순환(문제를 균등 순환하고 그 안에서 행 선택)으로 바꾸는 것은 버그 수정이 아니라
> **다른 처치**이므로, 원한다면 L4/L5에서 자체 arm으로 세운다. 논문 서술은 "문제 단위 정규화"가
> 아니라 위의 실제 동작으로 쓴다.

> **F2 — 위 상쇄의 따름정리: 현재 cookbook의 S3-tok arm은 S3(0.3)과 손실이 대수적으로 동일하다.**
> 두 정규화는 한 window에 improved 행이 2개 이상 있을 때만 다른데, sampler는 m_per_batch=1에서
> 그런 window를 만들지 않는다(예외는 smoke급 cliff-fill fallback뿐). seed·데이터 stream도 같으므로
> S3-tok run은 커널 비결정성 수준의 seed-twin이다 — 4~6 GPU-h를 써서 S3를 한 번 더 얻는다.
> (단위 테스트 `test_s3_tok_normalization`은 한 배치에 cliff 여러 행을 직접 넣어 분모 전환만 검증하므로
> 이 축퇴를 가리지 못한다.)
>
> **선택지**: (a) arm 폐기 — F1이 이미 "$m_C{=}1$에서 정규화 단위는 무효"를 보여주므로 그 관찰
> 자체를 결과로 쓴다 (권고); (b) `--override train.sft.cliff.m_per_batch=2`를 함께 걸어 재설계 —
> 값 override라 fresh fork dir 안에서만 hash가 바뀌어 실험 중에도 안전하지만, S3와 window 구성이
> 달라지므로 짝을 맞추려면 S3도 `m_per_batch=2`로 한 번 더 돌려야 한다(2 arm 추가); (c) L4 연기.
> **2026-08-26 기준 미결.** 나머지 L3 arm(S0/S1/S1′/S3/S4-v0)은 영향 없음 — S0·S1은 cliff off라
> 층화 sampler를 아예 쓰지 않고, S3·S1′·S4-v0은 F1의 A-내부 가중 잔차(≤2배)만 받는다.

### 4.3 $L_N$ — bounded unlikelihood (v1)

negative 행 $k$의 토큰별 관측 확률과 unlikelihood:

$$
p_{k,t}(\theta) = \pi_\theta\!\bigl(y_{k,t}\,\bigm|\,y_{k,<t}\bigr) = e^{-\mathrm{ce}_{k,t}(\theta)},
\qquad
u_{k,t}(\theta) = -\log\!\Bigl(1 - \min\bigl(p_{k,t}(\theta),\, 1-\delta\bigr)\Bigr)
$$

$\delta$ = `negative.delta` = 0.02. 항 전체 ($L_C$와 같은 구조, $\mathrm{ce} \to u$, $n_q \to n^-_q$):

$$
L_N \;=\;
\frac{\displaystyle \sum_{k \in B_N} \frac{1}{n^-_{q(k)}\,W_k} \sum_{t \in V_k} w_{k,t}\, u_{k,t}(\theta)}
     {\displaystyle \sum_{k \in B_N} \frac{1}{n^-_{q(k)}}}
$$

$m_N=1$이면 F1과 동일하게 $L_N = \frac{1}{W_k}\sum_t w\,u$ 로 축약(그 step의 negative 1개의 평균 unlikelihood).

**성질 (구현 그대로, train.py:333–337):**
- $u$는 fp32로 계산: `p = exp(-ce).clamp(max=1-δ)`, `u = -log1p(-p)`.
- **유계**: $u_{k,t} \le -\log\delta \approx 3.91$ nat — $-\log(1-p)$의 $p\to1$ 발산을 clamp가 차단.
- **자기 제한 gradient**: 관측 토큰 자신의 logit $z$에 대해 $\partial u/\partial z = p$.
  즉 모델이 확신하는(p 큰) attractor 토큰일수록 세게 내리고, 이미 낮은 토큰은 거의 건드리지 않는다.
  단 $p > 1-\delta$인 초확신 토큰은 clamp가 미분을 끊어 **gradient가 0** — 손실도 상수 $-\log\delta$.
  (하한 유계와 맞바꾼 dead zone; δ를 줄이면 zone은 줄고 상한은 커진다.)
- 다른 후보 logit $z'$에 대해서는 $\partial u/\partial z' = -p\,p'/(1-p)$ — 대안 토큰을 확률 비례로 올린다.
- negative 행은 label이 살아 있어도 CE 항($L_S, L_C$)에는 slice mask로 절대 들어가지 않는다.

### 4.4 $L_G$ — displacement guard

구제 행 $j$의 **비가중 completion 평균 NLL** (region 가중 없이, completion 마스크 × 유효 label만;
train.py:341–343 — $\mathrm{ref}$와 단위를 맞추기 위해 비가중):

$$
\overline{\mathrm{ce}}_j(\theta) \;=\; \frac{1}{T^{c}_j} \sum_{t \in c_j \cap V_j} \mathrm{ce}_{j,t}(\theta),
\qquad T^{c}_j = |c_j \cap V_j|
$$

기준값 $\mathrm{ref}_j$ = filters 단계 C(y) scoring pass의 `s_mean` = **같은 continuation을
scoring policy가 매긴 토큰당 평균 NLL** (candidate_scores.jsonl에서 `{qid}:{base_sample_idx}:{attempt_idx}`
키로 join; guard가 켜져 있으면 이 파일이 없을 때 하드 에러 → `filter.selection.always_score=true` 필요).
scoring policy는 그 iteration의 현재 policy이므로 **iter 0에서는 정확히 base $\pi_{\theta_0}$**
(L2/L3는 전부 iter-0 실험이라 원문서의 $\overline{\mathrm{ce}}_j(\theta_0)$ 표기와 일치;
k>0 loop에서는 rollout policy = iter_{k-1} ckpt의 NLL이라는 각주가 붙는다).

$$
L_G \;=\;
\frac{\displaystyle \sum_{j \in B_C^{\mathrm{ref}}} \frac{1}{n_{q(j)}}\,
      \max\!\bigl(0,\ \overline{\mathrm{ce}}_j(\theta) - \mathrm{ref}_j\bigr)}
     {\displaystyle \sum_{j \in B_C^{\mathrm{ref}}} \frac{1}{n_{q(j)}}}
,\qquad
B_C^{\mathrm{ref}} = \{\, j \in B_C : \mathrm{ref}_j \text{ join됨} \,\}
$$

- **원문서 §3은 이 항을 정규화 없는 합으로 썼지만, 구현은 $L_C$와 같은 비율 정규화다** (§7 정오표).
  $m_C=1$이면 어차피 $L_G = \max(0, \overline{\mathrm{ce}}_j - \mathrm{ref}_j)$로 동일.
- margin 없는 순수 hinge (DPOP처럼 여유폭 없음): 구제가 기준 시점보다 조금이라도 덜 likely해지는
  순간부터 선형 페널티. 잘 학습되면(NLL이 ref 아래) 0 — 평상시 비활성.
- ref 미join 행은 제외되고 `guard/skipped_ref`로 카운트(무단 탈락 감시).

### 4.5 v0 — 손실 항이 아니라 SFT→DPO 2단계 (S4-v0 arm)

`negative.mode=v0`는 위 $L$의 어떤 항도 바꾸지 않는다. **학습 단계 자체가 하나 늘어난다**
(`train.objective=sft+dpo`, [train.py:661–667](../src/expert_iter/train.py#L661-L667)):

**1단계 — SFT.** cliff objective를 그대로 켠 채 돈다:

$$
L^{\text{v0}}_{\text{SFT}} \;=\; (1-\rho)\,L_S \;+\; \rho\,\bigl(L_C + L_G\bigr)
$$

$\mu\cdot L_N$이 없는 이유는 $\mu$를 0으로 둬서가 아니라 **negative 행이 아예 만들어지지 않기** 때문이다
(build_dataset은 `neg_mode == "v1"`일 때만 `source="negative"` 행을 쓴다) → $\mathcal{D}_N=\varnothing$ →
sampler가 $m_N=0$으로 자동 강등(경고 1줄), 그 칸은 solved로 채워진다.
config 검증상 `negative.mode != off`는 `cliff.enabled=true`를 요구하므로 **v0도 반드시 cliff objective 위에서 돈다.**

**2단계 — DPO.** 1단계 결과 체크포인트에서 초기화하고, 참조 모델 $\pi_{\text{ref}}$ = 그 SFT 결과다
(`dpo_init = out_dir`). 쌍 $(y_w, y_l)$에 대한 표준 `trl.DPOTrainer` sigmoid 손실:

$$
L_{\text{DPO}} \;=\; -\log \sigma\!\left(\beta\left[
\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
-\log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\right]\right)
$$

- $y_w$ (chosen) = 구제 성공의 continuation (EOS 보장), $x$ = prompt(+anchor, 현재 빈 열).
- $y_l$ (rejected) = `train.dpo.rejected_selection`이 정한다
  ([build_dataset.py:176–190](../src/expert_iter/build_dataset.py#L176-L190)):
  - `base_pick` (legacy 기본): anchor 단계가 고른 `base_sample_idx` rollout을 **같은 anchor 뒤로 잘라** 사용.
  - `modal_wrong` (**S4-v0가 쓰는 값**): 그 문제의 최빈 오답(attractor)을 가진 rollout. 그것이
    `base_sample_idx`와 다른 샘플이면 anchor가 그 prefix가 아니므로 **응답 전체**가 rejected가 된다
    (sequence-level negative; anchor가 빈 지금은 어차피 동일). 유효한 최빈 오답 실패가 없는 문제는
    `base_pick`으로 fallback하고 `dpo_rejected_fallback`으로 카운트된다.
- 하이퍼파라미터(`train.dpo`): lr 5e-7, $\beta$ 0.1, epochs 1, global_batch_size 16, loss_type sigmoid,
  max_grad_norm 1.0.

**v0 ≠ v1인 지점** (같은 신호의 두 구현이 아니라 별개 arm이다):

| | v0 | v1 |
|---|---|---|
| 어디서 도는가 | SFT 끝난 뒤 **별도 DPO 단계** | SFT 손실 **안**의 $\mu L_N$ 항 |
| 참조 모델 | SFT 결과 $\pi_{\text{ref}}$ 필요 (메모리 2배) | 없음 |
| 정규화 | 쌍 단위 균등 — $\rho$·$n_q$·window 층화 **전부 무관** | $\rho$ 괄호 안, $1/(n^-_q W_k)$ |
| 부정 신호 형태 | 상대적 (chosen 대비 rejected를 낮춤) | 절대적 (rejected 확률만 낮춤, 유계) |
| positive 보호 | DPO 항 자체가 chosen을 같이 올림 | $L_G$ hinge가 담당 |
| 코드 | 기존 경로 재사용 (zero-code) | 신규 손실 분기 |

L3 1라운드는 **v0만** 돈다 (`bash scripts/l3_arm.sh S4v0 <frozen>`), v1은 L4 arm이다.

---

## 5. 분산 구현 — 정규화가 topology-불변인 이유

- **분자는 local, 분모는 global.** `_get_num_items_in_batch`(train.py:390)가 각 optimizer window의
  compute_loss들보다 **먼저 한 번** 호출되어(transformers 5.7 계약, 검증됨) 4-분모 벡터
  $[D_S, D_C, D_N, D_G]$ + legacy 스칼라를 **단 한 번의 all-gather**로 전 rank·전 accum에 대해 모으고
  trainer에 stash한다. 각 micro-step의 compute_loss는 local 분자를 이 global 분모로 나누므로,
  micro loss들의 합 = 위 §4의 항 정의와 **정확히** 일치 (micro_batch/grad_accum/rank 분할 방식과 무관 —
  `tests/test_cliff_objective.py`의 accum-topology 불변성·rank-shard 분모 가산성 테스트).
- $D$의 정의 (구현 그대로): $D_S = \sum_{B_S} W_i$; $D_C = \sum_{B_C} 1/n_q$ (S3-tok이면 $\sum_{B_C} W_j$);
  $D_N = \sum_{B_N} 1/n^-_q$; $D_G = \sum_{B_C^{\mathrm{ref}}} 1/n_q$. 분모 0인 항은 0으로 zero-guard.
- **DP 보정**: `average_tokens_across_devices` + world_size>1이면 loss에 ×world_size —
  전역 분모를 쓰는 상태에서 DP all-reduce가 gradient를 평균내는 것을 되돌리는 계수 (legacy 경로와 동일,
  2-GPU zero2에서 경험 검증).
- **로깅**: window당 성분 평균 `loss/solved · loss/cliff · loss/negative · loss/guard`,
  카운터 `cliff/rows · negative/rows · guard/skipped_ref`.
  원문서 §7이 요구한 **group별 gradient-norm share 로깅은 아직 미구현** — ρ는 loss share일 뿐,
  구제의 per-token NLL이 solved의 ~3배인 상태에서 `max_grad_norm=1.0` clipping과 상호작용하므로
  gradient share의 실현 여부는 현재 계기로는 안 보인다 (열린 항목).

---

## 6. 하이퍼파라미터

| 기호 | config 키 | 기본값 | 탐색 (§3 원안) | §8 패널 수정 후 1라운드 |
|---|---|---|---|---|
| $\rho$ | `train.sft.cliff.rho` | 0.1 | {legacy≈0.03–0.06, 0.1, 0.3} | **{legacy, 0.3}** (`scripts/rho_legacy.py`가 frozen dataset 토큰 질량에서 legacy 값 산출) |
| $\mu$ | `cliff.negative.mu` | 0.1 | {0, 0.1, 0.3} | **0 고정**, S4는 μ-v0(sft+dpo) 단일 arm |
| $m_C$ | `cliff.m_per_batch` | 1 | — | 1 (S3-tok을 살리려면 ≥2 필요 — F2). **L5: `auto`** = 모든 improved 행을 epoch당 ≥1회 보는 최소값 (§9.4) |
| $m_N$ | `cliff.negative.m_per_batch` | 1 | — | v1일 때만 |
| $\delta$ | `cliff.negative.delta` | 0.02 | — | — |
| $n_q$ 상한 | `filter.max_per_question` | 2 | — | — |
| $n^-_q$ 상한 | `cliff.negative.max_per_question` | 8 | — | — |
| guard | `cliff.guard.enabled` | true | — | on (`always_score` 필요) |
| $G$ | `train.sft.global_batch_size` | 32 | — | — |
| region 가중 | `train.sft.region_weights` | p0/a0/c1/s1 | — | — |

주의: 어떤 `train.*` 키를 바꿔도 run 전체 config hash가 바뀌어 기존 `.done`이 전부 무효화된다 —
arm은 반드시 `scripts/fork_run.py`/`l3_arm.sh`로 fresh run dir에.

---

## 7. 정오표 — 원문서 §3 표기 vs 구현

1. **$\mathrm{ce}$의 실체**: $\mathrm{ce}_{i,t} = -\log \pi_\theta(y_{i,t}\mid y_{i,<t})$ (shifted, fp32),
   그리고 모든 항의 분자에는 region 가중 $w_{i,t}$가 곱해진다(기본 0/1이라 §3 표기와 수치 동일).
   $T$는 일반형에서 가중 질량 $W$.
2. **$L_G$ 정규화**: §3는 $\sum_j (1/n_q)\,\mathrm{hinge}_j$ (합), 구현은 $D_G = \sum 1/n_q$로 나눈
   **가중 평균**. $m_C=1$에서는 동일하므로 실험상 무해하나 표기는 이 문서가 맞다.
3. **F1**: $m_C=1$에서 $1/n_q$은 매 window 상쇄되어 **손실 전체에서 inert**($L_C$·$L_G$·$L_N$ 모두).
   길이 정규화 $1/W_j$는 두 경로 모두에서 유효. 문제 단위 질량은 sampler 방문 빈도가 결정하고
   행 단위 균등 순환이라 문제 질량 ∝ $n_q$ (≤2배, 의도와 반대 방향). "성공 수 무관 1 단위"는 미실현.
   **sampler는 고치지 않기로 결정(2026-08-26)** — 현 처치가 잘 정의되어 있고 A→B 전이 질문을
   훼손하지 않으며, 문제 단위 순환은 수정이 아니라 다른 처치이므로 L4/L5의 별도 arm 사안.
   실측 크기(어려운 : 쉬운 문항 = 1 : 2)와 L5 `auto`에서의 해소는 §9.2/§9.4.
4. **F2**: 그 따름정리로 cookbook의 S3-tok arm(= `per_question_norm=false`만 변경)은 S3(0.3)의
   seed-twin. 폐기(권고) / `m_per_batch=2` 짝으로 재설계 / L4 연기 중 **미결**.
5. **guard 기준의 policy**: §3의 $\theta_0$(base)는 iter-0에서만 엄밀히 성립.
   k>0에서 `s_mean`은 그 iteration의 scoring policy(= iter_{k-1} ckpt) 기준.
6. **gradient-norm share 로깅 미구현** (§5) — §7 위험 항목이 아직 열려 있음.
7. **negative의 EOS**: build_dataset이 `ensure_eos`를 부르지 않지만 rollout이 이미 EOS를 포함하므로
   $L_N$은 종결 토큰에도 걸린다(오답 궤적의 60–72%에서 clamp 아래 = gradient 유효). 코드 주석이
   말하는 보호는 존재하지 않으며, **EOS 유지가 현재 결정**이다(§1). 플래그·검정 arm은 L4.

## 8. 코드 맵

| 대상 | 위치 |
|---|---|
| region 가중·마스크·slice/n_q/ref/completion 채널 | train.py:53–121 (`WeightedCausalCollator`) |
| window 층화·순환·same-qid negative·절단 | train.py:128–239 (`StratifiedWindowSampler`) |
| 4항 합성·unlikelihood·hinge·zero-guard·성분 로깅 | train.py:271–388 (`compute_loss`, `log`) |
| 전역 분모 1-gather | train.py:390–434 (`_get_num_items_in_batch`) |
| $\mathrm{ref}_j$ join (s_mean) | build_dataset.py:67–94 |
| negative 구성 (`ensure_eos` 미호출 — 단 §1 참조) / 최빈 오답 추출 | build_dataset.py:96–119 / 233–254 |
| $n_q, n^-_q$ 스탬프 (merge·exclude 후) | build_dataset.py:280–285 |
| config 정의 (`CliffTermCfg` 등) | config.py:575–648 |
| v0 (sft+dpo + modal_wrong rejected) | build_dataset.py `_build_dpo_pairs`, config `train.dpo.rejected_selection`; SFT→DPO 순차 실행 train.py:661–667 |
| EOS unlikelihood 압력 측정 (CPU, diag 덤프 기반) | [scripts/diag_eos_pressure.py](../scripts/diag_eos_pressure.py) → `runs/toy_cliff/_diag_summaries/eos_pressure.json` |
| L3 arm 실행 | [scripts/l3_arm.sh](../scripts/l3_arm.sh) (fork + train/eval + B readout) |
| 검증 | tests/test_cliff_objective.py (topology 불변성, 분모 가산성, legacy byte-identical); GPU smoke `runs/smoke_20260824_221441` |

---

## 9. 왜 cliff 항만 문항(sequence) 단위인가 — 길이 실측과 근거 (2026-08-31 추가)

§4.1/§4.2의 두 정규화가 **왜 다른가**를 한곳에 모은다. 원문서 §2의 길이 추정(solved ≈1.8k / 구제 ≈5.6k,
2000문항 toy mix)은 L3 실측과 다르므로 여기 숫자가 기준이다. 모든 값은 L3 S3 arm이 실제로 학습한 행
(`runs/L3_S3_20260826_011420/iter_0/dataset/train_sft.jsonl`, B 제외 후 solved 5,366 / improved 118,
converted A 문항 82)에서 잰 것. loss 토큰 = `len(input_ids) − prompt_len` (improved 행의 `anchor_len`은 전부 0 확인).

### 9.1 실측 — 두 slice의 길이·NLL·문항당 행 수

| slice | 행 | loss 토큰 평균 | 중앙 | p10 | p90 | 최대 | 토큰 합 | 토큰 share | 행 share |
|---|---|---|---|---|---|---|---|---|---|
| solved | 5,366 | 2,941 | 2,196 | 644 | 6,491 | 12,606 | 15.78M | 97.2% | 97.8% |
| improved(구제) | 118 | 3,819 | 2,758 | 630 | 8,679 | 12,071 | 0.45M | **2.8%** | 2.2% |

- **평균 격차는 1.3배**에 그친다 (원문서의 3배 추정은 toy operator·shortest 선택 전 값). 큰 것은 **slice 안의
  산포**다 — 구제는 p10 630 → p90 8,679로 **14배**, solved도 644 → 6,491로 10배.
- **per-token NLL은 base 하에서 ~3배**: 구제의 $\mathrm{ref}_j$(= base $s_\text{mean}$) 평균 0.327 / 중앙 0.297;
  학습 로그 첫 기록(epoch 0.03, warm-up lr ≤3.6e-6) `loss/solved` 0.088 vs `loss/cliff` 0.260.
  즉 구제 토큰 하나는 solved 토큰 하나보다 gradient 신호가 ~3배 크다 — 토큰 pooling에서 구제의
  실효 몫이 토큰 share(2.8%)보다 커지는 이유이자, 그 몫이 길이·NLL이라는 데이터 속성에 끌려다니는 이유.
- legacy 토큰-질량 share (`scripts/rho_legacy.py`): **0.0278** (S3 학습셋, A만 118행). L3 문서의
  0.0556은 B 제외 전 frozen 셋(A+B 243행) 값 — 둘 다 맞고 데이터셋이 다르다.

**문항당 행 수 $n_q$와 난이도·길이** (converted A 82문항):

| $n_q$ | 문항 | 구제 길이 평균 / 중앙 | base floor 정답 수/64 | ≥1정답 (64) | adapter 정답 후보/문항 |
|---|---|---|---|---|---|
| 1 | 46 (**56%**) | **4,727** / 4,317 | 1.74 | 0.587 | 1.00 |
| 2 | 36 (44%) | 3,239 / 1,810 | **3.22** | 0.583 | **4.06** |

- $n_q{=}2$인 문항은 **더 쉬운 cliff**다: base가 64샘플에서 맞힌 수가 1.9배, adapter의 정답 후보가 4배.
  $n_q$는 "그 문항의 중요도"가 아니라 operator 샘플링에서 몇 개가 verifier를 통과했는가의 우연이고,
  그 우연은 쉬운 쪽으로 기운다.
- **어려운(singleton) 문항의 구제가 더 길다** (4.7k vs 3.2k). 길이는 operator의 장황함만이 아니라
  난이도도 일부 반영한다 — 아래 9.3 ①에서 이 점을 다룬다.

### 9.2 분석 — 길이가 dose를 정하면 무슨 일이 일어나나

§4.2 ★의 전개($L^\text{plain} = (1-\beta_j)L_S + \beta_j m_j$, $\beta_j = W_j/(\Sigma+W_j)$)에 실측 길이를 넣으면
(window = solved 31행 × 2,941 = 91.2k 토큰 + 구제 1행):

| 구제 1행의 길이 | 평범한 SFT에서 그 step의 구제 몫 $\beta_j$ | S3 ($\rho$=0.3) |
|---|---|---|
| p10 630 | 0.7% | 30% |
| 중앙 2,758 | 2.9% | 30% |
| 평균 3,819 | 4.0% | 30% |
| p90 8,679 | 8.7% | 30% |
| 최대 12,071 | 11.7% | 30% |
| epoch 평균 (대부분 window에 구제 0개) | **2.8%** (= legacy share) | 30% |

같은 "구제 1개"인데 **길이만으로 17배**(0.7% → 11.7%) 출렁인다. 어느 문항을 얼마나 배우는지가
그 문항의 bridge가 몇 토큰이었는가로 정해지는 것이다. S3는 이 계수를 떼어내고 상수 $\rho$를 꽂는다.

**slice 안에서** ($m_C > 1$인 L5 regime, §9.4) 토큰 pooling(S3-tok)과 문항 단위(S3)의 차이 — window에
cliff 행 4개가 들었다고 하면:

| 행 | 문항 | 길이 | $n_q$ | S3-tok 몫 $W_j/\sum W$ | S3 몫 $(1/n_q)/\sum(1/n_q)$ |
|---|---|---|---|---|---|
| $j_1$ | $q_1$ | 8,000 | 1 | **50%** | 33% |
| $j_2$ | $q_2$ | 2,000 | 1 | **12.5%** | 33% |
| $j_3, j_4$ | $q_3$ | 3,000 ×2 | 2 | 37.5% (행 2개 합) | 33% (½+½) |

토큰 단위에서는 $q_1$이 $q_2$의 4배를 배우고, 이유는 구제가 길다는 것 하나다. 문항 단위에서는 셋이 같다.

**$n_q$ 편향을 epoch 질량으로 환산** (9.1의 실측: singleton 1행 4.7k vs $n_q{=}2$ 2행 × 3.2k = 6.5k):

| 방식 | 어려운 문항($n_q{=}1$) : 쉬운 문항($n_q{=}2$) |
|---|---|
| legacy 토큰 pooling | 4.7k : 6.5k = **1 : 1.4** |
| S3, $m_C{=}1$ (L3 실제 — 행 단위 순환, $1/n_q$ inert, F1) | 방문 1 : 2 = **1 : 2** |
| S3, $m_C>1$ (L5 `auto` — $1/n_q$ 유효) | **1 : 1** |

즉 L3의 S3가 실제로 투여한 문항 가중은 이 축에서 legacy보다 **더** 쉬운 쪽으로 기울어 있었다(F1이
말한 "의도와 반대 방향"의 크기). L5에서 처음 1:1이 된다.

### 9.3 근거 — 왜 cliff는 문항 1표이고 solved는 토큰 1표인가

토큰 단위 정규화는 "모든 토큰이 같은 무게 = 같은 한 표"다. 행들이 서로 교환 가능한 i.i.d. 텍스트라면 그게
MLE이고 옳다. cliff 행은 그 전제를 세 가지 이유로 어긴다.

1. **길이는 문항의 속성이 아니라 operator의 산물이다.** 구제 길이는 bridge 프롬프트가 얼마나 길게 썼는가로
   정해지고(9.1: slice 안 14배 산포), sequence-level 신호 $\Lambda$는 사실상 길이 통계였다
   (원문서 §2: Spearman(Λ, 길이) = −0.82). 토큰으로 가중하면 "operator가 장황할수록 그 문항을 많이 배운다"가
   된다. 단, 9.1이 보여주듯 길이는 난이도와도 상관한다(singleton 4.7k vs 3.2k). 그렇더라도 어려운 문항에
   가중을 더 주고 싶다면 그것은 **명시적 난이도 가중**으로 해야지 장황함을 대리변수로 쓸 일이 아니다 —
   그리고 9.2의 환산이 보여주듯 토큰 pooling은 결과적으로 어려운 문항에 **덜** 준다(1 : 1.4).
2. **$n_q$는 operator 샘플링의 우연이고 쉬운 cliff 쪽으로 기운다** (9.1: $n_q{=}2$ 문항은 base 정답 1.9배,
   adapter 정답 후보 4배). 행/토큰 단위로 세면 쉬운 cliff가 더 많이, converted의 56%인 가장 어려운
   singleton이 가장 적게 학습된다. 또한 같은 문항의 구제 2개는 독립 증거가 아니라 **같은 사실을 두 번 적은
   것**이다(dedup 후 shortest 2개).
3. **측정 단위와 학습 단위가 같아야 한다.** endpoint는 문항별 attractor mass와 문항별 paired sign test — 문항이
   1표다. 학습에서 긴 구제 문항에 4표를 주고 평가에서 1표로 세면 objective와 논문의 estimand가 어긋난다.

**solved를 토큰 단위로 두는 이유**는 반대로 단순하다. (a) solved는 처치가 아니라 기질(substrate)이고 이미
문항당 shortest ≤4로 정리된 표준 SFT 데이터다. (b) 무엇보다 **현행 loss 그대로여야** S3 vs S1의 차이가
cliff 항 하나로 국한된다 — solved 정규화까지 바꾸면 두 arm의 차이에 두 번째 변수가 들어간다.

**비용.** 문항 단위에서는 긴 구제의 **개별 토큰**이 약하게 학습된다 — 8k 행의 토큰 하나는 2k 행의 토큰보다
¼ 무게. "긴 풀이에 내용이 더 많다"고 믿으면 손해인데, ①의 실측이 그 믿음을 지지하지 않고, `filter.selection.method=shortest`가
어차피 짧은 구제를 고르므로 노출은 작다.

**검증 상태 — 주의.** L3는 "자기 길이로 나누기 + $\rho$ 고정 + 매 step 공급"을 **묶음으로** legacy와 대조한
것이다(S3 −11.0pp vs S1 −4.0pp). 문항 단위 vs 토큰 단위 자체는 **분리 검정된 적이 없다** — $m_C{=}1$에서 두
식이 대수적으로 같기 때문(F2). 이 절의 근거는 실측 + 논증이지 arm 결과가 아니며, 분리 검정은 $m_C>1$인
L5 regime에서 `per_question_norm=false` 짝으로만 가능하다.

### 9.4 L3에서 실현된 것, L5에서 실현되는 것 (`cliff.m_per_batch: auto`)

F1(§7-3) 요약: $m_C{=}1$이면 window에 cliff 행이 하나라 $1/n_q$이 분자·분모에서 상쇄되고, 살아남는 것은
$1/W_j$뿐이다. L3의 S3는 그래서 9.3 ①만 실현했고 ②는 못 했다(9.2 환산표의 1 : 2). 참고로 L3 규모
(5,366/118)는 $m_C{=}1$로도 173 windows ≥ 118행이라 **coverage 1.0**(행당 ~1.5회/epoch)이었다 — 굶은 게
아니라 문항 균등화가 안 된 것이다.

L5는 solved : cliff 비가 훨씬 작아 $m_C{=}1$이면 구제 대부분이 학습에 못 들어간다. `auto` =
$n_\text{win}(m_C)\cdot m_C \ge |\mathcal{D}_C^\text{rows}|$를 만족하는 최소 $m_C$
(`StratifiedWindowSampler._auto_m_c`, fill이 줄면 $n_\text{win}$도 변하므로 스캔). smoke run 실측:

| run (solved/cliff 행) | $m_C{=}1$ coverage | `auto` $m_C$ | windows/epoch | coverage |
|---|---|---|---|---|
| L3 S3 (5,366/118) | 1.00 | 1 | 173 | 1.00 |
| L5 lspo 500문항 iter2 (1,237/110) | 0.35 | **3** | 42 | 1.00 |
| L5 lspo 500문항 iter0 (526/61) | 0.26 | 4 | 18 | 1.00 |
| L5 staged mix300 (194/153, smoke) | 0.04 | 15 | 11 | 1.00 |

$m_C > 1$이면 §4.2의 $L_C$가 원식 그대로 작동한다: window 안에서 행 $j$의 몫은 $(1/n_q)/\sum_{B_C} 1/n_q$,
sampler가 행을 균등 순환하므로 문항 $q$의 epoch 질량 $\propto n_q \times 1/n_q = 1$. **"성공 수·길이와 무관하게
cliff 하나 = 1 단위"는 L5에서 처음 온전히 실현된다.** 논문 서술 시 L3 arm과 L5 arm의 cliff 항이 이 점에서
다르다는 것을 각주로 남길 것. (본 run의 실제 $m_C$는 `iter_k/logs/train.log`의 `[train] cliff objective on:` 줄.)
