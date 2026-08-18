# 연구 메모: Cliff RL의 신호 제조 — Bridge-Prefix Curriculum + 그룹 주입

2026-08-15. 배경: toy cliff 3-arm 결과(CONTROL 24.3% / LSPO 30.8% / BRIDGE 42.1% 구제,
BRIDGE vs CONTROL McNemar p=0.0034) 이후, BRIDGE+RL(GRPO)이 zero-advantage 76.5%로
신호 없이 도는 문제에 대한 다음 단계 제안.

---

## 0. 진단 — 현행 RL이 실패하는 이유

### "support"란 무엇인가

어떤 분포의 support = 그 분포가 0이 아닌 확률을 주는 사건들의 집합. 여기서는:

> **현재 정책이 x에서 rollout할 때, 정답 궤적이 실제로 뽑힐 확률이 의미 있게 있는가?**

on-policy RL(GRPO 포함)은 **자기가 방금 뽑은 샘플만** 강화/억제할 수 있다. 정답 궤적이 한 번도 안 뽑히면 "이걸 더 하라"고 가리킬 대상 자체가 없다. cliff는 정의상 p̂(x)≈0 — 즉 정답 궤적이 현재 정책의 (실효적) support 밖에 있다. 이게 "support가 없다"의 뜻이다.

수치로: p=0.03, K=8일 때 그룹에 정답·오답이 섞일 확률 = 1−(1−p)⁸−p⁸ ≈ **21.7%**. GRPO는 그룹 보상이 전부 같으면 advantage=0 → gradient=0이므로, 나머지 78.3%의 스텝은 아무 일도 안 한다. 관측치(zero-advantage 76.5%, 실효 스텝 16/68 ≈ 23.5%)와 정확히 일치 — **GRPO는 정상 동작 중이고, 뽑을 정답이 없을 뿐이다.**

따라서 처방 순서:
1. **rollout 분포에 성공 궤적을 넣어준다** (시작 상태를 옮김 → 방법 A)
2. **이미 확보한 성공을 그룹에 재사용한다** (off-policy 주입 → 방법 B)
3. 그 다음에야 objective 세부 튜닝 (§5)

objective를 아무리 바꿔도 1·2 없이는 0에서 신호를 만들 수 없다.

---

## 1. 방법 A — Bridge-Prefix Reverse-Curriculum RL (BPC-RL)

### 1.1 무엇인가

RL rollout의 시작 상태를 `x`가 아니라 `x + z⁺[:ℓ]`로 바꾼다.

- z⁺ = 그 문제의 **verifier를 통과한 self-generated 궤적** (bridge 궤적 또는 verified 후보, §3의 소스 우선순위).
- ℓ = step 경계(`"\n\n"`)에서 자른 prefix 길이. 문제별로 고르고, 학습이 진행되면 0으로 줄인다.

직관: "백지에서 완주"(p≈0.03)가 아니라 "옳은 길 중간에서 이어 완주"로 바꾸면 성공 확률을 원하는 수준으로 끌어올릴 수 있다. GRPO 그룹에 정답·오답이 섞이기 시작하고, gradient가 생긴다. ℓ→0 annealing으로 최종 목표인 `p(y|x)`에 수렴시킨다 (R3, Xi et al. 2024의 reverse curriculum과 같은 뼈대).

**중요한 성질: prefix는 손실이 걸리는 타깃이 아니라 컨텍스트다.** GRPO 손실은 completion 토큰에만 걸린다. prefix 텍스트 자체는 절대 학습 타깃이 되지 않고, ℓ=0 시점의 정책만 최종 산출물이다.

### 1.2 원안("RL 프롬프트에 reference 추가")과의 차이 2가지

| 원안 | 정제판 | 근거 |
|---|---|---|
| gold y\* 텍스트를 힌트로 | **self-generated z⁺의 prefix** | 3-arm 결과가 직접 근거: gold 텍스트 조건화(LSPO)는 비유의(+6.5%p, p=0.248), self-distribution 궤적(BRIDGE)만 유의(p=0.0034). gold prefix는 모델에게 OOD 문맥이라 이어쓰기가 부자연스럽다. 또 y\*를 프롬프트에 넣지 않는 repo 불변식이 유지된다. |
| 힌트 고정 | **ℓ→0 annealing** | 고정 힌트는 p(y\|x,힌트)를 학습 → 테스트 시 분포 이탈. annealing이 이 간극을 점진적으로 닫는다. |

### 1.3 알고리즘 (구체)

```
입력: cliff 문제 집합 Q, 문제별 prefix 소스 z⁺(qid) (§3), fit adapter φ₀
파라미터: 목표 밴드 [τ_lo, τ_hi] = [0.2, 0.8], 프로브 횟수 m=8, 그리드 크기 4

# ── 초기 prefix 선택 (project-back α* 스윕과 동일한 모양) ──
for each qid:
    L(qid) = z⁺(qid)의 step 경계 중 균등 4개  (예: 0%, 25%, 50%, 75% 지점)
한 번의 vLLM pool로 모든 (qid, ℓ ∈ L(qid))를 m회씩 생성·채점   # fit 불필요, 생성만
ℓ*(qid) = min{ℓ : p̂(ℓ) ≥ τ_lo}      # "최소 개입" — α* = min{α: P̂≥τ}와 같은 원칙

# ── curriculum RL 라운드 ──
repeat:
    RL 행 = { (x_qid + z⁺[:ℓ*(qid)]) : qid ∈ Q }    # build_rl_rows의 anchor 주입 경로 그대로
    GRPO 몇 epoch (현행 improve.rl 설정)
    재프로브: p̂(qid, ℓ*(qid)) 갱신
    if p̂ ≥ τ_hi: ℓ*(qid) ← 한 단계 짧은 경계      # 뒤로 물러나기
    if ℓ*(qid) == 0 and p̂ ≥ τ_lo: qid 졸업
until 전원 졸업 or 예산 소진
출력: 어댑터 φ_E → 후보 샘플링은 기존처럼 ℓ=0 (bare x)에서
```

### 1.4 왜 구현 부담이 작은가

- `build_rl_rows`(lora_rl.py:53–77)가 **이미 anchor_token_ids를 RL 시작 상태로 주입**한다. z⁺ prefix를 `AnchorRecord` 형태로 공급하면 lora_rl.py는 무수정. step 경계에서 자르면 텍스트 재렌더링 seam도 안전(seam 카운터가 설계된 케이스가 정확히 `"\n\n"` 종결 anchor).
- z⁺ 토큰 id는 `improve/adapters/pairs_*.jsonl`의 `input_ids[prompt_len:]`로 지금도 복구 가능. 깔끔한 수정은 `bridges.jsonl`에 `token_ids` 필드 추가.
- prefix 프로브는 `_probe_correct_counts`(lora_sft.py:377) + (qid, α) per-request 스윕 패턴(lora_sft.py:546–582) 재사용.
- 주의 1: bridge-sourced anchor가 `build_dataset._build_dpo_pairs`의 `(qid, base_sample_idx)` join을 오염시키지 않게 가드.
- 주의 2: RL 행에 열 추가 시 `rl_key` 캐시 키에도 반영.

### 1.5 예측과 판정 기준

- 예측: p̂∈[0.2, 0.8] 밴드에서는 zero-advantage 그룹 확률이 K=8 기준 0.8~17% (p=0.5면 0.8%). **76.5% → 한 자릿수 %**로 급감, 실효 스텝이 거의 전부가 된다.
- 판정 지표: ① zero-advantage 비율 ② 문제별 annealing 곡선·ℓ=0 졸업률 ③ RL 후 어댑터의 rescue rate(bare x, 16샘플) vs BRIDGE-only — 같은 fit adapter·`--reuse-rollout`로 paired McNemar.

### 1.6 novelty 포지셔닝 — 2026-08-15 수정: 선행연구 정밀 대조 후 재작성

**경고: 메커니즘 자체는 이미 출판되어 있다.** 아래 두 논문이 §1.1–1.3의 뼈대를 거의 그대로 커버한다 (본문 확인 완료):

| 논문 | 메커니즘 | 우리 제안과의 겹침 |
|---|---|---|
| **BREAD** (arXiv 2506.17211) | GRPO 변형. 그룹 전멸 시 Episode Anchor Search: expert trace를 ~10 episode로 나눠 **이진 탐색**으로 "새 그룹 성공률이 목표 범위에 드는 최단 힌트"를 찾아 질문에 붙임. self-paced curriculum. pass@3=0 hard subset(500문제) 실험 포함 | ℓ\* = min{ℓ : p̂ ≥ τ_lo} 밴드 탐색, 전부-실패 트리거, 자동 curriculum — §1.3과 사실상 동일 |
| **Prefix-RFT** (arXiv 2507.01679, ICML'26) | GRPO 그룹 N개 중 1개를 demo-prefix + 정책-continuation hybrid로 교체. prefix 길이 l ~ U(low, high), low를 cosine-decay로 0 근처까지 감쇠. prefix 토큰은 PPO-clip 가중 + entropy top-20%만 업데이트 | annealing 스케줄 + 그룹 혼합(§4의 방법 B) — 거의 동일. demo는 OpenR1-Math의 R1-생성 CoT |

따라서 BPC-RL을 **새 메커니즘으로 주장하면 안 된다.** 두 논문은 인용 + baseline이다.

**남는 기여 — prefix의 출처(source) 질문:**

1. 두 논문 모두 prefix가 **expert 궤적**이다 (BREAD: ground-truth/대형모델 trace, Prefix-RFT: DeepSeek-R1 CoT). **expert vs self-generated prefix ablation은 어느 쪽에도 없다.** 우리의 LSPO-vs-BRIDGE 결과(p=0.0034)가 정확히 이 축의 증거다.
2. **BREAD가 우리 가설의 방증을 스스로 남겼다**: 그들의 "GRPO w/ Expert Trace" baseline(전체 expert trace를 그룹에 주입)이 BREAD보다 나빴고, 원인을 "student-expert 궤적의 distribution gap"으로 추정한다(§4.1). 그들은 격차를 관찰하고 힌트를 짧게 해서 우회했다; 우리는 **출처를 자기 분포(verifier-통과 z⁺)로 바꿔 원인을 제거**한다.
3. **전제 조건 격차**: 두 논문은 학습 가능한 expert CoT의 존재를 가정한다. 우리 세팅은 terse gold y\*만 있고 expert CoT가 없다 — bridge 단계가 y\*를 자기-스타일 궤적으로 **합성**한다. teacher가 없는 도메인(Lean 등)과 EI 자기개선 폐루프(train→eval)는 두 논문 밖이다.

**수정된 한 줄 스토리**: *"prefix-RL 메커니즘(BREAD/Prefix-RFT)은 주어진 것으로 두고, 그들이 열어둔 질문에 답한다 — prefix는 어디서 와야 하는가: expert CoT가 없는 cliff 세팅에서 self-generated privileged bridge prefix가 gold-prefix를 이긴다."*

**필수 실험 (재포지셔닝의 성립 조건)**: gold-y\*-prefix arm(BREAD식 EAS 재현) vs bridge-prefix arm, 같은 예산·같은 fit adapter·`--reuse-rollout`, paired McNemar. 이 예측이 틀리면(gold prefix도 동등하면) 방법 기여는 접고 C(y) 선택·EI 폐루프 쪽으로 기여의 무게를 옮긴다.

---

## 2. Reference-언급 문제 — 괜찮은가?

### 2.1 기존 실험에서 왜 언급이 사라졌나

bridge 생성 시 G5 스크린(regex rules·LLM judge)은 **둘 다 OFF**였고(BRIDGE.yaml 확인), z⁺는 정답 verifier만 통과했다. 그런데도 —

- bridge 궤적의 45.7%가 "reference solution"류를 언급했지만,
- 그 z⁺로 LoRA를 **가중치로만** 학습한 뒤 bare x에서 샘플링한 최종 후보의 언급률은 1.4% = CONTROL 바닥과 동일.

메커니즘: 언급은 privileged 프롬프트("여기 참고 풀이가 있다…")에 **조건화된** 행동이다. 어댑터가 (x → z⁺)를 학습해도, 추론 시 문맥에 그 프롬프트가 없으면 언급을 재현할 유인이 약하다. 오염이 가중치를 통과하며 세탁된 것.

### 2.2 BPC-RL은 무엇이 다른가 — 위험이 재도입된다

BPC-RL은 z⁺ **텍스트를 문맥에 그대로 되돌려 넣는다**. prefix에 "the reference solution says…"가 있으면:

1. **completion 오염**: 이어 쓰는 rollout이 언급을 메아리칠 수 있고, 그 completion이 강화될 수 있다.
2. **문맥 의존 학습**: 언급이 든 문맥에서만 잘 푸는 정책이 될 위험 — annealing이 닫아야 할 간극이 커진다.

즉 "가중치 세탁"에 기대던 안전장치가 prefix 경로에는 없다. → **prefix 소스를 스크리닝해야 한다.** 다행히 비용이 거의 없다는 것을 실측했다.

### 2.3 실측 (default_BRIDGE_20260813_202912, kept z⁺ 313개, G5 regex 패턴 적용)

| 측정 | 값 |
|---|---|
| kept z⁺ 중 언급 있음 | 154/313 (49.2%) |
| z⁺ 보유 86문제 중 **언급-없는 z⁺가 1개 이상** | **59** |
| 전부 언급하는 문제 | 27 — 그중 22개는 첫 언급이 전체의 25% 이후 (truncation으로 prefix 사용 가능) |

### 2.4 완화책 — prefix 소스 우선순위 (커버리지 실측 포함)

| 순위 | 소스 | 언급 위험 | 커버리지 기여 |
|---|---|---|---|
| ① | **verified 후보** (bare x에서 어댑터로 샘플·verifier 통과, 세 arm union) | 1.4% (이미 세탁됨) | 62문제 |
| ② | **언급-없는 z⁺** (기존 G5 regex를 prefix 선택 시에만 적용) | regex 통과분 | +α → ①∪② = **83문제** |
| ③ | **첫 언급 직전 step 경계에서 자른 z⁺** | prefix 내 언급 0 | +9 → **92/107** |
| ④ | privileged bridge 재시도 (n↑, retry_temperature 스윕; 이번엔 leakage_rules ON으로 생성) | regex 통과분 | 남은 15문제 일부 |
| ⑤ | 그래도 없으면 RL에서 제외 | — | 어차피 gradient 0이던 문제 |

스크리닝 비용: 무스크리닝 커버리지 93 → 스크리닝 후 92. **1문제 차이.** 안 할 이유가 없다.

이중 방어선(기존 인프라 그대로): 최종 후보는 어차피 G5 leakage 게이트·C(y) 선택을 다시 통과한 뒤에만 학습 데이터가 되므로, prefix 단계에서 새는 것이 있어도 학습 텍스트까지는 못 간다. 원하면 3중으로 RL reward에 regex 페널티(언급 completion → reward 0)도 한 줄로 추가 가능.

남는 잔여 위험(정직하게): 언급이 없어도 z⁺는 y\*에서 **유래한 내용**을 담는다 — 그게 rescue의 작동 원리 자체다. 지키는 불변식은 "y\* 원문·언급이 프롬프트/학습 텍스트에 나타나지 않는다"이고, prefix는 손실이 안 걸리는 컨텍스트이며 ℓ→0으로 소멸한다. gold y\* 폴백(§3 옵션 4)을 쓰지 않는 한 이 불변식은 유지된다.

---

## 3. z⁺ 없는 문제 처리 — 옵션 정리

§2.4의 우선순위 사슬이 곧 답이다. 요약:

- **옵션 1 (기본 채택)**: 세 arm verified 성공 union을 prefix 소스에 추가 — 비용 0, leakage 무결, 커버리지 62문제 확보.
- **옵션 2**: 언급-스크리닝된 z⁺(무결 59 + truncation 22) — 합계 92/107.
- **옵션 3**: 남은 15문제는 bridge 재시도(leakage_rules ON) 후, 그래도 없으면 제외. 제외해도 잃는 것은 "curriculum이 뚫었을 가능성"뿐(현행 RL에서도 gradient 0).
- **옵션 4 (비권장)**: gold y\* 텍스트 prefix 폴백. 커버리지 100%이지만 LSPO 결과상 효과가 의문이고 y\*-프롬프트 불변식에 예외를 만든다. 채택하지 않는다.

---

## 4. 방법 B — 그룹 주입 objective (off-policy guided GRPO, LUFFY 계열)

### 4.1 무엇인가

GRPO 그룹 K개 전부를 on-policy로 뽑는 대신 **1개를 verified 성공 궤적으로 교체**한다. 그룹 안에 reward=1 멤버가 항상 있으므로 **zero-advantage 그룹이 구조적으로 없다.**

- 주입된 성공 멤버: 양의 advantage → 그 궤적의 likelihood를 올리는 imitation-형 gradient (그룹 baseline이 걸린 self-imitation).
- 나머지 K−1개 on-policy 실패: 음의 advantage → 실패 모드 억제.
- off-policy 멤버의 importance ratio π_θ/q는 q(privileged 생성 분포)를 모르므로 정확 계산 불가 → LUFFY(2025)식 policy shaping: ratio를 clip/상수화한 NLL-형 가중으로 근사.

**주의 (2026-08-15 추가): 이 방법도 신규가 아니다.** Prefix-RFT가 정확히 이 그룹 혼합을 하고(§1.6 표), LUFFY는 전체 궤적 주입, BREAD는 "GRPO w/ Expert Trace"를 baseline으로 이미 실험해 **expert trace 주입은 adaptive 힌트보다 나쁘다**는 결과까지 냈다(distribution gap이 원인이라 추정). 우리가 시도한다면 유일한 변주는 "주입되는 궤적이 expert가 아니라 자기 분포의 z⁺"라는 점 — 즉 §1.6과 같은 source ablation의 한 축으로만 의미가 있고, 독립 기여로는 못 세운다.

### 4.2 구현 강도 2단계

1. **가벼운 근사 (권장 1차)**: 그룹은 안 건드리고 `GRPO loss + λ·NLL(성공 버퍼)` 보조항. 버퍼는 이미 디스크에 있다(`filtered/kept.jsonl` union + 스크리닝된 z⁺). Self-Imitation Learning(Oh et al. 2018)과 동형이고, 커스텀 loss 선례(`WeightedSFTTrainer.compute_loss`)가 repo에 있다.
2. **정식 그룹 주입**: `GRPOTrainer` subclass로 생성 직후 그룹 텐서에 삽입. pinned trl 1.3 내부 API 의존 — 착수 전 GPU 프로브 필수(api_notes.md 방식).

### 4.3 A와의 관계

직교·상보: **A는 시작 상태(탐색 분포), B는 그룹 구성(credit assignment).** A만으로 zero-advantage가 해소되면 B는 불필요할 수 있다. A 먼저 → A의 zero-advantage가 여전히 높거나 annealing 후반(ℓ→0)에 신호가 다시 마르면 B를 ablation으로 추가.

---

## 5. 기타 objective 후보 (참고, 후순위)

| 후보 | 요지 | 판단 |
|---|---|---|
| pass@K-aware advantage | 진짜 목표가 rescue(=pass@16)인데 GRPO는 pass@1 최적화. 다양성 보상 advantage 변형 | 연구 가치 있으나 support 해결 후에만 의미 |
| dense shaping (privileged divergence 보상) | 이진 보상에 step-level shaped 항 | reward hacking 위험, 비권장 |
| GFlowNet trajectory-balance | sparse binary reward + off-policy에 정합한 amortized posterior | 전면 custom loss — pinned trl에선 위험 大, 장기 옵션 |

## 6. 권장 로드맵 (2026-08-15 수정: source-ablation 중심으로 재편)

1. (기존 계획) BRIDGE + anchor(privileged_divergence) 실험.
2. **Prefix-source ablation이 핵심 실험**: 같은 prefix-RL 기계 위에서 3-arm paired 비교 —
   - (a) no-prefix (현행 RL = 대조군)
   - (b) **gold-y\*-prefix** (BREAD식 EAS 재현 = 선행연구 baseline)
   - (c) **bridge-z⁺-prefix** (우리 제안)
   같은 fit adapter, 같은 예산, `--reuse-rollout`, McNemar. 예측: (c) > (b) — LSPO-vs-BRIDGE의 prefix 버전. 지표: zero-advantage 비율, ℓ=0 졸업률, rescue rate.
3. (b)≈(c)로 나오면 방법 기여는 접고, C(y) 학습가능성 선택·EI 폐루프(rescue→train→eval)로 기여의 무게 이동. (c)가 이기면 BREAD/Prefix-RFT 대비 "prefix source가 관건"이라는 주장으로 집필.
4. 그룹 주입(B)은 독립 기여가 아니라 ablation으로만 (§4.1 주의 참조).
5. 최종 판정은 EI 외부 루프 train→eval.
