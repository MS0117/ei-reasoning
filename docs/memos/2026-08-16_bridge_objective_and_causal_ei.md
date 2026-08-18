# BRIDGE objective와 causal EI 연구 방향

## 핵심 결론

새 objective 하나만 만드는 것이 최선은 아니다. 가장 좋은 연구 방향은 두 층이다.

1. 현재 BRIDGE fit의 잘못된 weighting을 고치는 inner objective
2. bridge를 “정답 텍스트”가 아니라 “어떤 능력 방향으로 모델을 움직이는 parameter intervention”으로 활용하는 전체 EI 방법

새 loss는 필요하지만, 논문의 주인공은 두 번째 방향이 되어야 한다.

## 1. Objective는 어디에 적용해야 하는가

파이프라인을 세 단계로 분리해야 한다.

```text
privileged bridge 생성
        ↓ bridge 선택
transient LoRA fit
        ↓ bare-x candidate 생성·검증
final EI student 학습
```

| 단계 | 권장 방법 |
|---|---|
| Bridge 생성/선택 | verifier correctness + post-update utility |
| Transient LoRA fit | 새로운 question-balanced bridge objective |
| Final EI train | 우선 기존 SFT 또는 SFT+DPO로 고정 |

새 objective는 우선 transient bridge LoRA를 학습할 때 적용하는 것이 맞다.

- `313 pairs / 86 qids`라는 multi-positive 구조가 존재하는 곳이 이 단계다.
- 현재 token-global CE의 길이·개수 가중 문제가 발생하는 곳도 이 단계다.
- 이 단계의 목적은 cliff에서 bare-x 정답 후보를 만들어내는 것이다.
- Final EI objective까지 동시에 바꾸면 좋은 후보를 만들었기 때문인지 outer trainer가 좋아졌기 때문인지 분리할 수 없다.

Final EI는 먼저 기존 `SFT+DPO`로 고정하는 것이 좋다. 현재 코드에도 이미 지원된다: [train.py](../../src/expert_iter/train.py).

### 1.1 Question-balanced CE baseline

단순 question-balanced CE가 첫 baseline이다.

\[
L_{\mathrm{QBCE}}
=
\frac{1}{|Q_B|}
\sum_q
\frac{1}{|B_q|}
\sum_{z\in B_q}
\frac{1}{|z|}
\sum_t-\log\pi_{\theta+\phi}(z_t\mid x_q,z_{<t})
\]

이렇게 하면 다음이 성립한다.

- 문제마다 동일한 가중치
- 문제당 bridge가 1개든 4개든 동일한 가중치
- 500토큰과 5,000토큰 trajectory도 동일한 가중치

이는 현재 [lora_fit.py](../../src/expert_iter/lora_fit.py)의 global token mean과 다르다.

### 1.2 Question-balanced one-of-many bridge preference

그보다 한 단계 더 나간 objective는 모든 bridge를 외우는 것이 아니라 정답 bridge 중 하나의 확률을 실패 mode보다 올리는 것이다.

\[
s_\phi(r)=
\frac1{|r|}
\sum_t
\log
\frac{\pi_{\theta+\phi}(r_t\mid x,r_{<t})}
     {\pi_\theta(r_t\mid x,r_{<t})}
\]

\[
L_{\mathrm{set}}
=
\frac1{|Q_B|}
\sum_q
\operatorname{softplus}
\left(
m-\operatorname{LME}_{z\in B_q}s_\phi(z)
+\operatorname{LME}_{f\in F_q}s_\phi(f)
\right)
+\lambda L_{\mathrm{QBCE}}+\beta KL
\]

여기서:

- \(B_q\): verified bridges
- \(F_q\): 기존 동일 문제의 실패 rollout
- LME: 개수로 정규화한 log-mean-exp

이 objective는 “한 문제에서 하나라도 성공”인 rescue/pass@K와 SFT보다 더 직접적으로 정렬된다.

더 temporal하게 만들려면 adapter 크기 \(\alpha\phi\), \(\alpha\in\{0.25,0.5,1\}\)에서도 이 loss를 계산한다. 그러면 최종 강한 adapter에서만 정답이 되는 것이 아니라 작은 parameter 이동부터 success direction이 나타나도록 학습한다.

다만 이것은 좋은 objective이지만 단독 novelty는 약하다. [HDPO](https://arxiv.org/abs/2603.23871)가 이미 동일한 `cliff prompts` 정의와 privileged self-distillation을 사용하고, [RSTG](https://arxiv.org/abs/2608.00782)도 all-wrong group에만 dense teacher signal을 적용한다. 따라서 “cliff에 새로운 KD/DPO loss”만으로는 부족하다.

## 2. First-Hit Weight-Space Bridge

가장 먼저 시험할 전체 방법은 First-Hit Weight-Space Bridge다.

현재는 3-step SFT가 끝난 최종 bridge adapter 하나에서만 16개를 생성한다. 대신 실제 LoRA 학습 경로를 활용한다.

```text
base θ
  → step-1 adapter
  → step-2 adapter
  → step-3 adapter
```

각 checkpoint에서 bare-x로 후보를 생성하고, 문제별로 처음 정답이 나타난 checkpoint의 trajectory를 EI 데이터로 채택한다.

핵심 가설은 다음과 같다.

> 최종적으로 강하게 bridge-fit된 모델의 정답보다, base에서 가장 조금 움직였을 때 처음 나타난 정답이 원래 학생에게 더 가까우며 downstream에서 배우기 쉽다.

이는 현재의 \(C(y)\) 문제와 직접 연결된다. 최종 BRIDGE 출력은 많이 구제하지만 학생에게 낯설다. First-hit solution은 rescue를 유지하면서 C를 낮출 가능성이 있다.

필수 통제는 다음과 같다.

- 전체 생성 budget은 여전히 16
- checkpoint마다 16개가 아니라 총 16개를 분배
- 정답을 발견한 seed와 확인 seed를 분리
- endpoint-only와 first-hit을 동일 budget으로 비교
- 최종 conversion뿐 아니라 실제 train→eval 비교

Checkpoint sampling 자체는 이미 [Temporal Sampling](https://arxiv.org/abs/2505.20196)이 reasoning 능력의 temporal forgetting을 복구하는 데 사용했다. 따라서 novelty는 단순 checkpoint ensemble이 아니라 다음의 결합에 있어야 한다.

- privileged bridge가 만든 transient LoRA path
- 최소 parameter 이동에서의 first-hit
- 그 trajectory를 final EI 학습 데이터로 consolidation

## 3. Post-Update Transfer Graph

가장 강한 flagship 방향은 bridge의 가치를 correctness나 C-score가 아니라 실제 업데이트 이후의 전이 효과로 정의하는 것이다.

Bridge \(z\)로 1–3번 작은 LoRA update를 한다.

\[
\phi'_z=\phi-\eta\nabla_\phi L(x,z)
\]

그 후 같은 문제가 아니라, 분리된 다른 문제들을 bare-x로 평가한다.

\[
U(z)=
\Delta\operatorname{Pass@K}
(\mathcal H_q;\theta+\phi'_z)
-\lambda\operatorname{Forgetting}(\mathcal R)
-\gamma\operatorname{Cost}(z)
\]

- \(\mathcal H_q\): source 문제와 분리된 관련/held-out 문제
- \(\mathcal R\): 일반 능력 보존용 replay set
- 모든 rollout prompt는 x-only
- gold는 verifier에서만 사용

같은 문제에서 평가하면 memorization utility가 되므로 반드시 cross-problem 또는 cross-fold여야 한다.

이를 transfer matrix로 만들 수 있다.

\[
T_{j\rightarrow i}
=
\operatorname{Pass@K}_i(\theta+\phi_j)
-\operatorname{Pass@K}_i(\theta)
\]

그리고 pooled adapter 하나를 모두에게 사용하는 대신 다음을 수행한다.

1. Bridge gradient를 8–16개 skill cluster로 묶는다.
2. Cluster별 동일 budget micro-LoRA를 만든다.
3. 5-fold cross-fit으로 다른 cliff 문제에 대한 \(T_{j\to i}\)를 측정한다.
4. 여러 문제에 양의 전이를 주는 bridge cluster를 선택한다.
5. 각 cliff에 가장 유용한 adapter/cluster를 routing한다.
6. 여기서 생성된 bare-x 정답만 final EI 데이터에 넣는다.

현재도 자기 bridge가 전혀 없던 21문제 중 4문제가 pooled BRIDGE로 구제됐다. 작지만, cross-problem transfer가 실제로 존재한다는 중요한 단서다. 지금처럼 모든 bridge를 무작정 pooling하지 말고 그 전이 구조를 측정하자는 것이다.

이 방법의 핵심 주장은 다음과 같이 잡을 수 있다.

> Correct bridge가 아니라, verified transient update 이후 다른 x-only cliff에 전이되는 bridge가 좋은 teacher이다.

[LARK](https://arxiv.org/abs/2605.30651)와 RSR 계열은 trajectory의 learnability를 선택하고, [RLT](https://arxiv.org/abs/2506.08388)는 student understanding을 teacher reward로 사용한다. 따라서 방어 가능한 차별점은 반드시 다음 세 요소여야 한다.

- 실제 transient LoRA update를 intervention으로 수행
- 동일 문제가 아닌 cross-fitted x-only 문제에서 utility 측정
- 그 transfer graph로 bridge를 선택·routing하고 EI에 consolidation

즉 일반적인 learning-to-teach가 아니라 cliff에서의 causal update utility다.

## 4. RL은 bridge 이후에만 사용

RL을 포기할 필요는 없지만 cliff에 직접 적용하면 안 된다. 전체 EI에서는 문제를 매번 동적으로 routing해야 한다.

\[
\hat p_q=0
\quad\Rightarrow\quad
\text{Bridge/first-hit/transfer}
\]

\[
0<\hat p_q<1
\quad\Rightarrow\quad
\text{GRPO/RLOO}
\]

\[
\hat p_q\approx1
\quad\Rightarrow\quad
\text{skip 또는 replay/KL}
\]

Cliff가 측정값이라는 점을 반영하면 hard `0/8` 대신 Beta posterior로 다음 그룹이 mixed reward를 가질 확률을 계산할 수도 있다.

\[
P_{\rm effective}(q)
=
1-\mathbb E[p_q^K+(1-p_q)^K]
\]

이 값이 높은 문제에만 GRPO를 적용한다. 현재 RL의 zero-advantage 80.4%는 이러한 routing 없이 고정 107문제를 모두 순회한 결과다. Bridge의 역할은 RL 자체를 대체하는 것이 아니라 문제를 `cliff → slope`로 옮겨 RL gradient가 생기게 만드는 것이다.

## 5. 권장 실험 순서

1. 기존 bridge buffer를 고정하고 재생성하지 않는다.
2. 다음 네 fit을 비교한다.
   - BRIDGE-1 + question-balanced CE
   - BRIDGE-4 + question-balanced CE
   - BRIDGE-4 + set preference
   - 현재 BRIDGE-4 token-global CE
3. 3-step checkpoint를 저장해 budget-matched first-hit sampling을 실행한다.
4. 8–16개 bridge gradient cluster만으로 작은 transfer-graph pilot을 한다.
5. 최상위 두 방법의 x-only verified outputs로 동일한 final `SFT+DPO`를 수행한다.
6. 200 holdout 및 외부 benchmark에서 train→eval을 평가한다.

### 판정 기준

- `BRIDGE-4 QB > BRIDGE-1 QB`: multiple verified modes가 BRIDGE의 진짜 장점
- First-hit이 rescue를 유지하면서 C와 downstream 학습을 개선: temporal parameter path가 유효
- Post-update utility가 C, 길이, NLL보다 실제 train→eval을 잘 예측: flagship 가능
- Conversion만 높고 held-out gain이 없음: BRIDGE는 데이터 생성 도구일 뿐 학습 방법은 아님

## 최종 추천

권장 조합은 다음과 같다.

> Question-balanced inner objective + first-hit trajectory harvesting + post-update transfer routing

Anchor는 필요한 ablation이고, vanilla GRPO는 slope 전용 후속 단계다. 가장 중요한 발상의 전환은 bridge를 “학습할 긴 정답”이 아니라 “학생을 어느 방향으로 얼마나 움직여야 정답 support가 생기는지 보여주는 transient parameter probe”로 보는 것이다.
