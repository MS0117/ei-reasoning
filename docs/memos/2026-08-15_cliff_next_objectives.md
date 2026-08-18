## 핵심 판단

1. `x(+anchor)+reference 일부`로 성공 궤적을 만들어 RL 신호를 살리는 방향은 맞습니다. 다만 raw gold reference를 붙여 vanilla GRPO를 돌리는 방식은 권하지 않습니다.
2. 다음 강한 baseline은 **verified model-native bridge prefix + adaptive curriculum + OC-GRPO**입니다.
3. 더 novel한 주력 objective로는 **all-wrong 그룹에만 적용하는 reference-contrastive on-policy distillation**이 가장 잘 맞습니다.
4. RL을 포기한다면 **anchor-local bridge preference/set objective**가 가장 구현 대비 효과가 좋아 보입니다.

제가 산출물을 확인한 시점에는 RL 자체가 107/107까지 끝났고, step 중복을 제거하면 reward 평균 0.0280, zero-advantage 80.4%, 실효 업데이트 21/107이었습니다. 즉 예상대로 알고리즘 문제가 아니라 support 문제입니다. [현재 RL 메타데이터](/shared/minsu/ei-reasoning/runs/toy_cliff/default_BRIDGE_20260815_194535/iter_0/improve/adapters/pooled_c0_rl/d3084ff8a57f102c/rl_meta.json)

먼저 해석을 한 가지 다듬으면, “SFT가 안 된다”기보다는 **gold-target SFT가 안 맞는다**가 정확합니다. BRIDGE도 SFT인데 유의한 효과를 냈으므로, 핵심 변수는 objective 종류보다 target trajectory가 모델의 분포에 얼마나 맞는가입니다. 또한 LSPO의 \(p=.248\)은 “CONTROL과 동등함이 증명됨”이 아니라 “차이를 검출할 증거가 부족함”입니다.

## 1) Reference를 RL rollout에 추가할 것인가

### 권장 형태

Anchor 적용 시 실제 목표 상태를

\[
s_q=x_q+a_q
\]

라 하고, verifier를 통과한 anchored bridge continuation \(z_q^+\)의 step-prefix를 \(h_q\)라 두십시오. Rollout은

\[
y\sim\pi_{\phi_{\rm old}}(\cdot\mid s_q+h_q)
\]

에서 생성하되, 최종적으로 최적화하고 싶은 것은 \(\pi_\phi(\cdot\mid s_q)\)입니다.

Prefix 우선순위는 다음이 좋습니다.

1. `x+a`에서 이미 answer-blind로 생성된 verified candidate prefix
2. 같은 anchor에서 생성된 bridge \(z^+\) 중 reference 언급과 조기 정답 노출이 없는 prefix
3. reference에서 추출한 answer-free 구조적 hint: 목표, 제약조건, 다음 subgoal
4. raw gold prefix는 oracle upper-bound ablation으로만 사용

문제별로 여러 step-boundary 길이를 probe해 최소 prefix를 선택합니다.

\[
\ell_q^*=\min\{\ell:\hat p_q(\ell)\in[0.2,0.8]\}
\]

\(G=8\)일 때 \(p\in[0.2,0.8]\)면 zero-variance 확률은 최대 약 16.8%, \(p=0.5\)에서는 0.8%까지 떨어집니다. 이후 성공률이 높아지면 prefix를 한 단계씩 줄이고, 최종 후보는 반드시 \(h=\varnothing\), 즉 원래 `x+a`에서 뽑습니다.

### Vanilla GRPO가 아니라 OC-GRPO여야 하는 이유

Guided prompt로 생성하고 같은 guided prompt 아래에서 loss를 계산하면 \(\pi(y\mid x,a,h)\)를 학습하는 것이지, 배포 시 필요한 \(\pi(y\mid x,a)\)를 학습하는 것이 아닙니다. Guidance token에 loss를 주지 않는 것만으로도 이 mismatch는 해결되지 않습니다.

[OC-GRPO](https://arxiv.org/abs/2607.19313)는 다음 ratio를 사용합니다.

\[
\rho^{\mathrm{OC}}_t=
\frac{\pi_\phi(y_t\mid s_q,y_{<t})}
     {\pi_{\phi_{\rm old}}(y_t\mid s_q+h_q,y_{<t})}.
\]

즉 guided rollout을 쓰면서 gradient numerator는 unguided 상태에서 계산합니다. 현재 TRL GRPO 경로에는 이 두-context forward가 없으므로 custom trainer가 필요합니다.

이 계열 자체는 이미 [BREAD](https://arxiv.org/abs/2506.17211), PrefixRL, OC-GRPO 등과 매우 가깝습니다. 따라서 **강한 baseline으로는 필수지만, prefix curriculum 단독의 novelty는 낮습니다.** 프로젝트 고유성은 gold prefix 대신 “privileged하게 생성됐지만 model-native인 verified bridge”를 쓴다는 데 있습니다.

### 오염 해석은 조금 더 보수적으로

최종 candidate의 reference 명시 언급률 1.4%는 “표면 문자열 누출이 전파되지 않았다”는 좋은 증거입니다. 그러나 semantic leakage나 backward rationalization까지 없다는 뜻은 아닙니다. 최근 [Answer-Conditioned CoT 연구](https://arxiv.org/abs/2607.14552)는 gold answer를 보고 만든 verifier-correct reasoning trace가 downstream 학습을 오히려 해칠 수 있음을 보였습니다.

따라서 reference mention 외에도 다음을 봐야 합니다.

- 정답이 논증보다 먼저 등장하는 비율
- gold와의 semantic/n-gram overlap
- matched-reference / shuffled-reference / wrong-reference 대조
- 실제 problem-disjoint EI train→eval

## 2) 더 좋은 objective

### A. 가장 novel한 후보: Target-Specific Privilege Residual Distillation

Actor는 끝까지 `x+a`만 보고 rollout합니다. 따라서 correct rollout을 한 번도 만들지 못해도 inference occupancy는 정확히 유지됩니다. 같은 student prefix에서 frozen teacher만 privileged context를 봅니다.

\[
q_t^+(v)=\mu(v\mid x,a,y^*,\hat y_{<t})
\]

\[
q_t^-(v)=\mu(v\mid x,a,y_{\sigma(q)}^*,\hat y_{<t})
\]

여기서 \(y_{\sigma(q)}^*\)는 길이와 수학 도메인을 맞춘 다른 문제의 reference입니다. Bare teacher를 \(q_t^0\)라 두고 target-specific residual을

\[
r_t(v)=\operatorname{clip}
\left(\log q_t^+(v)-\frac1M\sum_m\log q_{t,m}^-(v),-c,c\right)
\]

\[
\tilde q_t(v)\propto q_t^0(v)\exp(\alpha r_t(v))
\]

로 정의한 뒤 student가 \(\tilde q_t\)를 따르도록 JS/KL loss를 줍니다.

핵심은 objective routing입니다.

\[
L=
\begin{cases}
L_{\rm GRPO}, & 0<\sum_iR_i<G\\
\lambda L_{\rm TPRD}, & \sum_iR_i=0\\
0\text{ 또는 KL regularization}, & \sum_iR_i=G
\end{cases}
\]

현재 낭비되는 80.4%의 all-wrong 그룹만 dense privileged signal로 바꾸고, correctness ordering이 존재하는 그룹에서는 verifier GRPO를 그대로 보존합니다.

이 방향은 [OPSD](https://arxiv.org/abs/2601.18734)와 all-wrong 그룹만 선택적으로 증류하는 [RSTG](https://arxiv.org/abs/2608.00782)를 강한 baseline으로 둬야 합니다. 중요한 차별점은 [OP²SD](https://arxiv.org/abs/2608.09228)가 지적한 “다른 문제 reference를 줘도 비슷한 효과가 나는 context confound”를 matched-minus-shuffled residual로 직접 제거한다는 것입니다.

또한 teacher가 student support 밖의 토큰만 밀지 않도록, teacher mass가 student top-\(K\) 안에 충분히 있는 위치에만 loss를 주는 것이 좋습니다. 이는 [Token Teachability](https://arxiv.org/abs/2605.26844)의 결과와도 맞습니다.

집중적으로 검색한 범위에서는 이 정확한 조합은 찾지 못했지만, 신규성 보장은 더 넓은 related-work 검토가 필요합니다.

### B. 가장 실용적인 custom loss: Anchor-local Existential Bridge Optimization

RL 성공을 기다리지 말고 이미 있는 verified bridge와 실패 suffix를 직접 비교합니다.

- \(B_q\): 동일한 `x+a` 이후의 verified bridge continuation 집합
- \(F_q\): 동일 anchor 이후의 실패 continuation 집합
- \(s_\phi(z)\): 길이 보정된 trajectory log-likelihood

\[
M_q^+=\tau_+\log\sum_{z\in B_q}\exp(s_\phi(z)/\tau_+)
\]

\[
M_q^-=\tau_-\log\sum_{f\in F_q}\exp(s_\phi(f)/\tau_-)
\]

\[
L_{\rm AEBO}=\sum_q
\operatorname{softplus}(m-M_q^++M_q^-)+\beta\,KL(\pi_\phi\|\mu).
\]

작은 \(\tau_+\)는 모든 bridge를 평균적으로 외우는 대신, 현재 정책이 가장 쉽게 접근할 수 있는 correct mode 하나에 질량을 옮깁니다. 이는 “한 개라도 성공”인 rescue/pass@16 목표와 SFT보다 직접적으로 맞습니다.

첫 구현은 더 단순하게 할 수 있습니다.

- chosen: verified \(z^+\)
- rejected: 동일 anchor 이후 실제 실패 suffix
- prompt: `x+a`
- continuation-only DPO

저장소의 downstream DPO pair join도 거의 같은 구조입니다. [build_dataset.py](/shared/minsu/ei-reasoning/src/expert_iter/build_dataset.py:97) 효과가 확인되면 single-pair DPO를 위 set-valued objective로 확장하면 됩니다.

### C. 최종 EI 목적에 맞춘 trajectory 선택

현재 \(C(y)\)를 단순 최소화하면 CONTROL처럼 “이미 익숙하지만 별 정보가 없는” 궤적을 선호할 수 있습니다. 반대로 BRIDGE의 높은 \(C\)는 informative novelty일 수도, 그냥 학습 불가능함일 수도 있습니다.

따라서 correctness를 hard constraint로 두고 다음을 Pareto 기준으로 쓰는 편이 낫습니다.

- student support와의 정렬
- 충분한 surprisal/informativeness
- downstream update utility

최근 [RSR](https://arxiv.org/abs/2601.14249)은 단순 likelihood보다 alignment와 informativeness의 균형이 downstream 성능을 더 잘 예측한다고 보고했습니다.

더 세팅 고유하게는 candidate \(z\)의 EI update gradient와 privileged validation/gold gradient의 정렬을 사용할 수 있습니다.

\[
U(z)\approx
\left\langle
\nabla_\theta L_{\rm privileged},
\nabla_\theta L_z
\right\rangle .
\]

Gold는 gradient scoring에만 사용하고 실제 학습 텍스트에는 넣지 않습니다. 이건 계산량은 크지만 “잘 푸는 trajectory”가 아니라 “학생을 실제로 개선하는 trajectory”를 직접 고른다는 점에서 최종 목표와 가장 잘 맞습니다.

## 권장 실행 순서

1. Anchor 실험은 최소한 다음 2×2로 수행합니다.

   - CONTROL / no-anchor
   - CONTROL / privileged-divergence anchor
   - BRIDGE / no-anchor
   - BRIDGE / privileged-divergence anchor

   Anchor 자체가 gold-informed intervention이므로 anchored BRIDGE만 추가하면 anchor 효과와 bridge 효과가 섞입니다.

2. 동일한 anchored BRIDGE pair와 동일한 pre-RL adapter에서 정확히 분기합니다.

   - BRIDGE-only
   - vanilla GRPO
   - minimal bridge-prefix OC-GRPO
   - selective OPSD 또는 TPRD

3. TPRD 전체 구현 전에 기존 trajectory에서

\[
H(y)=\frac1T\sum_t[\log q_t^+(y_t)-\log q_t^-(y_t)]
\]

가 within-qid correct/incorrect를 구분하는지 AUROC를 측정합니다. 구분력이 없으면 구현하지 않고 AEBO/DPO로 갑니다.

4. Conversion 이후 실제 EI에서는 random, min-\(C\), RSR/gradient-utility selection을 비교합니다.

5. 최종 평가는 problem-disjoint train→eval과 큰 \(K\)의 pass@64 정도를 함께 봅니다. 0/8 cliff는 저확률과 진짜 frontier 밖을 구분하지 못하기 때문입니다.

## 구현 전에 반드시 고칠 점

- RL reward는 현재 completion만 채점하지만, 최종 candidate는 `anchor+completion`을 채점합니다. [lora_rl.py](/shared/minsu/ei-reasoning/src/expert_iter/lora_rl.py:241), [lora_sft.py](/shared/minsu/ei-reasoning/src/expert_iter/lora_sft.py:599)  
  Anchor/prefix RL 전에 verifier 입력을 일치시켜야 합니다. Prefix 안에 final answer가 들어가 trivial reward가 생기지 않도록 answer step 이전에서 잘라야 합니다.

- `x+a+h+completion` 전체 길이를 검사해야 합니다. 현재 검증은 completion 중심이어서 20,480 context에서 16,384 completion을 쓰면 prompt·anchor·hint에 약 4K밖에 남지 않습니다.

- RL은 decoded anchor를 재토큰화하지만 최종 샘플링은 token-ID splice입니다. Prefix가 길어지면 seam mismatch가 커질 수 있으므로 별도 token-ID 경로 또는 최소한 `seam_strict=true`가 필요합니다.

- RL 전용 prefix map을 따로 두어야 합니다. 기존 anchor를 덮으면 privileged prefix가 최종 candidate와 EI dataset에 남습니다.

- 최신 RL run은 기존 BRIDGE-only와 bridge 자체가 달랐습니다(313/86 대 315/87). 다음 비교는 반드시 동일 fit adapter에서 분기해야 합니다.

- DDP 두 rank가 reward curve에 같은 step을 각각 기록하므로 집계 시 `unique_by(step)`가 필요합니다.

한 줄로 정리하면, 가장 좋은 연구 스토리는 이렇습니다.

> Cliff에서 필요한 것은 binary RL의 hyperparameter 조정이 아니라 성공 support의 제조다. Model-native bridge로 최소한의 support를 만들고, actor에는 reference를 노출하지 않은 채 matched-reference 고유 token residual만 all-wrong 상태에 증류한 뒤, 실제 EI student utility로 trajectory를 선택한다.
