# Group-advantage objective (`train.objective: gadv`) — 명세 (2026-09-03)

배경: `docs/L3_results_20260826.md`, 2026-09-01 bench sweep. 모든 L3 arm은 gradient의 절반을
8/8 solved 문항(GRPO가 0을 주는 곳)에, S3는 30%를 rescue에 고정 투여했고, base 자신의 frontier
실패에는 음의 항이 없었다. 이 objective는 GRPO/Reinforce-Rej의 gradient 배치(0/G·G/G 제외,
frontier 중심, 문항별 양·음 상쇄)를 offline EI 학생에 옮기되, GRPO가 가질 수 없는 두 가지를
더한다: (1) rescue를 base 실패 8개와 한 그룹으로 묶는 cliff 그룹, (2) 오답 질량의 attractor 배분(γ).
코드: `src/expert_iter/gadv.py`(그룹·advantage), `build_dataset.py`(gadv 분기), `train.py`
(`GadvCollator`, `GadvTrainer`, θ₀ pre-pass 콜백, `run_gadv`), `config.py`(`GadvCfg`).

## 0. Optimizer window 하나의 손실

$$L=\frac{1}{N_{\text{tok}}}\sum_{y\in\text{window}}\sum_{t}m_{y,t}\cdot\Big[-\min\big(\rho_{y,t}A_y,\ \operatorname{clip}(\rho_{y,t},1-\epsilon_{lo},1+\epsilon_{hi})A_y\big)\Big]
\;+\;w_G\cdot\frac{\sum_{y\in\text{rescue},\,\text{ref}_y\ge0}\operatorname{relu}(\overline{\mathrm{ce}}_y-\text{ref}_y)/n_q}{D_G}$$

- $m_{y,t}$ = completion 토큰 × region weight(기본 1), $N_{\text{tok}}=\sum m$ 은 **전역 window**
  (모든 rank·모든 accumulation micro-batch)에서 한 번의 collective로 모은다 → micro-batch/accum/DP
  topology 불변(`tests/test_loss_invariance.py::test_gadv_accum_topology_invariance`).
- $\rho_{y,t}=\pi_\theta(y_t\mid x,y_{<t})/\pi_{\theta_0}(y_t\mid x,y_{<t})$, θ₀ = trainer가 로드한
  초기 가중치. `clip.enabled=false`면 $-A_y\log\pi_\theta$.
- $\rho=1$에서 gradient는 $-A\nabla\log\pi$: $A>0$이면 SFT, $A<0$이면 NSR 형태(확신 토큰은 $(1-\pi)$로
  감쇠). unlikelihood($\pi/(1-\pi)$ 계수)가 아니다 — S4-v1의 diffusion·길이 폭증 원인이 그 형태였다.
- clip: $A>0$ 토큰은 $\rho>1+\epsilon_{hi}$에서, $A<0$ 토큰은 $\rho<1-\epsilon_{lo}$에서 gradient 0
  (PPO의 pessimistic bound). 손실 **값**은 $\rho=1$에서 토큰당 $-A$이고 $-A\log\pi$가 아니다(gradient만 같다).
- guard($w_G$ = `guard_weight`): rescue 행의 평균 completion CE가 C(y) pass의 `s_mean`을 넘지 않게 하는
  hinge(기존 cliff guard와 동일 형태·정규화, $D_G=\sum 1/n_q$ 전역 gather). 같은 문항의 실패 8개가 rescue와
  산문을 공유하므로 rescue가 밀려 내려가는 displacement를 막는다.
- world_size 배수: legacy 경로와 같은 이유(DP all-reduce 평균 보정).

## 1. 그룹 구성 (`gadv.build_gadv_examples`, rollout n=8 기준)

| 문항 | 그룹 | 행 |
|---|---|---|
| k=8 | 없음 (`solved_floor>0`이면 ≤`solved_floor_max_per_question`개 정답 행을 A=floor로) | – |
| 1≤k≤7 frontier | base rollout 8개 | clean-correct 전부(`correct_max_per_question`, 기본 8) `source=solved`, 나머지 `source=wrong` |
| k=0, kept rescue R≥1 | 실패 8개 + rescue R개 | rescue `source=improved`(guard ref 조인), 실패 `source=wrong` |
| k=0, rescue 없음 | 없음 | – |

- clean-correct = verifier correct **AND** `finish_reason=="stop"` (partition과 같은 정의). truncated-correct는
  오답이자 답 없음 bucket.
- 행은 `solved.jsonl`(shortest-k)이 아니라 `rollout/rollouts.jsonl`을 두 번 스트리밍해 만든다
  (1차: 소속만, 2차: 선택된 행의 token id). `partition.solved_selection/solved_keep_max`는 무시된다.
- truncated 실패(`finish_reason!="stop"`)는 `wrong_truncated_max_per_question`으로 문항당 cap된 뒤
  그룹이 짜인다(seed stream 별도, 기본 8 = 무캡). n과 문항의 음의 총량은 그대로고 남은 행에 재배분된다.
  근거(InT freeze, B 제외): 1,055행이 토큰의 23%(전부 16,384)인데 음의 질량은 12%(None bucket 싱글턴).
  `configs/methods/arms/budget_gadv.yaml` = 1 epoch × wrong≤4 × truncated≤1 = 54.3M 토큰, 275 step(S1 282).
- `source=wrong` 행에는 EOS를 **붙이지 않는다**(vLLM이 stop에서 이미 넣은 stop 토큰은 남는다;
  `wrong_drop_terminal_eos=true`가 ablation). 정답·rescue 행은 `ensure_eos`.
- `data.exclude_train_qids`(B)는 그 문항의 모든 행을 제거한다. `train.gadv.accumulate`(기본 false)가
  SFT 행의 iteration 누적을 정하고, `data.accumulate`는 gadv 아래서 `train_dpo.jsonl`만 관장한다.

## 2. Advantage (`gadv.group_advantages`, 순수 함수)

$p_q=n_q^+/G_q$ ($G_q$=8 frontier, $8+R$ rescue). 정답 멤버 $A_i=(1-p_q)\,c_i$ (base 1, rescue
$c$=`rescue_dose`). 양의 총량 $M_q=\sum A_i$ (cap이 걸리면 **학습되는 행의 합**). 오답 멤버 $j$:

$$f_j=\frac{\#\{j': a_{j'}=a_j\}}{n_q^-}\ (\text{None/truncated는 }1/n_q^-),\qquad
A_j=-\lambda\,M_q\,\frac{f_j^{\gamma}}{\sum_{j'}f_{j'}^{\gamma}}$$

λ=`neg_scale`(기본 1 → 문항 zero-sum), γ=`gamma`. 항등식(테스트로 고정): $\sum_j A_j=-\lambda M_q$;
γ=0이고 cap이 안 걸리면 $A_j=-p_q$(Dr.GRPO); 오답이 전부 같은 답이면 γ와 무관하게 균등.
답 bucket은 verifier의 `extracted_answer` 문자열 그대로(sympy repr, `_modal_wrong_failures`와 같은 관례).

예시(rescue 1, 실패 8 = X×6, Y×2, γ=1): rescue +0.889, X 각 −0.133, Y 각 −0.044 (합 −0.889).
GRPO 균등이면 −0.111씩. S3는 rescue에 문항당 solved의 7.6배를 실었고 실패에는 0을 줬다.

## 3. θ₀ pre-pass (`train.make_gadv_prepass_callback`)

`TrainerCallback.on_train_begin` — accelerate/DeepSpeed prepare **후**, 첫 optimizer step **전**
(verl의 recompute old_log_probs). `trainer.model_wrapped`(zero2: 복제 파라미터 엔진)로 no-grad
forward, rank별 strided 행 분담 → `all_gather_object`로 교환 → `trainer._old_logp[row_idx]`
(completion 토큰만, `cache_dtype`, CPU). fp32 log-softmax는 2048 위치 chunk로 계산(16k×152k fp32
복사 회피). k=0에서 rollout 정책=θ₀이므로 step 0에 ρ≡1. 비용: L2 세트(≈33M completion 토큰)에서
≈130 MB, 2×A100 ≈ 15분. clip 비활성이면 생략.

**왜 batch 채널이 아니라 `row_idx` 조회인가:** `Trainer._prepare_input`이 float 입력을 전부 모델 dtype
(bf16)으로 캐스팅한다(transformers 5.7). advantage·θ₀ log-prob은 trainer 쪽 fp32 테이블에서
`row_idx`(Dataset 위치 열, `run_gadv`가 검증)로 읽는다.

## 4. 감시 지표 (wandb/log)

`loss/pos`, `loss/neg`, `loss/guard`, `rows/pos|neg|rescue`, `guard/skipped_ref`,
`clip/frac_pos`(A>0 & ρ>1+ε_hi 토큰 질량 비율), `clip/frac_neg`, `ratio/mean|max`(rank-local),
`gadv/n_tok|pos_mass|neg_mass`(직전 window의 전역 값). 판독: epoch 2에서 `clip/frac_neg`가 과반이면
음의 항이 꺼진 것 — `eps_hi`(clip-higher), `epochs: 1`, 또는 epoch 경계 re-pass(콜백에 `on_epoch_begin`
추가)로 대응. dataset 쪽은 `iter_k/dataset/stats.json["gadv"]`(버킷별 문항·행 수, advantage 통계,
`zero_sum_max_abs_residual`, truncated/None 수, cap 발동 수).

## 5. 설정 블록

```yaml
train:
  objective: gadv
  sft: {cliff: {enabled: false}}   # 상호 배타 (validate가 거부)
  gadv:
    gamma: 1.0                # 0 = GRPO 균등, 3 = attractor 집중
    rescue_dose: 1.0
    neg_scale: 1.0            # 1 = zero-sum
    solved_floor: 0.0
    solved_floor_max_per_question: 1
    correct_max_per_question: 8
    wrong_max_per_question: 8
    wrong_truncated_max_per_question: 8   # 16k truncated 실패의 문항당 cap (0 = 제거); budget arm은 1
    clip: {enabled: true, eps_lo: 0.2, eps_hi: 0.2}
    guard_weight: 1.0         # >0 이면 filter.selection.always_score 또는 c_score 필요
    accumulate: false
    wrong_drop_terminal_eos: false
    prepass_batch_size: 1
    cache_dtype: float32
```
`train.sft.{lr, epochs, global/micro batch, region_weights, ...}`는 그대로 쓴다.

## 6. 실행

```bash
# L3 arm (frozen L2 fork + B readout) — 프리셋 overlay
PRESET=configs/methods/arms/gadv.yaml bash scripts/l3_arm.sh P runs/L2_freeze_20260825_040504 0,1
PRESET=configs/methods/arms/s3.yaml   bash scripts/l3_arm.sh P runs/L2_freeze_20260825_040504 0,1  # == S3, rho를 YAML에서
PRESET=configs/methods/arms/budget_gadv.yaml bash scripts/l3_arm.sh P <FROZEN> 0,1   # S1/S3 예산(1 epoch, wrong≤4, trunc≤1)
# ablation: overlay 뒤에 --override가 붙는다
PRESET=configs/methods/arms/gadv.yaml ARM_TAG=g0 bash scripts/l3_arm.sh P <FROZEN> 0,1 --override train.gadv.gamma=0
# L5 loop
bash scripts/run.sh -c configs/methods/l5_gadv.yaml -b        # init_from: last (헤더 참조)
# smoke (0.6B, 1 GPU): DeepSpeed 없는 single backend에서 pre-pass + clip 경로 확인
bash scripts/smoke.sh <GPU> configs/methods/smoke_gadv.yaml
```
overlay = `scripts/fork_run.py --overlay`: sparse YAML을 frozen snapshot에 deep-merge(중첩 dict 병합,
리스트/스칼라 대체), 그 뒤 `--override`. 알 수 없는 키는 여전히 하드 에러.

## 7. 위험·미결

1. DeepSpeed 엔진의 no-grad forward + `all_gather_object`(~130 MB)는 코드 읽기로만 검증됨 — smoke가 확인 지점.
2. `row_idx == Dataset 위치` 가정: `run_gadv`가 한 번, `compute_loss`가 배치마다(캐시 길이 ≠ completion 길이) 검사.
3. drift 규모: 논문(RAFT++/W-REINFORCE)은 buffer당 ≤4 step, lr 1e-6; 여기는 2 epoch ≈343 step, lr 1e-5.
   ε=0.2 clip이 epoch 2에서 대부분 걸릴 수 있다(§4 대응).
4. `init_from: base`인 k≥1에서는 rollout 정책(ckpt_{k−1})≠θ₀(base)라 step 0에 ρ≠1 — L5 프리셋은
   `init_from: last`. 비교 arm의 init도 같이 바뀌므로 귀속 시 주의.
5. `wrong` 행의 stop 토큰이 음의 advantage 아래 놓인다(NSR 감쇠로 S4-v1보다 훨씬 약함). 길이 감시,
   `wrong_drop_terminal_eos` ablation.
6. 답 bucket은 exact string: 형식만 다른 동치 오답은 다른 bucket. None/truncated는 singleton.
7. `train.gadv` 필드 추가로 기존 run의 config hash가 전부 바뀐다. fork는 자동 restamp; in-place 재개는
   `scripts/restamp_config_hash.py --apply`.
8. 미검증(GPU): 전부. CPU에서 검증된 것: advantage 항등식, 그룹 구성, 손실=참조(ρ=1)·clip 밖 gradient 0·
   rank-shard 가법성·topology 불변·pre-pass 정렬, legacy 경로 byte-identical(기존 suite 통과).
