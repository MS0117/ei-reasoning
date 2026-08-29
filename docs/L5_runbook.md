# L5 실행 안내서 (2026-08-29)

L5 본안을 다른 서버에서 실행하기 위한 절차. 설계 근거는
`docs/L5_protocol_20260829.md`, 결과 계보는 `docs/L3_results_20260826.md`.

**중요**: 5개 arm은 **같은 코드 리비전**으로 돌려야 한다. vLLM 풀은 배치 구성이 커널
수치를 바꾸므로 코드가 다르면 표본이 달라지고 arm 간 비교가 깨진다(프로토콜 §2-1).
중간에 `git pull` 하지 말 것.

---

## 0. 시작 전 확인

```bash
cd <repo>
bash scripts/setup.sh --skip-lean          # 최초 1회 (uv 기반, sudo 불필요)
.venv/bin/python -m pytest -q -m "not slow"   # 483개 통과해야 정상
.venv/bin/python scripts/check_env.py         # GPU/vLLM/flash-attn 확인
```

- Python은 3.11 고정 (flash-attn 휠이 cp311 전용). 항상 `.venv/bin/python`을 쓰고
  시스템 python을 쓰지 말 것.
- 필요한 데이터 파일이 repo에 이미 있는지:
  ```bash
  ls -la data/mixes/openr1_mix6k_qwen3-4b-2507.jsonl \
         data/mixes/openr1_mix8k_qwen3-4b-2507_cliff_holdout.jsonl
  ```
  두 파일이 5개 arm 전부의 학습셋/holdout이다. 없으면 알려줄 것.
- **GPU 지정**: `-g 0,1` 처럼 넘기거나, 스케줄러(srun 등)가 이미 GPU를 제한하고
  있으면 `-g`를 생략한다. 생략 시 보이는 GPU를 전부 쓴다.
- **wandb는 없어도 된다.** API 키가 없으면 자동으로 offline로 내려가고
  (`runs/<name>/wandb/`에 로컬 저장), 논문 숫자는 전부 파일로 남는다.

---

## 1. 훈련 — 5개 arm

GPU를 공유하므로 **한 번에 하나씩** 돌린다. 각 명령은 새 타임스탬프 run 디렉터리를
만들며(덮어쓰기 없음), `-b`는 백그라운드 실행이다.

```bash
# 1) Ours — staged bridge operator + cliff objective S3
bash scripts/run.sh -c configs/methods/l5_staged_dpo_s3.yaml -b

# 2) LSPO — gold y*로 transient LoRA fit (operator 대조)
bash scripts/run.sh -c configs/methods/l5_lspo.yaml -b

# 3) Gold-in-loop — cliff 행을 gold 원문으로 (데이터원 대조)
bash scripts/run.sh -c configs/methods/l5_gold_inloop.yaml -b

# 4) RFT / ReST-EM — solved만 학습 (표준 baseline)
bash scripts/run.sh -c configs/methods/l5_rft.yaml -b

# 5) Gold SFT — offline 전량 y* (rollout 없음, 전용 런처)
bash scripts/l5_gold_sft.sh -b
```

**오버라이드를 붙이지 말 것.** 데이터·하이퍼파라미터가 전부 yaml에 고정되어 있고,
arm 간 정렬이 그것에 의존한다.

arm 간 의도적으로 다른 것은 셋뿐이다:

| | 값 | 왜 |
|---|---|---|
| `loop.iterations` | 1~4번 arm은 3, Gold SFT는 1 | offline SFT는 반복이 무의미 |
| cliff objective | Ours/LSPO는 S3, 나머지는 현행 loss | baseline은 표준 SFT로 읽는다 |
| `train.sft.epochs` | 1~4번 arm은 2, Gold SFT는 6 | 정적 데이터라 어느 값도 완전 정합이 안 됨. `l5_gold_sft.yaml` 헤더 참조 |

Gold SFT는 rollout이 없어 ~3시간이므로, epochs 2 변형도 같이 돌려 둘 중 강한 쪽을
baseline 행으로 쓰는 것을 권한다:
```bash
bash scripts/l5_gold_sft.sh -b -- --override train.sft.epochs=2
```

### run 디렉터리 이름

각 명령은 `runs/<yaml의 run.name>_<타임스탬프>/`를 만든다. 예:

| arm | yaml | 만들어지는 디렉터리 (예) | 최종 iteration |
|---|---|---|---|
| Ours | `l5_staged_dpo_s3.yaml` | `runs/l5_staged_dpo_s3_20260901_093000/` | `iter_2` |
| LSPO | `l5_lspo.yaml` | `runs/l5_lspo_20260903_141500/` | `iter_2` |
| Gold-in-loop | `l5_gold_inloop.yaml` | `runs/l5_gold_inloop_20260906_082000/` | `iter_2` |
| RFT | `l5_rft.yaml` | `runs/l5_rft_20260908_110000/` | `iter_2` |
| Gold SFT | `l5_gold_sft.yaml` | `runs/l5_gold_sft_20260909_170000/` | **`iter_0`** |

타임스탬프는 실행할 때 정해지므로, 아래 §2에서는 **실제 만들어진 이름으로 바꿔 쓰거나**
쉘 glob을 쓴다(arm당 run이 하나뿐일 때만 안전):

```bash
ls -d runs/l5_*_2026*        # 만들어진 run 이름 확인
ARM=$(ls -d runs/l5_rft_2026* | tail -1)    # 변수에 담아 쓰면 오타가 없다
echo $ARM
```

진행 확인:
```bash
tail -f logs/loop_<run 이름>.log
```

**중간에 죽으면 같은 이름으로 재개**한다. stage/shard/행 단위로 이어서 돈다:
```bash
bash scripts/run.sh -c configs/methods/<같은 yaml> -r <run 이름>
```

### 대략의 소요

| stage | 500문항 실측 | 6k 환산(문항 비례) |
|---|---|---|
| rollout | 4,342s | ~15h |
| partition | 82s | ~17분 |
| anchor | 56s | ~11분 |
| improve (operator arm) | 3,251~14,303s | 수 시간 ~ 하루 |
| train | 875~2,966s | 수 시간 |
| eval (holdout 300×4) | 1,175s | ~1h |

2×A100 기준 무거운 arm 하나가 3 iteration에 **2~3일**, RFT는 ~1.5일,
Gold SFT는 ~3시간. 더 빠른 GPU면 비례해서 줄어든다.

---

## 2. 훈련 후 — arm마다 두 가지

**훈련이 완전히 끝난(`[loop] done`) arm에만** 실행한다. 자동이 아니다.

최종 iteration은 arm 1~4가 `iter_2`, Gold SFT는 `iter_0`이다.

### 2-1. headline cliff 지표 (논문의 주 지표)

```bash
ARM=runs/l5_staged_dpo_s3_20260901_093000     # <- 실제 이름으로
LAST=iter_2                                    # Gold SFT만 iter_0

.venv/bin/python scripts/cliff_reroll.py \
    --run-dir $ARM \
    --qids-file holdout --n 32 --passes 1 \
    --model-path $ARM/$LAST/ckpt \
    --out $ARM/headline
```

5개 arm 전부:
```bash
for ARM in runs/l5_staged_dpo_s3_* runs/l5_lspo_* runs/l5_gold_inloop_* runs/l5_rft_*; do
  .venv/bin/python scripts/cliff_reroll.py --run-dir "$ARM" \
      --qids-file holdout --n 32 --passes 1 \
      --model-path "$ARM/iter_2/ckpt" --out "$ARM/headline"
done
ARM=$(ls -d runs/l5_gold_sft_* | tail -1)      # Gold SFT는 iter_0
.venv/bin/python scripts/cliff_reroll.py --run-dir "$ARM" \
    --qids-file holdout --n 32 --passes 1 \
    --model-path "$ARM/iter_0/ckpt" --out "$ARM/headline"
```

holdout cliff 300문항(`questions/holdout.jsonl`)을 최종 체크포인트로 32번씩 다시 풀린다.

`cliff_reroll`은 **채점 결과만** 남긴다(`verdicts.jsonl` + `summary.json`). 논문에 쓰는
지표는 `scripts/attractor_mass.py`가 그 verdicts에서 계산한다 — `mean_p_top1`(attractor
mass, 주 지표), `mean_pass_rate`(= avg@32), `frac_pass_gt0`(≥1정답 coverage),
`mean_p_top2`, `n_wrong_kinds`. 문항별 paired 비교와 sign test도 그 스크립트가 한다.

> **반드시 확인**: `--model-path`를 빠뜨리면 에러가 아니라 **학습 전 base 모델로 조용히
> 돈다** — arm의 숫자라고 생각한 게 실은 floor가 된다. 끝나면:
> ```bash
> grep -o '"model_path": "[^"]*"' runs/<arm run>/headline/summary.json
> #   arm  -> runs/<arm run>/iter_2/ckpt  여야 정상
> ```

### 2-2. 벤치마크

```bash
bash scripts/eval_bench.sh runs/l5_staged_dpo_s3_20260901_093000/iter_2/ckpt -b
#                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 실제 이름으로
bash scripts/eval_bench.sh runs/l5_gold_sft_20260909_170000/iter_0/ckpt -b   # Gold SFT만 iter_0
```

aime24/25/26 + hmmt25를 n=64로, math500_hard를 n=8로 채점한다(arm당 ~4시간).
결과는 `runs/bench/<slug>_<timestamp>/`에 별도로 떨어진다.

> **확인**: 끝나면 `runs/bench/<...>/iter_0/benchmark_eval/metrics.json`의
> `model_path`가 넘긴 ckpt를 가리키는지 볼 것. 스크립트가
> `>>> NOTE: MODEL argument wins over the config's model.base` 를 찍어주지만,
> 과거에 엉뚱한 모델을 채점한 사고가 있었던 자리다.

### 2-3. base floor — **전체에서 딱 한 번만**

arm별로 돌리지 말 것. 모든 arm이 같은 floor와 비교한다.

```bash
.venv/bin/python scripts/cliff_reroll.py \
    --run-dir runs/<아무 arm run> \
    --qids-file holdout --n 32 --passes 2 \
    --out runs/floor_holdout
```

arm의 headline과 **완전히 같은 조건**(같은 holdout 300문항, 같은 32샘플, 같은
max_tokens 16384, 같은 verifier)에서 **모델만 학습 전 base**로 돌린 것이다. 그래서
문항별 paired 비교가 성립하고, 5개 arm이 같은 holdout을 쓰므로 한 번이면 전부에 쓰인다.

왜 필요한가: cliff는 "0/8 정답"으로 뽑은 문항이라 **다시 샘플링만 해도 상당수가 뚫린다**
(평균 회귀). L3 실측으로 base가 공짜로 ≥1정답 29%, avg@32 0.020을 준다. floor 없이는
arm의 숫자에서 이 몫을 빼낼 수 없다.

`--passes 2`인 이유: 두 pass는 같은 base 모델의 독립적인 두 표본이라, 그 차이가
**측정 절차 자체의 노이즈**를 재는 null 검정이 된다(L3에서 Δattractor 0.0pp, p=0.78 —
절차가 가짜 신호를 만들지 않는다는 증거).

> 여기서는 `--model-path`를 **일부러 생략**한다(= cfg.model.base). 확인:
> ```bash
> grep -o '"model_path": "[^"]*"' runs/floor_holdout/summary.json
> #   -> "Qwen/Qwen3-4B-Instruct-2507"  여야 정상
> ```

---

## 3. 결과 전송

체크포인트는 arm당 26GB(7.9GB × 3)라 보내지 않는다. 분석에 필요한 파일은 run당 ~5MB뿐이다.

```bash
bash scripts/collect_run_artifacts.sh \
    runs/l5_staged_dpo_s3_* runs/l5_lspo_* runs/l5_gold_inloop_* \
    runs/l5_rft_* runs/l5_gold_sft_* runs/floor_holdout runs/bench/*
# -> l5_artifacts_<timestamp>.tar.gz  (5 arm 합쳐 ~10MB)
```

**§2를 먼저 다 끝낸 뒤에** 실행할 것 — headline/floor/bench 결과가 함께 담긴다.

담기는 것: 동결 config, `metrics.jsonl`, 각 stage의 stats/report JSON,
train 로그(loss·guard 곡선), 모든 `verdicts.jsonl`(샘플별 채점), questions 분할.

체크포인트는 **그쪽 서버에 그대로 두면 된다.** 나중에 추가 분석이 필요하면 그때 요청.

---

## 4. 자주 걸리는 것

| 증상 | 원인 / 대처 |
|---|---|
| `run config mismatch for runs/...` | 같은 run 이름에 다른 오버라이드를 준 것. 오버라이드를 빼거나 새 이름으로. |
| `[wandb] no API key ... falling back to offline` | 정상. 무시해도 된다. |
| `[pool] 0/N results` 가 한참 유지 | 정상. shard 하나가 끝나야 카운트가 오른다. 로그가 안 늘고 GPU가 0%면 그때 확인. |
| GPU OOM | `--override engine.gpu_memory_utilization=0.90` (기본 0.95). 그래도 나면 0.85. |
| flash-attn 로드 실패 | GPU 아키텍처 불일치. `scripts/check_env.py` 결과와 함께 알려줄 것. |
| 학습 중 중단 | 같은 이름으로 `-r` 재개. stage/shard/행 단위로 이어진다. |

**하지 말 것**: 실행 중 `git pull`, arm마다 다른 오버라이드, `runs/` 안의 파일 수동 편집,
같은 GPU에서 두 arm 동시 실행.
