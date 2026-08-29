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

진행 확인:
```bash
tail -f logs/loop_<run.name>_<timestamp>.log
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
.venv/bin/python scripts/cliff_reroll.py \
    --run-dir runs/<arm run> \
    --qids-file holdout --n 32 --passes 1 \
    --model-path runs/<arm run>/iter_2/ckpt \
    --out runs/<arm run>/headline
```

holdout cliff 300문항을 최종 체크포인트로 32번씩 다시 풀린다.

### 2-2. 벤치마크

```bash
bash scripts/eval_bench.sh runs/<arm run>/iter_2/ckpt -b
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

`--passes 2`인 이유: pass_0은 기준선, 두 pass의 차이가 **측정 절차 자체의 노이즈**를
재는 null 검정이 된다(L3에서 Δattractor 0.0pp, p=0.78).

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
