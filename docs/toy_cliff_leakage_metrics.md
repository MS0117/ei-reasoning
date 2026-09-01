# Privileged 오염 지표 — 정의·측정법·결과 (toy cliff 250, 2026-09-01)

bridge 연산자는 gold 풀이 y\*를 **보여주는 프롬프트**로 궤적 z\*를 생성한다. 그 z\*로 transient LoRA를
fit하고, 어댑터는 **y\*가 없는 bare x**에서 후보를 뽑는다. 이 문서는 "y\*의 정보가 어디까지 흘러나오는가"를
재는 네 지표를 정의하고 측정 결과를 남긴다.

측정 대상: `runs/toy_cliff_2/` 라운드 1(4스텝·24샘플, 250 cliff)의 5개 arm과 그 bridge 궤적.
전부 **전수**(샘플링 없음), CPU. 재현 스크립트는 §6.

관련 문서: [toy_cliff_lora_fits.md](toy_cliff_lora_fits.md)(실험 전체), [toy_cliff_playbook.md](toy_cliff_playbook.md)(실행법).

---

## 1. 지표 정의

네 지표는 서로 다른 종류의 누출을 잡는다. **말로 언급했는가**(refN/refB)와 **말없이 베꼈는가**
(gold재현/근사복사)는 독립적이며, 실제로 상관이 낮다(§4).

### 1-1. refN — 참조 언급 (narrow regex)

> **"참조 풀이라는 것이 존재한다"고 궤적이 명시적으로 말했는가.**

`solution`/`reference` 같은 **단어 자체가 아니라 구(phrase)** 를 본다. 수학 글에서 "solution"은
일상어이기 때문이다 — bridge 궤적의 **70.6%**에 `solution`이, **56.7%**에 `reference`가 등장하지만
대부분 `no solution`, `the only solution is $c=1$`, `Solution:` 같은 용법이다. 단어 매칭은 쓸 수 없다.

```python
NARROW = re.compile(
    r"reference (solution|answer)|given solution|provided solution|"
    r"the solution (states|says|given)|official solution|"
    r"according to the (solution|reference)|as (given|shown) in the solution", re.I)
```

실제로 잡힌 문자열 분포 (bridge 2,360개에서 매치 횟수):

| 문자열 | 횟수 |
|---|---|
| `reference solution` | 7,008 |
| `reference answer` | 57 |
| `given solution` | 40 |
| `provided solution` | 39 |
| `the solution says` | 36 |
| `according to the reference` | 7 |
| 기타(`the solution given`, `official solution`, …) | 5 |

압도적으로 `reference solution`이다. bridge 프롬프트가 y\*를 그 이름으로 제시하기 때문이다.

오탐 방지 확인 — `solution`은 있는데 NARROW엔 안 걸린 문맥(정상적으로 제외됨):

```
… So values are 2, 4, 1 — never 3. ❌ So no solution. Thus, k = 3 is invalid.
… which is not integer, so no integer solution for n in this case.
… So only solution is c = 1. Thus, the only constant solution is f(x) = 1 …
```

### 1-2. refB — 참조 언급 (broad regex)

NARROW가 놓친 완곡한 표현을 추가로 잡는 확장판. 실측에서 발견한 누락 사례
(`This is known from the problem's context and its solution.`)를 계기로 만들었다.

```python
BROAD = NARROW의 모든 항 + re.compile(
    r"(the|its|this) solution('s)? (states|says|given|claims|shows|uses|suggests|indicates|mentions)|"
    r"according to (the )?(solution|reference|answer key)|"
    r"as (given|shown|stated) in the (solution|reference)|"
    r"from (the )?(problem'?s )?(context and its|known) solution|"
    r"the (intended|model|book) (solution|answer)", re.I)
```

BROAD가 추가로 잡은 것: `the intended answer`(4), `as given in the reference`(3),
`from the known solution`(3), `from known solution`(2), `the intended solution`(1).

**bridge에서는 NARROW 52.5% → BROAD 53.0%로 거의 차이가 없다**(bridge는 대놓고 `reference solution`이라
쓴다). 반면 **LoRA 후보에서는 refN 0~0.7% vs refB 3.1~6.9%로 5배 이상 벌어진다** — 어댑터를 통과한
뒤 남는 잔여 언급은 완곡한 형태다. 그래서 **후보의 오염률을 볼 때는 refB를 봐야 하고, refN은 하한**이다.

두 regex 모두 **재현율의 하한**이다. 사람이 읽어야만 잡히는 표현은 놓친다.

### 1-3. gold재현 — gold 8-gram 재현율

> **gold 풀이를 8토큰 단위로 쪼갰을 때, 그중 몇 %가 그 궤적 안에 그대로 나타나는가.**

```python
def toks(s):  # 단어 / 숫자 / 기호를 각각 한 토큰으로
    return re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", s.lower())
def grams(t, n): return {tuple(t[i:i+n]) for i in range(len(t)-n+1)}

gold재현 = |grams(궤적,8) ∩ grams(gold,8)| / |grams(gold,8)|
```

0 = 무관, 1 = gold를 통째로 포함. 표에는 **중앙값**을 싣는다(평균은 소수의 복사 사례에 끌려간다).

분모가 gold라는 점이 중요하다 — "궤적이 gold를 얼마나 흡수했나"를 재지, "궤적이 얼마나 gold스러운가"를
재지 않는다. 궤적이 길어져도 값이 부풀지 않는다.

### 1-4. 근사복사 — 표절 판정

> **gold재현 ≥ 0.30 이거나, gold와 공통 30-gram(≈100자 이상 연속 일치)이 존재.**

```python
근사복사 = (gold재현 >= 0.30) or bool(grams(궤적,30) & grams(gold,30))
```

두 조건은 서로 다른 복사를 잡는다. **재현율 조건**은 gold 전체를 흩어 옮긴 경우(요약·재배열),
**30-gram 조건**은 한 문단을 통으로 옮긴 경우다. 8-gram(≈25자)은 수학 표현에서 우연히 겹치므로
표절 판정에는 쓸 수 없고, 30-gram이면 우연 일치가 사실상 없다.

> 이전 분석에서 쓴 `LCS ≥ 200자`(최장 공통 부분문자열) 기준은 계산이 O(n·m)이라 전수 측정이 불가능해
> 30-gram 집합 교차로 대체했다. **30-gram 기준이 더 민감하다**(연속 100자면 잡음) — 같은 데이터에서
> LCS 기준 6% vs 30-gram 기준 17~27%. 두 문서의 숫자를 섞어 읽지 말 것.

**기저율**: CONTROL(베이스가 y\*를 전혀 안 보고 스스로 맞힌 궤적)의 근사복사가 **3.7%**다. 이것이
"우연히 겹치는 수학 표현"의 바닥이고, 모든 값은 이 대비로 읽어야 한다.

### 1-5. (참고) 답@10% / 답@25% / show-that

같은 표에 실린 구조 지표. **오염 지표가 아니다** — §5의 경고 참조.

- **답@10% / @25%**: 정답 문자열이 궤적의 앞 10% / 25% 지점에 처음 등장하는 비율(공백·`$`·`\dfrac` 정규화 후).
  "답을 먼저 선언하고 정당화하는" 구조를 잡으려 한 지표.
- **show-that**: `we need/want/have/must to show|prove|verify|confirm`, `let's verify|check|confirm`,
  `to verify|confirm|check that` 중 하나라도 있으면 1.

---

## 2. 결과 — bridge 궤적 (LoRA 학습 *이전*)

y\*를 보여주는 프롬프트로 base가 생성한 궤적. `runs/toy_cliff_2/default_BRIDGE_20260830_114141`.

| 행 | n | 문제 | len | 답@10% | 답@25% | show | gold재현 | **근사복사** | refN | refB |
|---|---|---|---|---|---|---|---|---|---|---|
| 생성 전체 | 2,360 | 250 | 5,522 | 28% | 43% | 30% | .074 | **26.4%** | 52.5% | 53.0% |
| 정답 (검증 통과) | 1,302 | 212 | 5,568 | 21% | 39% | 31% | .074 | **27.8%** | 60.9% | 61.3% |
| 정답 & 참조 **미언급** | 509 | 131 | 3,929 | 24% | 38% | 28% | .061 | **17.3%** | 0% | 1.0% |
| 정답 & 참조 언급 | 793 | 177 | 6,620 | 19% | 39% | 33% | .088 | **34.6%** | 100% | 100% |
| **채택 = LoRA 학습 타깃** | 742 | 212 | 5,086 | 20% | 36% | 30% | .073 | **27.4%** | 55.5% | 55.8% |

- **verifier 게이트가 오염을 농축시킨다**: 참조 언급이 전체 52.5% → 정답만 60.9%. 참조를 실제로 참조한
  궤적이 더 자주 맞기 때문이다.
- **`keep_selection: shortest`가 약한 오염 필터로 작동한다**: 61% → 55%. 언급하는 궤적이 1.7배 길기
  때문(6,620 vs 3,929 토큰)이며, 오염을 겨냥한 규칙이 아니라 부수 효과다.
- **★ "참조 미언급 = 깨끗함"이 아니다.** 참조를 한 번도 언급하지 않은 정답 bridge의 **17.3%가 gold를
  100자 이상 그대로 옮겨 쓴다.** 말없이 베끼는 경로가 따로 있고, regex 필터는 이를 전혀 못 잡는다.

## 3. 결과 — LoRA 이후 후보 (bare x, 학생 훈련에 들어갈 궤적)

| arm | n | 문제 | len | 답@10% | 답@25% | show | gold재현 | **근사복사** | refN | refB |
|---|---|---|---|---|---|---|---|---|---|---|
| **CONTROL** (기저율) | 108 | 65 | 5,971 | 27% | 36% | 28% | .007 | **3.7%** | 0% | 4.6% |
| LSPO (gold LoRA) | 144 | 81 | 5,585 | 24% | 37% | 36% | .008 | **3.5%** | 0.7% | 6.9% |
| BRIDGE | 626 | 125 | 4,255 | 31% | 41% | 19% | .036 | **8.3%** | 0.6% | 3.2% |
| STAGED | 380 | 127 | 4,658 | 34% | 41% | 21% | .028 | **8.9%** | 0.5% | 3.4% |
| STAGED_DPO | 416 | 123 | 4,451 | 31% | 43% | 25% | .032 | **9.6%** | 0% | 3.1% |

**가중치 채널을 지나며 세탁된다 — 두 종류 모두.**

```
참조 언급   학습 타깃 55.5%  →  후보 refN 0~0.6% / refB 3.1~3.4%
gold 복사   학습 타깃 27.4%  →  후보 8.3~9.6%          (기저율 3.7%)
gold재현    학습 타깃 .073   →  후보 .028~.036
```

- 참조 언급은 **거의 완전히** 사라진다(refB 기준으로도 CONTROL 4.6%보다 낮다).
- **gold 문자열 복사는 완전히는 안 사라진다**: 27.4% → 8~10%, 기저율의 **2.2~2.6배**. 이것이 현재 학생
  데이터에 남아 있는 유일한 실측 오염이다.
- LSPO는 gold를 직접 fit하는데도 후보의 근사복사가 3.5%(기저율)다. 4스텝 gold 어댑터는 gold를 복제할
  만큼 학습되지 않는다(라운드 2에서 6스텝을 주자 후보가 타깃 쪽으로 끌려갔다 — `toy_cliff_lora_fits.md` §11).

---

## 3-2. 분포 거리 — 학습 타깃과 후보의 base-policy NLL

문자열 누출(§1~3)과 별개로, **그 궤적이 학생(=base) 분포에서 얼마나 먼가**를 같은 단위로 잰다.

**측정법 (GPU 불필요).** LoRA는 B 행렬이 0으로 초기화되므로 **step 0의 logits은 베이스 그대로**다.
따라서 cold fit의 `fit_meta.json` → `loss_per_step[0]`이 곧 **그 fit 타깃의 base-policy NLL**(응답
토큰당, 토큰 가중)이다. 별도 스코어링 패스 없이 이미 기록돼 있다.

후보 쪽은 filters가 잰 `s_mean`인데 이것은 **시퀀스별 평균의 평균**이라 가중이 다르다. 아래 표는
후보를 **토큰 가중으로 다시 집계**해 fit loss와 같은 정의로 맞춘 값이다(시퀀스 가중 대비 +0.03~0.05).

| arm | 학습 타깃 | 타깃 base NLL | 타깃 토큰 | 후보 base NLL | 후보 n | 차이 |
|---|---|---|---|---|---|---|
| CONTROL | — (fit 없음) | — | — | **0.387** | 108 | — |
| LSPO | gold y\* | **1.238** | 129K | 0.463 | 144 | **+0.775** |
| BRIDGE | bridge z\* (kept) | 0.533 | 3.77M | 0.501 | 626 | +0.032 |
| STAGED | bridge z\* (kept, stage-1) | 0.523 | 3.79M | 0.490 | 380 | +0.032 |
| STAGED_DPO | bridge z\* (kept, stage-1) | 0.521 | 3.80M | **0.369** | 416 | +0.152 |

청크별 원값 (`loss_per_step[0]`):

| arm | 청크 | base NLL |
|---|---|---|
| LSPO | pooled_c0 / c1 / c2 | 1.290 / 1.176 / 1.267 |
| BRIDGE | pooled_c0 / c1 / c2 | 0.547 / 0.486 / 0.596 |
| STAGED | stage1_c0 / c1 / c2 | 0.538 / 0.500 / 0.577 |
| STAGED_DPO | stage1_c0 / c1 / c2 | 0.538 / 0.494 / 0.582 |

**읽을 점**

- **bridge 타깃(0.52~0.53)은 gold 타깃(1.238)의 절반 이하다.** bridge가 base 자기 분포에서 나왔으니
  당연하지만, 이 2.4배가 LSPO가 bridge 계열에 크게 지는 이유의 유력한 후보다 — 어댑터가 4스텝으로
  좁히기엔 gold가 너무 먼 타깃이다.
- **BRIDGE/STAGED는 타깃과 후보가 사실상 같다**(0.53 vs 0.50). 어댑터가 bridge 분포를 거의 그대로
  재생산한다는 뜻이다. "후보가 타깃보다 더 on-policy하다"가 아니라 **동등**이다.
- **LSPO만 타깃과 후보가 크게 벌어진다**(1.238 → 0.463). 4스텝 gold 어댑터는 gold를 복제하지 못하고
  base 근처에서 답을 뽑는다. 라운드 2에서 6스텝을 주자 후보 `s_mean`이 1.286 — **타깃 자리까지 끌려갔고**
  conversion은 안 늘었다([toy_cliff_lora_fits.md](toy_cliff_lora_fits.md) §11). 스텝이 적을 땐 타깃이
  멀어도 후보는 base 근처, 스텝이 늘면 타깃을 따라간다.
- **STAGED_DPO만 후보가 타깃보다 뚜렷이 가깝다**(0.521 → 0.369, −0.152). 같은 bridge를 chosen으로 쓰고도
  후보를 base 쪽으로 당기는 것은 DPO뿐이다.

**측정 못 한 것**

| 항목 | 왜 |
|---|---|
| bridge **전체/정답** 궤적의 NLL | fit loss는 **kept(채택된 것)**만 커버한다. 나머지 궤적은 스코어링 패스(GPU, ~15분)가 필요 — `adapters/pairs_*.jsonl`의 `input_ids`를 그대로 태우면 후보와 동일 정의로 나온다 |
| bridge의 `s_tail` / `C(y)` | fit loss는 평균 하나만 남기고 토큰별 logprob을 안 남긴다. CVaR 복원 불가 |
| stage-2 타깃의 base NLL | stage-2 fit은 warm-start라 `loss_per_step[0]`이 **stage-1 어댑터 기준**이지 base가 아니다(표에서 제외). DPO stage-2의 0.693은 base NLL이 아니라 −log 0.5, 즉 선호 손실의 초기값이다 |

---

## 4. 두 오염 축은 독립이다

| 구간 | refN | 근사복사 |
|---|---|---|
| bridge 정답 & 참조 언급 | 100% | 34.6% |
| bridge 정답 & 참조 미언급 | 0% | **17.3%** |

참조를 언급하지 않아도 복사는 절반 수준으로 일어난다. **regex로 거르면 오염의 절반만 걸러진다.**
"bridge를 참조 미언급만 필터링해서 쓰면 깨끗하다"는 가정은 성립하지 않으며, 그 필터는 동시에
**길이 필터**(3,929 vs 6,620 토큰)로도 작동해 짧은 궤적만 남기는 편향을 만든다.

---

## 5. ⚠ 답@10%는 오염 지표가 아니다 (기준선 확인)

`gold 원문` 자체를 같은 지표로 재면:

| 행 | n | len | 답@10% | 답@25% | show | gold재현 |
|---|---|---|---|---|---|---|
| **gold solution (원문)** | 250 | 541 | **31%** | 38% | **6%** | 1.000 |

**gold 원문의 답@10%가 31%로 우리 후보(24~34%)·CONTROL(27%)과 같은 수준이다.** 즉 "답이 앞에 나온다"는
오염이 아니라 **간결한 수학 풀이 글의 일반적 성질**이다(교과서 풀이는 답을 먼저 말하고 검산한다).
5개 arm의 답@10%는 CONTROL 대비 전부 유의하지 않다(z = −0.43 ~ 1.10, p = 0.27 ~ 0.67).

반면 **show-that은 변별력이 있다**: gold 원문 6% vs bridge 30% vs 후보 19~25%. bridge가 gold보다 5배
높은 것은 "y\*를 보고 그쪽으로 논증하는" 서술이 실재한다는 뜻이고, LoRA를 지나며 줄어든다.

> 이 문서 이전의 분석에서 답@10%를 "답 의존 구조"의 증거로 쓴 대목이 있다. gold 기준선을 재기 전이었고,
> **철회한다.** 구조 신호로는 show-that만 남는다.

---

## 6. 재현

```bash
.venv/bin/python <스크립트>   # 전부 CPU, 수 분
```

원천 파일 (arm별 `runs/toy_cliff_2/<run>/iter_0/` 아래):

| 무엇 | 어디 |
|---|---|
| bridge 궤적 메타 (correct/kept/n_tokens) | `improve/bridge/bridges.jsonl` |
| bridge 궤적 **본문** | `improve/pool/bridge*/out_*.jsonl` — `rid = "<qid>:bridge"`, `samples[i]`가 `sample_idx=i` |
| LoRA 후보 (본문 + correct) | `improve/improved.jsonl` |
| CONTROL의 정답 판정 | `filtered/candidate_scores.jsonl`의 key — `improved.jsonl`의 `correct`가 `self_resample`에서는 None이라 여기서 조인해야 한다 |
| gold y\* | `data/cliff_sets/openr1_default_cliff450_k16_with_gold.jsonl`의 `meta.gold_solution` |

주의: bridge 본문은 `bridges.jsonl`에 없다(메타만). 풀 출력과 `(qid, sample_idx)`로 조인해야 한다.

---

## 7. 요약

| 질문 | 답 |
|---|---|
| bridge가 참조를 언급하나 | 학습 타깃의 **55%**. verifier가 농축시키고(52→61%), shortest가 6pp 되돌린다 |
| 그게 후보로 새나 | **아니다.** refN 0~0.6%, refB 3.1~3.4% — CONTROL(4.6%)보다도 낮다 |
| 말없이 gold를 베끼나 | **그렇다.** 학습 타깃 27%, 참조 미언급 것만 봐도 **17%** |
| 그게 후보로 새나 | **일부.** 8~10%, 기저율 3.7%의 **2.2~2.6배** |
| 참조 미언급 필터로 충분한가 | **아니다.** 복사 축을 전혀 못 잡고(17% 통과), 길이 편향을 만든다 |
| 답을 먼저 쓰는 게 오염인가 | **아니다.** gold 원문도 31%, CONTROL도 27% |
| 남는 구조 신호는 | **show-that**: gold 6% / bridge 30% / 후보 19~25% |
