# SOAR Week 3 Champion Notes | Cao Yi: Quantization Accuracy Stability & Dataset Analysis

## Editor's Note

This week, Cao Yi (SlightwindSec) defended his title for the third consecutive week. As quantization optimizations approached performance bottlenecks, he shifted focus from raw speed to accuracy stability and engineering foundations — addressing accuracy jitter, improving the evaluation pipeline, and refining calibration data.

The champion score now sits at **97% of the top bounty target**. Every incremental step is critical. From mixed-precision quantization in earlier weeks, to this week's focus on **accuracy jitter control** and **local fast evaluation tooling**, he transformed "luck-dependent" unstable submissions into predictable, controlled optimization. He also performed deep analysis of the evaluation dataset and redesigned the quantization calibration strategy as groundwork for future score improvements.

---

## About the Author

**Cao Yi (SlightwindSec)** — LLM inference framework developer and model performance optimization engineer at a domestic compute platform. 2+ years of front-line engineering experience in inference acceleration and low-bit quantization.

---

## Week 3 Progress

> "Very fortunate to three-peat. Quantization feels like it's hitting a ceiling. This week I made almost no forward progress on speed — instead I spent time adjusting the calibration dataset and doing infrastructure work."

---

## 01 — Accuracy Jitter Problem

Last week I described encountering **accuracy instability** with quantized inference: same code, same config, different submission batches produce different correctness scores. Bad luck means taking an accuracy penalty coefficient.

With only **3 submission attempts per day**, unstable results waste opportunities. The root cause: my initial calibration dataset was the simplest possible — there's still optimization headroom there.

Evaluating quantization quality is non-trivial:

1. **Generative tasks** accurately reflect real inference quality, but are slow. With jitter you need multiple runs averaged — doubling time cost.

2. **MMLU** gives fast first-token accuracy, but diverges from generative task behavior. "Flip" phenomena mean accuracy numbers can be misleading — a "false prosperity" where first-token accuracy looks good but doesn't reflect true quantization degradation vs. the FP16 baseline. *(ref: "Accuracy is Not All You Need")*

**Solution**: Compare logit distributions of output tokens between the quantized model and the BF16 baseline. Using SGLang's `logprobs` API — no source code modification needed — this gives stable, fast pre-submission accuracy verification.

---

## 02 — Logprob-Based Fast Accuracy Evaluation

The core idea: **compare logprobs distributions between quantized model and BF16 baseline**, rather than running full benchmark suites.

### Two-Phase Flow

**Phase 1 — Baseline Collection** (run once):
- Use BF16 model to greedily decode a fixed set of built-in prompts
- Extract top-256 logprobs for the last 128 tokens
- Save as sparse baseline files at batch sizes 1 / 8 / 32

**Phase 2 — Quantized Weight Evaluation**:
- Start server, send identical prompts
- Compare against baseline, output three metrics:

| Metric | Meaning |
|---|---|
| KL Divergence | Lower = quantized output distribution closer to BF16 |
| Cosine Similarity | Higher = better alignment |
| Top-1 Agreement | % of positions where argmax matches |

### Design Details

- **Top-256 sparse representation**: SGLang doesn't support full-vocabulary logprobs. Top-256 covers >99.9% of probability mass; remaining probability is smoothed uniformly.
- **Zero GPU overhead**: Baseline saved as numpy arrays on CPU. Evaluation only sends HTTP requests to the running server — extremely lightweight, can run anytime.
- **Known limitation**: Only measures Prefill-phase accuracy error. Does not cover error propagation during Decode.

This tool screens out clearly broken configurations before wasting a submission slot.

---

## 03 — Evaluation Dataset Analysis

The official accuracy evaluation dataset has **150 samples across 5 task types (30 each)**:

| Task Type | Count | Notes |
|---|---|---|
| Task A | 30 | |
| Task B | 30 | |
| Task C | 30 | |
| Task D | 30 | High proportion of low-semantic-coherence data |
| Task E | 30 | Includes random words, UUIDs |

**Key finding**: Low-semantic-coherence data (random words, UUIDs) is over-represented. When building a calibration dataset:
- Mirror the 5-task distribution
- **Reduce** the proportion of random/UUID-style samples
- **Increase** semantically coherent, continuous-context samples

Excessive random/UUID data likely harms the calibration process.

---

## 04 — NVFP4 Quantization

Current approach: **NVFP4 format for weight quantization**.

NVFP4 is NVIDIA's 4-bit floating point format introduced with the **Blackwell architecture**. Since the competition hardware is Blackwell, it gets **native hardware acceleration** — the primary reason to use it.

### Format

- **E2M1** structure: 1 sign bit, 2 exponent bits, 1 mantissa bit
- Representable range: approximately **-6 to 6**

### Dual-Scale Mechanism

NVFP4's key accuracy advantage is a two-level scaling scheme:

| Level | Scope | Scale Type |
|---|---|---|
| Local | Per 16 weights (group) | FP8 (E4M3) correction factor |
| Global | Entire tensor | FP32 scale factor |

This dual-scale approach captures local weight distribution more accurately than single-scale FP4, preserving more precision at the same compression ratio.

---

## 05 — Summary

This week's progress:
- Built a **local fast accuracy evaluation tool** based on logprob distribution comparison
- Analyzed the **evaluation dataset distribution** across 5 task types
- Optimized the **calibration dataset** composition

Next: more directions to explore. Hoping for a bigger breakthrough next week.

---

## Key Takeaways for Our Optimization

| Insight | Action for us |
|---|---|
| Logprob KL divergence as fast pre-submission accuracy check | Build similar tool before submitting FP8 |
| Random/UUID samples hurt calibration | Avoid them in any calibration data we collect |
| NVFP4 (E2M1 + dual-scale) is the current leader's quantization | This is our Roadmap #6 — high effort but Blackwell-native |
| 97% of top bounty achieved by the leader | We need ~45% speed reduction on S1; FP8 is step 1 |
| Per-tensor scale causes accuracy jitter vs per-channel | Consider upgrading to per-channel FP8 if accuracy is marginal |
