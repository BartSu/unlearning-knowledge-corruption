# Storyline — The 10-Beat Narrative Thread

Companion to `speaker_notes.md` (which is slide-by-slide). **This file is the story.** Memorise the 10 beats and the causal links between them; slides become backup.

If your mind goes blank, fall back to the causal chain — each beat **forces** the next one.

---

## The whole story in three sentences

> Unlearning a forget set damages unrelated knowledge on three semantic-distance layers. The damage is a **property of the data**, not only of the algorithm. Therefore we should **audit the forget set before unlearning** — and a Ridge regressor on 12 geometric features is enough to do the audit.

Three sentences. That's the entire contribution.

---

## Beat 1 · Problem: unlearning hurts unrelated data *(p3)*

**What to say**:
> "LLM unlearning is needed whenever copyright, privacy or toxic content has to be deleted from a trained model. But unlearning is **not surgical** — forgetting $\mathcal{D}_f$ damages knowledge unrelated to $\mathcal{D}_f$. We call that **knowledge corruption**."

**Why this beat exists**: set up the object we're studying. No method yet.

**Causal link to Beat 2**:
> "And it's hard to study systematically — which is why the phenomenon hasn't been characterised before."

---

## Beat 2 · Why it's hard: quadratic cost *(p3, end)*

**What to say**:
> "To establish the pattern and compare 100 candidate forget sets, we need 100 unlearn runs plus an N×N cross-PPL matrix. Cost scales **quadratically** — our N=100 took several GPU-hours."

**Why this beat exists**: explain why this problem is open; justify why your method matters.

**Causal link to Beat 3**:
> "So first we built a controlled testbed that makes the three layers naturally observable."

---

## Beat 3 · Setup: triplet schema + cross-matrix *(p4)*

**What to say**:
> "WikiText-103 clustered by HDBSCAN into 10 semantic clusters. From each cluster I sample 10 **triplets**; each triplet has three 50-text splits — `train` is the forget set, `validation` is retain neighbours, `test` is probe — all in the same cluster but disjoint. Every checkpoint is evaluated on every triplet's test — an N×N cross-PPL matrix."

**Why this beat exists**: trust the numbers that follow. Listeners need to know the data is clean.

**Causal link to Beat 4**:
> "With that testbed, the first thing that jumps out is a **structure**."

---

## Beat 4 · Observation 1: three layers decay monotonically, and L3 is above 1 *(p5, p13, p14)*

**What to say**:
> "Ratio `r = unlearned-PPL / base-PPL`. Geo-mean per layer gives one number each:
> **L1 forget 1.96×**, **L2 locality 1.32×**, **L3 spillover 1.19×**.
> Monotone decay — further from the forget set, less damage.
>
> But here's the punchline: **L3 is 1.19 — above 1** — and **74% of cross-topic samples** still see PPL raised. Spillover is not a tail event."

**Why this beat exists**: the **observation** half of the paper.

**Causal link to Beat 5**:
> "And that damage is very **uneven** across forget sets — which is the second half of Act I."

---

## Beat 5 · Observation 2: the forget set matters *(p15, p16)*

**What to say**:
> "Same algorithm, same hyper-parameters, **only the data differs** — L1 varies by **1.6×** across the 100 forget sets; log-scale spread 1.7×. A 'storm' cluster around triplet 073 concentrates the worst damage on all three layers."

**Why this beat exists — this is the pivot of the whole paper**:
> "The damage is a **property of the data**, not only of the algorithm. Which means: if we could characterise that property from the data side, we'd be able to **predict** corruption before running unlearn."

**Causal link to Beat 6**:
> "This reframes the question."

---

## Beat 6 · Framing shift: benchmark → audit *(p18)*

**What to say** — say this one **slowly**, it is the paper's pitch:
> "Instead of running unlearn and then measuring corruption, let's **audit the forget set first** — look at its geometry — and use that to rank the three layers of risk. **No unlearn. No evaluation text.** Only the forget set itself."

**Why this beat exists**: new question, new object. You have earned the audience's attention; now cash in.

**Causal link to Beat 7**:
> "Concretely, here's the auditor."

---

## Beat 7 · Method: 12 geometric features + Ridge LOO *(p19)*

**What to say**:
> "Twelve features from four families of the forget set's sentence embeddings:
> **spread** (variance, pairwise distance),
> **similarity** (pairwise cosine),
> **location and scale** (centroid norm),
> **shape** (effective rank, isotropy).
> Ridge regression, α=1, leave-one-out over 100 forget sets."

**Why this beat exists**: show the method is **minimal** — no fancy architecture, no tuning. This is a credibility move.

**Causal link to Beat 8**:
> "And here's whether it works."

---

## Beat 8 · Result: it works, L2 is predictable, L3 lifts at n=100 *(p20)*

**What to say** — stop and deliver:
> "All three layers **beat** the mean-baseline (R² = −0.02).
> L1 R² = +0.25, ρ = +0.53.
> **L2 R² = +0.71, ρ = +0.84, CI 0.76 to 0.89.**
> L3 R² = +0.22, ρ = +0.49, CI excludes zero.
>
> Same-topic collateral — **L2 locality** — is almost perfectly rank-predictable from forget-set geometry alone. And L3, which at n=10 was R² = −0.46, lifts to +0.22 once we hit n=100 — proving the signal is real, not noise."

**Why this beat exists**: the answer. Everything up to here was setup.

**Causal link to Beat 9**:
> "Before you think this replaces unlearning — it doesn't. Scope matters."

---

## Beat 9 · Positioning: warner, not replacement *(p21)*

**What to say** — say this deliberately:
> "The auditor is a **cheap coarse-screening warner** — not a replacement. It can **rank** who is worse, **screen** for red/yellow/green, and **save compute** — one second of geometry vs. GPU-hours. It **cannot** give absolute scores, cannot score an unlearner, cannot replace the final unlearn."

**Why this beat exists**: precise scope is what makes the claim reviewer-proof. Without this beat the audience will ask "so can I skip unlearning?" and you lose.

**Causal link to Beat 10**:
> "Who asked these questions before? Two papers shape the framing."

---

## Beat 10 · Positioning in the literature + next steps *(p23, p24, p25)*

**What to say**:
> "**Dang 2024** predicted adversarial robustness from dataset-side features — that's the paradigm I borrowed. **Ko 2025** showed unlearning leaves adjacent knowledge holes — that's the phenomenon I quantify. **Dang told me *how* to predict from the data side; Ko told me *what* to predict.**
>
> Four singles I'd like to lift: single base model, single unlearner, n=100, single metric (PPL not QA). Next steps: NPO or GradDiff as a second unlearner, QA-label metrics. Target venue NeurIPS or ICML."

**Closing line** — look up from the slide, say crisp, then stop:
> "Corruption **has layers**, geometry **predicts them**, audit **lets you triage 100 forget sets in seconds instead of days**."

---

## The causal chain on one page

```
 Problem: unlearn hurts unrelated data
   ↓  "and it's hard to study —"
 Why hard: N² cost to benchmark N forget sets
   ↓  "so we built a clean testbed —"
 Setup: triplets + N×N matrix
   ↓  "the first thing that jumps out is a structure —"
 Obs 1: three layers, L3>1, 74% of bystanders hit
   ↓  "and the damage is very uneven across forget sets —"
 Obs 2: 1.6× spread in L1 → damage is a property of the data
   ↓  "which reframes the question —"
 Shift: benchmark → audit (from data side, no unlearn)
   ↓  "concretely, here's the auditor —"
 Method: 12 geom features + Ridge LOO
   ↓  "and here's whether it works —"
 Result: three CIs exclude 0; L2 R²=0.71; L3 lifts at n=100
   ↓  "but it is not a replacement —"
 Scope: warner, not replacement (rank ✓ / absolute ✗)
   ↓  "who asked these questions before —"
 Related + limits + next: Dang + Ko, four singles, NeurIPS target
   ↓
 Close: "Corruption has layers, geometry predicts them, audit lets you
         triage 100 forget sets in seconds instead of days."
```

Each arrow is a **causal transition** you can say out loud. If any arrow doesn't feel natural to you, that's where the story has a gap and you should stop and re-think the beat above it.

---

## 3 hardest Q&A rehearsals

**Q1**: "But to audit a single forget set I don't need 100 unlearn runs, do I?"
> "Correct — one unlearn + N-way eval is enough for one profile. The N² cost shows up when you want to *compare* N candidates or *train* the audit. That training happens **once**; deployment is seconds per new forget set."

**Q2**: "Why would forget-set geometry predict unlearning damage?"
> "Intuition: LLMs encode knowledge in embedding neighbourhoods. A **compact** forget set sits in one knowledge pocket — unlearn hits that pocket cleanly. A **sprawling** forget set overlaps many pockets — damage leaks further. L2 R²=0.71 is that intuition quantified."

**Q3**: "Is n=100 enough?"
> "Honestly it's the smallest scale where all three layer CIs exclude zero and L3 lifts from negative R² into positive. L2 is very tight — CI 0.76 to 0.89. L3 is still the weakest and still on the 'keep scaling' list for the next round."

---

## Memorise just these

1. **The whole story in three sentences** (top of this file).
2. **The L2 result line**: "L2 R² 0.71, rho 0.84, CI 0.76 to 0.89 — geometry alone, no unlearn."
3. **The scope line**: "Cheap coarse-screening warner, not a replacement."
4. **The closing line**: "Corruption has layers, geometry predicts them, audit triages 100 forget sets in seconds instead of days."

If everything else blurs out, these four sentences plus the causal chain carry the talk.
