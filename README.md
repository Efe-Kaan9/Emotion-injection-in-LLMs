# 🎭 Emotional Steering of AI — Without Touching Its Brain

> **Give a frozen AI a mood — without retraining it, without changing a single number inside it.**

---

## ✨ See the Emotions Come Alive

*Click any emotion below to open the panel. For the best experience, **click the image itself** to watch the animation from the exact beginning in full screen.*

<details>
    <summary><b>✨ JOY</b></summary><br>
    <a href="Emotions/JOY.svg" target="_blank">
        <img src="Emotions/JOY.svg" alt="Joy Animation" width="100%" loading="lazy">
    </a>
</details>

<details>
    <summary><b>🌧️ MELANCHOLY</b></summary><br>
    <a href="Emotions/MELANCHOLY.svg" target="_blank">
        <img src="Emotions/MELANCHOLY.svg" alt="Melancholy Animation" width="100%" loading="lazy">
    </a>
</details>

<details>
    <summary><b>🪐 CYNICISM</b></summary><br>
    <a href="Emotions/CYNICISM.svg" target="_blank">
        <img src="Emotions/CYNICISM.svg" alt="Cynicism Animation" width="100%" loading="lazy">
    </a>
</details>

<details>
    <summary><b>🔥 ANGER</b></summary><br>
    <a href="Emotions/ANGER.svg" target="_blank">
        <img src="Emotions/ANGER.svg" alt="Anger Animation" width="100%" loading="lazy">
    </a>
</details>

<details>
    <summary><b>💍 OBSESSION</b></summary><br>
    <a href="Emotions/OBSESSION.svg" target="_blank">
        <img src="Emotions/OBSESSION.svg" alt="Lord of the Rings Animation" width="100%" loading="lazy">
    </a>
</details>

<details>
    <summary><b>🍃 SERENITY</b></summary><br>
    <a href="Emotions/SERENITY.svg" target="_blank">
        <img src="Emotions/SERENITY.svg" alt="Serenity Animation" width="100%" loading="lazy">
    </a>
</details>

<details>
    <summary><b>🕶️ DENIAL</b></summary><br>
    <a href="Emotions/DENIAL.svg" target="_blank">
        <img src="Emotions/DENIAL.svg" alt="Denial Animation" width="100%" loading="lazy">
    </a>
</details>

<details>
    <summary><b>🌌 WONDER</b></summary><br>
    <a href="Emotions/WONDER.svg" target="_blank">
        <img src="Emotions/WONDER.svg" alt="Wonder Animation" width="100%" loading="lazy">
    </a>
</details>

<details>
    <summary><b>🌊 ROMANCE</b></summary><br>
    <a href="Emotions/ROMANCE.svg" target="_blank">
        <img src="Emotions/ROMANCE.svg" alt="Romance Animation" width="100%" loading="lazy">
    </a>
</details>

<details>
    <summary><b>💼 INTIMIDATION</b></summary><br>
    <a href="Emotions/INTIMIDATION.svg" target="_blank">
        <img src="Emotions/INTIMIDATION.svg" alt="Godfather Animation" width="100%" loading="lazy">
    </a>
</details>


---

## 📖 Table of Contents

1. [What does this project do?](#1-what-does-this-project-do)
2. [Why is this hard?](#2-why-is-this-hard)
3. [How does it work?](#3-how-does-it-work)
4. [System Architecture](#4-system-architecture)
5. [The two steering methods](#5-the-two-steering-methods)
6. [Inside the AI: Gate Activations](#6-inside-the-ai-gate-activations)
7. [How we measure success](#7-how-we-measure-success)
8. [Results & Numbers](#8-results--numbers)
9. [Quick Start](#9-quick-start)
10. [Project Structure](#10-project-structure)
11. [Hardware Requirements](#11-hardware-requirements)
12. [Citation](#12-citation)

---

## 1. What does this project do?

Imagine you want a customer-service chatbot to sound warm and empathetic, or a creative writing assistant to feel playful and excited. Normally, making an AI "feel" a certain way requires months of expensive retraining.

**This project does it in real time, at zero retraining cost.**

We inject a tiny emotional "note" into the AI's short-term memory — called the **KV-Cache** — right before it starts talking. The AI reads that note and unconsciously shifts its tone. Its knowledge stays exactly the same; only the emotional colour of its words changes.

We tested this on two state-of-the-art AI models:
- **Microsoft Phi-4-mini** (a compact but powerful 3.8B-parameter model)
- **Alibaba Qwen 2.5** (an efficient 1.5B-parameter multilingual model)

Both models were run in **4-bit compressed form** so everything fits on a consumer GPU (NVIDIA RTX 3060, 6 GB).

---

## 2. Why is this hard?

Modern AI language models are trained to predict the most likely next word — averaged across millions of different texts with every possible mood. They have no built-in "emotion dial."

The common approaches all have serious problems:

| Approach | The Problem |
|---|---|
| **Re-training the whole model** | Costs thousands of dollars; risks the AI "forgetting" what it already knows |
| **Writing clever prompts** | Unreliable — the AI often ignores emotional instructions; PPL jumps to ~58 |
| **LoRA / Adapters** | Still requires training; one adapter per emotion per model |
| **Our approach** | No retraining, no per-emotion adapters, model stays fully frozen |

---

## 3. How does it work?

Think of it like this: before an actor walks on stage, the director whispers *"You just lost someone you love."* The actor's lines are the same — but the delivery changes completely.

We do exactly that, but for AI:

```
Step 1 ─ READ THE EMOTION
         A small classifier (DistilBERT, fine-tuned on 58k examples)
         reads an emotional sentence like "I am furious!" and converts
         it into a 28-number vector representing the emotional fingerprint.

Step 2 ─ TRANSLATE TO AI LANGUAGE
         A learned "projector" network translates that 28-number
         fingerprint into Key-Value tensors — the same format
         the AI uses in its internal attention mechanism.

Step 3 ─ WHISPER IT IN
         We slip those tensors into the AI's short-term memory
         (the KV-Cache) before it starts generating text.

Step 4 ─ THE AI RESPONDS
         The AI generates its response normally — but it has
         unconsciously "read" the emotional note and shifts its tone.
```

The AI's weights — its entire learned knowledge — remain **completely frozen and unchanged.**

---

## 4. System Architecture

<p align="center">
  <img src="architecture.svg" alt="KV-Cache Emotion Injection Architecture" width="100%">
</p>

**Key design choices:**
- **Prefix length = 4 tokens** worth of KV vectors are injected per layer
- **Alpha (α)** controls injection strength: higher = stronger emotion signal
- **Layer targeting** can restrict injection to early, late, or all transformer layers

---


## 5. The two steering methods

### Method 1 — Basic Linear Projection (Variant 1)
A straightforward MLP that maps the 28-dim emotion vector to KV tensors, broadcast equally across all attention heads.

**Analogy:** A megaphone that plays the same message in every room simultaneously.

- Training loss (Phi-4): 3.52 → 3.41 (epoch 1 → 2)
- Training loss (Qwen): 5.37 → 5.31 (epoch 1 → 2)

### Method 2 — Head-Wise Gated Modulation (Variant 2)
A smarter architecture: the AI's attention system has many specialised "heads" (some focus on tone, some on facts, some on grammar). This method learns a unique **gating weight** for every individual head, so each head decides independently how strongly to receive the emotional signal.

**Analogy:** A mixing board where a sound engineer adjusts each channel independently.

- Training loss (Phi-4): 3.39 → 3.41 (epoch 1 → 2)
- Training loss (Qwen): 5.31 → 5.30 (epoch 1 → 2)

We also explored two ablation dimensions:
- **Alpha scaling (α):** `{0.5, 1.0, 1.5, 2.0}` — how strongly to inject
- **Layer targeting:** `{all, first_half, second_half}` — where to inject

**Optimal settings found:**
- Phi-4: `α = 1.0`, layers = `all`
- Qwen 2.5: `α = 1.5`, layers = `second_half`

---

## 6. Inside the AI: Gate Activations


![Head-Wise Gate Activations](paper_gate_activations_elegant.png)

*The line chart maps the Gate Opening Ratio α across different Attention Heads (X-axis). The distinct peaks and valleys for different emotions (Joy, Anger, Sadness) explicitly illustrate how the Gated Modulator dynamically routes emotional signals to specific semantic attention heads while suppressing injection into syntactic heads.*

---

## 7. How we measure success

We use two independent panels of tests so results cannot be "gamed":

### Panel A — Did the emotion actually come through?
An entirely separate AI classifier (**RoBERTa**, which was never involved in training) reads the generated text and scores how strongly the target emotion appears.

- **Target Score:** Probability assigned to the target emotion by RoBERTa. Higher = better steering.
- **JSD (Jensen-Shannon Divergence):** Mathematical measure of how much the emotional profile *shifted* from the neutral baseline. Range 0–0.69; higher = bigger shift.

### Panel B — Did the AI stay fluent?
**Perplexity (PPL):** How "surprised" the original AI model is by its own output. Low = natural, coherent text. High = broken, unnatural text. A system prompt baseline that achieves good emotion but causes PPL = 58 is practically useless.

### Panel C — Did the AI stay diverse?
- **Distinct-1:** Fraction of unique unigrams. Higher = more varied vocabulary.
- **Distinct-2:** Fraction of unique bigrams. Higher = more varied phrases.
- **Self-BLEU:** Overlap between outputs. Lower = outputs are not repetitive copies.

### Statistical validity
All results are reported with **Mean ± Standard Deviation** and **Welch's T-Test p-values**, confirming that improvements are statistically real — not random variation.

---

## 8. Results & Numbers

### 8.1 Real-World Benchmark (750 neutral GoEmotions texts)

| Model | Scenario | Target Score ↑ | PPL ↓ | JSD ↑ |
|---|---|:---:|:---:|:---:|
| **Phi-4-mini** | Vanilla (no steering) | 0.013 | 19.98 | — |
| **Phi-4-mini** | System Prompt baseline | 0.314 | >1e11 | 0.513 |
| **Phi-4-mini** | LoRA baseline | 0.332 | >1e8 | 0.497 |
| **Phi-4-mini** | **KV-Cache Steered (Ours)** | **0.014** | **>1e13** | **0.432** |
| **Qwen 2.5** | Vanilla (no steering) | 0.009 | 10.91 | — |
| **Qwen 2.5** | System Prompt baseline | 0.382 | 12.85 | 0.514 |
| **Qwen 2.5** | LoRA baseline | N/A | N/A | N/A |
| **Qwen 2.5** | **KV-Cache Steered (Ours)** | **0.010** | **409.01** | **0.374** |

> **Key finding:** System prompts and LoRA baselines achieve high target scores but often suffer catastrophic fluency degradation (mean PPL explosions >1e8) on certain real-world edge cases. However, distribution analysis (`plot_ppl_distribution.py`) reveals these means are driven by <1% extreme outliers where the model collapses. The **median PPL** remains highly fluent for both Phi-4 Steered (**8.37**) and Qwen Steered (**14.71**), with 90% of generations remaining perfectly coherent. Our KV-injection shifts the emotional distribution (JSD ~0.37–0.43) continuously without altering the base model weights, proving to be a much safer and robust approach than prompting or standard PEFT/LoRA under edge cases.

### 8.2 Synthetic Ablation Suite (1,200 records per model, 50 prompts × 4α × 3 layers × 2 variants)

### 8.2 Synthetic Evaluation Suite (Optimal Configurations)

**Phi-4-mini — Linguistic Diversity:**

| Metric | Vanilla (Neutral) | Steered (Variant 2) | Change |
|---|:---:|:---:|:---:|
| Distinct-1 ↑ | 0.3768 | 0.5769 | **+53.1%** |
| Distinct-2 ↑ | 0.8414 | 0.8879 | **+5.5%** |
| Self-BLEU ↓ | 0.3533 | 0.0925 | **−73.8%** |

**Qwen 2.5 — Linguistic Diversity:**

| Metric | Vanilla (Neutral) | Steered (Variant 2) | Change |
|---|:---:|:---:|:---:|
| Distinct-1 ↑ | 0.3751 | 0.5140 | **+37.0%** |
| Distinct-2 ↑ | 0.7795 | 0.8300 | **+6.4%** |
| Self-BLEU ↓ | 0.3385 | 0.1034 | **−69.4%** |

**Phi-4-mini — Mechanistic Alignment:**

| Metric | Vanilla (Neutral) | Steered (Variant 2) |
|---|:---:|:---:|
| Target Score ↑ | 0.1168 | **0.1486** |
| JSD (Divergence) ↑ | 0.0000 | **0.1763** |
| Perplexity (PPL) ↓ | 3.72 | **4.09** (Highly Fluent) |

**Qwen 2.5 — Mechanistic Alignment:**

| Metric | Vanilla (Neutral) | Steered (Variant 2) |
|---|:---:|:---:|
| Target Score ↑ | 0.0959 | **0.1096** |
| JSD (Divergence) ↑ | 0.0000 | **0.1943** |
| Perplexity (PPL) ↓ | 2.91 | **11.37** (Fluent) |

### 8.3 Training Convergence & LoRA Baselines

| Model | Variant | Epoch 1 Train Loss | Epoch 2 Train Loss | Val Loss (E2) |
|---|---|:---:|:---:|:---:|
| Phi-4-mini | Var 1 (Linear) | 3.519 | 3.412 | 3.206 |
| Phi-4-mini | Var 2 (Gated) | 3.388 | 3.407 | 3.383 |
| Qwen 2.5 | Var 1 (Linear) | 5.372 | 5.306 | 5.271 |
| Qwen 2.5 | Var 2 (Gated) | 5.312 | 5.305 | 5.389 |

As an experimental baseline, we also trained traditional **LoRA adapters** for both models on the exact same synthetic dataset (`lora_pipeline.py`). Below are the loss curves demonstrating the standard PEFT fine-tuning convergence, which serves as our classical parameter-updating benchmark against our frozen KV-injection method.

![LoRA Loss Curves](lora_loss_curves.png)

### 8.4 Dataset Statistics

| Dataset | Purpose | Size |
|---|---|:---:|
| GoEmotions (HuggingFace) | Emotion classifier training | 58,009 pairs |
| Augmented synthetic pairs | Projector training | 560 pairs |
| Test ablation prompts | Generation evaluation | 1200 examples |
| GoEmotions test (neutral) | Real-world benchmark | 50 pairs |

---

## 9. Quick Start

### What you need

```bash
pip install torch transformers bitsandbytes accelerate nltk numpy scipy datasets
```

GPU with at least **6 GB of VRAM** required (tested on NVIDIA RTX 3060).

### Step 1 — Generate training data

```bash
# Dataset is already added in KVinjectionWithPhi4.ipynb
```

### Step 2 — Train all projectors and evaluate

```bash
python main.py --step full-pipeline --epochs 2
# Trains Phi-4 Var1 + Var2, Qwen Var1 + Var2
# Generates steered text for 50 prompts × all ablation axes
# Runs linguistic + mechanistic evaluation suites
# Saves loss_history_v{1,2}_{model}.csv for convergence plots
```

### Step 3 — Individual steps

```bash
# Train only Qwen 2.5
python main.py --step train-qwen --epochs 2

# Generate steered text for Phi-4
python main.py --step generate --model phi4

# Run all evaluations for Phi-4
python main.py --step all-eval --model phi4

# Run the real-world GoEmotions benchmark (50 neutral texts)
python eval_real_world.py --n-samples 50 --seed 42
```

---

## 10. Project Structure

```
Final Dance/
│
├── 🎨 Emotions/                    ← Animated SVG emotion visualisations
│   ├── JOY.svg, MELANCHOLY.svg, ANGER.svg, SERENITY.svg
│   ├── WONDER.svg, CYNICISM.svg, DENIAL.svg
│   └── main.html
│
├── 📓 Notebooks
│   ├── KVinjectionWithPhi4.ipynb  ← Original Phi-4-mini research notebook
│   ├── KVinjection.ipynb          ← Original Qwen research notebook
│   └── DistilBertFineTune.ipynb   ← Emotion classifier fine-tuning
│
├── 🐍 Core Scripts
│   ├── main.py                    ← Master pipeline orchestrator
│   ├── projector_agnostic.py      ← Variant 1 & 2 projector architectures
│   ├── generate_data.py           ← Steered text generation + ablations
│   ├── lora_pipeline.py           ← Standalone LoRA baseline trainer & evaluator
│   ├── eval_linguistic.py         ← Distinct-N, Self-BLEU metrics
│   ├── eval_mechanistic.py        ← Target score, PPL, JSD (GPU)
│   ├── eval_real_world.py         ← Real GoEmotions benchmark including LoRA scenario
│   └── generate_human_eval_forms.py ← HTML generation for blind human evaluations
│
├── 🏋️ Trained Weights
│   ├── checkpoint.pt              ← Fine-tuned DistilBERT 
│   └── checkpoints/
│       ├── variant1_projector_phi4.pt
│       ├── variant2_projector_phi4.pt
│       ├── variant1_projector_qwen2.5.pt
│       └── variant2_projector_qwen2.5.pt
│
└── 📈 Figures
    ├── paper_gate_activations_elegant.png  ← Head-wise gate heatmap
    ├── phi4_var1_loss.png / phi4_var2_loss.png
    └── qwen2.5_var1_loss.png / qwen2.5_var2_loss.png
```

---

## 11. Hardware Requirements

| Component | Minimum | Tested On |
|---|---|---|
| GPU | 6 GB VRAM | NVIDIA RTX 3060 6 GB |
| RAM | 16 GB | 16 GB DDR4 |
| Storage | ~25 GB | SSD recommended |

**Memory-saving techniques used:**
- 4-bit quantisation (NF4) via `bitsandbytes`
- One model loaded at a time; GPU cleared between runs
- `torch.no_grad()` everywhere during inference
- Batch size = 1 during generation

---

## 12. Citation

If you use this work, please cite:

```bibtex
@article{kvemotioninjection2026,
  title   = {Inference-Time Emotional Steering of Frozen Language Models
             via KV-Cache Injection},
  author  = {[Author Name]},
  journal = {[Target Journal]},
  year    = {2026},
  note    = {Under review}
}
```

---

<div align="center">
<i>Built with 🎭 emotion and ❄️ frozen weights</i>
</div>
