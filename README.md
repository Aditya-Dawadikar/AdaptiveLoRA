# 🧠 Adaptive LoRA: Dynamic Rank Allocation for Efficient Fine-Tuning

Adaptive LoRA is a novel extension to the popular LoRA (Low-Rank Adaptation) technique for parameter-efficient fine-tuning of large language models. Unlike traditional LoRA, which uses a fixed rank for all adapter layers, Adaptive LoRA dynamically allocates ranks per layer based on real-time training signals — specifically, **gradient norms** and **weight deltas**.

---

## 🔍 Motivation

LoRA injects trainable low-rank matrices into pre-trained transformer layers to reduce the number of trainable parameters. However, using the same rank across all layers can be sub-optimal:

* Lower layers may not need high capacity
* Higher layers could benefit from more parameters
* Static ranks waste resources or miss performance opportunities

**Adaptive LoRA** solves this by **assigning rank per layer**, balancing efficiency and effectiveness.

---

## 🚀 Key Features

* 📉 **Dynamic Rank Assignment** using:

  * Gradient Norm: short-term learning activity
  * Weight Delta: long-term adaptation tracking
* 🧠 **Per-Layer Importance Scoring**: $s_i = \alpha G_i + \beta W_i$
* ⚙️ Plug-and-play with HuggingFace Transformers
* 🧪 Built-in warmup, monitoring, and fine-tuning workflow
* 🪶 \~14% less memory usage than fixed-rank LoRA
* 🎯 Up to **+1.3% accuracy improvement** on SST-2 (DistilBERT)

---

## 📦 Installation

```bash
pip install transformers datasets scikit-learn
```

---

## 🧩 Usage

### 1. Import & Initialize

```python
from adaptive_lora import AdaptiveLoRA
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
trainer = AdaptiveLoRA(model, train_dataset, val_dataset)
accuracy = trainer.train()
```

### 2. Custom Configuration

```python
trainer = AdaptiveLoRA(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    r=4,
    alpha=16,
    monitor_alpha=0.5,
    monitor_beta=0.5,
    min_r=2,
    max_r=16,
    warmup_steps=200,
    num_train_epochs=3
)
```

---

## 📊 Results

| Model         | Accuracy (%) | Time (min) | Memory (MB) | GPU Util (%) |
| ------------- | ------------ | ---------- | ----------- | ------------ |
| LoRA (rank=4) | 89.35        | 5.63       | 1496        | 25.6         |
| Adaptive LoRA | **90.07**    | 5.93       | **1276**    | 33.6         |

✅ Adaptive LoRA improves accuracy and reduces memory with negligible training time overhead.

---

## 🔧 How It Works

1. **Warmup Phase**: Run for `N` steps (e.g., 200–500) and record:

   * Gradient norms per layer
   * Weight deltas (change in weights)

2. **Scoring & Rank Assignment**:

   * Normalize gradients and deltas
   * Compute layer score: $s_i = \alpha G_i + \beta W_i$
   * Map scores to ranks: $r_i \in [r_{\text{min}}, r_{\text{max}}]$

3. **Resume Training**: Replace LoRA adapters with new ranks and train as usual.

---

## 🧪 Experiments

* Dataset: SST-2 (GLUE benchmark)
* Model: DistilBERT-uncased
* Hardware: NVIDIA A100 (Colab Pro)
* Evaluation: Accuracy, Memory Usage, Training Time

---

## ⚗️ Ablation Studies

* **Gradient Norm vs Weight Delta**:

  * Best accuracy with `α=1, β=0`
  * Moderate correlation between metrics (Pearson ≈ −0.37)
* **Warmup Step Size**:

  * Optimal: 200 steps
  * Overfitting observed beyond 500 steps

---

## 📁 Project Structure

```
adaptive_lora/
├── adaptive_lora.py         # Main class and logic, with hooks
└── README.md                # This file
```

---

## 📈 Future Work

* Evaluate on QA/Summarization tasks
* Extend to BERT, RoBERTa, DeBERTa
* Automate α/β selection via reinforcement learning
* Layer saliency-based visualization

---

## 📚 References

* [LoRA: Hu et al., 2021](https://arxiv.org/abs/2106.09685)
* [AdaLoRA: Zhang et al., 2023](https://arxiv.org/abs/2303.10512)
* \[DyLoRA, ALoRA, AutoLoRA, etc.]

---
