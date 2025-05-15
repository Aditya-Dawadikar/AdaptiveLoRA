import os
import gc
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score


class AdaptiveLoRALinear(nn.Module):
    def __init__(self, base_layer, r=4, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.randn(r, base_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(base_layer.out_features, r) * 0.01)

        device = self.lora_A.device
        self.initial_A = self.lora_A.detach().clone().to(device)
        self.initial_B = self.lora_B.detach().clone().to(device)

        self.grad_norm_history = []
        self.weight_change_history = []

        self.lora_A.register_hook(self._capture_grad_hook('A'))
        self.lora_B.register_hook(self._capture_grad_hook('B'))

    def forward(self, x):
        result = self.base(x)
        if self.r > 0:
            lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
            result += self.scaling * lora_out
        return result

    def _capture_grad_hook(self, which):
        def hook(grad):
            norm = grad.norm().item()
            if which == 'A':
                self.grad_norm_history.append(('A', norm))
            elif which == 'B':
                self.grad_norm_history.append(('B', norm))
        return hook

    def compute_weight_change(self):
        delta_A = (self.lora_A - self.initial_A).norm().item()
        delta_B = (self.lora_B - self.initial_B).norm().item()
        total_change = delta_A + delta_B
        self.weight_change_history.append(total_change)
        return total_change

    def average_grad_norm(self):
        norms = [n for (_, n) in self.grad_norm_history if np.isfinite(n)]
        return sum(norms) / len(norms) if norms else 0.0


class AdaptiveLoRA:
    def __init__(self, model, train_dataset, val_dataset, r=4, alpha=16, dropout=0.05,
                 monitor_alpha=0.5, monitor_beta=0.5, min_r=2, max_r=16,
                 warmup_steps=200, num_train_epochs=3):
        self.model = self._inject_lora(model, r, alpha, dropout)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.monitor_alpha = monitor_alpha
        self.monitor_beta = monitor_beta
        self.min_r = min_r
        self.max_r = max_r
        self.warmup_steps = warmup_steps
        self.num_train_epochs = num_train_epochs

    def _inject_lora(self, model, r, alpha, dropout):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and ('q_lin' in name or 'v_lin' in name):
                parent = model
                for part in name.split('.')[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, name.split('.')[-1], AdaptiveLoRALinear(module, r, alpha, dropout))
        self._freeze_except_lora(model)
        return model.to("cuda")

    def _freeze_except_lora(self, model):
        for name, param in model.named_parameters():
            if 'lora_' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _assign_adaptive_ranks(self):
        modules = [m for m in self.model.modules() if isinstance(m, AdaptiveLoRALinear)]
        grad_scores = [m.average_grad_norm() for m in modules]
        weight_scores = [m.compute_weight_change() for m in modules]

        def normalize(scores):
            finite = [s for s in scores if np.isfinite(s)]
            min_s, max_s = min(finite), max(finite)
            return [(s - min_s) / (max_s - min_s) if np.isfinite(s) and max_s > min_s else 0.5 for s in scores]

        norm_grads = normalize(grad_scores)
        norm_weights = normalize(weight_scores)
        scores = [self.monitor_alpha * g + self.monitor_beta * w for g, w in zip(norm_grads, norm_weights)]

        min_score, max_score = min(scores), max(scores)
        for i, m in enumerate(modules):
            norm = (scores[i] - min_score) / (max_score - min_score) if max_score > min_score else 0.5
            new_r = int(self.min_r + norm * (self.max_r - self.min_r))
            new_r = max(self.min_r, min(self.max_r, new_r))

            device = m.lora_A.device
            m.lora_A = nn.Parameter(torch.randn(new_r, m.base.in_features, device=device) * 0.01)
            m.lora_B = nn.Parameter(torch.randn(m.base.out_features, new_r, device=device) * 0.01)
            m.initial_A = m.lora_A.detach().clone().to(device)
            m.initial_B = m.lora_B.detach().clone().to(device)
            m.r = new_r
            m.grad_norm_history.clear()
            m.weight_change_history.clear()

    def _train(self, trainer):
        trainer.train()
        outputs = trainer.predict(self.val_dataset)
        preds = np.argmax(outputs.predictions, axis=-1)
        acc = accuracy_score(self.val_dataset['labels'], preds)
        return acc

    def train(self):
        warmup_args = TrainingArguments(
            output_dir="./adaptive_lora_warmup",
            max_steps=self.warmup_steps,
            per_device_train_batch_size=16,
            learning_rate=5e-4,
            weight_decay=0.01,
            logging_steps=50,
            report_to="none",
            eval_strategy="no",
            fp16=True
        )

        trainer = Trainer(
            model=self.model,
            args=warmup_args,
            train_dataset=self.train_dataset
        )

        trainer.train()
        self._assign_adaptive_ranks()

        finetune_args = TrainingArguments(
            output_dir="./adaptive_lora_finetune",
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=5e-4,
            weight_decay=0.01,
            logging_steps=50,
            report_to="none",
            evaluation_strategy="epoch",
            save_strategy="no",
            fp16=True
        )

        trainer = Trainer(
            model=self.model,
            args=finetune_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=-1))}
        )

        return self._train(trainer)
