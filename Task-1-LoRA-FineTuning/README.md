# Task 1: Fine-Tuning a Small Language Model Using LoRA

## 1. Project Overview

The objective of this task was to fine-tune a small open-weight language model (under 1.5 billion parameters) using Low-Rank Adaptation (LoRA) for a banking customer support intent classification task.

The experiment uses **Qwen2.5-0.5B-Instruct** as the base model and the Banking77 dataset, restricted to six banking-related customer support intents.

The goal was to compare the original model against the LoRA fine-tuned model using a held-out test dataset of 240 examples.

### Key Results

| Metric | Base Model | Fine-Tuned Model |
|---|---:|---:|
| Accuracy | 54.17% | **94.58%** |
| Precision (Macro) | 57.74% | **94.68%** |
| Recall (Macro) | 54.17% | **94.58%** |
| Macro F1-score | 52.65% | **94.55%** |

The fine-tuned model improved accuracy by **40.41 percentage points**, demonstrating the effectiveness of parameter-efficient fine-tuning for this classification task.

---

## 2. Base Model and Technologies

**Base Model:** Qwen/Qwen2.5-0.5B-Instruct

**Fine-Tuning Method:** LoRA (Low-Rank Adaptation)

**Frameworks and Libraries:**
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT
- Hugging Face Datasets
- Scikit-learn
- Pandas
- NumPy
- Accelerate

**Execution Environment:** Google Colab with GPU acceleration.

---

## 3. Dataset Preparation

The experiment uses the official Banking77 dataset, selecting six customer support intents relevant to banking withdrawals and transfers.

### Selected Intent Categories

1. `cash_withdrawal_charge`
2. `cash_withdrawal_not_recognised`
3. `declined_cash_withdrawal`
4. `pending_cash_withdrawal`
5. `failed_transfer`
6. `card_swallowed`

### Dataset Distribution

| Dataset | Number of Examples |
|---|---:|
| Training | 765 |
| Validation | 86 |
| Testing | 240 |
| **Total** | **1,091** |

The test dataset contains 40 examples for each of the six categories.

The training dataset was used for fine-tuning, while the validation dataset was used to monitor training performance and select the best checkpoint.

The 240-example test dataset was held out for the final model comparison.

### Dataset Class Distribution

| Intent | Training | Validation | Testing |
|---|---:|---:|---:|
| cash_withdrawal_charge | 159 | 18 | 40 |
| cash_withdrawal_not_recognised | 144 | 16 | 40 |
| declined_cash_withdrawal | 155 | 18 | 40 |
| pending_cash_withdrawal | 129 | 14 | 40 |
| failed_transfer | 123 | 14 | 40 |
| card_swallowed | 55 | 6 | 40 |
| **Total** | **765** | **86** | **240** |

---

## 4. Baseline Evaluation

Before fine-tuning, the original Qwen2.5-0.5B-Instruct model was evaluated on all 240 test examples.

### Baseline Results

| Metric | Score |
|---|---:|
| Accuracy | 0.5417 |
| Precision | 0.5774 |
| Recall | 0.5417 |
| Macro F1 | 0.5265 |

The original model achieved approximately 54.17% accuracy.

### Baseline Observations

- The model performed relatively well on `failed_transfer` and `pending_cash_withdrawal`.
- It struggled to distinguish between similar withdrawal-related categories.
- The `card_swallowed` category received an F1-score of zero.
- The model showed limitations in understanding banking-specific intent distinctions without task-specific fine-tuning.

These observations motivated the LoRA fine-tuning experiment.

---

## 5. LoRA Configuration

LoRA was selected because it allows efficient fine-tuning without updating all the parameters of the original model.

Instead of training the complete model, LoRA introduces small trainable adapter matrices into selected attention projection modules.

### Hyperparameters

| Parameter | Value |
|---|---|
| Base Model | Qwen2.5-0.5B-Instruct |
| Fine-Tuning Method | LoRA |
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Training Batch Size | 4 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 16 |
| Warmup Steps | 7 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |

### Rationale

- **Rank = 8:** Provides a lightweight adapter with sufficient capacity for the classification task.
- **Alpha = 16:** Provides a LoRA scaling factor of 2 (alpha divided by rank).
- **Dropout = 0.05:** Helps reduce overfitting during adapter training.
- **Target Modules:** The query, key, value and output projection layers were selected to adapt the attention mechanism.
- **Learning Rate = 2e-4:** Used as the starting learning rate for adapter optimisation.
- **Three Epochs:** Allows the model to learn from the training dataset multiple times while monitoring validation loss.

### Trainable Parameters

| Parameter | Count |
|---|---:|
| Total Model Parameters | 495,114,112 |
| Trainable Parameters | 1,081,344 |
| Trainable Percentage | 0.2184% |

Only approximately 0.2184% of the model parameters were trainable.

This demonstrates the parameter efficiency of LoRA.

---

## 6. Training Process and Results

The training dataset was converted into instruction-response examples using the Qwen chat template.

Each example contained:
- A banking customer message.
- An instruction to classify the message.
- The correct intent name as the expected assistant response.

Prompt tokens were masked using `-100` so that training loss was calculated only on the expected answer tokens.

A custom data collator handled padding and batch preparation.

### Training History

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 0.058731 | 0.041999 |
| 2 | 0.032050 | 0.021182 |
| 3 | 0.018942 | 0.024390 |

### Training Metrics

| Metric | Actual Result |
|---|---:|
| Training Runtime | 160.51 seconds |
| Total Optimizer Steps | 144 |
| Training Samples per Second | 14.30 |
| Training Steps per Second | 0.897 |
| Overall Training Loss | 0.046358 |

The validation loss decreased from 0.041999 in epoch 1 to 0.021182 in epoch 2, before increasing to 0.024390 in epoch 3.

The training configuration used `load_best_model_at_end=True` and selected the checkpoint with the lowest validation loss.

The best checkpoint corresponded to epoch 2.

The trained LoRA adapter and tokenizer were saved for reuse.

---

## 7. Fine-Tuned Model Evaluation

After training, the fine-tuned model was evaluated on the same 240 held-out test examples used for baseline evaluation.

### Fine-Tuned Results

| Metric | Score |
|---|---:|
| Accuracy | 0.945833 |
| Precision (Macro) | 0.946836 |
| Recall (Macro) | 0.945833 |
| Macro F1 | 0.945522 |

The fine-tuned model correctly classified 227 out of 240 examples, with 13 incorrect predictions.

### Base vs Fine-Tuned Comparison

| Metric | Base Model | Fine-Tuned Model | Improvement |
|---|---:|---:|---:|
| Accuracy | 0.5417 | 0.945833 | +0.404133 |
| Precision | 0.5774 | 0.946836 | +0.369436 |
| Recall | 0.5417 | 0.945833 | +0.404133 |
| Macro F1 | 0.5265 | 0.945522 | +0.419022 |

The accuracy improvement was approximately **40.41 percentage points**.

The macro F1-score improved from 52.65% to 94.55%.

These results indicate that LoRA fine-tuning substantially improved the model's ability to recognise the selected banking customer support intents.

---

## 8. Failure Analysis

The fine-tuned model made 13 incorrect predictions out of 240 test examples.

Three representative failure cases were examined.

### Failure Case 1: Withdrawal Accounting Confusion

**Customer Message:**

"Where's accounting for my cash withdrawal?"

- Actual Intent: `pending_cash_withdrawal`
- Predicted Intent: `cash_withdrawal_not_recognised`

**Reason for Failure:**

The message is ambiguous and does not explicitly explain whether the withdrawal is pending or unrecognised. The model confused the withdrawal accounting/status query with an unrecognised transaction.

### Failure Case 2: Pending vs Declined Withdrawal

**Customer Message:**

"My account was charged for a withdraw I tried to make that was decline."

- Actual Intent: `pending_cash_withdrawal`
- Predicted Intent: `declined_cash_withdrawal`

**Reason for Failure:**

The message contains conflicting signals: the withdrawal was declined, but the account was charged. The model focused on the declined transaction wording rather than the unresolved charged-withdrawal context.

### Failure Case 3: Failed Transfer vs Declined Withdrawal

**Customer Message:**

"I want to know why I have been unable to withdrawn funds."

- Actual Intent: `declined_cash_withdrawal`
- Predicted Intent: `failed_transfer`

**Reason for Failure:**

The message does not specify the transaction channel or the reason for the withdrawal failure. The model confused a withdrawal failure with a transfer failure.

### Failure Analysis Summary

The remaining classification errors mainly involve:
- Ambiguous customer messages.
- Overlapping banking terminology.
- Confusion between pending and declined transactions.
- Similar language used for failed withdrawals and failed transfers.

Potential improvements include collecting more representative examples, improving class coverage, and testing alternative prompt formats and LoRA configurations.

---

## 9. GPU Memory Calculation for a 7B Model

This section provides a theoretical comparison of full fine-tuning, LoRA and QLoRA for a model containing 7 billion parameters.

Assumptions:
- Model parameters: 7 billion.
- BF16 model weights: 2 bytes per parameter.
- BF16 gradients: 2 bytes per parameter.
- FP32 master weights: 4 bytes per parameter.
- Adam optimiser states: 8 bytes per parameter (two FP32 states).

### A. Full Fine-Tuning

Full fine-tuning updates all model parameters.

Memory calculation:

- Model weights: 7B × 2 bytes = 14 GB.
- Gradients: 7B × 2 bytes = 14 GB.
- Master weights: 7B × 4 bytes = 28 GB.
- Optimiser states: 7B × 8 bytes = 56 GB.

**Total:**

7B × (2 + 2 + 4 + 8) = 112 GB.

This excludes activations, temporary buffers and other runtime overhead.

### B. LoRA Fine-Tuning

LoRA freezes the original model weights and updates only the adapter parameters.

- Frozen BF16 base model weights: 14 GB.
- Adapter weights, gradients and optimiser states: comparatively small.
- Additional activation and runtime memory is required.

The base model weight storage alone is approximately 14 GB.

### C. QLoRA Fine-Tuning

QLoRA uses 4-bit quantization for the frozen base model and trains LoRA adapters.

Memory calculation:

7B × 0.5 bytes = 3.5 GB.

The raw 4-bit base weights require approximately 3.5 GB, excluding quantization metadata, adapters, activations and runtime overhead.

### Comparison

| Method | Base Model Weight Storage | Overall Training Memory |
|---|---:|---|
| Full Fine-Tuning | 14 GB | Approximately 112 GB plus activations |
| LoRA | 14 GB | Base weights plus adapters and activations |
| QLoRA | 3.5 GB raw | Quantized base plus adapters and activations |

The calculations demonstrate why LoRA and QLoRA can make fine-tuning large language models more memory-efficient.

Actual GPU requirements depend on sequence length, batch size, optimiser implementation, checkpointing and quantization configuration.

---

## 10. Conclusion

This experiment demonstrated the effectiveness of parameter-efficient fine-tuning for a narrow banking customer support classification task.

The original Qwen2.5-0.5B-Instruct model achieved 54.17% accuracy on the held-out test set.

After LoRA fine-tuning, accuracy increased to 94.58%, with a macro F1-score of 94.55%.

Only 1,081,344 parameters out of 495,114,112 total parameters were trainable, representing approximately 0.2184% of the model.

The experiment also identified remaining classification challenges involving ambiguous messages and overlapping banking intents.

Overall, the results show that LoRA can significantly improve the performance of a small language model on a specialised classification task without requiring full-model fine-tuning.

---

## 11. Project Artifacts

The Task 1 project includes:

- `Task1_LoRA_FineTuning.ipynb` – implementation and experiment notebook.
- `data/train.csv` – training dataset.
- `data/validation.csv` – validation dataset.
- `data/test.csv` – held-out test dataset.
- `results/fine_tuned_lora_adapter/` – trained LoRA adapter and tokenizer files.
- `results/experiment_metrics.json` – experiment configuration and metrics, if exported.
- `results/training_history.json` – training history, if exported.
- `results/baseline_metrics.csv` – baseline evaluation metrics.
- `results/fine_tuned_predictions.csv` – fine-tuned model predictions.
- `results/model_comparison.csv` – base vs fine-tuned comparison.
- `results/failure_cases.csv` – misclassified test examples.
- `results/failure_analysis.json` – failure analysis.

**Note:** File names listed above should be checked against the actual files in the repository before submission.

### Reproducibility

The experiment was executed in Google Colab using Hugging Face Transformers, PEFT and PyTorch.

The saved LoRA adapter can be loaded on top of the original Qwen2.5-0.5B-Instruct base model for inference without repeating the fine-tuning process.

**Final Result: 94.58% test accuracy using LoRA fine-tuning.**