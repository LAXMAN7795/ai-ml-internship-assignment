# AI/ML Engineering Internship Assignments

A collection of practical AI/ML engineering assignments covering Large Language Model fine-tuning, Retrieval-Augmented Generation (RAG), and local LLM inference.

This repository documents the implementation, experiments, evaluation results, and technical learnings from the internship assignments.

## Assignments

### Task 1 – Fine-Tuning a Small Language Model Using LoRA

**Status:** Completed

Fine-tuned Qwen2.5-0.5B-Instruct using Low-Rank Adaptation (LoRA) for banking customer support intent classification.

Key highlights:
- Banking77 dataset with six selected intent categories.
- 765 training, 86 validation and 240 test examples.
- LoRA rank: 8; alpha: 16.
- Only 0.2184% of model parameters were trainable.
- Compared original and fine-tuned model performance.
- Analysed classification failures and theoretical GPU memory requirements.

**Final test accuracy: 94.58%**

[View Task 1](./Task1-LoRA-Fine-Tuning/README.md)

### Task 2 – Build RAG from Scratch

**Status:** Pending

Planned implementation:
- Document ingestion and processing.
- Text chunking and embeddings.
- Retrieval implementation without LangChain or LlamaIndex.
- Recall@5 evaluation.
- Retrieval failure analysis.

[View Task 2](./Task2-RAG-From-Scratch/README.md)

### Task 3 – Local LLM Inference and Quantization

**Status:** Pending

Planned implementation:
- Run a local language model.
- Compare two quantization configurations.
- Measure inference speed and memory usage.
- Evaluate prompt quality.
- Analyse KV cache and quantization.
- Estimate inference capacity for multiple concurrent users.

[View Task 3](./Task3-Local-LLM-Inference/README.md)

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT
- Hugging Face Datasets
- Scikit-learn
- NumPy
- Pandas
- Google Colab
- Git and GitHub

Additional libraries will be documented as the assignments progress.

## Repository Structure

```text
ai-ml-internship-assignment/
├── Task1-LoRA-Fine-Tuning/
├── Task2-RAG-From-Scratch/
├── Task3-Local-LLM-Inference/
├── README.md
├── requirements.txt
└── .gitignore
```

## Execution Environment

Experiments are developed and executed in Google Colab. Notebooks, source code, evaluation results and relevant artifacts are organised locally using Visual Studio Code and maintained in this repository.

## Author

**Laxman Sannu Gouda**

- GitHub: https://github.com/LAXMAN7795
- LinkedIn: https://www.linkedin.com/in/laxman-gouda/

---

This repository is intended for educational and technical demonstration purposes.
