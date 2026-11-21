# AI/ML Internship Assignment – End-to-End Projects

A collection of three practical AI/ML projects covering **LoRA fine-tuning, Retrieval-Augmented Generation (RAG), and local Large Language Model (LLM) inference with quantization**.

The projects demonstrate hands-on experience with small language models, NLP, embeddings, vector retrieval, model evaluation, GPU memory optimization, and inference benchmarking.

---

## Projects Overview

| Task | Project | Main Objective | Status |
|---|---|---|---|
| 1 | LoRA Fine-Tuning | Fine-tune a small LLM for banking intent classification | Completed |
| 2 | RAG From Scratch | Build and evaluate a RAG pipeline without LangChain or LlamaIndex | Completed |
| 3 | Local LLM Inference | Compare FP16 and 4-bit quantization performance | Experiments completed |

---

# Task 1: LoRA Fine-Tuning for Banking Intent Classification

## Objective

Fine-tune a small open-weight language model using Low-Rank Adaptation (LoRA) for a narrow banking customer-support intent classification task.

**Base Model:** `Qwen/Qwen2.5-0.5B-Instruct`

**Dataset:** Banking77 – selected six intent categories.

### Selected Intent Categories

1. `cash_withdrawal_charge`
2. `cash_withdrawal_not_recognised`
3. `declined_cash_withdrawal`
4. `pending_cash_withdrawal`
5. `failed_transfer`
6. `card_swallowed`

### Dataset Split

| Dataset | Samples |
|---|---:|
| Training | 765 |
| Validation | 86 |
| Testing | 240 |
| **Total** | **1,091** |

The test set contained 40 examples per class and was held out for final evaluation.

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Dropout | 0.05 |
| Learning rate | 0.0002 |
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Optimizer | AdamW |
| Weight decay | 0.01 |

### Training Summary

- Optimizer steps: 144
- Training duration: 160.51 seconds
- Trainable parameters: 1,081,344
- Total parameters: 495,114,112
- Trainable parameter percentage: 0.2184%
- Best validation checkpoint: Epoch 2

## Evaluation Results

| Metric | Base Model | Fine-Tuned Model |
|---|---:|---:|
| Accuracy | 54.17% | **94.58%** |
| Macro Precision | 57.74% | **94.68%** |
| Macro Recall | 54.17% | **94.58%** |
| Macro F1 | 52.65% | **94.55%** |

### Key Findings

- Test accuracy improved by 40.41 percentage points.
- Macro F1 improved by 41.90 percentage points.
- The fine-tuned model correctly classified 227 out of 240 test examples.
- Only 0.2184% of the model parameters were trainable through LoRA.

**Conclusion:** Parameter-efficient fine-tuning substantially improved classification performance on the selected banking intent dataset.

[View Task 1 – LoRA Fine-Tuning](./Task1-LoRA-Fine-Tuning/)

---

# Task 2: Retrieval-Augmented Generation (RAG) From Scratch

## Objective

Build a complete RAG pipeline from scratch using Python without using LangChain or LlamaIndex.

The system retrieves relevant document passages and uses a language model to generate answers based on the retrieved context.

## Source Document

**Document:** NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0)

**Organization:** National Institute of Standards and Technology (NIST)

**Document Length:** 47 pages

**Official PDF:** https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf

## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| PDF extraction | PyMuPDF |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Embedding dimensions | 384 |
| Similarity calculation | NumPy cosine similarity |
| Generation model | Qwen/Qwen2.5-0.5B-Instruct |
| Retrieval | Dense vector retrieval |
| Evaluation | Source-page Recall@5 |

## RAG Pipeline

1. Load the NIST PDF document.
2. Extract text page by page while preserving page numbers.
3. Split extracted text into overlapping chunks.
4. Generate vector embeddings for each chunk.
5. Convert the user question into an embedding.
6. Calculate cosine similarity between query and document embeddings.
7. Retrieve the top five relevant chunks.
8. Pass retrieved context and the question to the language model.
9. Generate an answer using the retrieved context and provide source-page information.

## Chunking Experiment

Two chunk configurations were evaluated with an overlap of 40 words.

| Parameter | Configuration A | Configuration B |
|---|---:|---:|
| Chunk size | 200 words | 400 words |
| Overlap | 40 words | 40 words |
| Total chunks | 114 | 66 |
| Evaluation questions | 15 | 15 |
| Top-K | 5 | 5 |

## Retrieval Evaluation Results

| Metric | 200-word Chunks | 400-word Chunks |
|---|---:|---:|
| Successful retrievals | 14/15 | 12/15 |
| Failed retrievals | 1 | 3 |
| Source-page Recall@5 | **93.33%** | **80.00%** |

The 200-word chunk configuration outperformed the 400-word configuration by 13.33 percentage points.

**Evaluation note:** A question counted as successfully retrieved when at least one of the top five retrieved chunks came from its expected source PDF page. This measures source-page Recall@5, not exact passage-level relevance.

## Retrieval Failure Analysis

Two important failure cases were analyzed:

- **Main purpose of the NIST AI RMF:** The expected page was not retrieved in the top five for either chunk configuration.
- **MANAGE function responsibilities:** With 400-word chunks, the expected page was missed because other semantically related passages ranked higher.

### Proposed Improvements

- Remove title-only and low-information chunks.
- Improve chunk boundaries.
- Combine BM25 keyword retrieval with dense vector retrieval.
- Add a cross-encoder reranker.
- Improve query formulation and expansion.

## BM25 and Reranking

**BM25:** A keyword-based retrieval algorithm that ranks documents using query-term relevance. It is useful for exact terms, technical terminology, and acronyms.

**Reranking:** A second-stage retrieval process that scores query-passage pairs and reorders retrieved candidates based on relevance.

These techniques were discussed as potential improvements; they were not part of the measured baseline retrieval experiment.

## Generation Quality Observations

Qwen2.5-0.5B-Instruct was used to generate answers from retrieved context.

The model produced useful answers for several questions but sometimes generated unsupported or incomplete details.

This demonstrated that successful retrieval does not automatically guarantee faithful answer generation.

[View Task 2 – RAG From Scratch](./Task2-RAG-From-Scratch/)

---

# Task 3: Local LLM Inference and Quantization

## Objective

Run a local language model using two different precision settings and compare inference speed, GPU memory consumption, and generated outputs.

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`

**Execution Environment:** Google Colab

**GPU:** NVIDIA Tesla T4

**GPU Memory:** 14.56 GB

## Technology Stack

| Component | Version / Technology |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.22.1 |
| Accelerate | 1.15.0 |
| Bitsandbytes | 0.50.2 |
| CUDA | Available |

## Quantization Configurations

### Configuration A: FP16

- Precision: 16-bit floating point
- Data type: `torch.float16`
- Quantization: None

### Configuration B: 4-bit Quantization

- Library: Bitsandbytes
- Quantization type: NF4
- Double quantization: Enabled
- Compute data type: FP16

## Benchmark Methodology

The same 10 prompts were tested using both configurations.

Topics included:

- Artificial intelligence
- Machine learning and deep learning
- Python programming
- Overfitting
- SQL joins
- Retrieval-Augmented Generation
- Precision and recall
- Cloud computing
- Generative AI
- Transformer architecture

Generation settings:

| Parameter | Value |
|---|---|
| Maximum new tokens | 128 |
| Decoding strategy | Greedy decoding |
| Sampling | Disabled |
| Warm-up | One generation before measurement |
| Prompts | 10 |

## Benchmark Results

| Metric | FP16 | 4-bit NF4 |
|---|---:|---:|
| Prompts tested | 10 | 10 |
| Total output tokens | 1,280 | 1,255 |
| Total generation time | 39.89 seconds | 85.02 seconds |
| Overall generation speed | **32.09 tokens/sec** | **14.76 tokens/sec** |
| Peak allocated GPU memory | 2.89 GB | 1.09 GB |

### Key Findings

- 4-bit quantization reduced peak allocated GPU memory by approximately 62.3%.
- FP16 achieved approximately 2.17 times the generation speed of 4-bit in this experiment.
- 4-bit quantization reduced memory requirements but was slower on the Tesla T4 for this workload.

The first 4-bit prompt generated 103 tokens, while the other nine generated 128 tokens each.

## Understanding Quantization

Quantization reduces the numerical precision used to represent model weights.

- FP16 uses 16-bit values.
- 4-bit quantization uses a lower-bit representation with additional metadata and runtime overhead.
- Quantization can reduce GPU memory consumption.
- Inference speed depends on hardware and implementation; lower precision does not always mean faster inference.

## Understanding KV Cache

KV cache stands for Key-Value cache.

Transformer models store previously computed attention keys and values during autoregressive generation.

Benefits:

- Avoids repeatedly calculating attention information for previous tokens.
- Improves generation efficiency.
- Supports efficient autoregressive decoding.

KV cache consumes GPU memory. Longer contexts and more active requests require additional cache capacity.

## Proposed Architecture for 20 Concurrent Users on a 24 GB GPU

The following is a proposed deployment approach, not a measured load-test result.

| Component | Proposed Approach |
|---|---|
| Model | Suitable quantized LLM |
| Inference server | vLLM |
| Request handling | Continuous batching |
| KV cache | PagedAttention |
| Memory management | Controlled context and output lengths |
| Request management | Queue and admission control |
| Monitoring | GPU memory, latency and throughput |

Actual capacity depends on model size, sequence lengths, KV cache requirements, batching, and GPU performance. A concurrent load test is required to validate 20-user capacity.

[View Task 3 – Local LLM Inference](./Task3-Local-LLM-Inference/)

---

# Repository Structure

```text
ai-ml-internship-assignment/
│
├── Task1-LoRA-Fine-Tuning/
│   ├── data/
│   ├── results/
│   ├── notebooks/
│   └── README.md
│
├── Task2-RAG-From-Scratch/
│   ├── data/
│   │   └── documents/
│   ├── results/
│   ├── notebooks/
│   └── README.md
│
├── Task3-Local-LLM-Inference/
│   ├── results/
│   │   ├── prompt_comparison.csv
│   │   ├── prompt_comparison.json
│   │   └── metrics_summary.json
│   ├── notebooks/
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure above describes the intended repository organization. Include only files that are actually present in your repository.*

---

# Overall Learning Outcomes

Through these three projects, I gained practical experience in:

- Fine-tuning open-weight language models using LoRA.
- Preparing and evaluating classification datasets.
- Measuring accuracy, precision, recall, and F1-score.
- Extracting PDF documents and building a RAG pipeline.
- Text chunking, embeddings, and cosine similarity.
- Evaluating retrieval performance using Recall@5.
- Analyzing retrieval failures and answer-grounding issues.
- Understanding BM25 and cross-encoder reranking.
- Running local LLM inference.
- Comparing FP16 and 4-bit quantization.
- Measuring inference speed and GPU memory consumption.
- Understanding KV cache and multi-user inference architecture.

---

# Final Conclusion

These projects demonstrate practical implementation and evaluation across three important areas of modern AI engineering: parameter-efficient fine-tuning, retrieval-augmented generation, and optimized local LLM inference.

The experiments showed that LoRA fine-tuning significantly improved banking intent classification, smaller chunks achieved better source-page retrieval performance in the evaluated RAG dataset, and 4-bit quantization reduced GPU memory consumption while FP16 delivered higher inference throughput on the tested Tesla T4.

The work also highlighted practical challenges such as ambiguous classification examples, retrieval failures, hallucinations in small language models, and the need to validate concurrent inference capacity through load testing.

**All reported metrics represent recorded experiments. Proposed improvements and deployment plans are identified separately from measured results.**

---

## Author

**Laxman Sannu Gouda**

- [LinkedIn](https://www.linkedin.com/in/laxman-gouda/)
- [GitHub](https://github.com/LAXMAN7795)
- [Portfolio](https://laxman7795.github.io/Portfolio-Laxman-Sannu-Gouda/)
