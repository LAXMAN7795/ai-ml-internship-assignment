# Task 3: Local LLM Inference and Quantization

## 1. Project Overview

This project evaluates local Large Language Model (LLM) inference using two different model precision settings:

- FP16 (16-bit floating point)
- 4-bit quantization using bitsandbytes NF4

The objective is to compare inference speed, GPU memory consumption, and generated outputs using the same model and 10 prompts.

The project also explains KV cache management and proposes an approach for serving 20 concurrent users on a 24 GB GPU.

## 2. Hardware and Software Environment

### Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA Tesla T4 |
| GPU memory | 14.56 GB |
| Execution environment | Google Colab |

### Software

| Component | Version / Technology |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.22.1 |
| Accelerate | 1.15.0 |
| Bitsandbytes | 0.50.2 |
| CUDA | Available |

## 3. Model Details

**Model:** Qwen/Qwen2.5-1.5B-Instruct

The same model was used for both experiments to ensure a fair comparison.

Two configurations were evaluated:

### Configuration A: FP16

- Model precision: FP16
- Data type: `torch.float16`
- Device: Tesla T4 GPU
- Quantization: None

### Configuration B: 4-bit Quantization

- Quantization library: bitsandbytes
- Quantization type: NF4
- Double quantization: Enabled
- Compute data type: FP16

## 4. Benchmark Methodology

A set of 10 prompts covering different topics was used for both configurations.

The prompts covered:

1. Artificial intelligence
2. Machine learning and deep learning
3. Python programming
4. Overfitting
5. SQL joins
6. Retrieval-Augmented Generation
7. Precision and recall
8. Cloud computing
9. Generative AI applications
10. Transformer architecture

### Generation settings

| Parameter | Setting |
|---|---|
| Maximum new tokens | 128 |
| Decoding strategy | Greedy decoding |
| Sampling | Disabled |
| Warm-up | One generation before measurement |
| Benchmark prompts | 10 |
| Measurement | Generation time and output tokens |

Generation speed was calculated as:

**Tokens per second = Generated output tokens / Generation time**

Peak GPU allocated memory was recorded during the benchmark.

## 5. Experimental Results

### FP16 Results

| Metric | Result |
|---|---:|
| Prompts tested | 10 |
| Total output tokens | 1,280 |
| Total generation time | 39.89 seconds |
| Overall generation speed | 32.09 tokens/second |
| Peak GPU allocated memory | 2.89 GB |

### 4-bit Quantization Results

| Metric | Result |
|---|---:|
| Prompts tested | 10 |
| Total output tokens | 1,255 |
| Total generation time | 85.02 seconds |
| Overall generation speed | 14.76 tokens/second |
| Peak GPU allocated memory | 1.09 GB |

The first 4-bit prompt generated 103 tokens, while the remaining nine prompts generated 128 tokens each.

## 6. FP16 vs 4-bit Comparison

| Metric | FP16 | 4-bit |
|---|---:|---:|
| Model | Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct |
| Prompts | 10 | 10 |
| Output tokens | 1,280 | 1,255 |
| Generation time | 39.89 s | 85.02 s |
| Tokens/second | 32.09 | 14.76 |
| Peak allocated memory | 2.89 GB | 1.09 GB |

### Key observations

**Memory consumption:**

The 4-bit configuration reduced peak allocated GPU memory from 2.89 GB to 1.09 GB, approximately 62.3% less memory.

**Inference speed:**

FP16 achieved 32.09 tokens/second, while 4-bit achieved 14.76 tokens/second.

FP16 was approximately 2.17 times faster in this experiment.

**Overall finding:**

4-bit quantization significantly reduced GPU memory consumption but also reduced inference speed on the Tesla T4 in this benchmark.

Quantization does not always improve inference speed. Performance depends on GPU architecture, quantization implementation, and computational overhead.

## 7. Understanding Quantization

Quantization reduces the numerical precision used to represent model weights.

### FP16

FP16 represents model weights using 16 bits per value.

Advantages:
- Good inference performance on supported GPUs.
- Higher numerical precision than 4-bit quantization.
- No additional weight quantization overhead.

Disadvantages:
- Requires more memory than 4-bit quantization.

### 4-bit Quantization

4-bit quantization represents model weights using approximately 4 bits per value, with additional metadata and runtime overhead.

Advantages:
- Reduces model memory requirements.
- Allows larger models to fit into limited GPU memory.
- Can make local deployment more accessible.

Disadvantages:
- Can reduce inference speed depending on hardware.
- May introduce numerical approximation and affect output quality.

## 8. Understanding KV Cache

KV cache stands for Key-Value cache.

Transformer models use attention mechanisms that calculate Key and Value representations for tokens.

During autoregressive generation, the model stores previously calculated Key and Value tensors in memory.

### Why is KV cache important?

Without KV caching, the model would repeatedly calculate information for previously processed tokens.

With KV caching:
- Previously calculated Key and Value tensors are reused.
- Repeated computation is reduced.
- Token generation becomes more efficient.

### KV cache memory considerations

- KV cache consumes GPU memory.
- Longer input and output sequences require more cache memory.
- Concurrent requests need separate cache storage for their active sequences.
- Efficient cache management is important when serving multiple users.

## 9. Serving 20 Concurrent Users on a 24 GB GPU

The following is a proposed deployment architecture. It was not load-tested as part of this experiment.

### Proposed approach

| Component | Proposed solution |
|---|---|
| Model | Memory-efficient quantized LLM |
| GPU | 24 GB GPU |
| Inference server | vLLM |
| Request handling | Continuous batching |
| KV cache | PagedAttention |
| Memory optimization | Quantization and controlled context lengths |
| Request management | Queue and admission control |
| Monitoring | GPU utilization, memory, latency and throughput |

### How it would work

1. Load a suitable quantized model into GPU memory.
2. Start an optimized inference server such as vLLM.
3. Accept requests from multiple users.
4. Use continuous batching to process active requests efficiently.
5. Manage KV cache memory using PagedAttention.
6. Set maximum input and output token limits.
7. Queue requests when available GPU capacity is insufficient.
8. Monitor throughput, latency, and GPU memory.

### Important considerations

Supporting 20 concurrent users does not guarantee that all 20 requests can generate tokens simultaneously at maximum speed.

Actual capacity depends on:
- Model size and architecture.
- Input and output sequence lengths.
- KV cache requirements.
- GPU memory and compute performance.
- Batch size and serving configuration.

A load test is required to validate the actual 20-user capacity.

## 10. Limitations

- Experiments were performed on a single Tesla T4 GPU.
- Only one model was evaluated.
- Only 10 prompts were used.
- Each generation was capped at 128 new tokens.
- The first 4-bit prompt generated fewer tokens than the other prompts.
- The benchmark used sequential requests, not concurrent requests.
- The proposed 20-user deployment architecture was not load-tested.
- Generated answer quality was recorded but not independently scored.

## 11. Future Improvements

- Evaluate larger models.
- Compare additional quantization methods such as 8-bit.
- Test different batch sizes.
- Perform concurrent inference benchmarking.
- Measure time to first token (TTFT).
- Measure inter-token latency.
- Evaluate answer quality across precision settings.
- Deploy the model using vLLM and test real concurrent workloads.

## 12. Project Structure

```text
Task3-Local-LLM-Inference/
│
├── notebooks/
│   └── Task3_Local_LLM_Inference.ipynb
│
├── results/
│   ├── prompt_comparison.csv
│   ├── prompt_comparison.json
│   └── metrics_summary.json
│
└── README.md
```

## 13. Conclusion

This experiment compared FP16 and NF4 4-bit inference using Qwen2.5-1.5B-Instruct on an NVIDIA Tesla T4 GPU.

FP16 achieved 32.09 tokens/second with 2.89 GB peak allocated GPU memory. The 4-bit configuration achieved 14.76 tokens/second with 1.09 GB peak allocated GPU memory.

The results demonstrate the memory efficiency of 4-bit quantization and the inference-speed advantage of FP16 in this particular hardware and software environment.

The project also explored KV cache management and a proposed architecture for serving multiple concurrent users on a 24 GB GPU.