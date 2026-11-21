# Task 2: Retrieval-Augmented Generation (RAG) From Scratch

## 1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline from scratch using Python, without using frameworks such as LangChain or LlamaIndex.

The system retrieves relevant information from a public document and uses a small open-weight language model to generate answers based on the retrieved context.

**Objective:** Build and evaluate a RAG system, compare retrieval performance using two chunk sizes, analyze retrieval failures, and understand techniques that can improve retrieval quality.

## 2. Dataset and Source Document

**Document:** NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0)

**Source:** National Institute of Standards and Technology (NIST)

**Official PDF:** https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf

- Total pages: 47
- Document type: Public PDF
- Content: AI risk management, trustworthy AI characteristics, governance, mapping, measurement, and management.

The PDF was downloaded and its text was extracted page by page while preserving the original page numbers for source attribution.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Development environment | Google Colab |
| PDF text extraction | PyMuPDF |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Generation model | Qwen/Qwen2.5-0.5B-Instruct |
| Similarity calculation | NumPy cosine similarity |
| Data processing | Python, NumPy |
| Retrieval method | Dense vector retrieval |
| Evaluation | Recall@5 |

No LangChain or LlamaIndex libraries were used.

## 4. RAG Pipeline Architecture

The project follows these steps:

1. **Document Loading:** Load the NIST AI RMF PDF.
2. **Text Extraction:** Extract text from all 47 pages while preserving page numbers.
3. **Chunking:** Split the extracted text into overlapping chunks using two configurations.
4. **Embedding Generation:** Convert each chunk into a 384-dimensional vector using the embedding model.
5. **Vector Retrieval:** Convert the user question into an embedding and calculate cosine similarity against document chunk embeddings.
6. **Top-K Retrieval:** Retrieve the five most relevant chunks.
7. **Answer Generation:** Pass the retrieved context and question to Qwen2.5-0.5B-Instruct.
8. **Source Attribution:** Include the retrieved PDF page numbers with the generated answer.

## 5. Chunking Strategy

Two chunk configurations were tested. Both used an overlap of 40 words.

| Configuration | Chunk Size | Overlap | Total Chunks |
|---|---:|---:|---:|
| Small chunks | 200 words | 40 words | 114 |
| Large chunks | 400 words | 40 words | 66 |

The chunking process preserved PDF page numbers to support source attribution and retrieval evaluation.

## 6. Embedding and Retrieval

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`

- Embedding dimension: 384
- Embeddings were normalized.
- NumPy dot product was used to calculate cosine similarity between query and document embeddings.
- Top 5 chunks were retrieved for each question.

## 7. Evaluation Dataset

A manually prepared evaluation dataset containing 15 questions was used.

Each question included:
- A manually written question.
- An expected answer or relevant source passage.
- The expected PDF page number.

The questions covered the main sections of the NIST AI RMF, including risk, risk tolerance, trustworthy AI, bias, GOVERN, MAP, MEASURE, and MANAGE.

## 8. Retrieval Evaluation Results

Retrieval performance was evaluated using source-page Recall@5.

A question was counted as successfully retrieved when at least one of the five retrieved chunks came from its expected source page.

| Metric | 200-word chunks | 400-word chunks |
|---|---:|---:|
| Total questions | 15 | 15 |
| Successful retrievals | 14 | 12 |
| Failed retrievals | 1 | 3 |
| Recall@5 | 93.33% | 80.00% |

**Result:** The 200-word chunk configuration performed better, achieving a 13.33 percentage-point improvement over the 400-word configuration.

### Why did smaller chunks perform better?

- Smaller chunks preserve more focused information.
- Relevant passages are less likely to be diluted by unrelated content.
- Specific questions can match more precise passages.
- Larger chunks may contain multiple topics, reducing retrieval precision for narrow questions.

For this document and evaluation dataset, 200-word chunks provided better retrieval performance.

## 9. Retrieval Failure Analysis

Two important retrieval failure cases were examined.

### Failure Case 1: Main purpose of the NIST AI RMF

- Question: What is the main purpose of the NIST AI RMF?
- Expected source page: 9
- Retrieved pages: 1, 2, 6, 37, 8
- Result: Expected page was not retrieved in the top five.

**Possible reason:**
The question is broad and semantically related to multiple sections of the document. Title pages and introductory sections also contain related terminology, which can influence similarity rankings.

**Potential improvements:**
- Remove title-only and low-information chunks.
- Improve chunk boundaries.
- Use hybrid keyword and semantic retrieval.
- Apply a reranker to reorder retrieved passages.

### Failure Case 2: Responsibilities of the MANAGE function

- Question: What are the responsibilities of the MANAGE function?
- Expected source page: 37
- Retrieved pages with 400-word chunks: 28, 25, 46, 27, 26
- Result: Expected page was not retrieved in the top five.

**Possible reason:**
The larger chunks contain broader information from multiple sections. Related governance and risk-management terminology may have received higher similarity scores than the specific MANAGE passage.

**Potential improvements:**
- Use smaller chunks for specific questions.
- Improve page-aware chunk boundaries.
- Add BM25 keyword retrieval.
- Use a cross-encoder reranker.

## 10. BM25 and Reranking

### BM25

BM25 is a keyword-based information retrieval algorithm. It ranks documents based on the relevance of query terms to the document text.

**Advantages:**
- Effective for exact keyword matches.
- Useful for acronyms, technical terms, and specific phrases.
- Does not require embedding models.

### Reranking

A reranker is a second-stage retrieval component that evaluates the relevance of the query against retrieved passages.

A cross-encoder reranker processes the query and each candidate passage together and assigns a relevance score.

**Advantages:**
- Improves the ordering of retrieved passages.
- Helps distinguish closely related passages.
- Can improve the quality of the final context passed to the language model.

A hybrid retrieval system combining BM25 and dense vector retrieval, followed by reranking, could improve the current pipeline.

## 11. RAG Answer Generation

**Generation model:** `Qwen/Qwen2.5-0.5B-Instruct`

The generation component receives:
- The user's question.
- The top five retrieved document chunks.
- Instructions to answer using the supplied context.

The model was tested against all 15 evaluation questions.

### Observations

- Several answers correctly captured information from the NIST document.
- Some answers contained unsupported details or overgeneralizations.
- The model struggled with certain questions involving precise definitions and lists.
- Broad questions were more difficult when the expected source page was not retrieved.
- Tightening the prompt improved some answers but did not eliminate hallucinations.

This demonstrates that successful retrieval does not automatically guarantee a fully grounded generated answer.

## 12. Key Findings

1. The RAG pipeline was implemented without LangChain or LlamaIndex.
2. The NIST AI RMF 1.0 document was successfully processed across 47 pages.
3. Two chunk sizes were compared using 15 manually prepared questions.
4. The 200-word chunk configuration achieved 93.33% source-page Recall@5.
5. The 400-word chunk configuration achieved 80.00% source-page Recall@5.
6. Smaller chunks performed better for this document and test dataset.
7. Retrieval failures showed the importance of chunk quality and query relevance.
8. The small generation model sometimes produced unsupported information, even when relevant context was retrieved.

## 13. Limitations

- Evaluation was performed using only one document.
- The evaluation dataset contained 15 manually written questions.
- Recall@5 measured source-page matching, not exact passage-level relevance.
- The system used dense retrieval without BM25 or a reranker in the evaluated pipeline.
- The small generation model did not consistently produce fully grounded answers.
- The page-wise chunking strategy can produce short chunks on pages with limited text.

## 14. Future Improvements

- Implement hybrid retrieval using BM25 and dense embeddings.
- Add a cross-encoder reranker.
- Improve chunking using semantic boundaries.
- Remove low-information title and repeated-header chunks.
- Expand the evaluation dataset.
- Evaluate answer faithfulness and answer correctness separately from retrieval recall.
- Experiment with a stronger generation model.
- Add more robust citation validation.

## 15. Project Structure

```text
Task2-RAG-From-Scratch/
│
├── data/
│   ├── documents/
│   │   └── nist_ai_rmf_1_0.pdf
│   ├── extracted_pages.json
│   ├── extracted_text.txt
│   ├── chunks_200.json
│   ├── chunks_400.json
│   ├── embeddings_200.npy
│   ├── embeddings_400.npy
│   └── evaluation_questions.json
│
├── results/
│   ├── retrieval_results_200.json
│   ├── retrieval_results_400.json
│   ├── rag_answers.json
│   ├── metrics_summary.json
│   └── failure_analysis.md
│
├── notebooks/
│   └── task2_rag_experiment.ipynb
│
└── README.md
```

## 16. Conclusion

A complete basic Retrieval-Augmented Generation pipeline was developed from scratch using Python, sentence-transformer embeddings, NumPy cosine similarity, and Qwen2.5-0.5B-Instruct.

The experiment demonstrated that chunk size has a measurable impact on retrieval performance. The 200-word configuration achieved 93.33% source-page Recall@5, outperforming the 400-word configuration at 80.00%.

The project also highlighted two important challenges in RAG systems: retrieving the correct source passage and generating answers that remain faithful to the retrieved context.

These findings provide a foundation for improving the system through hybrid retrieval, reranking, better chunking, and stronger answer-grounding techniques.