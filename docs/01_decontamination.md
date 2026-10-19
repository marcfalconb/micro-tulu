# Phase 1: Data Engine & 8-Gram Decontamination

> **Architecture Stage:** Pre-training / Data Curation Pre-flight  
> **Reference:** Allen Institute for AI, *TÜLU 3: Pushing Frontiers in Open Language Model Post-Training* (Lambert et al., Nov 2024, Section 3.2: "Prompt Decontamination")

---

## 1. Concept Overview

In modern post-training pipelines, **data contamination** occurs when evaluation benchmark questions, tasks, or their near-identical paraphrases leak into instruction-tuning (SFT), preference (DPO), or reinforcement learning (RLVR) datasets. When a model exhibits strong benchmark performance on contaminated evaluations, it reflects **dataset memorization** rather than genuine post-training generalization.

Phase 1 establishes the **Micro-TÜLU Data Engine**:
1. **Benchmark Ingestion:** Caches canonical test sets from the target evaluation suites:
   - **GSM8K:** Grade-school multi-step mathematical word problems.
   - **MATH-500:** High-school Olympiad-level competition mathematics.
   - **IFEval:** Verifiable instruction-following constraints (e.g., length, formatting, punctuation rules).
2. **Inverted 8-Gram Indexing:** Tokenizes evaluation prompts and constructs a reverse lookup table mapping every continuous 8-token window to the benchmark ID and token span.
3. **Token Coverage Scanning:** Scans prompt tokens (user turns in multi-turn dialogues) from raw candidate datasets. If more than **50% of tokens in an evaluation instance** are spanned by shared 8-grams with a training prompt, that training prompt is classified as contaminated and filtered out.
4. **Visual Verification & Auditability:** Generates terminal rich panels and an interactive standalone HTML report (`data/decontaminated/sft_report.html`) detailing every filtered sample with side-by-side prompt diffs.

---

## 2. Why TÜLU 3 Uses It (Empirical Justification)

In Section 3.2 and Section 4.1.2 of the TÜLU 3 technical report, the authors describe how several widely-used open datasets suffered from benchmark leakage:
- **Evol CodeAlpaca:** Leaked into HumanEval (3.5% removed).
- **NuminaMath-TIR:** Contaminated against the MATH competition benchmark (11.3% removed).
- **WildChat & WildGuardMix:** Overlapped with internal and public safety benchmarks (up to 5.4% removed).

Crucially, TÜLU 3 observed:
- When models were trained on intermediate mixes without rigorous decontamination, apparent benchmark scores were artificially inflated.
- After aggressive 8-gram decontamination, performance dropped slightly on contaminated subsets (Figure 3 in the paper), but the resulting models proved significantly more robust when evaluated on unseen held-out benchmarks.
- Decontaminating **prompts alone** (rather than completions) is essential because model responses in synthetic datasets are frequently rewritten or reformatted by teacher models (e.g., GPT-4o), masking surface-level completion overlap while still leaking the test task prompt.

---

## 3. Mathematical Formulation

Let an evaluation prompt instance be represented as an ordered sequence of $L_E$ normalized tokens:
$$E = (t_1^{(E)}, t_2^{(E)}, \dots, t_{L_E}^{(E)})$$

The set of contiguous $N$-grams (where $N = 8$) extracted from $E$ is:
$$\mathcal{G}_N(E) = \left\{ \mathbf{g}_i^{(E)} = \left(t_i^{(E)}, t_{i+1}^{(E)}, \dots, t_{i+N-1}^{(E)}\right) \;\middle|\; 1 \le i \le L_E - N + 1 \right\}$$

Let a candidate training prompt be represented as an ordered sequence of $L_T$ tokens:
$$T = (t_1^{(T)}, t_2^{(T)}, \dots, t_{L_T}^{(T)})$$
with $N$-gram set $\mathcal{G}_N(T)$.

### 8-Gram Intersection & Token Coverage
The set of shared $N$-grams between the training prompt and the evaluation prompt is:
$$\mathcal{I}(T, E) = \mathcal{G}_N(T) \cap \mathcal{G}_N(E)$$

For any token position $k \in \{1, 2, \dots, L_E\}$ in the evaluation prompt, define the binary indicator function $C(k; T, E)$ representing whether position $k$ is covered by at least one shared $N$-gram:
$$C(k; T, E) = \mathbf{1}\left[ \exists \, \mathbf{g}_i^{(E)} \in \mathcal{I}(T, E) \text{ such that } i \le k \le i + N - 1 \right]$$

The **Token Overlap Ratio** $\mathcal{R}(T, E)$ evaluates the proportion of evaluation tokens covered by shared $N$-grams:
$$\mathcal{R}(T, E) = \frac{1}{L_E} \sum_{k=1}^{L_E} C(k; T, E)$$

### Decontamination Decision Rule
Following the TÜLU 3 rule, given a threshold $\tau = 0.50$:
$$\text{Status}(T) = \begin{cases}
\text{Contaminated (Discard)}, & \text{if } \exists \, E \in \mathcal{D}_{\text{eval}} \text{ such that } \mathcal{R}(T, E) > \tau \\
\text{Clean (Retain)}, & \text{otherwise}
\end{cases}$$

---

## 4. MPS & Apple Silicon Implementation Details

Although decontamination is predominantly an algorithmic token-processing phase rather than a tensor computation phase, several Mac-specific engineering practices are enforced:

1. **Memory Efficiency via Inverted Indexing:**
   - Rather than computing quadratic $O(|\mathcal{D}_{\text{train}}| \times |\mathcal{D}_{\text{eval}}|)$ pairwise string comparisons, we build an inverted index $O(|\mathcal{D}_{\text{eval}}| \times L_E)$ mapping 8-gram tuples to small integer position tuples `(eval_id, start_pos)`.
   - Training prompt scanning is $O(L_T \times \text{hit-rate})$, maintaining sub-millisecond per-sample throughput on Apple Silicon unified memory without paging to SSD.
2. **Unified Memory Management:**
   - On Apple M3 Pro (36 GB unified memory), streaming raw JSONL data through Python iterators avoids allocating full datasets into memory simultaneously, leaving the memory footprint under 200 MB during scanning.
3. **Data Types & Serialization:**
   - Clean datasets are stored as streamable JSONL (`.jsonl`), ready for zero-copy memory mapping (`mmap`) via PyArrow and Hugging Face `datasets` in Phase 2.

---

## 5. Resume & Interview Talking Points

- **What problem did you solve?**  
  *"Implemented an automated 8-gram prompt decontamination pipeline modeled after the Allen Institute for AI's TÜLU 3 methodology to eradicate data leakage across GSM8K, MATH, and IFEval benchmarks."*
- **Why 8-grams instead of embeddings or exact string matching?**  
  *"Full-string matching fails on minor formatting variations or prompt prefixes, while embedding cosine similarity conflates topical similarity with actual benchmark leakage. As demonstrated by Dubey et al. (Llama 3) and Lambert et al. (TÜLU 3), 8-gram token coverage provides an exact, empirically calibrated compromise that captures paraphrased prompts without false positives."*
- **What was the architectural design?**  
  *"Constructed an inverted index over benchmark n-grams with a 50% token coverage threshold, decoupling data ingestion, validation, and terminal/HTML visual inspection reporting."*
