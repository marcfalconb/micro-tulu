# Phase 5: Standardized Evaluation Harness

> **Architecture Stage:** Post-Training Stage 4 (Benchmarking & Progression Tracking)  
> **Reference:** Allen Institute for AI, *TÜLU 3: Pushing Frontiers in Open Language Model Post-Training* (Lambert et al., Nov 2024, Section 2.2: "TÜLU 3 Evaluation", Section 7: "Overall Evaluation", Appendix F: "Evaluation Principles")

---

## 1. Concept Overview

A fundamental challenge in language model post-training is determining whether training improvements reflect **true capability generalization** or merely benchmark overfitting and style mimicry.

Phase 5 establishes the **Micro-TÜLU Standardized Evaluation Harness**:
1. **Bipartite Evaluation Architecture:**
   - **Development Suite:** Evaluates skills directly optimized during training using decontaminated test splits:
     * **GSM8K:** Multi-step arithmetic and algebraic word problem solving.
     * **MATH-500:** Complex high-school and Olympiad mathematics.
     * **IFEval:** Strict, verifiable instruction-following constraints.
   - **Unseen (Held-out) Suite:** Evaluates multi-task knowledge and out-of-distribution reasoning that the model was never directly trained on:
     * **MMLU:** Massive Multitask Language Understanding across STEM, Humanities, and Social Sciences.
2. **Deterministic & CoT Assessment:**
   - Evaluates models using greedy decoding (`temperature=0.0`) to produce deterministic, reproducible generations.
   - Extracts responses using regex and LaTeX parsers, checking mathematical equivalence via **SymPy** and verifying IFEval rules programmatically.
3. **End-to-End Progression Scoreboard:**
   - Simultaneously benchmarks all model checkpoints across the post-training lifecycle:
     $$\text{Base Foundation} \longrightarrow \text{SFT (Stage 1)} \longrightarrow \text{DPO (Stage 2)} \longrightarrow \text{RLVR (Stage 3)}$$
   - Generates an interactive visual HTML scoreboard (`docs/evaluation_scoreboard.html`) with comparative SVG bar charts, metric tables, and qualitative cross-stage response diffs.

---

## 2. Why TÜLU 3 Uses It (Empirical Justification)

In Section 2.2 and Section 7 of the TÜLU 3 report, the authors establish crucial principles for evaluating post-trained models:
- **Separation of Development and Unseen Suites:**
  - Optimizing solely against development benchmarks leads to overfitting on prompt distributions. Tracking performance on held-out tasks (like MMLU and safety benchmarks) ensures the post-training pipeline expands capabilities without catastrophic forgetting.
- **TÜLU 3 Progression Trajectory (Table 1 & Figure 2 in paper):**
  - **Base Model $\rightarrow$ SFT:** Triggers the largest discrete jump in instruction following (+30–40% on IFEval) and chat usability.
  - **SFT $\rightarrow$ DPO:** Further refines response quality, aligns safety/helpfulness, and boosts win-rates on human preferences.
  - **DPO $\rightarrow$ RLVR:** Specifically elevates reasoning and precision on verifiable tasks (boosting GSM8K from 76% to >81%) without degrading MMLU or conversational safety.
- **Generative vs. Multiple-Choice Evaluation:**
  - Rather than assessing log-likelihood perplexity over multiple-choice labels (which rewards superficial token priors), TÜLU 3 emphasizes generative evaluation where models generate complete answers followed by strict verification.

---

## 3. Mathematical Formulation

### Benchmark Metrics & Macro-Averaging
Let the evaluation benchmark set be partitioned into the Development suite $\mathcal{B}_{\text{dev}}$ and Unseen suite $\mathcal{B}_{\text{unseen}}$:
$$\mathcal{B} = \mathcal{B}_{\text{dev}} \cup \mathcal{B}_{\text{unseen}} = \{\text{GSM8K}, \text{MATH-500}, \text{IFEval}\} \cup \{\text{MMLU}\}$$

For each benchmark $b \in \mathcal{B}$ comprising $N_b$ test instances $\{(x_i, y_i^*)\}_{i=1}^{N_b}$, the model $\pi$ generates completion $\hat{y}_i = \text{argmax}_{y} \pi(y \mid x_i)$.

The accuracy for benchmark $b$ is:
$$\text{Acc}_b(\pi) = \frac{1}{N_b} \sum_{i=1}^{N_b} \mathbf{1}\left[ \mathcal{V}_b(\hat{y}_i, y_i^*) = 1 \right]$$
where $\mathcal{V}_b$ is the deterministic domain verifier function:
$$\mathcal{V}_{\text{math}}(\hat{y}, y^*) = \mathbf{1}\left[ \text{SymPy}\left(\text{extract}(\hat{y}) - y^*\right) == 0 \right]$$
$$\mathcal{V}_{\text{ifeval}}(\hat{y}, c) = \mathbf{1}\left[ \text{satisfies}(\hat{y}, c) \right]$$
$$\mathcal{V}_{\text{mmlu}}(\hat{y}, y^*) = \mathbf{1}\left[ \text{extract\_choice}(\hat{y}) == y^* \right]$$

### Macro-Averaged Suite Scores
$$\text{Score}_{\text{dev}}(\pi) = \frac{1}{|\mathcal{B}_{\text{dev}}|} \sum_{b \in \mathcal{B}_{\text{dev}}} \text{Acc}_b(\pi)$$
$$\text{Score}_{\text{unseen}}(\pi) = \frac{1}{|\mathcal{B}_{\text{unseen}}|} \sum_{b \in \mathcal{B}_{\text{unseen}}} \text{Acc}_b(\pi)$$
$$\text{Score}_{\text{overall}}(\pi) = \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \text{Acc}_b(\pi)$$

The **Post-Training Capability Delta** between stage $k$ and stage $k-1$ is:
$$\Delta \text{Score}_k = \text{Score}_{\text{overall}}(\pi_k) - \text{Score}_{\text{overall}}(\pi_{k-1})$$

---

## 4. MPS & Apple Silicon Implementation Details

1. **Sequential Model Loading & Unloading:**
   - To benchmark four successive checkpoints (`Base`, `SFT`, `DPO`, `RLVR`), loading all models into Apple Silicon unified memory simultaneously would require $>16\text{ GB}$ of static parameter weights plus generation KV caches.
   - `src/eval/evaluate.py` iterates over stages sequentially:
     - Loads model and LoRA adapter weights in native `torch.bfloat16`.
     - Executes greedy batched generation on MPS with SDPA.
     - Unloads weights, deletes references, and calls `torch.mps.empty_cache()` before instantiating the subsequent stage.
   - This keeps memory usage capped under $3.5\text{ GB}$ throughout the entire multi-model evaluation suite.
2. **Deterministic Greedy Decoding:**
   - By enforcing `do_sample=False`, the Metal kernel executes greedy argmax decoding, producing 100% deterministic, reproducible evaluation trajectories.

---

## 5. Resume & Interview Talking Points

- **What did you build for the evaluation stage?**  
  *"Constructed a standardized multi-stage evaluation harness modeled after TÜLU 3, benchmarking model checkpoints across both Development (GSM8K, MATH-500, IFEval) and Unseen held-out suites (MMLU) using SymPy and deterministic rule verifiers."*
- **Why is a bipartite (Dev vs. Unseen) evaluation harness necessary?**  
  *"Measuring progress solely on tasks seen during post-training masks overfitting and fails to detect degradation on core reasoning. A bipartite suite guarantees that capabilities gained in SFT, DPO, and RLVR transfer to unseen tasks without catastrophic forgetting."*
- **How was the progression scoreboard designed?**  
  *"Implemented sequential model benchmarking on Apple Silicon MPS with automated unified memory garbage collection, generating an interactive HTML dashboard featuring multi-bar comparison curves and qualitative response diffs across the entire post-training lifecycle."*
