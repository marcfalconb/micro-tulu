# Phase 2: Supervised Fine-Tuning (SFT) & Equal Token-Weighting

> **Architecture Stage:** Post-Training Stage 1 (Instruction Adaptation)  
> **Reference:** Allen Institute for AI, *TÜLU 3: Pushing Frontiers in Open Language Model Post-Training* (Lambert et al., Nov 2024, Section 4: "Supervised Finetuning", Section 4.3.2: "Batch Aggregation")

---

## 1. Concept Overview

Supervised Fine-Tuning (SFT)—also known as **instruction tuning**—adapts a raw next-token prediction foundation model (such as `Llama-3.2-1B`) into a conversational assistant capable of following complex multi-turn user requests.

Phase 2 establishes the **Micro-TÜLU SFT Engine**:
1. **Chat Template Formatting:** Applies the exact TÜLU 3 Jinja2 chat template (Appendix B.3, Figure 27) with `<|system|>`, `<|user|>`, and `<|assistant|>` delimiters.
2. **Prompt Masking:** In multi-turn dialogues, system prompts, user questions, and delimiter tokens are masked with label `-100`. Gradients and cross-entropy losses are computed **exclusively on assistant completion tokens**.
3. **Equal Token-Weighting (Sum-Loss):** Standard training libraries average loss per sequence before accumulating gradients, giving short sequences disproportionate weight. TÜLU 3 enforces Sum-Loss, weighting every target token equally across all batches and accumulation steps.
4. **Parameter-Efficient LoRA on MPS:** Fine-tunes low-rank decomposition matrices ($r=16, \alpha=32$) over attention projection weights (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) in native `torch.bfloat16` with SDPA attention on Apple Silicon Unified Memory.
5. **Visual Verification:** Live terminal sparkline loss progression and an interactive standalone HTML report (`checkpoints/sft/training_report.html`) containing an SVG loss trajectory curve and qualitative before-and-after generation comparisons.

---

## 2. Why TÜLU 3 Uses It (Empirical Justification)

In Section 4.3.2 ("Batch Aggregation") of the TÜLU 3 paper, the authors highlight a critical flaw in standard Hugging Face / DeepSpeed training pipelines:
- **The Batch Aggregation Discrepancy:** Transformers default loss averaging computes the mean across active tokens within each sequence/micro-batch, and then averages across micro-batches:
  $$\mathcal{L}_{\text{mean}} = \frac{1}{B} \sum_{b=1}^B \left( \frac{1}{N_b} \sum_{i=1}^{N_b} \ell_{b, i} \right)$$
- If sample 1 has $N_1 = 10$ tokens and sample 2 has $N_2 = 1000$ tokens:
  - Each token in sample 1 receives gradient weight $\frac{1}{10 B}$.
  - Each token in sample 2 receives gradient weight $\frac{1}{1000 B}$ (100× smaller!).
- In addition, changing `gradient_accumulation_steps` or the number of distributed GPUs implicitly changes the token loss weightings, causing performance to diverge across hardware setups (e.g., TPUs vs. GPUs).

TÜLU 3 solved this by switching to **Sum-Loss (Equal Token-Weighting)**:
- Removes the sequence-level normalization denominator, ensuring every token contributes equally regardless of sequence length or batching configuration.
- Empirical results (Figure 5 in the paper) demonstrated that Sum-Loss achieved superior average benchmark performance across GSM8K, IFEval, and MMLU compared to standard Mean-Loss, particularly when combined with an optimal learning rate.

---

## 3. Mathematical Formulation

### Standard Autoregressive Cross-Entropy
Given an input sequence $X = (x_1, x_2, \dots, x_L)$ with target labels $Y = (y_1, y_2, \dots, y_L)$ where $y_i \in \mathcal{V} \cup \{-100\}$, the per-token cross-entropy error at position $t$ is:
$$\ell_t = -\log P_\theta(x_{t+1} = y_t \mid x_1, \dots, x_t)$$

Define the active indicator mask $m_t = \mathbf{1}[y_t \ne -100]$.

### Mean-Loss (Standard Baseline)
For a batch of $B$ sequences, Mean-Loss normalizes within each sequence $b$ by that sequence's active token count $N_b = \sum_{t} m_{b, t}$:
$$\mathcal{L}_{\text{seq}}^{(b)} = \frac{1}{\max(1, N_b)} \sum_{t=1}^{L} m_{b, t} \, \ell_{b, t}$$
$$\mathcal{L}_{\text{mean}} = \frac{1}{B} \sum_{b=1}^B \mathcal{L}_{\text{seq}}^{(b)}$$
Notice the effective gradient weight for a token in sequence $b$:
$$\frac{\partial \mathcal{L}_{\text{mean}}}{\partial \ell_{b, t}} = \frac{1}{B \cdot N_b}$$
Tokens in short responses receive massively amplified gradients compared to tokens in long, complex multi-step reasoning chains.

### TÜLU 3 Sum-Loss (Equal Token-Weighting)
Sum-Loss computes the unnormalized loss sum across all sequences and normalizes globally by the total active tokens $N_{\text{total}} = \sum_{b=1}^B N_b$:
$$\mathcal{L}_{\text{sum}} = \frac{\sum_{b=1}^B \sum_{t=1}^{L} m_{b, t} \, \ell_{b, t}}{\sum_{b=1}^B N_b}$$

Taking the partial derivative with respect to any individual token loss $\ell_{b, t}$:
$$\frac{\partial \mathcal{L}_{\text{sum}}}{\partial \ell_{b, t}} = \frac{1}{N_{\text{total}}}$$
**Theorem:** Under Sum-Loss, every active token across all sequences and accumulation steps has **identical gradient sensitivity**, making training robust to prompt length variance and invariant to micro-batch sizing.

### LoRA Parameterization
For each linear projection weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA parameterizes the weight update via low-rank decomposition:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$
where $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with $r \ll \min(d, k)$. During forward propagation on input $h$:
$$h' = W_0 h + \frac{\alpha}{r} B A h$$
Only matrices $A$ and $B$ receive gradient updates, reducing trainable parameters to $<0.3\%$ of the base model.

---

## 4. MPS & Apple Silicon Implementation Details

Fine-tuning a 1.2B parameter model on an Apple M3 Pro requires specialized memory and compute configurations:

1. **Native BF16 Precision:**
   - Apple Silicon's Neural Engine and Metal Performance Shaders natively support Brain Floating Point 16 (`torch.bfloat16`).
   - BF16 retains the dynamic range of FP32 (8-bit exponent) while cutting memory consumption by 50% compared to standard float32, preventing underflow/overflow during gradient accumulation.
2. **PyTorch SDPA (Scaled Dot-Product Attention):**
   - By specifying `attn_implementation="sdpa"`, PyTorch invokes native Metal optimized attention kernels, avoiding materializing the $O(L^2)$ attention matrix in memory.
3. **Gradient Checkpointing:**
   - By calling `model.gradient_checkpointing_enable()`, intermediate activation tensors are dropped during the forward pass and recomputed on-demand during the backward pass. This reduces activation memory by $>65\%$, allowing sequences of up to 2,048 tokens on 36 GB Unified Memory.
4. **Dynamic Batch Padding:**
   - Rather than padding all sequences to a fixed `max_seq_len = 2048`, our `SFTDataCollator` pads each micro-batch to the maximum length of sequences *in that specific micro-batch* (rounded to a multiple of 8 for Metal SIMD lane alignment). This prevents wasting unified memory bandwidth on trailing zeros.

---

## 5. Resume & Interview Talking Points

- **What did you implement in the SFT stage?**  
  *"Implemented an instruction-tuning pipeline for Llama-3.2 on Apple Silicon MPS using PEFT LoRA with the exact TÜLU 3 token-weighting formulation (Sum-Loss) to eliminate batch aggregation artifacts."*
- **What is the difference between Mean-Loss and Sum-Loss?**  
  *"Standard cross-entropy implementations average loss at the sequence level, which unintentionally assigns inversely proportional gradient weights based on sequence length—a short 10-token prompt gets 100× more gradient weight per token than a 1,000-token prompt. As demonstrated in TÜLU 3 Section 4.3.2, Sum-Loss normalizes across total tokens globally, ensuring uniform token gradients and preventing gradient accumulation divergence."*
- **How was prompt masking implemented?**  
  *"Constructed an incremental conversation tokenizer using the TÜLU 3 Jinja2 chat template where user questions and formatting headers are masked with -100, ensuring cross-entropy optimization only computes over assistant response tokens."*
