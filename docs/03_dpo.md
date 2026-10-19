# Phase 3: Length-Normalized Direct Preference Optimization (DPO)

> **Architecture Stage:** Post-Training Stage 2 (Preference Alignment)  
> **Reference:** Allen Institute for AI, *TÜLU 3: Pushing Frontiers in Open Language Model Post-Training* (Lambert et al., Nov 2024, Section 5: "Preference Tuning", Section 5.4.1: "Hyperparameter and Algorithm Design", Eq 6)

---

## 1. Concept Overview

Direct Preference Optimization (DPO) aligns language models with human preferences directly from pairwise rankings $(x, y_c, y_r)$ without needing an explicit neural reward model or reinforcement learning rollouts.

Phase 3 establishes the **Micro-TÜLU DPO Engine**:
1. **Length-Normalized Loss (Eq. 6):** Standard DPO exhibits an implicit **verbosity bias**, rewarding overly lengthy, superficial responses simply because total log-probability grows with token length. TÜLU 3 normalizes log-probabilities by response lengths $|y_c|$ and $|y_r|$, eradicating length exploitation.
2. **Offline Reference Log-Probability Caching:** The reference policy $\pi_{\text{ref}}$ is frozen during training. Rather than loading two full models into memory simultaneously, we precompute $\log \pi_{\text{ref}}(y_c \mid x)$ and $\log \pi_{\text{ref}}(y_r \mid x)$ offline once, **halving active memory consumption** during DPO training.
3. **Split Forward Passes:** Decouples chosen and rejected sequences into separate forward passes, preventing peak activation spikes on Apple Silicon Unified Memory.
4. **Visual Metrics & Auditability:** Live terminal tracking of the **implicit reward margin** ($R_c - R_r$) and preference accuracy, accompanied by an interactive standalone HTML dashboard (`checkpoints/dpo/training_report.html`).

---

## 2. Why TÜLU 3 Uses It (Empirical Justification)

In Section 5.4.1 of the TÜLU 3 paper, the authors comprehensively ablated preference algorithms:
- **DPO vs. PPO vs. SimPO vs. Length-Normalized DPO:**
  - Standard DPO: Achieved an average score of $55.2$ on an early TÜLU 3 checkpoint.
  - SimPO (without reference model): Reached $52.9$ (underperformed baseline).
  - PPO (with full reward model + online sampling): Achieved $55.5$, but required **7× more GPU compute** (28 hours across two nodes vs. 4 hours for DPO).
  - **Length-Normalized DPO (`DPO-norm`):** Achieved the highest score of **$57.3$** (Table 18 in the paper), surpassing both standard DPO and PPO.
- **Why Length Normalization Matters:**  
  Human evaluators and LLM judges (such as GPT-4o) frequently prefer longer answers. Under unnormalized DPO, the policy exploits this heuristic by padding output tokens. Normalizing log-likelihood by $|y|$ forces the model to focus on substantive reasoning quality rather than output verbosity.

---

## 3. Mathematical Formulation

### Standard Direct Preference Optimization
Under the Bradley-Terry preference model, given prompt $x$, chosen response $y_c$, and rejected response $y_r$, the standard DPO objective minimizes:
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \beta \log \frac{\pi_\theta(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)} \right) \right]$$
where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function, and $\beta$ controls the KL penalty strength relative to the reference policy $\pi_{\text{ref}}$.

### TÜLU 3 Length-Normalized DPO (Equation 6)
To mitigate length bias, TÜLU 3 normalizes the log-ratio of each completion by its corresponding token length $|y|$:
$$\mathcal{L}_{\text{DPO-norm}}(\theta) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}} \left[ \log \sigma \left( \frac{\beta}{|y_c|} \log \frac{\pi_\theta(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \frac{\beta}{|y_r|} \log \frac{\pi_\theta(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)} \right) \right]$$

### Implicit Reward and Margin Formulation
We define the implicit normalized rewards:
$$R_\theta(x, y_c) = \frac{\beta}{|y_c|} \left( \log \pi_\theta(y_c \mid x) - \log \pi_{\text{ref}}(y_c \mid x) \right)$$
$$R_\theta(x, y_r) = \frac{\beta}{|y_r|} \left( \log \pi_\theta(y_r \mid x) - \log \pi_{\text{ref}}(y_r \mid x) \right)$$

The **Implicit Reward Margin** is:
$$\Delta R = R_\theta(x, y_c) - R_\theta(x, y_r)$$
The loss simplifies computationally to:
$$\mathcal{L}_{\text{DPO-norm}} = \text{softplus}(-\Delta R) = \log\left(1 + e^{-\Delta R}\right)$$
During training, as $\Delta R$ increases (chosen reward exceeds rejected reward), the loss monotonically approaches zero.

---

## 4. MPS & Apple Silicon Implementation Details

1. **Offline Reference Log-Prob Caching (`cache_logprobs.py`):**
   - Standard DPO keeps two simultaneous models in memory: $\pi_\theta$ (trainable) and $\pi_{\text{ref}}$ (frozen).
   - In Micro-TÜLU, we run an offline inference pass over the decontaminated preference pairs with $\pi_{\text{ref}}$ in eval mode (`torch.no_grad()`), recording $\log \pi_{\text{ref}}(y_c \mid x)$ and $\log \pi_{\text{ref}}(y_r \mid x)$ directly in the JSONL dataset.
   - During training, $\pi_{\text{ref}}$ is completely discarded from memory. Active memory is **cut by 50%**, leaving full unified memory available for LoRA gradient backpropagation.
2. **Split Forward Passes (`train_dpo.py`):**
   - In standard implementations, chosen and rejected sequences are concatenated along the batch dimension: $[2B, L]$.
   - On Apple Silicon MPS, large concatenated batches trigger substantial temporary unified memory allocations during attention computation.
   - We execute two smaller split forward passes:
     1. Forward pass on $y_c$ -> returns $\log \pi_\theta(y_c \mid x)$
     2. Forward pass on $y_r$ -> returns $\log \pi_\theta(y_r \mid x)$
     3. Combined vectorized loss computation and single backward pass.
   - This halves peak activation memory during backprop.

---

## 5. Resume & Interview Talking Points

- **What did you implement in the DPO phase?**  
  *"Engineered a Length-Normalized DPO training pipeline based on TÜLU 3 (Lambert et al., 2024), optimizing preference alignment without neural reward models."*
- **Why is Length-Normalized DPO superior to standard DPO?**  
  *"Standard DPO sums log-probabilities over the entire sequence, inadvertently penalizing concise responses and incentivizing verbosity hacking. Normalizing by response token length $|y|$ enforces reward parity and yielded TÜLU 3's highest overall benchmark score (57.3 vs 55.2)."*
- **How did you optimize memory for Apple Silicon?**  
  *"Architected offline reference log-probability caching and split forward passes. Because the reference policy is static, precomputing its log-probabilities reduced active VRAM by 50% and eliminated redundant forward passes during training."*
