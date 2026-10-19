# Phase 4: Reinforcement Learning with Verifiable Rewards (RLVR)

> **Architecture Stage:** Post-Training Stage 3 (Reasoning & Verifiable Alignment)  
> **Reference:** Allen Institute for AI, *TÜLU 3: Pushing Frontiers in Open Language Model Post-Training* (Lambert et al., Nov 2024, Section 6: "Reinforcement Learning with Verifiable Rewards", Section 6.2: "RLVR Recipe and Analyses")

---

## 1. Concept Overview

While Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) adapt a model's style and align general preferences, they rely on static teacher demonstrations. In complex reasoning tasks (such as multi-step mathematics or strict constraint following), models frequently suffer from hallucination and errors in intermediate derivation steps.

Standard RLHF trains a neural reward model (RM) to assign scores. However, neural reward models suffer from **Goodhart's Law** and **reward hacking**: the policy discovers syntactically plausible or sycophantic completions that score highly on the reward model despite being factually incorrect.

**Reinforcement Learning with Verifiable Rewards (RLVR)** replaces neural reward models with **deterministic Python rule verification functions**:
1. **Deterministic Verifiers (`src/envs/verifiers.py`):**
   - **Math Reasoning:** Parses candidate answers (e.g., `\boxed{...}`, `#### <number>`) and uses **SymPy** for symbolic mathematical equivalence, handling algebraic simplifications, fractions, decimals, and polynomials without fuzzy heuristics.
   - **IFEval Constraints:** Deterministically validates strict formatting rules (e.g., lowercase only, exact word counts, delimiter inclusion).
   - **Non-EOS Penalty:** If a rollout truncates before emitting an `<|end_of_text|>` token, a penalty of $-10.0$ is applied to disincentivize incomplete answers.
2. **Policy Optimization via Group-Relative Advantages (GRPO):**
   - Samples $G$ rollouts per prompt in parallel.
   - Normalizes rewards across the group to compute relative advantages $A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$.
   - Updates policy weights on Apple Silicon MPS with **zero auxiliary neural critic models**, saving gigabytes of unified memory.

---

## 2. Why TÜLU 3 Uses It (Empirical Justification)

In Section 6.2 and Figures 20–23 of the TÜLU 3 paper, the authors present key empirical discoveries:
- **Do Not Use Scores from Neural Reward Models (Figure 22):**
  - Training on verifiable rewards alone significantly outperformed hybrid setups that combined neural RM scores with verifiable rewards.
  - Adding neural RM scores injected variance and noise into policy updates, degrading downstream reasoning accuracy on GSM8K and MATH.
- **Targeted Skill Improvement Without Catastrophic Forgetting:**
  - RLVR yielded substantial gains on mathematical reasoning (GSM8K accuracy improved from $76.2\%$ to $81.4\%$) and instruction following without degrading conversational or safety scores.
- **Starting from DPO vs. SFT (Figure 20):**
  - While RLVR can improve both SFT and DPO checkpoints, initializing RLVR from a DPO checkpoint converged faster with significantly lower KL divergence penalty.
- **Dropout Must Be Disabled ($\text{Dropout} = 0.0$):**
  - Active dropout causes policy log-probabilities to fluctuate between rollout and optimization steps, corrupting advantage ratios. Setting dropout to $0.0$ ensures deterministic forward passes and stable training.

---

## 3. Mathematical Formulation

### The RLVR Objective
Given a prompt $x \sim \mathcal{D}$ and policy $\pi_\theta$, the model samples completions $y \sim \pi_\theta(x)$ and optimizes:
$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(x)} \left[ v(x, y) - \beta D_{\text{KL}}\left(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)\right) \right]$$
where $v(x, y)$ is the deterministic verifier reward:
$$v(x, y) = \begin{cases}
+1.0, & \text{if answer is verifiably correct} \\
-10.0, & \text{if response lacks EOS token (truncated)} \\
0.0, & \text{otherwise}
\end{cases}$$

### Group-Relative Policy Optimization (GRPO)
Rather than instantiating an auxiliary value network $V_\phi(x)$ (which incurs massive memory overhead), GRPO estimates baseline advantages directly across a group of $G$ sampled completions $\{y_1, y_2, \dots, y_G\}$ for prompt $x$:

1. Compute verifier reward for each sample: $R_i = v(x, y_i)$ for $i \in \{1, \dots, G\}$.
2. Group statistics:
   $$\mu_R = \frac{1}{G} \sum_{i=1}^G R_i, \qquad \sigma_R = \sqrt{\frac{1}{G} \sum_{i=1}^G (R_i - \mu_R)^2}$$
3. Group-Relative Advantage:
   $$A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$
   If all rollouts in the group achieve identical rewards (e.g. all correct or all incorrect), $A_i = 0$, preventing noisy parameter updates on uninformative prompts.

4. Policy Gradient Loss:
   For an autoregressive completion $y_i = (y_{i, 1}, \dots, y_{i, T_i})$:
   $$\mathcal{L}_{\text{RLVR}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \left( \sum_{t=1}^{T_i} \log \pi_\theta(y_{i, t} \mid x, y_{i, <t}) \right) A_i$$

Taking the gradient with respect to policy weights $\theta$:
$$\nabla_\theta \mathcal{L}_{\text{RLVR}} = - \frac{1}{G} \sum_{i=1}^G A_i \sum_{t=1}^{T_i} \nabla_\theta \log \pi_\theta(y_{i, t} \mid x, y_{i, <t})$$
Completions with above-average rewards ($A_i > 0$) have their token log-probabilities boosted, while completions with below-average rewards ($A_i < 0$) are penalized.

---

## 4. MPS & Apple Silicon Implementation Details

1. **Elimination of Critic / Value Network:**
   - On an Apple M3 Pro (36 GB unified memory), running PPO with an Actor $\pi_\theta$, Critic $V_\phi$, and Reference $\pi_{\text{ref}}$ would require maintaining three concurrent models, exhausting memory bandwidth.
   - By adopting group-relative advantages, **no Critic model is ever loaded**, reducing runtime memory by over $50\%$.
2. **Batched Parallel Rollouts:**
   - Group rollouts $\{y_1, \dots, y_G\}$ are generated in parallel using `model.generate(..., num_return_sequences=G, do_sample=True)` in native `torch.bfloat16`.
3. **Dropout = 0.0:**
   - Following TÜLU 3 Section 6.2, LoRA dropout is set to `0.0` during RLVR to guarantee that log-probabilities computed during the learning phase match the generation phase deterministically.
4. **Symbolic Math with SymPy:**
   - Rule verification occurs in native Python using compiled regular expressions and SymPy parsing in a separate CPU process, overlapping verification with GPU/MPS operations.

---

## 5. Resume & Interview Talking Points

- **What did you build for Phase 4?**  
  *"Implemented Reinforcement Learning with Verifiable Rewards (RLVR) based on the TÜLU 3 recipe, replacing fallible neural reward models with deterministic rule verifiers for mathematical reasoning and instruction following."*
- **Why avoid neural reward models for reasoning tasks?**  
  *"As documented in TÜLU 3 Section 6.2, neural reward models introduce noise and suffer from reward hacking—models learn to exploit superficial stylistic quirks to achieve high scores. Deterministic verifiers (such as SymPy symbolic equivalence and exact constraint checkers) provide ground-truth binary rewards that cannot be gamed."*
- **How was the reinforcement learning loop optimized for local hardware?**  
  *"Adopted Group-Relative Policy Optimization (GRPO) to compute advantage estimates across sample groups without allocating a neural value network, cutting GPU memory overhead and enabling scalable local training on Apple Silicon."*
