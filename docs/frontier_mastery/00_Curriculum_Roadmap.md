# 🗺️ Frontier LLM Mastery Curriculum (9-Month Roadmap)

**Timeline:** October 30, 2026 ➔ June 30, 2027 (8 Months)  
**Goal:** Master first-principles mathematics, PyTorch implementation, and architectural evolution of Large Language Models to pass technical interviews at frontier labs (Anthropic, DeepMind, OpenAI) and top-tier tech (Netflix, Meta, Microsoft).

---

## The Philosophy: Why This Structure?

Frontier labs do not test if you know how to call `model.generate()`. They test if you know **why** the gradients flow the way they do, **what** causes numerical instability in FP16, and **how** to implement a paper's equations purely in PyTorch. 

This curriculum is split into 4 Modules. Each module contains:
1. **Mathematical Foundations:** The "Why" (Derivations, Chain Rule, Matrix properties).
2. **Architectural Implementation:** The "How" (PyTorch `nn.Module` from scratch).
3. **Systems & Scaling:** The "Bottlenecks" (Memory bandwidth, KV-Cache, MoE routing).
4. **Interview prep:** LeetCode-style ML questions and paper-replication exercises.

---

## 📅 Module 1: The First-Principles Foundation (Nov – Dec)
**Focus:** Rebuilding the mental model for matrix calculus, gradients, and the original Transformer.

*   **Week 1-2: Matrix Calculus & Backpropagation from Scratch**
    *   **Math:** Chain rule for matrices, Jacobian of Softmax, cross-entropy derivatives.
    *   **Code:** Write a 2-layer MLP and backpropagate purely in NumPy (no PyTorch `autograd`).
*   **Week 3-4: Attention Mechanics & The Original Transformer**
    *   **Math:** Why divide by $\sqrt{d_k}$? Attention as a sequence-to-sequence permutation.
    *   **Code:** Implement `ScaledDotProductAttention`, `MultiHeadAttention`, and `TransformerBlock` in PyTorch.
*   **Week 5-6: Normalization, Residuals, & The GPT-2 Architecture**
    *   **Math:** LayerNorm vs RMSNorm (mean-centering vs variance scaling). Post-LN vanishing gradients vs Pre-LN stability.
    *   **Code:** Build the `nano_llm_from_scratch.ipynb` (Train a tiny character-level GPT on Shakespeare).

---

## 📅 Module 2: The Modern Decoder (Jan – Feb)
**Focus:** The 2023-2025 architectural shifts that define LLaMA 3, Mistral, and OLMo.

*   **Week 7-8: Positional Encodings (RoPE) & Activation Functions (SwiGLU)**
    *   **Math:** 2D rotational matrices, complex number multiplication for relative distance. Why SwiGLU empirically beats GELU.
    *   **Code:** Implement RoPE and SwiGLU in PyTorch. 
*   **Week 9-10: The Memory Wall (KV-Cache, MQA, GQA)**
    *   **Systems:** Compute-bound vs Memory-bandwidth bound. The exact memory formula for KV-Cache.
    *   **Code:** Implement Grouped-Query Attention (GQA) and a stateful KV-Cache for autoregressive decoding.
*   **Week 11-12: High-Performance Inference & Sampling**
    *   **Math:** Softmax temperature scaling, Top-P (Nucleus) vs Min-P sampling.
    *   **Code:** Implement a Speculative Decoding engine (draft/target rejection sampling).

---

## 📅 Module 3: Frontier Architectures (Mar – Apr)
**Focus:** The 2025-2026 leaps in extreme scale (DeepSeek, Kimi K3).

*   **Week 13-15: Mixture of Experts (MoE) & Routing**
    *   **Math:** Top-K gating, auxiliary load-balancing loss, expert collapse.
    *   **Code:** Implement a sparse MoE layer with a noisy Top-K router.
*   **Week 16-17: Latent Space Routing & Bounded Activations**
    *   **Math:** SiTU-GLU bounding ($|f(x)| \le 100$), Quantile Balancing.
    *   **Code:** Implement Kimi K3's Stable LatentMoE and SiTU-GLU.
*   **Week 18-20: Advanced Attention (MLA & AttnRes)**
    *   **Math:** Multi-Head Latent Attention (MLA) rank compression. Block Attention Residuals.
    *   **Code:** Implement an AttnRes layer computing softmax over depth.

---

## 📅 Module 4: Alignment & Post-Training (May – Jun)
**Focus:** Making the model useful (SFT, RLHF, DPO, RLVR).

*   **Week 21-22: Supervised Fine-Tuning (SFT) & LoRA**
    *   **Math:** Low-Rank Adaptation (LoRA) matrix factorization ($\Delta W = AB$). Loss masking.
    *   **Code:** Implement LoRA from scratch and apply it to a Linear layer.
*   **Week 23-25: Direct Preference Optimization (DPO & SimPO)**
    *   **Math:** Bradley-Terry model. DPO implicit reward derivation. Length-normalization.
    *   **Code:** Implement DPO and SimPO loss functions.
*   **Week 26-28: Reinforcement Learning with Verifiable Rewards (RLVR)**
    *   **Math:** Policy Gradients (REINFORCE), Group-Relative Advantage (GRPO / OlmoRL).
    *   **Code:** Implement a PPO/GRPO clipping loss function.
*   **Week 29-30: The Final Boss (Systems Integration)**
    *   **Prep:** Mock interviews. Reviewing all mathematical derivations.

---

## 🚀 Next Step (Beyond June): The CUDA Abyss
Once you master the PyTorch/Math level, the next frontier is hardware acceleration.
*   **Triton/CUDA:** Writing custom kernels for FlashAttention. Understanding GPU SRAM tiling, coalesced memory access, and thread block synchronization.
