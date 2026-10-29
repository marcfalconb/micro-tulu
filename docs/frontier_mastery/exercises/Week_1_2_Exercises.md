# Week 1-2 Exercises: Matrix Calculus & Backpropagation from Scratch

**Goal:** Prove you understand how gradients flow backward through a network without relying on `loss.backward()`.

---

## 📝 Problem 1: The Jacobian of Softmax

The Softmax function converts a vector of raw logits $z \in \mathbb{R}^K$ into a probability distribution $p \in \mathbb{R}^K$:
$$p_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

### Your Task (Math):
Derive the Jacobian matrix $J \in \mathbb{R}^{K \times K}$ where $J_{ij} = \frac{\partial p_i}{\partial z_j}$.
*Hint: You must consider two cases: $i = j$ and $i \neq j$. Use the quotient rule.*

---

## 📝 Problem 2: Cross-Entropy Loss Gradient

In classification, we compute the Cross-Entropy loss between the predicted probabilities $p$ and the true one-hot label $y$ (where $y_c = 1$ for the correct class $c$, and $0$ elsewhere):
$$L = - \sum_{i=1}^K y_i \log(p_i) = - \log(p_c)$$

### Your Task (Math):
Using the Chain Rule and your result from Problem 1, prove that the gradient of the loss with respect to the raw logits $z$ is beautifully simple:
$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

---

## 💻 Problem 3: NumPy MLP (No PyTorch Allowed)

You must build a 2-layer Multi-Layer Perceptron (MLP) for a 3-class classification problem. 
You cannot use PyTorch. You must use NumPy.

**Architecture:**
- Input $x \in \mathbb{R}^{D_{\text{in}}}$
- Hidden Layer $h_1 = \text{ReLU}(W_1 x + b_1)$  (where $W_1 \in \mathbb{R}^{D_{\text{hid}} \times D_{\text{in}}}$)
- Output Logits $z = W_2 h_1 + b_2$ (where $W_2 \in \mathbb{R}^{3 \times D_{\text{hid}}}$)
- Probabilities $p = \text{Softmax}(z)$

### Your Task (Code):
Complete the `backward` pass function.

```python
import numpy as np

class NumPyMLP:
    def __init__(self, d_in, d_hid, d_out):
        # Initialize weights randomly
        self.W1 = np.random.randn(d_hid, d_in) * 0.1
        self.b1 = np.zeros((d_hid, 1))
        self.W2 = np.random.randn(d_out, d_hid) * 0.1
        self.b2 = np.zeros((d_out, 1))
        
        # Cache for backward pass
        self.cache = {}

    def forward(self, x):
        """
        x: shape (d_in, batch_size)
        Returns: p, shape (d_out, batch_size)
        """
        self.cache['x'] = x
        
        z1 = self.W1 @ x + self.b1
        self.cache['z1'] = z1
        
        h1 = np.maximum(0, z1) # ReLU
        self.cache['h1'] = h1
        
        z2 = self.W2 @ h1 + self.b2
        self.cache['z2'] = z2
        
        # Softmax (numerically stable)
        z2_shifted = z2 - np.max(z2, axis=0, keepdims=True)
        exp_z2 = np.exp(z2_shifted)
        p = exp_z2 / np.sum(exp_z2, axis=0, keepdims=True)
        self.cache['p'] = p
        
        return p

    def backward(self, y, lr=0.01):
        """
        y: one-hot true labels, shape (d_out, batch_size)
        You must compute dW2, db2, dW1, db1 and update the weights.
        """
        m = y.shape[1] # batch size
        
        # 1. Gradient of Loss w.r.t logits (z2)
        # TODO: Implement using the result from Problem 2
        dz2 = self.cache['p'] - y 
        
        # 2. Gradient w.r.t W2 and b2
        # TODO: Implement dW2 and db2
        dW2 = None 
        db2 = None
        
        # 3. Gradient w.r.t h1
        # TODO: Implement dh1
        dh1 = None
        
        # 4. Gradient w.r.t z1 (Backprop through ReLU)
        # TODO: Implement dz1
        dz1 = None
        
        # 5. Gradient w.r.t W1 and b1
        # TODO: Implement dW1 and db1
        dW1 = None
        db1 = None
        
        # Update weights
        # self.W2 -= lr * dW2
        # self.b2 -= lr * db2
        # self.W1 -= lr * dW1
        # self.b1 -= lr * db1
```