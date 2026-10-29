# Week 1-2 Solutions: Matrix Calculus & Backpropagation from Scratch

## 📝 Problem 1 Solution: The Jacobian of Softmax

Let $S = \sum_{k=1}^K e^{z_k}$. Thus, $p_i = \frac{e^{z_i}}{S}$.

**Case 1: $i = j$**
Using the quotient rule:
$$ \frac{\partial p_i}{\partial z_i} = \frac{\frac{\partial (e^{z_i})}{\partial z_i} S - e^{z_i} \frac{\partial S}{\partial z_i}}{S^2} $$
Since $\frac{\partial S}{\partial z_i} = e^{z_i}$:
$$ \frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} S - (e^{z_i})^2}{S^2} = \frac{e^{z_i}}{S} - \left(\frac{e^{z_i}}{S}\right)^2 = p_i - p_i^2 = \mathbf{p_i(1 - p_i)} $$

**Case 2: $i \neq j$**
$$ \frac{\partial p_i}{\partial z_j} = \frac{0 \cdot S - e^{z_i} \frac{\partial S}{\partial z_j}}{S^2} = \frac{- e^{z_i} e^{z_j}}{S^2} = -\left(\frac{e^{z_i}}{S}\right)\left(\frac{e^{z_j}}{S}\right) = \mathbf{-p_i p_j} $$

Therefore, the Jacobian is:
$$ J_{ij} = p_i(\delta_{ij} - p_j) $$

---

## 📝 Problem 2 Solution: Cross-Entropy Loss Gradient

The loss is $L = - \sum_{k=1}^K y_k \log(p_k)$. 
By the Chain Rule:
$$ \frac{\partial L}{\partial z_i} = \sum_{k=1}^K \frac{\partial L}{\partial p_k} \frac{\partial p_k}{\partial z_i} $$

First, find $\frac{\partial L}{\partial p_k}$:
$$ \frac{\partial L}{\partial p_k} = -\frac{y_k}{p_k} $$

Now substitute the Jacobian from Problem 1:
$$ \frac{\partial L}{\partial z_i} = \sum_{k=1}^K \left(-\frac{y_k}{p_k}\right) p_k(\delta_{ki} - p_i) $$
Split the sum into $k=i$ and $k \neq i$:
$$ = \left(-\frac{y_i}{p_i}\right) p_i(1 - p_i) + \sum_{k \neq i} \left(-\frac{y_k}{p_k}\right) (-p_k p_i) $$
$$ = -y_i(1 - p_i) + \sum_{k \neq i} y_k p_i = -y_i + y_i p_i + p_i \sum_{k \neq i} y_k $$
Since $y$ is a one-hot vector (or any probability distribution), $\sum_{k=1}^K y_k = 1$. Therefore, $\sum_{k \neq i} y_k = 1 - y_i$.
$$ = -y_i + y_i p_i + p_i (1 - y_i) = -y_i + y_i p_i + p_i - p_i y_i = \mathbf{p_i - y_i} $$

*This incredibly elegant result is why Softmax and Cross-Entropy are paired together in Deep Learning!*

---

## 💻 Problem 3 Solution: NumPy MLP

```python
    def backward(self, y, lr=0.01):
        m = y.shape[1] # batch size
        
        # 1. Gradient of Loss w.r.t logits (z2)
        dz2 = self.cache['p'] - y # Shape: (d_out, m)
        
        # 2. Gradient w.r.t W2 and b2
        # z2 = W2 @ h1 + b2  => dz2/dW2 = h1^T
        dW2 = (dz2 @ self.cache['h1'].T) / m # Shape: (d_out, d_hid)
        db2 = np.sum(dz2, axis=1, keepdims=True) / m # Shape: (d_out, 1)
        
        # 3. Gradient w.r.t h1
        # z2 = W2 @ h1 + b2 => dz2/dh1 = W2^T
        dh1 = self.W2.T @ dz2 # Shape: (d_hid, m)
        
        # 4. Gradient w.r.t z1 (Backprop through ReLU)
        dz1 = dh1 * (self.cache['z1'] > 0) # Shape: (d_hid, m)
        
        # 5. Gradient w.r.t W1 and b1
        dW1 = (dz1 @ self.cache['x'].T) / m # Shape: (d_hid, d_in)
        db1 = np.sum(dz1, axis=1, keepdims=True) / m # Shape: (d_hid, 1)
        
        # Update weights
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
```