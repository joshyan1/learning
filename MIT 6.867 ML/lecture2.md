# Machine Learning - MIT OCW 6.867
## Lecture 2 - Perception, convergence, and generalization

*Recall*, we're looking at linear classifiers through the origin of the form
```math
f(x;\theta) = sign(\theta^T x)
```
where $\theta \in \mathbb{R}^d$ are the parameters that we estimate.

We will use the perceptron algorihtm to solve the estimation problem.
Let $k$ denote the number of parameters updates we've performed and $\theta^(k)$ the parameter vector after $k$ updates.

We start with $k = 0$ and $\theta^{(k)} = 0$. Then $\theta^{(k + 1)} = \theta^{(k)} + y_tx_t$ where $y_t(\theta^{(k)})^T x_t < 0$.

### Convergence in a finite number of updates
We'll show that the perceptron algorithm converges, helping us understand how lienar classifiers generalize to unseen images in the process. 

Assume that all (training) images have bounded Euclidean norms, that is $||x_t|| \leq R$ for all $t$ and some $R$. We'll also assume that $\exists$ a linear classifier with finite parameter values that corrctly classifies all (training) images. Formally, we assume there exists some $ \gamma > 0$ such that $y_t(\theta^*)^T x_t \geq \gamma$ for all $t = 1, \dots n$.

Note, we use $\gamma > 0$ to ensure that each example is classified correctly with a *finite margin*

Our proof utilizes the following results: 1) The inner product $(\theta^*)^T \theta^{(k)}$ increases at least linearly and 2) the squared norm $||\theta^{(k)}||^2$ increases at most linearly in the number of updates $k$. Then, we show the cosine of the angle between $\theta^*$ and $\theta^{(k)}$ increases from each update. Since cosine is bounded above by 1, it follows we have a finite number of updates.

1) Consider $(\theta^*)^T \theta^{(k)}$ before and afer each update. For the $k^{th}$ update, we have 
```math
(\theta^*)^T \theta^{(k)} = (\theta^*)^T \theta^{(k - 1)} + y_t(\theta^*)^T x_t \geq (\theta^*)^T \theta^{(k - 1)} + \gamma
```
Thus, after $k$ updates,
```math
(\theta^*)^T \theta^{(k)} \geq k\gamma
```

2) Our second claim follows from the fact that we only update following mistakes
```math
\begin{align*}
    ||\theta^{(k)}||^2 &= ||\theta^{(k-1)} + y_tx_t||^2 \\
    &= ||\theta^{(k-1)}||^2 + 2y_t(\theta^{(k-1)})^Tx_t + ||x_t||^2 \\
    &\leq ||\theta^{(k-1)}||^2 + ||x_t||^2 \\
    &\leq ||\theta^{(k-1)}||^2 + R^2
\end{align*}
```
Note, that $2y_t(\theta^{(k-1)})^Tx_t < 0$ when an update is made. Thus 
```math
||\theta^{(k)}||^2 \leq kR^2
```

It now follows that
```math
cos(\theta^*, \theta*{(k)}) = \frac{(\theta^*)^T \theta^{(k)}}{||\theta^(kk)|| \ ||\theta^*||} \geq \frac{k\gamma}{\sqrt{kR^2}||\theta^*||}
```
As cosine is bounded above by $1$, we get
```math
1 \geq \frac{k\gamma}{\sqrt{kR^2}||\theta^*||} \text{ or } k \leq \frac{R^2||\theta^*||^2}{\gamma^2}
```

### Margin and geometry
We'll look at this result a bit more. For example, $||\theta^*||^2/\gamma^2$ relates to how difficult the classification problem is. We claim that its inverse is the smallest distance in the image space from any image to the decision boundary of $\theta^*$. It measures how well the two classes of images are separated. This is called the geometric margin or $\gamma_{geom}$. Then, $\gamma_{geom}^{-1}$ measures how difficult the problem is.

$\gamma_{geom}$ is calculuated by measuring the decision from the decision boundary to an image $x_t$ where $y_t(\theta^*)^Tx_t = \gamma$. $\theta^*$ is normal to the decision boundary and the shortest path to a $x_t$ will be parallel to the normal. Then, $x_t$ such that $y_t(\theta^*)^T x_t = \gamma$ is among those closest to the boundary. 

Let's define $x(0) = x_t$, towards the boundary. This is 
```math
x(\zeta) = x(0) - \zeta \frac{y_t \theta^*}{||\theta^*||}
```
where $\zeta$ defines the length of the line segment. We now find the value of $\zeta$ such that $(\theta^*)^Tx(\zeta) = 0$ or $y_t(\theta^*)^T x(\zeta) = 0$. This is the point where the segment hits the decision boundary. We have
```math 
\begin{align*}
    y_t(\theta^*)^T x(\zeta) &= y_t(\theta^*)^T\left[x(0) - \zeta \frac{y_t \theta^*}{||\theta^*||}\right] \\
    &= y_t(\theta^*)^T[x_t - \zeta \frac{y_t \theta^*}{||\theta^*||}] \\
    &= y_t(\theta^*)^Tx_t - \zeta\frac{||\theta^*||^2}{||\theta^*||} \\
    &= \gamma - \zeta||\theta^*||
\end{align*}
```
This shows $\zeta$, our shortest distance is $\gamma/||\theta^*||$ as claimed. We can now more concisely bound our $k$ using $\gamma_{geom}$:
```math
k \leq \left(\frac{R}{\gamma_{geom}}\right)^2
```
Note that this result is not entirely dependent on $d$ nor $n$, but it's still tempting to interpret this bound as a measure of difficulty of the problem of learning linear classifiers. You will see this later in the form of a *VC-dimension*

### Generalization guarantees
Now, how does this perceptron classify general images? Suppose that all images and labels satisfy our $2$ assumptions: $||x_t|| \leq R$ and $y_t (\theta^*)^T x_T \geq \gamma$ for all $t$ and some $\theta^*$. In other words, we assume there exists a linear classifier that works. We created our previous bounds on the assumption that we repeat the same training set. 

What if we had an infinite sequence of arbitrary images? We will be bound by the same number of mistakes $k \leq (R/\gamma_{geom})^2$, meaning that after all these mistakes, we will classify all new images correctly. Then, our bound is dependent on *knowing* the labels to these **infinite** images.

### Maximum margin classifier
So far, the perceptron has been used to estimate a linear classifier. To do so, we assumed that there was such a classifier with a sufficiently large enough geometric margin. Such a classifier is a **Support Vector Machine** (SVM), which we will find directly.