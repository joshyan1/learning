# Machine Learning - MIT OCW 6.867
## Lecture 2 - Perception, convergence, and generalization

*Recall*, we're looking at linear classifiers through the origin of the form
$$ 
f(x;\theta) = sign(\theta^T x)
$$
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
$$ 
(\theta^*)^T \theta^{(k)} = (\theta^*)^T \theta^{(k - 1)} + y_t(\theta^*)^T x_t \geq (\theta^*)^T \theta^{(k - 1)} + \gamma
$$
Thus, after $k$ updates,
$$
(\theta^*)^T \theta^{(k)} \geq k\gamma$$

2) Our second claim follows from the fact that we only update following mistakes
$$
\begin{align*}
    ||\theta^{(k)}||^2 &= ||\theta^{(k-1)} + y_tx_t||^2 \\
    &= ||\theta^{(k-1)}||^2 + 2y_t(\theta^{(k-1)})^Tx_t + ||x_t||^2 \\
    &\leq ||\theta^{(k-1)}||^2 + ||x_t||^2 \\
    &\leq ||\theta^{(k-1)}||^2 + R^2
\end{align*}
$$
Note, that $2y_t(\theta^{(k-1)})^Tx_t < 0$ when an update is made. Thus 
$$
||\theta^{(k)}||^2 \leq kR^2
$$

It now follows that
$$
cos(\theta^*, \theta*{(k)}) = \frac{(\theta^*)^T \theta^{(k)}}{||\theta^(kk)|| \ ||\theta^*||} \geq \frac{k\gamma}{\sqrt{kR^2}||\theta^*||}
$$
As cosine is bounded above by $1$, we get
$$
1 \geq \frac{k\gamma}{\sqrt{kR^2}||\theta^*||} \text{ or } k \leq \frac{R^2||\theta^*||^2}{\gamma^2}
$$

### Margin and geometry
We'll look at this result a bit more. For example, $||\theta^*||^2/\gamma^2$ relates to how difficult the classification problem is. We claim that its inverse is the smallest distance in the image space from any image to the decision boundary of $\theta^*$. It measures how well the two classes of images are separated. This is called the geometric margin or $\gamma_{geom}$. Then, $\gamma_{geom}^{-1}$ measures how difficult the problem is.

$\gamma_{geom}$ is calculuated by measuring the decision from the decision boundary to an image $x_t$ where $y_t(\theta^*)^Tx_t = \gamma$. 

