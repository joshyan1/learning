# Machine Learning - MIT OCW 6.867
## Lecture 1

Suppose that we wanted to do binary classification on images of people based on features. This is a binary function $f: \mathbb{R} \rightarrow \{-1, 1\}$, where $d$ is the dimension (or many number of pixels) in the image. In essence, our classifier has a set of $n$ training vectors $x_1, \dots, x_n$ and binary outputs $y_1, \dots, y_n$. 

**What's a possible solution?** \
Suppose that $n = 50$ and our images are $128 \times 128$, with each pixel having intensity between $0 - 256$. Maybe we can find a single pixel $i$ such that each $x_j$ has a distinct value for $i$. Using this, we could make our function
$$ f_i(x^\prime) =  \begin{cases} 
      y_t & \text{if } x_{ti} = x^\prime \text{ for some } t = 1, \dots, n \\
      -1 & \text{otherwise} 
   \end{cases}
$$

Clearly, this works *perfectly* if our data is distinct. However, this doesn't generalize well. We want to generalize for unseen images.

### Model selection
The key to finding classifiers that generalize well is to constrain the set of possible binary functions we want to look at. In other words, we want to find a **class** of functions that, given work well on the training set, will work well on unseen images. Finding this class of functions is known as the **model selection** problem

### Linear classifiers through origin
For now, we fix our function class to **linear classifiers**. Formally, we consider functions of the form
$$ 
f(x; \theta) = sign(\theta_1x_1 + \dots + \theta_d x_d ) = sign(\theta^T x)
$$
where $\theta = [\theta_1, \dots, \theta_d]^T$ is a column vector of real valued parameters. These functions are parameterized by $\theta \in \mathbb{R}^d$, meaning that different values of this *parameter* could yield different *outputs* for some inputs $x$

Thinking geometrically, $\theta^T x = 0$ is a *decision boundary*, and our function predicts $1$, ($-1$ respectively), when $\theta^T x > 0$, ($\theta^T x < 0$).

Note that linear classifiers result in the loss of information or properties. For example, with images, the *promixity* of pixels to eachother is not considered for a linear classifier. If we permuted the pixels, this simply reorders our summation, resulting in the same output.

### The perceptron
We chose linear classifies as our function class. Now we need to find a function in this class that works. This is called the **estimation** problem. Formally, we want to find a linear classifier (a parameter $\theta$) that minimizes the **training error**
$$ 
\hat{E}(\theta) = \frac{1}{n}\sum_{t=1}^n (1 - \delta(y_t, f(x_t; \theta))) = \frac{1}{n}\sum_{t=1}^nLoss(y_t, f(x_t; \theta))
$$
where $\delta(y, y^\prime) = 1$ if $y = y^\prime$ and $0$ otherwise. This specific training error counts the average number of images where our expected output is different from our predicted output. Generally, we could compare our predictions using a loss function.

How do we decide $\theta$. The simplest algorithm that *accounts for mistakes* might be the *perceptron* update rule. Consider each training image one by one, and adjust the parameters as follows
$$
\theta^t \leftarrow \theta + y_tx_t \text{ if } y_t \neq f(x_t;\theta)
$$

Simply put, we change our parameters only if we make a mistake. To see this, a mistake occurs when $sign(y_t) \neq sign(\theta^T x_t)$ so $y_t \theta^T x_t < 0$. Now, with a mistake on $x_t$, then the next time we see $x_t$ using the updated parameter
$$ 
y_t(\theta + y_tx_t)^T x_t = y_t\theta^T x_t + ||x_t||^2
$$
This means, $y_t\theta^T x_t$ increases as a result of our adjustment, and so enough adjustments means that eventually we will properly classify this. Note, that this might steer the parameters to make mistakes on other images.

### Analysis of the perceptron algoirthm
This algorithm halts when all images are classified correctly. Then, if the images are possible to classify correctly with a linear classifier, this algorithm converges to such a classifier as will be show in **Lecture 2**