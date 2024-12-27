# Machine Learning - MIT OCW 6.867
## Lecture 4
### The Support Vector Machine and regularization

**Recall**, we introduced a relaxed SVM allowing for some examples to be misclassified.

Let's take a look at the relaxed optimization problem from the point of *regularization*. These problems are optimization problems with a desired objective and a **regularization penalty**, which are used to help stabilize the minimization of the objective or use prior knowledge about solutions. We will now cast our SVM optimization as a regularization problem.

To do so, lets solve the optimization problem with respect to the $\xi$ values for some fixed $\theta$ and $\theta_0$. In other words, we define a loss function for the $\xi_t$ values using specific costs of violating margin constraints. Then, our function is of $C\sum_t \xi_t$ as a function of $\theta$ and $\theta_0$.

Our loss function now is based on the *hinge loss* $Loss_h(z) = (1 - z)^+$. Our problem is now
```math
\text{minimize } \frac{1}{2}||\theta||^2 + C\sum^n_{t=1}\left((1-y_t(\theta^Tx_t = \theta_0)\right)^+
```
where $\left((1-y_t(\theta^Tx_t = \theta_0)\right) = \xi_t$. Here, $\frac{1}{2}||\theta||^2$ is a *regularization penalty* to stabilize the objective. When we have 0 loss, this penalty says to maximize the geometric margin.

### Logistic regression, maximum likelihood estimation
Another way to deal with noisy labels is to model how these labels are generated. One simple model is a logistic regression model. Here, we assign a probability distribution over the 2 labels so the furthe away an example is from a decision boundary, the more likely it is to be correct. Formally
```math
P(Y=1| x, \theta, \theta_0) = g(\theta^Tx + \theta_0)
```
where we define $g(z) = (1 + exp(-z))^{-1}$ to be the logistic function. To derive this, the intuition might be to say that the *log-odds* of the probabilities should be a linear function of the inputs
```math
log\frac{P(y=1x, \theta, \theta_0)}{P(Y= -1|x, \theta, \theta_0)} = \theta^Tx + \theta_0
```
Then, when we predict both probabilities to be equal, then the log-odds term is 0 and we get back our decision boundary. We will derive our form of the logistic function later.

To compare this logistic regression model with the SVM, we will rewrite $P(y|x, \theta, \theta_0). Note $1 - g(z) = g(-z)$. Then,
```math
P(y = -1|x, \theta, \theta_0) = 1 - P(y=1|x,\theta,\theta_0) = 1- g(\theta^Tx + \theta_0) = g\left(-(\theta^Tx + \theta_0)\right)
```
Therefore,
```math
P(y|x,\theta,\theta_0) = g(y(\theta^Tx + \theta_0))
```
This now gives a linear classifier that makes probabilistic predictions about labels. How do we train these models? It makes sense to try and maximize the probability that we predict the right labels for each example. We can assume independence, giving us the following likelihood functionn,
```math
L(\theta,\theta_0) = \prod^n_{t=1}P(y_t|x_t,\theta,\theta_0)
```
We call $L(\theta,\theta_0)$ the (conditional) likelihood function for a fixed data set. Maximizing $L(\theta,\theta_0)$ gives the *maximum likelihood estimates* of the parameters.

Maximum likelihood *estimators*, which are functions mapping data to parameter values, have nice properties. Assuming the right model class (LRM), and certain conditions, then the estimator is 
- *consistent*: we get the parameter values in the limit of a many training examples
- *efficient*: the fastest to converge to the correct parameter values *in the mean-squared sense*

If our assumptions are wrong, neither property may hold. These estimators are a subset of *M-estimators*, where better ones can be found.

For ease of use, we can *maximize* the log form of the likelihood function
```math
l(\theta,\theta_0) = \sum_{t=1}^n log\left(P(y_t|x_t,\theta,\theta_0)\right)
```

Alternatively, *minimize* the negative log
```math
\begin{align*}
-l(\theta,\theta_0) &= \sum_{t=1}^n -log\left(P(y_t|x_t,\theta,\theta_0)\right) \\
&= \sum_{t=1}^n -log\left(g(y_t(\theta^Tx_t + \theta_0))\right) \\
&= \sum_{t=1}^n -log\left[1+exp(-y_t(\theta^Tx_t + \theta_0))\right]
\end{align*}
```
This looks similar to sum of the hinge losses in the SVM approach. The loss depends only on the margin for each example. Note here that we have a clear *strength* for each prediction, ie, $P(y_t|x_t,\theta,\theta_0)$. Note also that these are not necessary *calibrated*, which means they correspond to observed frequencies.

The above minimization problem is convex, so we have many optimization methods available, including gradient descent. For stochastic gradient descent, we'd modify the parameters in response to each term in the sum. We need the following derivatives
```math
\begin{align*}
\frac{d}{d\theta_0} log\left[ 1 + exp\left( -y_t(\theta^Tx_t + \theta_0)\right) \right] &= -y_t \frac{exp\left(-y_t(\theta^Tx_t + \theta_0) 
  \right)}{1 + exp\left(-y_t(\theta^Tx_t + \theta_0)\right)} \\
  &= -y_t\left[1 - P(y_t|x_t,\theta,\theta_0)\right]
\end{align*}
```
and
```math
\frac{d}{d\theta} log\left[ 1 + exp\left( -y_t(\theta^Tx_t + \theta_0)\right) \right] = -y_tx_t\left[1 - P(y_t|x_t,\theta,\theta_0)\right]
```
We then update our parameters by selecting *random training examples* and moving the parameters in teh opposite direction of the derivatives:
```math
\begin{align*}
    \theta_0 &\leftarrow \theta_0 + \eta \cdot y_t\left[1 - P(y_t|x_t,\theta,\theta_0)\right] \\
    \theta &\leftarrow \theta + \eta \cdot y_tx_t\left[1 - P(y_t|x_t,\theta,\theta_0)\right]
\end{align*}
```
where $\eta \in \mathbb{P}$ is a small learning rate. Here, $1 - P(y_t|x_t,\theta,\theta_0)$ is the probability of making a mistake in the training label. These updates then resemble the perceptron mistake=driven updates, but here we "grade" our updates based on the proportion of making of a mistake.

This algorithm leads to no signifcant change on average when the gradient of the entire objective equals $0$. Setting it to $0$ is also a condition of optimality
```math
\begin{align*}

\end{align*}
