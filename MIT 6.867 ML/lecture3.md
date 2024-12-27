# Machine Learning - MIT OCW 6.867
## Lecture 3 - The Support Vector Machine

**Recall**, we based our perceptron algorithm on the fact that there existed a large margin classifier. This is the **Support Vector Machine**

Our intuition might be to find a *correct classifier*, then increasing the geometric margin until the classifier "locks in place" at the point where we cannot increase the margin any further. The solution is unique.

More formally, we will solve an optimization problem to maximize $\gamma_{geom}$. We will need this classifier to be correct for all training examples, ie. $y_t\theta^Tx_t \geq \gamma$ for all $t = 1, \dots, n$. We want to maximize $\gamma/||\theta||$ which is the geometric margin. Equivalently, we can minimize the inverse squared: $\frac{1}{2}(||\theta||/\gamma)^2$.

We now want to solve the following optimization problem:
```math
\text{minimize } \frac{1}{2}||\theta||^2/\gamma^2 \text{ subject to } y_t\theta^Tx_t \geq \gamma \text{ for all } t = 1, \dots, n
```
Since our problem indicates information only about the ratio $\theta/\gamma$. We can scale $\theta$ by a constant, which doesn't affect the decision boundary thereby letting us fix $\gamma = 1$ and to solve for $\theta$ in the following:

```math
\text{minimize } \frac{1}{2}||\theta||^2\text{ subject to } y_t\theta^Tx_t \geq 1 \text{ for all } t = 1, \dots, n
```
This is called **standard SVM form** and is a *quadratic programming* problem. The solution $\hat{\theta}$ is unique and produces the geometric margin of $1/||\hat{\theta}||$. Note, our fixing of $\gamma$ did not affect any values.

### General formulation, offset parameter
We allow for generalization but not forcing the decision boundary to cross the origin.
```math
f(x; \theta, \theta_0) = sign(\theta^Tx +\theta_0)
```
Here, $\theta$ is the normal to the separating hyper plane and $\theta_0 \in \mathbb{R}$ is an offset. Again, we find the equation for the separating hyper-plane by setting $\theta^Tx +\theta_0 = 0$. This is the **general equation** for a hyper-plane. 

Our optimization problems changes slightly:
```math
\text{minimize } \frac{1}{2}||\theta||^2\text{ subject to } y_t(\theta^Tx_t + \theta_0) \geq 1 \text{ for all } t = 1, \dots, n
```

We see this offset affects only the constraints and is equivalent to adding a constant component to our examples to our previous linear classifier.

### Properties of the maximum margin linear classifier
#### Benefits:
- These classifiers are based on the perceptron algorithm and therefore *good references*.
- This produces unique solutions on linearly separable training sets, ie. dividable by a hyper-plane
- The boundary is dependent on only a subset of the examples, which are those that appear on the margin. These examples are called *support vectors*.

A method for performance evaluation is *cross-validation*. This is a method of retraining the classifier with subsets of the original examples and testing them on the remaining examples. One version is the *leave-one-out cross-validation*. 

The method works as follows: for each example, train the classifier on all other examples and test the classifier for the single example. Sum all the errors.

Formally, let "$-i$" denote the parameters obtaining by excluding the $i^{th}$ example. Then:
```math
\text{leave-one-out CV error } = \frac{1}{n}\sum_{i=1}^n Loss\left(y_i, f(x_i; \theta^{-i}, \theta_0^{-i})\right)\
```
This is a method to gauge how well the classifier would generalize to each training example. Having a low leave-one-out CV error indicates that a classifier might generalize well.

What is the LOU CV error of a maximum magin linear classifier? Examples outside the margin would be classified correctly regardless of if they are in the training set. Support vectors, on the other hand, are key to defining the linear separator and may be misclassified if excluded from the training set.

Then, we have the following upper bound on the LOU CV error:
```math
LOU  \ CV \  error \leq \frac{\text{\# of support vectors}}{n}
```
Then, a *spare solution* (one with a small number of support vectors), is advantageous.

#### Problems:
One mislabeled example can radically change a classifier.

### Allowing misclassified examples, relaxation
Practically, labeling errors are common. When examples are hard to classify, we can't usually tell if its because a linear classifier doesn't fit the problem, or if the data is mislabelled. 

To permit errors, a simple way is to include "slack" variables for constraints in our problem. In other words, we associate a cost to how much a margin constraint is violated. Then, in our optimization problem, we want to minimize the costs of violating constraints **and** the norm of our parameter vector. This new problem is as follows
```math
\begin{align*}
    &\text{minimize }\frac{1}{2}||\theta||^2 + C\sum_{i=1}^n \xi_t \\
    &\text{subject to } y_t(\theta^T x_t + \theta_0) \geq 1 - \xi_t \land \xi_t \geq 0 \ \forall t=1,\dots,n
\end{align*}
```
where $\xi_t$ are slack variables. Essentially, for we have a trade-off between $C\xi_t$, the cost of a violation, and the reduction in our norm. Increasing the penalty $C$ forces our violations to be small, while a small $C$ allows for many violations. In this case, our set of support vectors are all examples on the margin, inside the margin, and misclassified.