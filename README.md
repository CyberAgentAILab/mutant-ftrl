# Mutation-Driven Follow the Regularized Leader for Last-Iterate Convergence in Zero-Sum Games
Code for reproducing results in the paper "Mutation-Driven Follow the Regularized Leader for Last-Iterate Convergence in Zero-Sum Games".

## About
In this study, we consider a variant of the Follow the Regularized Leader (FTRL) dynamics in two-player zero-sum games.
FTRL is guaranteed to converge to a Nash equilibrium when time-averaging the strategies, while a lot of variants suffer from the issue of limit cycling behavior, i.e., lack the last-iterate convergence guarantee.
To this end, we propose mutant FTRL (M-FTRL), an algorithm that introduces mutation for the perturbation of action probabilities.
We then investigate the continuous-time dynamics of M-FTRL and provide the strong convergence guarantees toward stationary points that approximate Nash equilibria under full-information feedback.
Furthermore, our simulation demonstrates that M-FTRL can enjoy faster convergence rates than FTRL and optimistic FTRL under full-information feedback and surprisingly exhibits clear convergence under bandit feedback.

## Installation
This code is written in Python 3.
To install the required dependencies, execute the following command:
```bash
$ pip install -r requirements.txt
```

## Run Experiments
In order to investigate the performance of M-FTRL in biased Rock-Paper-Scissors under full-information feedback, execute the following command:
```bash
$ python run_full_feedback_experiment.py
```

To evaluate M-FTRL via an experiment in biased Rock-Paper-Scissors under bandit feedback, execute the following command:
```bash
$ python run_bandit_feedback_experiment.py
```
