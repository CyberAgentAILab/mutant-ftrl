# Mutation-Driven Follow the Regularized Leader for Last-Iterate Convergence in Zero-Sum Games
Code for reproducing results in the paper "[Mutation-Driven Follow the Regularized Leader for Last-Iterate Convergence in Zero-Sum Games](https://proceedings.mlr.press/v180/abe22a.html)".

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

### For Docker User
Build the container:
```bash
$ docker build -t mutant-ftrl .
```
After build finished, run the container:
```bash
$ docker run -it mutant-ftrl
```

## Run Experiments
In order to investigate the performance of M-FTRL in biased Rock-Paper-Scissors under full-information feedback, execute the following command:
```bash
$ python run_experiment.py
```
In this experiment, the following options can be specified:
* `--game`: Name of a matrix game. The default value is `biased_rps`.
* `--num_trials`: Number of trials to run experiments. The default value is `1`.
* `--T`: Number of iterations. The default value is `10000`.
* `--feedback`: Type of feedback given to players. The default value is `full`.
* `--seed`: Random seed. The default value is `0`.
* `--random_init_strategy`: Whether to generate the initial strategy uniformly at random. The default value is `False`.

To evaluate M-FTRL via an experiment in biased Rock-Paper-Scissors under bandit feedback, execute the following command:
```bash
$ python run_experiment.py --feedback=bandit
```

## Citation
If you use our code in your work, please cite our paper:
```
@InProceedings{abe22mftrl,
  title = 	 {Mutation-driven follow the regularized leader for last-iterate convergence in zero-sum games},
  author =       {Abe, Kenshi and Sakamoto, Mitsuki and Iwasaki, Atsushi},
  booktitle = 	 {Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {1--10},
  year = 	 {2022},
  volume = 	 {180}
}
```