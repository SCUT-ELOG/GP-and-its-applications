"""Minimal SGP example used by the repository quick start."""

import numpy as np

from sgp import SGP


def main():
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = X[:, 0] ** 2

    model = SGP(
        pop_size=200,
        generations=20,
        seed=42,
        verbose=False,
    ).fit(X, y)

    print("发现的表达式：", model.best_expression_)
    print("R2：", model.score(X, y))
    model.plot_fitness("fitness_evolution.svg")


if __name__ == "__main__":
    main()
