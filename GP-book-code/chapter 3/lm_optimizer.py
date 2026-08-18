# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 3 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).
#This is Im_optimizer file.
import numpy as np
from scipy.optimize import least_squares

def residual_function(coefficients, X, y):
    """
    The residual function for the linear model.
    y_predicted = X @ coefficients
    residual = y_predicted - y
    """
    y_predicted = X @ coefficients
    return y_predicted - y

def lm_optimizer(simplified_matrix, training_outputs):
    """
    Performs Levenberg-Marquardt optimization to find the best coefficients for a linear model.
    This replaces the LM.java and LM_LDEPc_Optimizer logic.

    Args:
        simplified_matrix (np.ndarray): The matrix of independent register values (features).
        training_outputs (np.ndarray): The target values.

    Returns:
        np.ndarray: The optimized coefficients for the simplified matrix.
    """
    num_coeffs = simplified_matrix.shape[1]
    if num_coeffs == 0:
        return np.array([])

    # Initial guess for the coefficients
    a_initial = np.random.uniform(-10, 10, size=num_coeffs)

    try:
        # Use least_squares to find the optimal coefficients.
        result = least_squares(
            residual_function,
            a_initial,
            args=(simplified_matrix, training_outputs),
            method='lm',
            max_nfev=100  # From original Java code
        )
        return result.x
    except Exception as e:
        print(f"LM optimizer failed: {e}")
        # Return a zero vector of the correct shape if optimization fails
        return np.zeros_like(a_initial)