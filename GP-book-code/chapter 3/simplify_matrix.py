# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 3 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).
#This is simplify_matrix file.
import numpy as np

def get_independent_columns_indicator(matrix):
    """
    Identifies a set of linearly independent columns using rank analysis.
    Returns a boolean array indicating which columns to keep.
    This is a robust replacement for the logic in SimplifyMatrix.java.
    """
    num_cols = matrix.shape[1]
    indicator = np.zeros(num_cols, dtype=bool)

    try:
        rank = np.linalg.matrix_rank(matrix)
    except np.linalg.LinAlgError:
        # Handle cases where matrix rank cannot be computed
        return indicator

    if rank == 0:
        return indicator

    # Iteratively find a set of independent columns
    independent_cols_indices = []
    current_rank = 0
    for i in range(num_cols):
        # Try adding the current column
        potential_set_indices = independent_cols_indices + [i]
        sub_matrix = matrix[:, potential_set_indices]
        
        # If the rank increases, this column is independent of the previous ones
        try:
            new_rank = np.linalg.matrix_rank(sub_matrix)
            if new_rank > current_rank:
                independent_cols_indices.append(i)
                current_rank = new_rank
        except np.linalg.LinAlgError:
            # If rank computation fails, skip this column
            continue
        
        # Stop when we've found enough columns
        if len(independent_cols_indices) == rank:
            break
            
    indicator[independent_cols_indices] = True
    return indicator