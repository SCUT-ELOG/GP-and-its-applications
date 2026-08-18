# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 3 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).
#This is utils file.
import random
import numpy as np
import copy
import os

def randval(a, b):
    """
    Returns a random float in the range [a, b).
    """
    return random.uniform(a, b)

def save_text(filepath, text, append=False):
    """
    Save text to a file. Creates directory if it doesn't exist.
    """
    mode = 'a' if append else 'w'
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode, encoding='utf-8') as f:
            f.write(text)
    except IOError as e:
        print(f"Error writing to file {filepath}: {e}")
        exit(-1)

def copy_gene(gene_b):
    """
    Returns a deep copy of a Gene object.
    """
    return copy.deepcopy(gene_b)

def standard_deviation(x):
    """
    Calculates the population standard deviation of an array.
    """
    return np.std(x)

def protected_div(a, b):
    """
    Protected division to avoid division by zero.
    Works for scalars and arrays via broadcasting.
    Computes: a / b, but if |b| < 1e-6 then a / sqrt(1 + b^2)
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    denom = np.where(np.abs(b_arr) < 1e-6, np.sqrt(1.0 + b_arr * b_arr), b_arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = a_arr / denom
    return res