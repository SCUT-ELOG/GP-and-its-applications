# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 3 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).
#This is gene file.
import numpy as np
import constants

class Gene:
    """
    Represents a single individual (a gene) in the population.
    Translated from GENE.java.
    """
    def __init__(self):
        self.x = np.zeros(constants.NVARS)
        self.coefficients = np.zeros(constants.MAX_RG_NUMS)
        self.f = 1e10  # Training fitness, initialized to a large value
        self.tf = 1e10 # Testing fitness, initialized to a large value
        self.nodeCount = 0

    def __str__(self):
        return f"Gene(f={self.f:.6f}, tf={self.tf:.6f}, nodeCount={self.nodeCount})"