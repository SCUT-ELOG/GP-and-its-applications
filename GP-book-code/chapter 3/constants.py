# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 3 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

# This is constants file
"""
This file contains the constants for the MLDEP project, translated from Constants.java.
"""

MAXGENS = 150000
POPSIZE = 20
MAXEVALS = MAXGENS * POPSIZE
NVARS = 48
# Note from Java code: For F10-F11, MAX_RG_NUMS should be 60. For F12, MAX_RG_NUMS should be 80.
MAX_RG_NUMS = 100
CR_NUMS = 50
C_RATE = 0.05
MAXINPUTS = 365000
MAX_VARIABLES = 15
MAX_DEGREE = 4
FUNCTION_NUM = 6  # +, -, *, /, sin, cos. Original code commented out exp, log
MAX_RUNS = 30