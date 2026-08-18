# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 5 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

import numpy

# dimension
def ADD(list1, list2):
    return [0.5 * (i + j) for i, j in zip(list1, list2)]

def SUB(list1, list2):
    return [0.5 * (i + j) for i, j in zip(list1, list2)]

def MAX(list1, list2):
    return [0.5 * (i + j) for i, j in zip(list1, list2)]

def MIN(list1, list2):
    return [0.5 * (i + j) for i, j in zip(list1, list2)]

def MUL(list1, list2):
    return [i + j for i, j in zip(list1, list2)]

def DIV(list1, list2):
    return [i - j for i, j in zip(list1, list2)]

def EXP(list):
    return [0 for i in list]

def LOG(list):
    return [0 for i in list]

def SIN(list):
    return [0 for i in list]

def COS(list):
    return [0 for i in list]


class TreeNode(object):
    def __init__(self, symbol, dim=[0, 0, 0, 0, 0], dimGap=0, left=None, right=None):
        self.symbol = symbol
        self.dim = dim
        self.dimGap = dimGap
        self.left = left
        self.right = right


def buildTree(expression, variables):
    if len(expression) == 1:
        return TreeNode(expression, variables[expression[0]])

    operator = expression[0]

    if operator in ['add', 'sub', 'mul', 'div']:
        # binary
        i = 0
        count = 0
        left_begin = left_end = right_begin = right_end = -1
        while i < len(expression):
            if expression[i] == '(':
                if (count == 0):
                    left_begin = i + 1
                count += 1
            elif expression[i] == ')':
                count -= 1
                if (count == 0):
                    right_end = i - 1
            elif expression[i] == ',':
                if (count == 1):
                    left_end = i - 1
                    right_begin = i + 1
            i += 1
        node = TreeNode(operator)
        node.left = buildTree(expression[left_begin:left_end + 1], variables)
        node.right = buildTree(expression[right_begin:right_end + 1], variables)
        return node

    # 单目 统一建在右子树
    if operator in ['exp', 'log', 'sin', 'cos', 'sqrt']:
        node = TreeNode(operator)
        node.left = None
        node.right = buildTree(expression[2:-1], variables)
        return node

# 后序
def postorderComputeDim(root: TreeNode):
    dimGapSum = [0]
    def traversal(root: TreeNode,dimGapSum):
        if root == None:
            return
        if root.left == None and root.right == None: #叶子节点
            return root.dim
        elif root.left and root.right:  #双目运算符
            leftdim = traversal(root.left,dimGapSum)    # 左
            rightdim = traversal(root.right,dimGapSum)   # 右
            #计算dimension和dimGap
            root.dim = globals()[root.symbol.upper()](leftdim,rightdim)
            gap = [i-j for i,j in zip(leftdim,rightdim)]
            if root.symbol in ['mul','div']:
                root.dimGap = 0
            else:
                root.dimGap = sum([abs(i) for i in gap])
            dimGapSum[0] += root.dimGap
            # print(root.symbol,root.dimGap)
            return root.dim
        elif root.right:               #单目目运算符
            dim = traversal(root.right,dimGapSum)
            #计算dimension和dimGap
            root.dimGap = sum([abs(i) for i in root.right.dim])
            # print(root.dimGap)
            dimGapSum[0] += root.dimGap
            root.dim = [0,0,0]
            return root.dim
    traversal(root,dimGapSum)
    return dimGapSum




def computeDimGap(expr,variables):
    expression = expr.replace(' ', '').replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , ').split()
    # 解析表达式建树
    root = buildTree(expression,variables)
    # 后序计算dimension、dimGap同时计算dimGap总和
    dimGapSum = postorderComputeDim(root)
    # inorderTraversal(root)
    return dimGapSum[0]
