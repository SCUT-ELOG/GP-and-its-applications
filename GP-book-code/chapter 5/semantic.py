# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 5 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

from deap import creator,base,tools,gp,algorithms
# import MOGP as mogp
import math
from scipy import spatial
import numpy as np

baseStructure={'0','1','2',
'x1','x2','x3','x4','x5',
'exp(x1)','exp(x2)','exp(x3)','exp(x4)','exp(x5)',
'sqrt(x1)','sqrt(x2)','sqrt(x3)','sqrt(x4)','sqrt(x5)',
# 三角万能
'sin(x1)','sin(x2)','sin(x3)','sin(x4)','sin(x5)',
'cos(x1)','cos(x2)','cos(x3)','cos(x4)','cos(x5)', 
# 倍角
'sin(mul(2,x1))','sin(mul(2,x2))','sin(mul(2,x3))','sin(mul(2,x4))','sin(mul(2,x5))',
'cos(mul(2,x1))','cos(mul(2,x2))','cos(mul(2,x3))','cos(mul(2,x4))','cos(mul(2,x5))', 
# 两角�? sin(x +- y) = sin(x)cos(y) +- cos(x)sin(y)
'sin(add(x1,x1))','sin(add(x2,x2))','sin(add(x3,x3))','sin(add(x4,x4))','sin(add(x5,x5))', 
'sin(add(x1,x2))','sin(add(x1,x3))','sin(add(x1,x4))','sin(add(x1,x5))','sin(add(x2,x3))','sin(add(x2,x4))','sin(add(x2,x5))','sin(add(x3,x4))','sin(add(x3,x5))','sin(add(x4,x5))',
'sin(sub(x1,x2))','sin(sub(x1,x3))','sin(sub(x1,x4))','sin(sub(x1,x5))','sin(sub(x2,x3))','sin(sub(x2,x4))','sin(sub(x2,x5))','sin(sub(x3,x4))','sin(sub(x3,x5))','sin(sub(x4,x5))',
#       cos(x +- y) = cos(x)cos(y) +- sin(x)sin(y)
'cos(add(x1,x1))','cos(add(x2,x2))','cos(add(x3,x3))','cos(add(x4,x4))','cos(add(x5,x5))', 
'cos(add(x1,x2))','cos(add(x1,x3))','cos(add(x1,x4))','cos(add(x1,x5))','cos(add(x2,x3))','cos(add(x2,x4))','cos(add(x2,x5))','cos(add(x3,x4))','cos(add(x3,x5))','cos(add(x4,x5))',
'cos(sub(x1,x2))','cos(sub(x1,x3))','cos(sub(x1,x4))','cos(sub(x1,x5))','cos(sub(x2,x3))','cos(sub(x2,x4))','cos(sub(x2,x5))','cos(sub(x3,x4))','cos(sub(x3,x5))','cos(sub(x4,x5))',
# 积化和差 'sin(x)*cos(y)' 'cos(x)*sin(y)' 'cos(x)*cos(y)'    
'mul(sin(x1),cos(x1))','mul(sin(x2),cos(x2))','mul(sin(x3),cos(x3))','mul(sin(x4),cos(x4))','mul(sin(x5),cos(x5))',
'mul(sin(x1),cos(x2))','mul(sin(x1),cos(x3))','mul(sin(x1),cos(x4))','mul(sin(x1),cos(x5))','mul(sin(x2),cos(x3))','mul(sin(x2),cos(x4))','mul(sin(x2),cos(x5))','mul(sin(x3),cos(x4))','mul(sin(x3),cos(x5))','mul(sin(x4),cos(x5))'
'mul(cos(x1),sin(x1))','mul(cos(x2),sin(x2))','mul(cos(x3),sin(x3))','mul(cos(x4),sin(x4))','mul(cos(x5),sin(x5))',
'mul(cos(x1),sin(x2))','mul(cos(x1),sin(x3))','mul(cos(x1),sin(x4))','mul(cos(x1),sin(x5))','mul(cos(x2),sin(x3))','mul(cos(x2),sin(x4))','mul(cos(x2),sin(x5))','mul(cos(x3),sin(x4))','mul(cos(x3),sin(x5))','mul(cos(x4),sin(x5))'
'mul(cos(x1),cos(x1))','mul(cos(x2),cos(x2))','mul(cos(x3),cos(x3))','mul(cos(x4),cos(x4))','mul(cos(x5),cos(x5))',
'mul(cos(x1),cos(x2))','mul(cos(x1),cos(x3))','mul(cos(x1),cos(x4))','mul(cos(x1),cos(x5))','mul(cos(x2),cos(x3))','mul(cos(x2),cos(x4))','mul(cos(x2),cos(x5))','mul(cos(x3),cos(x4))','mul(cos(x3),cos(x5))','mul(cos(x4),cos(x5))'
# 和差化积
'add(sin(x1),sin(x1))','add(sin(x2),sin(x2))','add(sin(x3),sin(x3))','add(sin(x4),sin(x4))','add(sin(x5),sin(x5))',
'add(sin(x1),sin(x2))','add(sin(x1),sin(x3))','add(sin(x1),sin(x4))','add(sin(x1),sin(x5))','add(sin(x2),sin(x3))','add(sin(x2),sin(x4))','add(sin(x2),sin(x5))','add(sin(x3),sin(x4))','add(sin(x3),sin(x5))','add(sin(x4),sin(x5))'
'sub(sin(x1),sin(x2))','sub(sin(x1),sin(x3))','sub(sin(x1),sin(x4))','sub(sin(x1),sin(x5))','sub(sin(x2),sin(x3))','sub(sin(x2),sin(x4))','sub(sin(x2),sin(x5))','sub(sin(x3),sin(x4))','sub(sin(x3),sin(x5))','sub(sin(x4),sin(x5))'
'add(cos(x1),cos(x1))','add(cos(x2),cos(x2))','add(cos(x3),cos(x3))','add(cos(x4),cos(x4))','add(cos(x5),cos(x5))',
'add(cos(x1),cos(x2))','add(cos(x1),cos(x3))','add(cos(x1),cos(x4))','add(cos(x1),cos(x5))','add(cos(x2),cos(x3))','add(cos(x2),cos(x4))','add(cos(x2),cos(x5))','add(cos(x3),cos(x4))','add(cos(x3),cos(x5))','add(cos(x4),cos(x5))'
'sub(cos(x1),cos(x2))','sub(cos(x1),cos(x3))','sub(cos(x1),cos(x4))','sub(cos(x1),cos(x5))','sub(cos(x2),cos(x3))','sub(cos(x2),cos(x4))','sub(cos(x2),cos(x5))','sub(cos(x3),cos(x4))','sub(cos(x3),cos(x5))','sub(cos(x4),cos(x5))'
# x+y
'add(x1,x1)','add(x2,x2)','add(x3,x3)','add(x4,x4)','add(x5,x5)',
'add(x1,x2)','add(x1,x3)','add(x1,x4)','add(x1,x5)','add(x2,x3)','add(x2,x4)','add(x2,x5)','add(x3,x4)','add(x3,x5)','add(x4,x5)',
# x-y
'sub(x1,x2)','sub(x1,x3)','sub(x1,x4)','sub(x1,x5)','sub(x2,x3)','sub(x2,x4)','sub(x2,x5)','sub(x3,x4)','sub(x3,x5)','sub(x4,x5)',
# x*y
'mul(x1,x1)','mul(x2,x2)','mul(x3,x3)','mul(x4,x4)','mul(x5,x5)',
'mul(x1,x2)','mul(x1,x3)','mul(x1,x4)','mul(x1,x5)','mul(x2,x3)','mul(x2,x4)','mul(x2,x5)','mul(x3,x4)','mul(x3,x5)','mul(x4,x5)',
# x/y
'div(x1,x2)','div(x1,x3)','div(x1,x4)','div(x1,x5)','div(x2,x3)','div(x2,x4)','div(x2,x5)','div(x3,x4)','div(x3,x5)','div(x4,x5)',
# (a + b)^2 = a^2 + 2ab + b^2 
'mul(add(x1,x2),add(x1,x2))','mul(add(x1,x3),add(x1,x3))','mul(add(x1,x4),add(x1,x4))','mul(add(x1,x5),add(x1,x5))','mul(add(x2,x3),add(x2,x3))','mul(add(x2,x4),add(x2,x4))','mul(add(x2,x5),add(x2,x5))','mul(add(x3,x4),add(x3,x4))','mul(add(x3,x5),add(x3,x5))','mul(add(x4,x5),add(x4,x5))',
}

# library={'0','1','2',
#     'x',
#     'exp(x)',
#     'sqrt(x)',
#     #三角万能
#     'sin(x)',  sinα=2tan(α/2)/[1+tan^2(α/2)]
#     'cos(x)',   cosα=[1-tan^2(α/2)]/[1+tan^2(α/2)]
#     # 倍角
#     'sin(2*x)'  sin(2α)=2sinα·cosα
#     'cos(2*x)'  cos(2α)=cos^2(α)-sin^2(α)=2cos^2(α)-1=1-2sin^2(α)
#     #两角�?
#     'sin(x+y)',  sin(x+-y) = sin(x)cos(y) +- cos(x)sin(y) 
#     'sin(x-y)',
#     'cos(x+y)',  cos(x+-y) = cos(x)cos(y) +- sin(x)sin(y)
#     'cos(x-y)',
#     # 积化和差
#     'sin(x)*cos(y)'      sinα·cosβ=(1/2)[sin(α+β)+sin(α-β)] 
#     'cos(x)*sin(y)'     cosα·sinβ=(1/2)[sin(α+β)-sin(α-β)]
#     'cos(x)*cos(y)'     cosα·cosβ=(1/2)[cos(α+β)+cos(α-β)]
#     # 和差化积
#     'sin(x)+sin(y)', sin(x)+sin(y) = 2sin[(α+β)/2]cos[(α-β)/2]   
#     'sin(x)-sin(y)',
#     'cos(x)+cos(y)',
#     'cos(x)-cos(y)',  
#       #其他   
#       'x+y'
#       'x-y'
#       'x*y'
#       'x/y'
#       '(a + b)^2'       (a + b)^2 = a^2 + 2ab + b^2 
# }


# 计算库中构建输出值，保存在字典中
def generateLibrary(n_variables,baseStructure,data,toolbox):
    library = {}
    for base in baseStructure:
        flag = False
        if n_variables<5:
            for num in range(n_variables+1,5+1):
                if str(num) in base:
                    flag = True
        if flag:
            continue
        func = toolbox.compile(base)
        
        try:
            if n_variables == 1:
                sqerrors = ((func(x1)) for x1,y in data)
            if n_variables == 2:
                sqerrors = ((func(x1,x2)) for x1,x2,y in data)
            if n_variables == 3:
                sqerrors = ((func(x1,x2,x3)) for x1,x2,x3,y in data)
            if n_variables == 4:
                sqerrors = ((func(x1,x2,x3,x4)) for x1,x2,x3,x4,y in data)
            if n_variables == 5:
                sqerrors = ((func(x1,x2,x3,x4,x5)) for x1,x2,x3,x4,x5,y in data)
            obj_vec = [data for data in sqerrors]
        except OverflowError:
            obj_vec = float('inf')
            print('OverflowError',str(base))
        except ValueError:
            obj_vec = float('inf')
            print('ValueError',str(base))
        # obj_vec = [data for data in sqerrors]
        
        library[base] = obj_vec
        # print(obj_vec)
        # print('构件:',base,obj_vec)
    return library

def semanticSimply(individual,n_variables,data,toolbox,pset):
    # 构建�?
    LIBRARY = generateLibrary(n_variables,baseStructure,data,toolbox)
    # 遍历所有节�?
    sub_index=0
    while sub_index < len(individual):
        # 1、计算出每个子树的输出向�?
        slice_ = individual.searchSubtree(sub_index)
        if isinstance(individual[sub_index], gp.Terminal):
            sub_index+=1
            continue #此处跳出terminal节点
        sub_expr = gp.PrimitiveTree(individual[slice_])
        # print(sub_index,individual.searchSubtree(sub_index),sub_expr)
        func = toolbox.compile(sub_expr)
        try:
            if n_variables == 1:
                sqerrors = (func(x1) for x1,y in data)
            if n_variables == 2:
                sqerrors = (func(x1,x2) for x1,x2,y in data)
            if n_variables == 3:
                sqerrors = (func(x1,x2,x3) for x1,x2,x3,y in data)
            if n_variables == 4:
                sqerrors = (func(x1,x2,x3,x4) for x1,x2,x3,x4,y in data)
            if n_variables == 5:
                sqerrors = (func(x1,x2,x3,x4,x5) for x1,x2,x3,x4,x5,y in data)
            obj_vec = [data for data in sqerrors]
        except OverflowError:
            obj_vec = float('inf')
            print('OverflowError',str(individual))
        except ValueError:
            obj_vec = float('inf')
        # print('子树:',gp.PrimitiveTree(individual[slice_]),obj_vec)

        # 2、和库中构建输出值进行对比，计算向量差异
        candidateSubTree=''
        for key,value in LIBRARY.items():
            distance = sum(abs(a - b) for a, b in zip(value, obj_vec))
            if distance < 1:
                candidateSubTree = key
                break
        # 3、替�?
        subTree = gp.PrimitiveTree.from_string(candidateSubTree,pset)
        if  len(candidateSubTree) != 0:
            individual[slice_] = subTree
        sub_index+=1
    return individual

