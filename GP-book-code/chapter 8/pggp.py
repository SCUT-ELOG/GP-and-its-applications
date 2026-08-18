# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 8 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

import time
from src.nesymres.architectures.model import Model
import numpy as np
import operator
import math
import os
import random
from functools import partial
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import json, re
import omegaconf
import sympy
from src.nesymres.dclasses import FitParams, BFGSParams
from backpropagation import *
import warnings
import logging
import sys
from contextlib import contextmanager

warnings.filterwarnings("ignore")
# 禁用PyTorch Lightning的日志输出
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
# 禁用特定的升级警告
warnings.filterwarnings("ignore", message="Lightning automatically upgraded your loaded checkpoint")

@contextmanager
def suppress_stdout():
    """临时禁用标准输出"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

import re
import json
import omegaconf
import sympy
import copy
from sympy import expand

def handle_negatives(formula):
    # Define a string that matches a negative sign at the start, after an operator, or after an opening parenthesis
    negative_sign_pattern = r'((?<=^)|(?<=[*+/])|(?<=\())-'

    # Define a string that matches a number
    number_pattern = r'(\d+\.\d+|\d+)'

    # Define a string that matches a variable
    variable_pattern = r'([a-zA-Z_]\w*)'

    # Combine the patterns into two separate patterns
    negative_number_pattern = negative_sign_pattern + number_pattern
    negative_variable_pattern = negative_sign_pattern + variable_pattern

    # Replace negative numbers with 1
    formula = re.sub(negative_number_pattern, r'1', formula)

    # Replace negative variables with 1*variable
    formula = re.sub(negative_variable_pattern, r'1*\2', formula)

    return formula

def expand_power(expression):
    return str(expand(expression))
def expand_repeat(expression):
    def expand(match):
        base = match.group(1)
        exponent = int(match.group(2))
        return '*'.join([base] * exponent)

    def recursive_expand(expr):
        pattern = re.compile(r'\(([^()]+)\)\*\*(\d+)')
        simple_pattern = re.compile(r'([a-zA-Z_]*\d*)\*\*(\d+)')
        while '**' in expr:
            # for expressions in parentheses
            expr = re.sub(pattern, expand, expr)
            # for simple expressions without parentheses
            expr = re.sub(simple_pattern, expand, expr)
        return expr

    return recursive_expand(expression)

def reverse_add_sub(expr):
    # Helper function to find the index of the character that closes the first open parenthesis.
    def find_closing_paren(expr, open_pos):
        counter = 1
        for i in range(open_pos + 1, len(expr)):
            if expr[i] == '(':
                counter += 1
            elif expr[i] == ')':
                counter -= 1
                if counter == 0:
                    return i
        return -1  # If no closing parenthesis is found

    # Find the leading negative sign and determine the boundary of its term.
    if expr.lstrip().startswith('-'):
        # Skip whitespace and the leading negative sign
        pos = next((i for i, ch in enumerate(expr) if not ch.isspace()), None) + 1
        # We'll store the index positions where operators appear
        operators = []
        while pos < len(expr):
            char = expr[pos]
            if char == '(':
                # Find the matching closing parenthesis and skip the content inside
                closing_pos = find_closing_paren(expr, pos)
                if closing_pos < 0:
                    # No closing parenthesis found so break
                    break
                else:
                    # Skip past the closing parenthesis
                    pos = closing_pos
            elif char in '+-':
                # Found an operator, so we remember its position
                operators.append(pos)
                break
            # Move to the next character position
            pos += 1

        # If operators are found, determine the expression before and after the found operator
        if operators:
            op_pos = operators[0]
            return expr[op_pos + 1:] + ' - ' + expr[:op_pos].lstrip('-')
        else:
            # If there's no '+' or '-' after the leading '-', simply return the expression as it is
            return expr

    # If the expression does not start with a negative sign, return it unchanged
    return expr

def infix_to_prefix(expression):
    operators = set(['+', '-', '*', '/', 'sin', 'cos', 'tan','exp', 'ln','asin','acos','atan'])

    def is_operator(char):
        return char in operators

    def get_precedence(operator):
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2, 'sin': 3, 'cos': 3, 'tan': 3,'exp': 3, 'ln': 3,'asin': 3,'acos': 3,'atan': 3}
        return precedence.get(operator, 0)

    def infix_to_prefix_recursive(expr):
        stack = []
        output = []
        tokens = re.findall(r'([a-zA-Z_]+\d*|\*\*|\d+\.\d+|\d+|\S)', expr)
        # chatgpt改进之后的
        # tokens = re.findall(r'([a-zA-Z_]+\d*|\*+|\d+\.\d+|\d+|\S)', expr)

        for i, token in enumerate(reversed(tokens)):
            if re.match(r'[a-zA-Z_]+\d*', token):
                output.append(token)
            elif re.match(r'\d+\.\d+|\d+', token):
                output.append(token)
            elif token == ')':
                stack.append(token)
            elif token == '(':
                while stack and stack[-1] != ')':
                    output.append(stack.pop())
                stack.pop()
            elif is_operator(token):
                while stack and get_precedence(stack[-1]) > get_precedence(token):
                    output.append(stack.pop())
                stack.append(token)
            elif token == '**':
                output.append(token)
            elif token == '-' and (i == 0 or (i > 0 and (tokens[i - 1] in operators or tokens[i - 1] == '('))):
                # Handle negative sign as a unary operator for the entire number
                output.append('(-')
                stack.append(')')
            elif token == '-' and i > 0 and re.match(r'\d+\.\d+|\d+', tokens[i - 1]):
                # Handle negative constant
                if i < len(tokens) - 1 and tokens[i + 1] == '(':
                    # Subtract within parentheses
                    output.append('-')
                else:
                    output.append('*')
                    output.append('(-')
                    stack.append(')')

        while stack:
            output.append(stack.pop())

        return ''.join(reversed(output))

    return infix_to_prefix_recursive(expression)

def split_expr(expr):
    functions = ['sin', 'cos', 'tan','exp', 'ln', 'sqrt', 'pow', 'log', 'abs', 'sign', 'floor', 'ceil', 'round', 'trunc','asin','acos','atan']
    allowed_vars = ['x_0', 'x_1', 'x_2', 'x_3', 'c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7']

    # Create patterns for the functions and allowed variables, and escape them
    funcs_pattern = '|'.join(functions)
    allowed_vars_pattern = '|'.join(map(re.escape, allowed_vars))

    # Full regex pattern to match functions, allowed variables and general tokens
    pattern = fr'({funcs_pattern}|{allowed_vars_pattern}|[a-zA-Z]+(?!\d)|\d+\.\d+|\d+|\*|\S)'
    tokens = re.findall(pattern, expr)
    return tokens



def convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """
    prim = copy.copy(prim)
    #prim.name = re.sub(r'([A-Z])', lambda pat: pat.group(1).lower(), prim.name)    # lower all capital letters

    converter = {
        'sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'protectedDiv': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'mul': lambda *args_: "Mul({},{})".format(*args_),
        'add': lambda *args_: "Add({},{})".format(*args_)
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)

def stringify_for_sympy(f):
    """Return the expression in a human readable string.
    """
    string = ""
    stack = []
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string

def transformer_init():
    with open('jupyter/100M/eq_setting.json', 'r') as json_file:
        eq_setting = json.load(json_file)

    cfg = omegaconf.OmegaConf.load("jupyter/100M/config.yaml")
    return eq_setting, cfg


def get_res_transformer(X, y, BFGS, first_call=False):
    input_X = np.array(X)
    input_Y = np.array(y)
    X = torch.from_numpy(input_X)
    y = torch.from_numpy(input_Y)

    eq_setting, cfg = transformer_init()

    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )

    params_fit = FitParams(word2id=eq_setting["word2id"],
                           id2word={int(k): v for k, v in eq_setting["id2word"].items()},
                           una_ops=eq_setting["una_ops"],
                           bin_ops=eq_setting["bin_ops"],
                           total_variables=list(eq_setting["total_variables"]),
                           total_coefficients=list(eq_setting["total_coefficients"]),
                           rewrite_functions=list(eq_setting["rewrite_functions"]),
                           bfgs=bfgs,
                           beam_size=cfg.inference.beam_size
                           # This parameter is a tradeoff between accuracy and fitting time
                           )
    weights_path = "weights/100M.ckpt"
    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    fitfunc = partial(model.fitfunc, cfg_params=params_fit)

    if first_call:
        # 第一次调用，使用BFGS
        params_fit.bfgs.activated = True
        with suppress_stdout():
            _ = fitfunc(X, y)

        final_equation = model.get_equation()

        # 第二次调用，不使用BFGS
        params_fit.bfgs.activated = False
        with suppress_stdout():
            prefix_symbol_list = fitfunc(X, y)

        return prefix_symbol_list, final_equation

    if BFGS:
        try:
            params_fit.bfgs.activated = True
            with suppress_stdout():
                result = fitfunc(X, y)
            prefix_symbol_list = result['best_bfgs_preds']
        except (ValueError, RuntimeError):
            return [None, None]

        final_equation = model.get_equation()

        return prefix_symbol_list, final_equation
    else:
        params_fit.bfgs.activated = False
        with suppress_stdout():
            result = fitfunc(X, y)
        prefix_symbol_list = result['best_bfgs_preds']
        return prefix_symbol_list



def protectedMul(left, right):
    try:
        return left*right
    except OverflowError:
        return 1e7

def protectedDiv(left, right):
    if right == 0:
        return left
    res = left / right
    if res > 1e7:
        return 1e7
    if res < -1e7:
        return -1e7
    return res
def protectedExp(arg):
    if is_complex(arg):
        return 99999
    if arg > 10:
        arg = 10
    return math.exp(arg)

def protectedLog(arg):
    if abs(arg) < 1e-5:
        arg = 1e-5
    return math.log(abs(arg))


def protectedAsin(x):
    if x < -1.0 or x > 1.0:
        return 99999
    else:
        return math.asin(x)

def protectedSqrt(x):
    if x < 0:
        return 99999
    else:
        return math.sqrt(x)

def protectedAcos(x):
    if x < -1.0 or x > 1.0:
        return 99999
    else:
        return math.asin(x)

def protectedAtan(x):
    try:
        return math.atan(x)
    except Exception:
        return 99999


def convert_to_list(trimmed_eq, accurate_constant=False):

    token_list= []
    for i in range(n_variables):
        token_list.append('x_'+str(i+1))

    # 创建一个新列表而不是在迭代时修改原列表
    result = []
    for x in trimmed_eq:
        # 如果是字符串形式的表达式，需要先解析
        if isinstance(x, str) and ('+' in x or '-' in x or '*' in x or '/' in x or '(' in x or ')' in x):
            # 这是一个表达式，不是单个符号，跳过或进行特殊处理
            # 这里我们选择跳过，因为它不是一个有效的符号
            continue
        elif x == 'constant' :
            if not accurate_constant:
                result.append('rand505')
            else:
                result.append(x)
        elif x.startswith('x_'):
            index=int(x[-1])
            result.append('x_'+str(index-1))
        elif isinstance(x, (int, float)):
            # 如果是数字，直接转换为字符串
            result.append(str(x))
        elif isinstance(x, str) and x.replace('.', '').replace('-', '').lstrip('-').isdigit():
            # 如果是可以转换为数字的字符串
            result.append(x)
        else:
            result.append(x)

    return result



def get_creator():
    pset = gp.PrimitiveSet("MAIN", n_variables)
    rename_kwargs = {"ARG{}".format(i): 'x_'+str(i) for i in range(n_variables)}
    for k, v in rename_kwargs.items():
        pset.mapping[k].name = v
    pset.renameArguments(**rename_kwargs)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(protectedMul, 2, name='mul')
    pset.addPrimitive(protectedDiv, 2, name='div')
    pset.addPrimitive(protectedExp, 1, name="exp")
    pset.addPrimitive(protectedLog, 1, name="ln")
    pset.addPrimitive(protectedSqrt, 1, name="sqrt")
    pset.addPrimitive(operator.pow, 2, name="pow")
    pset.addPrimitive(operator.abs, 1, name="abs")
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.tan, 1)
    pset.addPrimitive(protectedAsin, 1,name='asin')
    pset.addPrimitive(protectedAcos, 1,name='acos')
    pset.addPrimitive(protectedAtan, 1, name='atan')
    pset.addEphemeralConstant("rand505", partial(random.uniform, -5, 5))

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    return pset, creator

def init_individual(pset, creator, trimmed_eq):

    plist=[]

    for t in trimmed_eq:
        if t in pset.mapping:
            plist.append(pset.mapping[t])

        elif t in [ '-3','-2','-1','0', '1', '2', '3', '4', '5']:

            if t not in pset.terminals[pset.ret]:
                pset.addTerminal(float(t), name=t)
                term = pset.terminals[pset.ret][-1]

            else:
                for i, term in enumerate(pset.terminals[pset.ret]):
                    if term.name == t:
                        break
                term = pset.terminals[pset.ret][i]()

            plist.append(term)


        elif t=='rand505':

            index_rand505=n_variables
            term = pset.terminals[pset.ret][index_rand505]()

            plist.append(term)

        else:
            value = float(t)

            pset.addTerminal(value, name=t)

            plist.append(pset.terminals[pset.ret][-1])

    individual = creator.Individual(plist)

    return individual

def is_complex(number):
    """
    Determine whether the given number is a complex number.

    :param number: The number to be checked.
    :return: True if the number is complex, False otherwise.
    """
    # A number is complex if it has a non-zero imaginary part
    return isinstance(number, complex) and number.imag != 0


def evalSymbReg(individual, pset, toolbox):
    wrong_mark=999999999
    
    # 处理表达式编译错误，包括语法错误和过于复杂的嵌套
    try:
        func = toolbox.compile(expr=individual)
    except (SyntaxError, RecursionError, MemoryError):
        return wrong_mark,
    
    sqerrors=[]
    for i, x in enumerate(input_X):
        try:
            try:
                try:
                    result=func(*x)
                except (AttributeError, ValueError, ZeroDivisionError):
                    return wrong_mark,

                if is_complex(result):
                    return wrong_mark,
                else:
                    tmp = (func(*x) - input_Y[i]) ** 2
                    sqerrors.append(tmp)
            except TypeError:
                return wrong_mark,

        except OverflowError:
            return wrong_mark,
    try:
        res = math.sqrt(math.fsum(sqerrors) / len(input_X))
    except TypeError:
        return  wrong_mark,

    return res,

def mutReplace(individual, pset, toolbox, creator):

    global  input_X
    node_index = random.randrange(len(individual))
    if len(individual) == 1:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)
    while individual[node_index].arity == 0:
        node_index = random.randrange(len(individual))
    try:
        semantic = backpropogation(individual, pset, (input_X, input_Y), node_index)
    except OverflowError:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)
    if semantic == 'nan':
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)

    try:
        symbol_list, prediceted_equation = get_res_transformer(input_X, semantic, BFGS=True)

    except (ValueError, RuntimeError):
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)

    if symbol_list is None:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)



    symbol_list=convert_to_list(symbol_list,accurate_constant=True)

    # 检查转换后的符号列表是否为空
    if not symbol_list:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)

    new_subtree=init_individual(pset, creator, symbol_list)
    
    # 检查新子树是否为空
    if not new_subtree:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)

    CT_slice = individual.searchSubtree(node_index)

    individual[CT_slice] = new_subtree

    return individual,

def mutate(individual, pset, creator, toolbox, p_subtree=0.05):

    if random.random() < p_subtree:

        return mutReplace(individual, pset=pset, toolbox=toolbox, creator=creator)
    else:

        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)


def run_single_experiment(filename=None, seed=8346, data_count=1000, generations=200, population_size=300):

    dataset_path = "dataset/"
    file_path = os.path.join(dataset_path, filename)
    
    global n_variables
    global input_X
    global input_Y
    file_contents = {}

    all_X = []
    all_Y = []

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            count = 0
            for line in file:
                if count >= data_count:  # 读取指定条数的数据
                    break
                
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                data = line.split()
                if len(data) < 2:  # 确保至少有输入和输出
                    continue
                    
                x = list(map(float, data[:-1]))
                y = float(data[-1])
                all_X.append(x)
                all_Y.append(y)
                count += 1
    else:
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    # 检查是否有足够的数据
    total_samples = len(all_X)
    if total_samples < 5:  # 至少需要5个样本
        raise ValueError(f"数据量不足，只读取到 {total_samples} 个样本，至少需要5个样本")
    
    # 按80%训练集和20%测试集分割数据，但确保至少有1个训练样本和1个测试样本
    train_size = max(1, int(total_samples * 0.8))
    if train_size >= total_samples:
        train_size = total_samples - 1
    
    train_X = np.array(all_X[:train_size])
    train_Y = np.array(all_Y[:train_size])
    test_X = np.array(all_X[train_size:])
    test_Y = np.array(all_Y[train_size:])

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    # 对输入特征进行标准化
    train_X_scaled = scaler_X.fit_transform(train_X)
    test_X_scaled = scaler_X.transform(test_X)
    
    # 对目标变量进行标准化
    train_Y_scaled = scaler_Y.fit_transform(train_Y.reshape(-1, 1)).flatten()
    test_Y_scaled = scaler_Y.transform(test_Y.reshape(-1, 1)).flatten()
    
    # 转换为列表格式供后续使用
    input_X = train_X_scaled.tolist()
    input_Y = train_Y_scaled.tolist()

    n_variables = len(input_X[0])


    pset, creator = get_creator()

    # fixme

    symbol_list, equation, =  get_res_transformer(input_X, input_Y, BFGS=False, first_call=True)

    total_variables = ["x_1", "x_2", "x_3"][:n_variables]

    # 使用标准化的测试数据进行预测
    X_dict = {x: test_X_scaled[:, idx] for idx, x in enumerate(total_variables)}
    y_pred_scaled = np.array(sympy.lambdify(",".join(total_variables), equation)(**X_dict))
    
    # 使用标准化后的数据计算MSE
    transformer_rmse = math.sqrt(mean_squared_error(test_Y_scaled.ravel(), y_pred_scaled.ravel()))

    # early stop
    if transformer_rmse < 1e-10:
        print('early stop')
        return

    # fixme
    trimmed_eq = convert_to_list(symbol_list, accurate_constant=False)


    # set toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    # toolbox.register("population", init_population, list, creator.Individual, eq=trimmed_eq, primitive_set=pset, num_individuals=20)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalSymbReg, pset=pset, toolbox=toolbox)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", mutate, pset=pset, creator=creator, toolbox=toolbox, p_subtree=0.025)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

    np.random.seed(seed)
    random.seed(seed)

    pop = toolbox.population(n=population_size)


    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)


    # 手动实现进化算法来跟踪适应度变化
    fitness_trend = []
    
    start_time = time.time()
    
    # 评估初始种群
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    
    # 记录初始代的最佳适应度
    current_best = min([ind.fitness.values[0] for ind in pop])
    fitness_trend.append(current_best)
    hof.update(pop)
    
    # 进化过程
    for gen in range(generations):
        # 选择
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # 交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:  # 交叉概率
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:  # 变异概率
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # 评估需要评估的个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # 替换种群
        pop[:] = offspring
        
        # 更新名人堂
        hof.update(pop)
        
        # 记录当前代的最佳适应度
        current_best = min([ind.fitness.values[0] for ind in pop])
        fitness_trend.append(current_best)
        
        # 每50代输出一次进度
        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"Generation {gen+1}: Best fitness = {current_best}")
    
    end_time = time.time()
    training_time = end_time - start_time

    last_generation_fitness = min([ind.fitness.values[0] for ind in pop])

    # 编译最佳个体，添加异常处理
    try:
        func = toolbox.compile(expr=hof[0])
    except (SyntaxError, ValueError, TypeError) as e:
        print(f"编译错误: {e}，使用简单线性函数")
        func = lambda *args: args[0] if args else 0
    
    # 计算测试集RMSE - 使用标准化后的数据，添加异常值处理
    test_predictions_scaled = []
    problematic_predictions_count = 0
    for row in test_X_scaled:
        try:
            pred = func(*row)
            # 检查异常值
            if np.isnan(pred) or np.isinf(pred) or np.iscomplex(pred):
                pred = 999999999
                problematic_predictions_count += 1
            test_predictions_scaled.append(pred)
        except Exception as e:
            pred = 999999999
            problematic_predictions_count += 1
            test_predictions_scaled.append(pred)
    
    if problematic_predictions_count > 0:
        print(f"警告: 在测试集预测中，有 {problematic_predictions_count} 个预测值被替换为 999999999。")
    
    test_rmse = np.sqrt(mean_squared_error(test_Y_scaled, test_predictions_scaled))
    print(f"测试集RMSE: {test_rmse:.6f}")
    
    return {
        'test_rmse': test_rmse,
        'fitness_trend': fitness_trend,
        'training_time': training_time,
        'best_individual': str(hof[0])
    }

# 指定的6个数据集
datasets = ["I.6.2","I.6.2b","I.12.4","I.14.3","I.14.4","I.25.13"]

# 随机种子0-9
seeds = list(range(10))

def run_experiments():
    for dataset in datasets:
        print(f"开始处理数据集: {dataset}")
        
        # 存储每次运行的结果
        results = []
        
        # 运行10次实验
        for seed in seeds:
            print(f"  运行种子 {seed}", end=" ... ")
            result = run_single_experiment(filename=dataset, seed=seed, data_count=10, generations=300, population_size=200)
            result['seed'] = seed
            results.append(result)
            print(f"完成 (RMSE: {result['test_rmse']:.6f})")

        if len(results) == 0:
            print(f"数据集 {dataset} 所有运行都失败")
            continue
            
        # 保存所有运行的完整结果
        dataset_results = {
            'dataset': dataset,
            'total_runs': len(results),
            'all_experiments': results
        }
        
        # 创建results目录（如果不存在）
        os.makedirs("results", exist_ok=True)
        
        # 保存到文件
        output_file = f"results/pggp_{dataset}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)
        
        print(f"数据集 {dataset} 结果已保存到 {output_file}")
        
        # 显示所有运行结果的摘要
        test_rmses = [r['test_rmse'] for r in results]
        print(f"  所有运行的RMSE结果:")
        for i, rmse in enumerate(test_rmses):
            print(f"    种子 {i}: {rmse:.6f}")
        print()

if __name__ == "__main__":
    run_experiments()