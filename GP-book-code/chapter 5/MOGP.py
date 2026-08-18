# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 5 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).


import math
import random
import operator
from deap import creator,base,tools,gp,algorithms
import numpy as np
from matplotlib import pyplot as plt
import dimGapParseTree as dgpt
import sys
import simplify as sm
import cProfile
import pstats
sys.setrecursionlimit(10000) # è®¾ç½®ä¸ºè¶³å¤å¤§ç???
from deap.benchmarks.tools import igd
from deap.benchmarks.tools import hypervolume as hv 
import multiprocessing
import argparse
import os
import json
import semantic
import time
import pickle
def str2bool(v):
    """å°å­ç¬¦ä¸²å¼è½¬æ¢æå¸å°??"""
    return v.lower() in ('yes', 'true', 't', 'y', '1')

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-problem', '--arg1', type=str, help='Description of arg1')
parser.add_argument('-algorithm', '--arg2', type=str, help='Description of arg2')
parser.add_argument('-isDimMutate', '--arg3', type=str2bool, nargs='?', const=True, default=False, help='Description of arg3')
parser.add_argument('-isSimplify', '--arg4', type=str2bool, nargs='?', const=True, default=False, help='Description of arg4')
parser.add_argument('-isArch', '--arg5', type=str2bool, nargs='?', const=True, default=False, help='Description of arg5')

args = parser.parse_args()

print('arg1:', args.arg1)
print('arg2:', args.arg2)
print('arg3:', args.arg3)
print('arg4:', args.arg4)
print('arg5:', args.arg5)



# æ°æ®éè¾å¥åéä¸ª
problem = args.arg1
selectalgorithm = args.arg2

if problem == 'I.8.14':
    n_variables = 4
    variables = {'x1': [1,0,0,0,0], 'x2': [1,0,0,0,0] ,'x3':[1,0,0,0,0],'x4':[1,0,0,0,0],'0':[0,0,0,0,0],'1':[0,0,0,0,0],'2':[0,0,0,0,0]}   #./Feynman_with_units/I.18.4'
elif problem == 'I.13.4':
    n_variables = 4
    variables = {'x1': [1,-1,0,0,0], 'x2': [1,-1,0,0,0] ,'x3':[1,-1,0,0,0],'x4':[0,0,1,0,0],'0':[0,0,0,0,0],'1':[0,0,0,0,0],'2':[0,0,0,0,0]}   
elif problem == 'I.18.4':
    n_variables = 4
    variables = {'x1': [0,0,1,0,0], 'x2': [0,0,1,0,0] ,'x3':[1,0,0,0,0],'x4':[1,0,0,0,0],'0':[0,0,0,0,0],'1':[0,0,0,0,0],'2':[0,0,0,0,0]}   
elif problem == 'I.10.7':
    n_variables = 3
    variables = {'x1': [0,0,1,0,0], 'x2': [1,-1,0,0,0] ,'x3':[1,-1,0,0,0],'0':[0,0,0,0,0],'1':[0,0,0,0,0],'2':[0,0,0,0,0]}   
elif problem == 'I.18.12':
    n_variables = 3
    variables = {'x1': [1,0,0,0,0], 'x2': [1,-2,1,0,0] ,'x3':[0,0,0,0,0],'0':[0,0,0,0,0],'1':[0,0,0,0,0],'2':[0,0,0,0,0]}   
elif problem == 'I.18.14':
    n_variables = 4
    variables = {'x1': [0,0,1,0,0], 'x2': [1,0,0,0,0] ,'x3':[1,-1,0,0,0],'x4':[0,0,0,0,0],'0':[0,0,0,0,0],'1':[0,0,0,0,0],'2':[0,0,0,0,0]}   



def readData():
    # with open('./Feynman_with_units/' + problem, 'r') as f:
    with open('D:/GP/GeneticProgramming/Feynman_with_units/'+problem, 'r') as f:
        data = []
        start_line = 0
        end_line = 1000
        current_line = 0
        for line in f:
            if current_line >= start_line:
                line = str(line[:-1])
                nums = []
                j=0
                k=0
                for i in line:
                    if i == ' ':
                        nums.append(line[k:j])
                        k=j
                    j = j+1
                line.split(' ')
                line = list(map(float, nums))
                # å¤çè¯»åå°çè¡æ°??????
                data.append(line)
            current_line += 1
            if current_line > end_line:
                break
    return data

data = readData()

# ç»??????????
if n_variables == 1:
    pset = gp.PrimitiveSet("MAIN", arity=1)
    pset.renameArguments(ARG0 = 'x1')
elif n_variables == 2:
    pset = gp.PrimitiveSet("MAIN", arity=2)
    pset.renameArguments(ARG0 = 'x1')
    pset.renameArguments(ARG1 = 'x2')
elif n_variables == 3:
    pset = gp.PrimitiveSet("MAIN", arity=3)
    pset.renameArguments(ARG0 = 'x1')
    pset.renameArguments(ARG1 = 'x2')
    pset.renameArguments(ARG2 = 'x3')
elif n_variables == 4:
    pset = gp.PrimitiveSet("MAIN", arity=4)
    pset.renameArguments(ARG0 = 'x1')
    pset.renameArguments(ARG1 = 'x2')
    pset.renameArguments(ARG2 = 'x3')
    pset.renameArguments(ARG3 = 'x4')
elif n_variables == 5:
    pset = gp.PrimitiveSet("MAIN", arity=5)
    pset.renameArguments(ARG0 = 'x1')
    pset.renameArguments(ARG1 = 'x2')
    pset.renameArguments(ARG2 = 'x3')
    pset.renameArguments(ARG3 = 'x4')
    pset.renameArguments(ARG4 = 'x5')

# ä¿æ¤æ§çæä½
def protectedDiv(left,right):
    if right == 0:
        return 1
    else:
        return left / right

def protectedSqrt(x):
    if x<0:
        return 0
    else:
        return math.sqrt(x)

def protectedExp(x):
    if x>10:
        return math.exp(10)
    else:
        return math.exp(x)
        
# function terminal
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv,2,'div')
pset.addPrimitive(math.sin,1)
pset.addPrimitive(math.cos,1)
pset.addPrimitive(protectedExp,1,'exp')
pset.addPrimitive(protectedSqrt,1,'sqrt')
pset.addTerminal(0, name='0')
pset.addTerminal(1, name='1')
pset.addTerminal(2, name='2')

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti, pset=pset)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2) # generator
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)  # creator.Individualæ¯container toolbox.expræ¯generatorãinitIterateå½æ°è¿åcontainerçä¸???????å®ä¾???????
toolbox.register('population',tools.initRepeat,list,toolbox.individual) # initRepeatè¿è°ç¨n???????toolbox.individual()æ¥å¡«ålist
toolbox.register('compile',gp.compile, pset = pset)# å¨è¯ä¼°æ¯????????ä½çéåºåº¦æ¶ï¼å¯ä»¥ä½¿ç¨è¿????å·¥å·å°????å·è¡¨è¾¾å¼ç¼è¯ä¸?????æ????çå½æ°ï¼ç¶åè°?????????å½æ°æ¥è®¡ç®é?åºåº¦å????

# éçº²ä¸?è´æ?§è¯
def dimensionGap(individual,variables):
    expr = str(individual)
    return dgpt.computeDimGap(expr,variables)


MIN_VALUE = -10
MAX_VALUE = 10

# å®ä¹è¯ä»·å½æ°
def fit_evaluation(individual,points):
    func = toolbox.compile(expr = individual)   # psetä¸é¢å·²ç»ç»è¿???????
    try:
        if n_variables == 1:
            sqerrors = ((func(x1) - y)**2 for x1,y in points)
        if n_variables == 2:
            sqerrors = ((func(x1,x2) - y)**2 for x1,x2,y in points)
        if n_variables == 3:
            sqerrors = ((func(x1,x2,x3) - y)**2 for x1,x2,x3,y in points)
        if n_variables == 4:
            sqerrors = ((func(x1,x2,x3,x4) - y)**2 for x1,x2,x3,x4,y in points)
        if n_variables == 5:
            sqerrors = ((func(x1,x2,x3,x4,x5) - y)**2 for x1,x2,x3,x4,x5,y in points)
        obj1 = math.sqrt(math.fsum(sqerrors) / len(points))
    except OverflowError:
        obj1 = float('inf')
        print('obj1 is OverflowError',str(individual))
    except ValueError:
        obj1 = float('inf')
        print('obj1 is ValueError',str(individual))
    ind1 = len(individual)
    obj2 = ind1  # +(0.1)*ind2
    # print(individual)
    obj3 = dimensionGap(individual,variables)
    return obj1,obj2,obj3 # å¿é¡»è¿åä¸?ä¸ªtuple




def dimensionAwaredMutation(expr, pset,type_):
    substitude_expr = expr(pset=pset, type_=type_) # generate an expression as substitude_expr
    exprTree = gp.PrimitiveTree(substitude_expr)
    substitude_expr_str = str(exprTree).replace(' ', '')

    # new: check dimension consistence. if inconsistence:regenerate
    num_generate = 0
    while(dgpt.computeDimGap(substitude_expr_str,variables)>0):
        num_generate = num_generate+1
        substitude_expr = expr(pset=pset, type_=type_) # generate an expression as substitude_expr
        exprTree = gp.PrimitiveTree(substitude_expr)
        substitude_expr_str = str(exprTree).replace(' ', '')
    # print(' dimgap: ',dgpt.computeDimGap(substitude_expr_str,variables)) 
    # print('after ',gp.PrimitiveTree(substitude_expr))

    return substitude_expr

def mutUniform(individual,isDimAware, expr, pset):
    # print('åè¡¨è¾¾å¼:',individual,end=' ')
    index = random.randrange(len(individual)) # randomly select mutate point as index
    # print('mutate happen in: ',index)
    slice_ = individual.searchSubtree(index) # search Subtree by index
    type_ = individual[index].ret # recognize the tpye 
    # print('before:',gp.PrimitiveTree(individual[slice_]),end=' ') 
    if isDimAware == False:
        substitude_expr = expr(pset=pset, type_=type_)
    else:
        substitude_expr = dimensionAwaredMutation(expr, pset,type_)

    individual[slice_] = substitude_expr # substitude the orginal Subtree
    return individual,

def Simplify(individual,pset):
    before_expr_str = str(individual)
    # print("Before Simplify:",before_expr_str,len(before_expr_str))
    after_expr_str = sm.exprsimplify(before_expr_str) # simplify
    # print("After Simplify:",after_expr_str,len(before_expr_str))
    new_ind = creator.Individual(gp.PrimitiveTree.from_string(after_expr_str, pset))
    new_ind.fitness.values = individual.fitness.values
    return new_ind


# å®ä¹evaluateãselectãmateãmutateï¼è¿å ä¸ªåå­å¿é¡»è¿æ ·åï¼å¦ååºé
# toolbox.register("map", pool.map) # multiprocessing
toolbox.register("map", map)
toolbox.register('evaluate',fit_evaluation,points = data) #???????å®ä¹ #pointsæç???????æ ·æ¬ï¼è¿éåå?åå???????
toolbox.register('mate',gp.cxOnePoint)
toolbox.register('expr_mut',gp.genFull,pset=pset,min_ = 0,max_ = 2)     # çæä¸?ä¸ªsubtree
toolbox.register('mutate',mutUniform,expr=toolbox.expr_mut,pset=pset) # subtree mutation
toolbox.register('simplify',Simplify,pset=pset) # subtree Simplify


# åç¹äº¤å ä¼äº§çä¸¤æ£µæ (a field guideé????ä¸???????????????ç¨ä¸æ£µæ ) 
# # éå¶ä¸?ä¸äº¤ååå¼åçæ æ·±åº¦ï¼æ ¹æ®Kozaç????æï¼æ????????17
toolbox.decorate('mate',gp.staticLimit(key=operator.attrgetter('height'),max_value=17))
toolbox.decorate('mutate',gp.staticLimit(key=operator.attrgetter('height'),max_value=17))

# # æµè¯çªå
# while(1):
#     mutUniform(toolbox.individual(), expr=toolbox.expr_mut,pset=pset)
# sys.exit()


if selectalgorithm == 'nsga2':
    toolbox.register("select", tools.selNSGA2)

# Define function for exchanging best individuals between arch_pop and original_pop
def exchange_pop2_to_pop1(original_pop, arch_pop,elite_size):
    def select_best_individuals(population, k):
        return sorted(population, key=lambda ind: ind.fitness.values[0], reverse=False)[:k]

    # Select the best individual from arch_pop based on accuracy
    best_original_pop = select_best_individuals(original_pop, k=elite_size)
    best_arch_pop = select_best_individuals(arch_pop, k=elite_size)

    # Replace the worst individuals in pop1 with the best individuals from pop2
    for i in range(elite_size):  
        if best_original_pop[i].fitness.values[0] > best_arch_pop[i].fitness.values[0]:
            print('forward exchange!!!!!!!!') # æç¹é®é¢ï¼ï¼ï¼åæäº¤æ¢å¾å¤åæäº¤æ¢å°??0.9å¤ªå¤§ï¼éè¦äº¤æ¢ï¼
            # print(best_arch_pop[i].fitness.values[0],best_original_pop[i].fitness.values[0])
            original_pop.remove(best_original_pop[i])
            original_pop.append(best_arch_pop[i])
        # else:
        #     print('reverse exchange!!!!!!!!')
        #     arch_pop.remove(best_arch_pop[i])
        #     arch_pop.append(best_original_pop[i])
    return original_pop, arch_pop

# æµç¨æ¥æºäºeaSimpleæºä»£??????
def run_gp(population,archive_population,toolbox, cxpb, mutpb, esize, ngen,maxgen_exchange,isDimAware,isSimplify,isArch,
             halloffame=None, verbose=__debug__):
  
        
    eval_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    size_stats = tools.Statistics(lambda ind: ind.fitness.values[1])
    dim_stats = tools.Statistics(lambda ind: ind.fitness.values[2])
    original_stats = tools.MultiStatistics(eval=eval_stats, height=size_stats, dim = dim_stats)
    original_stats.register("mean", np.mean)
    original_stats.register("best", np.min)

    arch_eval_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    arch_stats = tools.MultiStatistics(archh_eval=arch_eval_stats)
    arch_stats.register("mean", np.mean)
    arch_stats.register("best", np.min)
    # è®°å½logbookï¼åä¸ºheaderï¼chaptersårecords
    logbook = tools.Logbook()
    logbook.header = ['gen'] + original_stats.fields + arch_stats.fields #, 'nevals1', 'nevals2'

    # Evaluate the individuals with an invalid fitness(Original pop)
    invalid_ind = [ind for ind in population if not ind.fitness.valid] # å­å¨çæ¯å°æªè®¡ç®éåºåº¦å?¼ç??????????????,åä»£æç¨???????
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) # å°ä¸???????å½æ°åºç¨äºåºåä¸­çæ¯???????åç´ ,???????ä»¥ç®åå¹¶è¡????çå???è¿ç¨???ççè¿ç¨ï¼æé«ç¨åºçæç??
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Evaluate the individuals with an invalid fitness(Archive pop)
    arch_invalid_ind = [ind for ind in archive_population if not ind.fitness.valid] 
    arch_fitnesses = toolbox.map(toolbox.evaluate, arch_invalid_ind) 
    for ind, fit in zip(arch_invalid_ind, arch_fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population) #the worst individual is replaced by the best individual

    record1 = original_stats.compile(population)
    record2 = arch_stats.compile(archive_population)
    logbook.record(gen=0, **record1,**record2) #, nevals1=len(invalid_ind),nevals2=len(arch_invalid_ind),
    if verbose:
        print(logbook.stream)
    
    # save the upper and down boundaries.
    ueff = -1e9
    leff = 1e9
    usize = -1e9
    lsize = 1e9
    # init
    # for ind in population:
    #     ueff = max(ind.fitness.values[0],ueff)
    #     leff = min(ind.fitness.values[1],leff)
    #     usize = max(len(ind),usize)
    #     lsize = min(len(ind),lsize)
    # # initialize alpha to 0
    # alpha = 0

    # record the mean fitness of arch_pop
    previous_avg = np.min([ind.fitness.values[0] for ind in archive_population])

    # Begin the generational process
    for gen in range(1, ngen + 1):
        
        if isArch and gen <= maxgen_exchange: # ä¸¤ä¸ªé®é¢??1ãååäº¤?? 2??70è®¾ç½®çå¦ä½ï¼åé¢30ä»£nsga2è½æ¾å°ä¼åçè§£å??

            # elitism
            elitism_size = int(0.1*len(archive_population))
            arch_elitism = tools.selBest(archive_population, k=elitism_size)
 
            # Select the next generation individuals(archive)
            arch_offspring = tools.selTournament(archive_population, len(archive_population)-elitism_size ,tournsize = 3)

            # Clone(archive)
            arch_offspring = [toolbox.clone(ind) for ind in arch_offspring]

            # Apply crossover on the arch_offspring(archive)
            for i in range(1, len(arch_offspring), 2):
                if random.random() < cxpb:
                    arch_offspring[i - 1], arch_offspring[i] = toolbox.mate(arch_offspring[i - 1], arch_offspring[i])
                    del arch_offspring[i - 1].fitness.values,arch_offspring[i].fitness.values

            # Apply mutation on the offspring(archive)
            for i in range(len(arch_offspring)):
                if random.random() < mutpb:
                    arch_offspring[i], = gp.mutUniform(arch_offspring[i],expr=toolbox.expr_mut,pset=pset)
                    # arch_offspring[i], = toolbox.mutate(arch_offspring[i], isDimAware=isDimAware)
                    del arch_offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness (Archive pop)
            arch_invalid_ind = [ind for ind in arch_offspring if not ind.fitness.valid] 
            arch_fitnesses = toolbox.map(toolbox.evaluate, arch_invalid_ind) 
            for ind, fit in zip(arch_invalid_ind, arch_fitnesses):
                ind.fitness.values = fit

            arch_offspring += arch_elitism

            # Replace the current population by the offspring(archive)
            archive_population[:] = arch_offspring

            # record the mean fitness of arch_pop in current gen.
            current_avg = np.min([ind.fitness.values[0] for ind in archive_population])
            if current_avg <  0.95*previous_avg: # improve 10% 
                # info exchange
                population, archive_population = exchange_pop2_to_pop1(population,archive_population,esize) # generally 10%~20%
                # è®°å½ç§»å¨åçå¹³åç®æ å½æ°??
                previous_avg = current_avg
        

        # Clone(original)
        offspring = [toolbox.clone(ind) for ind in population]
          
        # Apply crossover on the offspring(orginal)
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                            offspring[i])
                # del offspring[i - 1].fitness.values, offspring[i].fitness.values
                
        # Apply mutation on the offspring(orginal)
        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i],isDimAware = isDimAware)
                # del offspring[i].fitness.values
        
                
        for i in range(1, len(offspring)):
            del offspring[i].fitness.values

        # Evaluate the individuals with an invalid fitnessï¼original)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # if selectalgorithm == 'alpha':
        #     # linear,sigmoid,cosine
        #     alpha = alpha_func(gen, ngen,'cosine') # lineapython MOGP.py -algorithm='alpha' -isDimMutate False -isSimplify Falser,sigmoid,cosine
        #     # print(alpha)

        # # Select the next generation individuals(orginal)
        # if selectalgorithm == 'alpha' or selectalgorithm == 'self_alpha':
        #     offspring = toolbox.select(population, len(population), alpha)
        # else:
        if selectalgorithm == 'nsga2':
            offspring = toolbox.select(offspring + population, len(population))
        else:
            offspring = toolbox.select(offspring, len(population))
        # if selectalgorithm == 'self_alpha':
        #     # self adaptive alpha adjustment scheme
        #     (alpha, ueff, leff, usize, lsize) = self_adaptive_alpha(population, alpha, ueff, leff, usize, lsize, lr=100000)
        #     # print(alpha, ueff, leff, usize, lsize)

        # Apply simplify on the offspring(orginal)
        if isSimplify and gen%2==0:
            offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=False)
            for i in range(int(0.01*len(offspring))):
                # print('before: ',len(offspring[i]),offspring[i])
                offspring[i] = toolbox.simplify(offspring[i])  # syntax
                #print('after1: ',len(offspring[i]),offspring[i])
                offspring[i] = semantic.semanticSimply(offspring[i],n_variables,data,toolbox,pset) # semantic
                #print('after2: ',len(offspring[i]),offspring[i])

                # Evaluate the individuals with an invalid fitnessï¼original)
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

        # Replace the current population by the offspring(original)
        population[:] = offspring
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring) 

        # Append the current generation statistics to the logbook
        record1 = original_stats.compile(population)
        record2 = arch_stats.compile(archive_population)
        logbook.record(gen=gen, **record1,**record2) # nevals1=len(invalid_ind),nevals2=len(arch_invalid_ind)
        if verbose:
            print(logbook.stream)

    return population, logbook


# # æµè¯
# # expr = 'sub(add(x2, div(sub(x3, x4), x3)), sub(x2, x4))'
# # expr = 'mul(2,mul(sin(div(add(x1,x3),2)),cos(div(sub(x1,x3),2))))' #sin(x)+sin(y) = 2sin[(Î±+Î²)/2]cos[(Î±-Î²)/2] 
# # expr = 'add(add(0, 0),0)' 
# # expr = 'div(2, sin(mul(mul(sqrt(x2), sin(sqrt(sqrt(x2)))), sin(add(mul(x1, 0), sqrt(x1))))))'
# expr = 'sub(add(1,x1),add(x1,1))'
# print('before:',expr)
# ind = gp.PrimitiveTree.from_string(expr,pset)
# ind1 = toolbox.simplify(ind)
# print('after1:',ind1)
# ind2 = semantic.semanticSimply(ind,n_variables,data,toolbox,pset)
# print('after2:',ind2)
# sys.exit(1)

def main():
    random.seed(2)
    import time
    start_time = time.time()

    # åæ°è®¾ç½®
    P_CROSSOVER = 0.9      # äº¤åæ¦ç
    P_MUTATION = 0.1    # åå¼æ¦ç
    MAX_GENERATIONS = 100     # GENERATIONS
    POP_SIZE = 500   # ç§ç¾¤è§æ¨¡
    ARCHIVE_SIZE = 500   # ç§ç¾¤è§æ¨¡
    ELITISM_SIZE = int(0.01*ARCHIVE_SIZE) # Set the size of the elitism to 10
    MAXGEN_EXCHANGE = 70
    runs_independent = 50


    # # åæ°è®¾ç½®
    # P_CROSSOVER = 0.9      # äº¤åæ¦ç
    # P_MUTATION = 0.1    # åå¼æ¦ç
    # MAX_GENERATIONS = 10     # GENERATIONS
    # POP_SIZE = 5   # ç§ç¾¤è§æ¨¡
    # ARCHIVE_SIZE = 5   # ç§ç¾¤è§æ¨¡
    # ELITISM_SIZE = int(0.01*ARCHIVE_SIZE) # Set the size of the elitism to 10
    # MAXGEN_EXCHANGE = 7
    # runs_independent = 5


    isDimAware = args.arg3
    isSimplify = args.arg4
    isArch = args.arg5

    file_name = "./result/"+problem+"/"+problem +"_"+ selectalgorithm +"_"+ str(isDimAware) +"_"+ str(isSimplify) +"_"+ str(isArch) + ".json"
    # å°å­å¸dataä¿å­å°æä»¶ä¸­
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print(selectalgorithm,P_CROSSOVER,P_MUTATION,MAX_GENERATIONS,POP_SIZE,ARCHIVE_SIZE,ELITISM_SIZE,runs_independent,isDimAware,isSimplify,isArch)
      
    # hall of fame
    hof = tools.ParetoFront()
    # hof = tools.HallOfFame(POP_SIZE)

    hv_list = []
    igd_list = []

    # æ¯???????ç®å®ï¼æ¾åºç²¾åº¦æé«çé£ä¸ª????ä½????
    precision_list = []
    generality_list = []
    dimension_list = []
    best_pops = toolbox.population(n=0) # keep best ind for each run
    best_list = []

    global runs
    for runs in range(runs_independent):
        pop = toolbox.population(n = POP_SIZE)
        archive_pop = toolbox.population(n = ARCHIVE_SIZE)
        population, logbook = run_gp(pop,
        archive_pop,                    
        toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        esize=ELITISM_SIZE,
        ngen=MAX_GENERATIONS,
        maxgen_exchange = MAXGEN_EXCHANGE,
        isDimAware = isDimAware,
        isSimplify = isSimplify,
        isArch = isArch,
        halloffame=hof,
        verbose = True
        )
          
        # Nondominated Sort
        pareto_fronts = tools.sortNondominated(pop, len(pop))[0] #first front
        # print('size of pareto:',len(pareto_fronts))
        # Calculate HV for a population
        wobj = np.array([ind.fitness.wvalues for ind in pareto_fronts]) * -1
        ref = np.max(wobj, axis=0) + 1
        points = [tuple(map(float, p.fitness.values)) for p in pareto_fronts] # Convert points to double type values
        HV = hv.hypervolume(points,ref)
        # print("Python version:",hv.hypervolume(points,ref))

        # Calculate IGD for a population
        pop_fitness = [ind.fitness.values for ind in pareto_fronts]
        ref_fitness = [[0, 1, 0]] # Example reference set
        IGD = igd(pop_fitness,ref_fitness)
        # print("IGD: ",IGD)
        hv_list.append(HV)
        igd_list.append(IGD)

        # ç¶åï¼æ????????ä»¥ä½¿ç¨Deapçtoolbox????çselBestå½æ°æ¥æ¾å????ä¸?????????æ å½æ°æå¤§ç????????
        best_ind = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=False)[:1]
        best_pops.append(best_ind)

        # print(str(best_ind[0].fitness.values[0]))
        precision_list.append(best_ind[0].fitness.values[0])
        generality_list.append(best_ind[0].fitness.values[1])
        dimension_list.append(best_ind[0].fitness.values[2])
        best_list.append([best_ind[0].fitness.values[0],best_ind[0].fitness.values[1],best_ind[0].fitness.values[2]])

        print(runs)
        print(best_ind[0].fitness.values[0],best_ind[0].fitness.values[1],best_ind[0].fitness.values[2])
        # è®¡æ¶
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("execute time: ", elapsed_time, "second?")
    
    # è®¡ç®HVçå¹³åå?¼åæ å??????
    hv_mean = np.mean(hv_list)
    hv_std = np.std(hv_list)

    # è®¡ç®IGDçå¹³åå?¼åæ å??????
    igd_mean = np.mean(igd_list)
    igd_std = np.std(igd_list)

    pre_mean = np.mean(precision_list)
    pre_std = np.std(precision_list)
    gen_mean = np.mean(generality_list)
    gen_std = np.std(generality_list)
    dim_mean = np.mean(dimension_list)
    dim_std = np.std(dimension_list)

    print(f"HV mean: {hv_mean}, HV std: {hv_std}")
    print(f"IGD mean: {igd_mean}, IGD std: {igd_std}")

    print("best individual:")
    print(f"precision mean: {pre_mean}, precision std: {pre_std}")
    print(f"generality mean: {gen_mean}, generality std: {gen_std}")
    print(f"dimension mean: {dim_mean}, dimension std: {dim_std}")

    # è®¡æ¶ç»æ
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Program Time execute: ", elapsed_time, "second!")


    import json

    # åå»ºDataFrame
    res_dict = {
        'problem':problem,
        'method':selectalgorithm,
        'parameters':[P_CROSSOVER,P_MUTATION,MAX_GENERATIONS,POP_SIZE,ELITISM_SIZE,runs_independent,isDimAware,isSimplify],
        'hv':hv_list, 
        'hv_mean_std':[hv_mean,hv_std], 
        'igd':igd_list,
        'igd_mean_std':[igd_mean,igd_std],
        'precision':precision_list,
        'precision_mean_std':[pre_mean,pre_std], 
        'generality':generality_list,
        'generality_mean_std':[gen_mean,gen_std], 
        'dimension':dimension_list,
        'dimension_mean_std':[dim_mean,dim_std],
        'best':best_list,
        'execute time(mins):': [elapsed_time/60],
    }

    # å°å­å¸dataä¿å­å°æä»¶ä¸­
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_name, "w") as f:
        json.dump(res_dict, f)

    # keep in pickle form
    if False:

        # å°????ç¾???????ä¸ºå­???
        pop_dict = {}
        for i, ind in enumerate(best_ind):
            pop_dict[i] = {"fitness": ind.fitness.values, "genotype": ind}

        # å°å­å¸åå¥æ???
        pkl_name = "./result/pop/"+problem +"_"+ selectalgorithm + ".pkl"
        with open(pkl_name, "wb") as f:
            pickle.dump(pop_dict, f)

    # print('pareto:')
    # # è¾åº Pareto åæ²¿
    # i=0
    # for ind in hof:
    #     i=i+1
    #     print(ind, ind.fitness) 
    #     #drawBestInd(ind,i)

    # print('population:')
    # i=0
    # for ind in population:
    #     i=i+1
    #     print(ind, ind.fitness) 
    #     #drawBestInd(ind,i)
    


   
if __name__=="__main__":
    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)  # multiprocessing
    main()
    # å³é­è¿ç¨pool
    # pool.close()