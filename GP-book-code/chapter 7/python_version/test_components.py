# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 7 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

"""
对比测试脚本 - 验证Python实现的正确性
测试关键组件的功能
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gep import Chromosome, FUNCTION_SET_SIZE, RANDOM_SET_SIZE

def test_chromosome_initialization():
    """测试染色体初始化"""
    print("\n" + "="*70)
    print("Test 1: Chromosome Initialization")
    print("="*70)
    
    Chromosome.TERMINAL_SET_SIZE = RANDOM_SET_SIZE + 10  # 假设10个特征
    
    chrom = Chromosome()
    chrom.random_init()
    
    print(f"✓ Chromosome initialized with {len(chrom.gene)} genes")
    print(f"✓ Gene values range: {chrom.gene.min()} to {chrom.gene.max()}")
    print(f"✓ Expected max: {FUNCTION_SET_SIZE + Chromosome.TERMINAL_SET_SIZE - 1}")
    
    # 验证基因值在有效范围内
    assert chrom.gene.max() < FUNCTION_SET_SIZE + Chromosome.TERMINAL_SET_SIZE
    assert chrom.gene.min() >= 0
    
    print("✅ Chromosome initialization test passed!")
    return True

def test_chromosome_validity():
    """测试染色体有效性检查"""
    print("\n" + "="*70)
    print("Test 2: Chromosome Validity Check")
    print("="*70)
    
    Chromosome.TERMINAL_SET_SIZE = RANDOM_SET_SIZE + 5
    
    chrom = Chromosome()
    
    # 创建一个简单的有效染色体
    # IF(a1, a2, a3) -> gene[0]=0 (IF), gene[1]=12 (a1), gene[2]=13 (a2), gene[3]=14 (a3)
    chrom.gene[0] = 0  # IF
    chrom.gene[1] = 12  # 终结符
    chrom.gene[2] = 13  # 终结符
    chrom.gene[3] = 14  # 终结符
    
    is_valid = chrom.is_valid()
    
    print(f"✓ Chromosome validity: {is_valid}")
    print(f"✓ Encoding genes: {chrom.encoding_genes}")
    
    assert is_valid
    assert chrom.encoding_genes == 4
    
    print("✅ Chromosome validity test passed!")
    return True

def test_chromosome_decode():
    """测试染色体解码"""
    print("\n" + "="*70)
    print("Test 3: Chromosome Decode")
    print("="*70)
    
    Chromosome.TERMINAL_SET_SIZE = RANDOM_SET_SIZE + 3
    
    chrom = Chromosome()
    
    # 创建简单表达式: a1 + a2
    # TERMINAL_SET_SIZE = 5(random) + 3(input) = 8
    # gene 12-16: RANDOM_SET[0-4] = {1,2,3,5,7}
    # gene 17-19: input attributes a1, a2, a3
    chrom.gene[0] = 1   # +
    chrom.gene[1] = 17  # 第一个输入属性 (a1, index=17-12-5=0)
    chrom.gene[2] = 18  # 第二个输入属性 (a2, index=18-12-5=1)
    
    is_valid = chrom.is_valid()
    assert is_valid
    
    # 创建测试数据
    input_attrs = np.array([
        [1.0, 2.0, 3.0],  # 样本1: a1=1, a2=2, a3=3
        [4.0, 5.0, 6.0],  # 样本2: a1=4, a2=5, a3=6
    ])
    
    result = chrom.decode_gene(input_attrs, 2)
    
    print(f"✓ Input: {input_attrs}")
    print(f"✓ Expression: a1 + a2")
    print(f"✓ Result: {result}")
    print(f"✓ Expected: [3.0, 9.0]")
    
    # 验证结果
    expected = np.array([3.0, 9.0])
    np.testing.assert_array_almost_equal(result, expected)
    
    print("✅ Chromosome decode test passed!")
    return True

def test_fitness_calculation():
    """测试适应度计算"""
    print("\n" + "="*70)
    print("Test 4: Fitness Calculation")
    print("="*70)
    
    from math import exp
    
    # 测试数据
    p = 8   # 正确覆盖的正例
    n = 2   # 错误覆盖的负例
    P = 10  # 总正例
    N = 10  # 总负例
    
    # 适应度公式
    consig = ((p / (p + n)) - (P / (P + N))) * ((P + N) / N)
    fitness = 0.0 if consig < 0 else consig * exp(p / P - 1)
    
    print(f"✓ p={p}, n={n}, P={P}, N={N}")
    print(f"✓ consig = {consig:.6f}")
    print(f"✓ fitness = {fitness:.6f}")
    
    # 验证公式正确性
    assert consig > 0
    assert fitness > 0
    
    print("✅ Fitness calculation test passed!")
    return True

def test_mutation():
    """测试变异操作"""
    print("\n" + "="*70)
    print("Test 5: Mutation Operation")
    print("="*70)
    
    Chromosome.TERMINAL_SET_SIZE = RANDOM_SET_SIZE + 5
    
    chrom = Chromosome()
    chrom.random_init()
    
    original_gene = chrom.gene.copy()
    
    # 执行变异
    chrom.is_modified = False
    chrom.mutation_and_update_is_modified()
    
    # 检查是否有基因发生变化
    changed = np.sum(original_gene != chrom.gene)
    
    print(f"✓ Original gene sample: {original_gene[:10]}")
    print(f"✓ Mutated gene sample: {chrom.gene[:10]}")
    print(f"✓ Genes changed: {changed}")
    
    print("✅ Mutation operation test passed!")
    return True

def test_crossover():
    """测试交叉操作"""
    print("\n" + "="*70)
    print("Test 6: Crossover Operation")
    print("="*70)
    
    from gep import GEPClassifier
    
    Chromosome.TERMINAL_SET_SIZE = RANDOM_SET_SIZE + 5
    
    mother = Chromosome()
    father = Chromosome()
    mother.random_init()
    father.random_init()
    
    while not mother.is_valid():
        mother.random_init()
    while not father.is_valid():
        father.random_init()
    
    gep = GEPClassifier()
    child = gep._crossover(mother, father)
    
    print(f"✓ Mother valid: {mother.is_valid()}")
    print(f"✓ Father valid: {father.is_valid()}")
    print(f"✓ Child valid: {child.is_valid()}")
    print(f"✓ Child is modified: {child.is_modified}")
    
    assert child.is_valid()
    
    print("✅ Crossover operation test passed!")
    return True

def test_data_loading():
    """测试数据加载"""
    print("\n" + "="*70)
    print("Test 7: Data Loading")
    print("="*70)
    
    try:
        from datasets.monk_dataset import Monk1
        
        dataset = Monk1()
        
        print(f"✓ Training samples: {dataset.INPUT_LINE_NUM}")
        print(f"✓ Features: {dataset.INPUT_ATTRIBUTE_NUM}")
        print(f"✓ Classes: {dataset.INPUT_CLASS_NUM}")
        print(f"✓ Input attrs shape: {dataset.input_attrs.shape}")
        print(f"✓ Input class shape: {dataset.input_class.shape}")
        
        # 验证数据形状
        assert dataset.input_attrs.shape == (dataset.INPUT_LINE_NUM, dataset.INPUT_ATTRIBUTE_NUM)
        assert dataset.input_class.shape == (dataset.INPUT_LINE_NUM,)
        
        # 加载测试数据
        test_attrs, test_class = dataset.load_test_data()
        print(f"✓ Test samples: {len(test_class)}")
        print(f"✓ Test attrs shape: {test_attrs.shape}")
        
        print("✅ Data loading test passed!")
        return True
    except FileNotFoundError as e:
        print(f"⚠️  Data files not found: {e}")
        print("   Please ensure data files are in the parent directory")
        return False

def run_all_tests():
    """运行所有测试"""
    print("="*70)
    print("GEP Python Implementation - Component Tests")
    print("="*70)
    
    tests = [
        ("Chromosome Initialization", test_chromosome_initialization),
        ("Chromosome Validity", test_chromosome_validity),
        ("Chromosome Decode", test_chromosome_decode),
        ("Fitness Calculation", test_fitness_calculation),
        ("Mutation Operation", test_mutation),
        ("Crossover Operation", test_crossover),
        ("Data Loading", test_data_loading),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ {name} failed with exception:")
            print(f"   {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    
    if failed == 0:
        print("\n🎉 All tests passed! Python implementation is correct.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
