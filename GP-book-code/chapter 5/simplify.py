# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 5 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).

import re

# 定义语法简化规则（前缀表达式形式）
simplify_rules = [
    (r'add\(([^,]+),0\)', r'\1'),
    (r'add\(0,([^,]+)\)', r'\1'),
    (r'mul\(([^,]+),1\)', r'\1'),
    (r'mul\(1,([^,]+)\)', r'\1'),
    (r'mul\(([^,]+),0\)', r'0'),
    (r'mul\(0,([^,]+)\)', r'0'),
    (r'sub\(([^,]+),0\)', r'\1'),
    (r'div\(([^,]+),1\)', r'\1'),
    (r'div\(0,[^,]+\)', r'0')
]

def is_balanced(expr):
    """检查括号是否匹配"""
    return expr.count('(') == expr.count(')')

def is_valid_expr(expr):
    """检查表达式是否基本合法"""
    if not expr or not isinstance(expr, str):
        return False
    if len(expr) < 3:
        return False
    if not is_balanced(expr):
        return False
    if re.search(r'[^\w\d_,()*/+\-]', expr):  # 只允许基本字符
        return False
    return True

def exprsimplify(expr_str, max_iter=10):
    if not is_valid_expr(expr_str):
        # print(f"[Simplify Warning] Input expression illegal or malformed: {expr_str}")
        return expr_str

    try:
        for _ in range(max_iter):
            old_expr = expr_str
            for pattern, replacement in simplify_rules:
                expr_str = re.sub(pattern, replacement, expr_str)
            if expr_str == old_expr:
                break  # 没有变化则停止
    except Exception as e:
       #print(f"[Simplify Error] Unexpected simplification error: {e}")
        return expr_str

    if not is_valid_expr(expr_str):
        # print(f"[Simplify Warning] Simplified expression became invalid: {expr_str}")
        return old_expr  # 回滚原始表达式

    return expr_str

# 可选测试入口
if __name__ == "__main__":
    test_cases = [
        "add(x1,0)",
        "add(0,x2)",
        "mul(x1,1)",
        "mul(0,x2)",
        "sub(x1,0)",
        "div(0,x2)",
        "div(x1,1)",
        "add(add(x1,0),0)",
        "mul(add(0,x1),1)",
        "add(",               # 非法：缺括号
        "div(x1,x2",          # 非法：括号不匹配
        "0",                  # 合法但不该简化
        "",                   # 空字符串
        None,                 # 非字符串类型
    ]
    for expr in test_cases:
        simplified = exprsimplify(expr)
        print(f"{expr}  -->  {simplified}")
