# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Jinghui Zhong
#
# Companion source code for Chapter 3 of the book
# "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).
#This is mldep file.
import numpy as np
import time
import os

import constants as C
import utils
from gene import Gene
from simplify_matrix import get_independent_columns_indicator
from lm_optimizer import lm_optimizer

class MLDEP:
    def __init__(self, function_id, job_id):
        self.function = function_id
        self.job = job_id
        
        self.population = [Gene() for _ in range(C.POPSIZE + 1)]
        self.new_population = [Gene() for _ in range(C.POPSIZE)]
        
        self.basic_rg = 0
        self.rg_nums = 0
        self.generation = 0
        self.terminal_num = 0
        self.training_cases = 0
        self.testing_cases = 0
        
        self.global_best_fit = 1e10
        
        self.training_inputs = np.zeros((C.MAXINPUTS, C.MAX_VARIABLES))
        self.training_outputs = np.zeros(C.MAXINPUTS)
        self.testing_inputs = np.zeros((C.MAXINPUTS, C.MAX_VARIABLES))
        self.testing_outputs = np.zeros(C.MAXINPUTS)
        
        self.rg_values = np.zeros((C.MAX_RG_NUMS, C.MAXINPUTS))
        self.basic_train_rg_values = np.zeros((C.MAX_RG_NUMS, C.MAXINPUTS))
        self.basic_test_rg_values = np.zeros((C.MAX_RG_NUMS, C.MAXINPUTS))
        self.const_values = np.zeros(C.CR_NUMS)
        self.node_numbers = np.zeros(C.MAX_RG_NUMS, dtype=int)
        self.register_indicator = None

    def read_data(self):
        train_file_path = os.path.join('Benchmark', 'new_compact_sample_dataset', f'F{self.function}_training_data.txt')
        test_file_path = os.path.join('Benchmark', 'new_compact_sample_dataset', f'F{self.function}_testing_data.txt')

        try:
            with open(train_file_path, 'r') as f:
                header = f.readline().split()
                self.training_cases, self.terminal_num = int(header[0]), int(header[1])
                data = np.loadtxt(f)
                self.training_inputs[:self.training_cases, :self.terminal_num] = data[:, :-1]
                self.training_outputs[:self.training_cases] = data[:, -1]

            with open(test_file_path, 'r') as f:
                header = f.readline().split()
                self.testing_cases = int(header[0])
                data = np.loadtxt(f)
                self.testing_inputs[:self.testing_cases, :self.terminal_num] = data[:, :-1]
                self.testing_outputs[:self.testing_cases] = data[:, -1]
        except (FileNotFoundError, IOError) as e:
            print(f"Error reading data file: {e}")
            exit(-1)

    def load_data_to_rg(self):
        s = 0
        self.basic_train_rg_values[s, :self.training_cases] = 1
        self.basic_test_rg_values[s, :self.testing_cases] = 1
        self.node_numbers[s] = 1
        s += 1
        
        for i in range(self.terminal_num):
            for j in range(2, C.MAX_DEGREE):
                self.basic_train_rg_values[s, :self.training_cases] = np.power(self.training_inputs[:self.training_cases, i], j)
                self.basic_test_rg_values[s, :self.testing_cases] = np.power(self.testing_inputs[:self.testing_cases, i], j)
                self.node_numbers[s] = 2 * j - 1
                s += 1
        self.basic_rg = s

    def _objective_impl(self, p, do_optim, is_training):
        cases, inputs, outputs, basic_rg_vals = (self.training_cases, self.training_inputs, self.training_outputs, self.basic_train_rg_values) if is_training else (self.testing_cases, self.testing_inputs, self.testing_outputs, self.basic_test_rg_values)

        self.rg_values[:self.terminal_num, :cases] = inputs[:cases, :self.terminal_num].T
        self.node_numbers[self.basic_rg : self.basic_rg + self.terminal_num] = 1

        k = 0
        for i in range(self.terminal_num, self.rg_nums):
            self.rg_values[i, :cases] = inputs[:cases, k % self.terminal_num].T
            self.node_numbers[self.basic_rg + i] = 1
            k += 1

        for i in range(C.NVARS // 4):
            k4 = i * 4
            op = int(abs(p.x[k4] - int(p.x[k4])) * C.FUNCTION_NUM)
            if op >= C.FUNCTION_NUM:
                op = C.FUNCTION_NUM - 1

            r1 = self.terminal_num + int(abs(p.x[k4 + 1] - int(p.x[k4 + 1])) * (self.rg_nums - self.terminal_num))
            if r1 >= self.rg_nums:
                r1 = self.rg_nums - 1

            # Left operand
            frac2 = abs(p.x[k4 + 2] - int(p.x[k4 + 2]))
            if frac2 < C.C_RATE:
                r2 = int(abs(p.x[k4 + 2] * C.CR_NUMS)) % C.CR_NUMS
                v2 = self.const_values[r2]
                nodeCount2 = 1
            else:
                r2 = int(frac2 * self.rg_nums)
                if r2 >= self.rg_nums:
                    r2 = self.rg_nums - 1
                v2 = self.rg_values[r2, :cases]
                nodeCount2 = self.node_numbers[self.basic_rg + r2]

            # Right operand
            frac3 = abs(p.x[k4 + 3] - int(p.x[k4 + 3]))
            if frac3 < C.C_RATE:
                r3 = int(abs(p.x[k4 + 3] * C.CR_NUMS)) % C.CR_NUMS
                v3 = self.const_values[r3]
                nodeCount3 = 1
            else:
                r3 = int(frac3 * self.rg_nums)
                if r3 >= self.rg_nums:
                    r3 = self.rg_nums - 1
                v3 = self.rg_values[r3, :cases]
                nodeCount3 = self.node_numbers[self.basic_rg + r3]

            # Update node count
            if op in (0, 1, 2, 3):
                self.node_numbers[self.basic_rg + r1] = nodeCount2 + nodeCount3 + 1
            elif op in (4, 5):
                self.node_numbers[self.basic_rg + r1] = nodeCount2 + 1
            
            # Compute value
            if op == 0: self.rg_values[r1, :cases] = v2 + v3
            elif op == 1: self.rg_values[r1, :cases] = v2 - v3
            elif op == 2: self.rg_values[r1, :cases] = v2 * v3
            elif op == 3: self.rg_values[r1, :cases] = utils.protected_div(v2, v3)
            elif op == 4: self.rg_values[r1, :cases] = np.sin(v2)
            elif op == 5: self.rg_values[r1, :cases] = np.cos(v2)

        if do_optim:
            self.lm_optimization(p)

        total_rg_count = self.basic_rg + self.rg_nums
        full_matrix = np.vstack([basic_rg_vals[:self.basic_rg, :cases], self.rg_values[:self.rg_nums, :cases]])
        predicted_y = p.coefficients[:total_rg_count] @ full_matrix
        
        error = np.sqrt(np.mean((outputs[:cases] - predicted_y)**2))
        std_dev = utils.standard_deviation(outputs[:cases])
        nrmse = (error / std_dev) if std_dev > 1e-6 else error
        nrmse = 0 if nrmse < 1e-6 else nrmse
        
        if is_training:
            p.f = nrmse
            total_node_count = 0
            for z in range(total_rg_count):
                if abs(p.coefficients[z]) > 1e-10:
                    total_node_count += (self.node_numbers[z] + 2 + 1)
            p.nodeCount = max(0, total_node_count - 1)
        else:
            p.tf = nrmse

    def print_formula(self, save_path, p):
        # 构建初始寄存器字符串
        str_of_rg = [""] * (self.rg_nums)
        for i in range(self.terminal_num):
            str_of_rg[i] = f"X{i}"
        k = 0
        for i in range(self.terminal_num, self.rg_nums):
            str_of_rg[i] = f"X{k}"
            k = (k + 1) % self.terminal_num

        # 解析基因并生成寄存器表达式
        for i in range(C.NVARS // 4):
            k4 = i * 4
            op = int(abs(p.x[k4] - int(p.x[k4])) * C.FUNCTION_NUM)
            if op >= C.FUNCTION_NUM:
                op = C.FUNCTION_NUM - 1
            r1 = self.terminal_num + int(abs(p.x[k4 + 1] - int(p.x[k4 + 1])) * (self.rg_nums - self.terminal_num))
            if r1 >= self.rg_nums:
                r1 = self.rg_nums - 1

            frac2 = abs(p.x[k4 + 2] - int(p.x[k4 + 2]))
            if frac2 < C.C_RATE:
                r2_idx = int(abs(p.x[k4 + 2] * C.CR_NUMS)) % C.CR_NUMS
                left_str = f"({self.const_values[r2_idx]:.6f})"
            else:
                r2_idx = int(frac2 * self.rg_nums)
                if r2_idx >= self.rg_nums:
                    r2_idx = self.rg_nums - 1
                left_str = str_of_rg[r2_idx]

            frac3 = abs(p.x[k4 + 3] - int(p.x[k4 + 3]))
            if frac3 < C.C_RATE:
                r3_idx = int(abs(p.x[k4 + 3] * C.CR_NUMS)) % C.CR_NUMS
                right_str = f"({self.const_values[r3_idx]:.6f})"
            else:
                r3_idx = int(frac3 * self.rg_nums)
                if r3_idx >= self.rg_nums:
                    r3_idx = self.rg_nums - 1
                right_str = str_of_rg[r3_idx]

            if op == 0:
                str_of_rg[r1] = f"({left_str}+{right_str})"
            elif op == 1:
                str_of_rg[r1] = f"({left_str}-{right_str})"
            elif op == 2:
                str_of_rg[r1] = f"({left_str}*{right_str})"
            elif op == 3:
                str_of_rg[r1] = f"mydiv({left_str}, {right_str})"
            elif op == 4:
                str_of_rg[r1] = f"sin({left_str})"
            elif op == 5:
                str_of_rg[r1] = f"cos({left_str})"

        # 线性组合输出
        c = p.coefficients.copy()
        if len(c) > 0 and abs(c[0]) > 1e-5:
            utils.save_text(save_path, f"{c[0]:.6f} * 1", append=True)
        s = 1
        for j in range(self.terminal_num):
            for deg in range(2, C.MAX_DEGREE):
                if s < len(c) and abs(c[s]) > 1e-5:
                    utils.save_text(save_path, f" + ({c[s]:.6f}) * pow(X{j}, {deg})", append=True)
                s += 1
        for j in range(self.rg_nums):
            if s < len(c) and abs(c[s]) > 1e-5:
                term = str_of_rg[j] if str_of_rg[j] else f"RG{j}"
                utils.save_text(save_path, f" + ({c[s]:.6f}) * {term}", append=True)
            s += 1

    def lm_optimization(self, p):
        total_rg_count = self.basic_rg + self.rg_nums
        original_matrix = np.vstack([
            self.basic_train_rg_values[:self.basic_rg, :self.training_cases],
            self.rg_values[:self.rg_nums, :self.training_cases]
        ]).T

        # indicator is a boolean array of size `total_rg_count`
        indicator = get_independent_columns_indicator(original_matrix)
        simplified_matrix = original_matrix[:, indicator]

        # First, zero out all coefficients
        p.coefficients.fill(0)

        if simplified_matrix.shape[1] > 0:
            # optimized_coeffs has shape (number of true values in indicator,)
            optimized_coeffs = lm_optimizer(simplified_matrix, self.training_outputs[:self.training_cases])
            
            # Get the integer indices where indicator is True
            true_indices = np.where(indicator)[0]
            
            # Ensure the number of coefficients matches the number of independent columns
            if len(true_indices) == len(optimized_coeffs):
                # Assign the optimized values to the full coefficient array at those indices
                p.coefficients[true_indices] = optimized_coeffs
            else:
                # This case should ideally not happen if logic is correct
                print(f"Warning: Mismatch in lm_optimization. Indicator count: {len(true_indices)}, Coeffs count: {len(optimized_coeffs)}")

    def objective(self, p, do_optim=True):
        self._objective_impl(p, do_optim, is_training=True)

    def test_objective(self, p):
        self._objective_impl(p, do_optim=False, is_training=False)

    def initialization(self):
        self.const_values = np.random.uniform(-10, 10, size=C.CR_NUMS)
        
        for i in range(C.POPSIZE):
            self.population[i].x = np.random.uniform(0, 1, size=C.NVARS)
            self.population[i].coefficients = np.random.uniform(-5, 5, size=C.MAX_RG_NUMS)
            self.objective(self.population[i], do_optim=True)
            
            if self.population[i].f < self.population[C.POPSIZE].f:
                self.population[C.POPSIZE] = utils.copy_gene(self.population[i])
                self.global_best_fit = self.population[C.POPSIZE].f

    def production(self):
        F, CR = 0.5, 0.1
        for i in range(C.POPSIZE):
            r_indices = np.random.choice(C.POPSIZE, 4, replace=False)
            p_best, p1, p2, p3, p4 = self.population[C.POPSIZE], self.population[r_indices[0]], self.population[r_indices[1]], self.population[r_indices[2]], self.population[r_indices[3]]
            
            rand_mask_x = np.random.rand(C.NVARS) < CR
            rand_mask_x[np.random.randint(C.NVARS)] = True
            mutant_x = p_best.x + F * (p1.x + p2.x - p3.x - p4.x)
            # 越界时重新随机（与Java一致），而非截断
            trial_x = np.where(rand_mask_x, mutant_x, self.population[i].x)
            out_of_bounds_x = (trial_x < -1) | (trial_x > 1)
            random_x = np.random.uniform(-1, 1, size=C.NVARS)
            trial_x = np.where(out_of_bounds_x & rand_mask_x, random_x, trial_x)
            self.new_population[i].x = trial_x

            rand_mask_c = np.random.rand(C.MAX_RG_NUMS) < CR
            rand_mask_c[np.random.randint(C.MAX_RG_NUMS)] = True
            mutant_c = p_best.coefficients + F * (p1.coefficients + p2.coefficients - p3.coefficients - p4.coefficients)
            # 越界时重新随机（与Java一致），而非截断
            trial_c = np.where(rand_mask_c, mutant_c, self.population[i].coefficients)
            out_of_bounds_c = (trial_c < -100) | (trial_c > 100)
            random_c = np.random.uniform(-100, 100, size=C.MAX_RG_NUMS)
            trial_c = np.where(out_of_bounds_c & rand_mask_c, random_c, trial_c)
            self.new_population[i].coefficients = trial_c

            self.objective(self.new_population[i], do_optim=True)

            if self.new_population[i].f < self.population[i].f:
                self.population[i] = utils.copy_gene(self.new_population[i])
                if self.population[i].f < self.population[C.POPSIZE].f:
                    self.population[C.POPSIZE] = utils.copy_gene(self.population[i])
                    self.global_best_fit = self.population[C.POPSIZE].f

    def run_de(self):
        self.read_data()
        self.load_data_to_rg()
        self.rg_nums = self.terminal_num + 5
        # 与 Java 控制台输出对齐：打印数据规模（训练样本数/变量数、测试样本数/变量数）
        print(f"{self.training_cases}\t{self.terminal_num}")
        print(f"{self.testing_cases}\t{self.terminal_num}")
        
        self.initialization()
        
        start_time = time.time()
        num_minutes = 10
        
        while (time.time() - start_time) < (num_minutes * 60):
            self.production()
            self.test_objective(self.population[C.POPSIZE])
            
            if self.generation % 200 == 0:
                best = self.population[C.POPSIZE]
                print(f"Gen: {self.generation}, F: {self.function}, Job: {self.job}, BestFit: {best.f:.6f}, TestFit: {best.tf:.6f}")

            if self.population[C.POPSIZE].f < 1e-6:
                break
            
            self.generation += 1
        
        print(f"\nFinished Job {self.job} for F{self.function}. Generations: {self.generation}")
        best_solution = self.population[C.POPSIZE]
        self.test_objective(best_solution)
        print(f"Final Best Fitness: {best_solution.f:.6f}, Final Test Fitness: {best_solution.tf:.6f}")

    def run_de_logged(self):
        # 目录与误差-时间日志
        exp_dir = f"ExperimentalData/F{self.function}"
        os.makedirs(exp_dir, exist_ok=True)
        err_time_path = os.path.join(exp_dir, f"F{self.function}_error_time_{self.job}.txt")
        utils.save_text(err_time_path, "Duration time\tError\r\n", append=False)

        # 数据与寄存器
        self.read_data()
        self.load_data_to_rg()
        self.rg_nums = self.terminal_num + 5

        # 固定种子（保证可复现，每轮不同种子保证独立性）
        import random as pyrand
        seed_val = self.function * 1000 + self.job
        pyrand.seed(seed_val)
        np.random.seed(seed_val)

        self.initialization()

        # 时间循环
        start_time = time.time()
        last_record = start_time
        num_minutes = 10
        
        while (time.time() - start_time) < (num_minutes * 60):
            self.production()
            self.test_objective(self.population[C.POPSIZE])
            
            if self.generation % 200 == 0:
                best = self.population[C.POPSIZE]
                # Java 风格：function 0 job generation train_fit test_fit
                print(f"{self.function}\t0\t{self.job}\t{self.generation}\t{best.f:.6f}\t{best.tf:.6f}")

            now = time.time()
            if (now - last_record) > 0.1:
                best = self.population[C.POPSIZE]
                utils.save_text(err_time_path, f"{now - start_time:.3f}\t{best.f:.6f}\r\n", append=True)
                last_record = now

            if self.population[C.POPSIZE].f < 1e-6:
                break
            
            self.generation += 1
        
        # 收尾：最佳解与文件输出
        best_solution = self.population[C.POPSIZE]
        self.test_objective(best_solution)

        best_path = os.path.join(exp_dir, f"F{self.function}_best_solution{self.job}.txt")
        utils.save_text(best_path, f"Training fitness = {best_solution.f:.6f}, Testing fitness = {best_solution.tf:.6f}\r\n", append=False)
        utils.save_text(best_path, "Formula of the best solution is:\r\n", append=True)
        self.print_formula(best_path, best_solution)
        utils.save_text(best_path, "\r\n\r\n", append=True)

        utils.save_text(best_path, "The gene of the best solution is:\r\n", append=True)
        line = []
        for i, val in enumerate(best_solution.x):
            line.append(f"{val:.6f}\t")
            if (i + 1) % 4 == 0:
                utils.save_text(best_path, "".join(line) + "\r\n", append=True)
                line = []
        if line:
            utils.save_text(best_path, "".join(line) + "\r\n", append=True)
        utils.save_text(best_path, "\r\n", append=True)

        utils.save_text(best_path, "The coefficient of the best solution is:\r\n", append=True)
        line = []
        total_rg_count = self.basic_rg + self.rg_nums
        coeffs = best_solution.coefficients[:total_rg_count]
        for i, val in enumerate(coeffs):
            line.append(f"{val:.6f}\t")
            if (i + 1) % 8 == 0:
                utils.save_text(best_path, "".join(line) + "\r\n", append=True)
                line = []
        if line:
            utils.save_text(best_path, "".join(line) + "\r\n", append=True)

        results_path = os.path.join(exp_dir, f"F{self.function}_results.txt")
        utils.save_text(results_path, f"{self.job + 1}\t{best_solution.f:.6f}\t{best_solution.tf:.6f}\r\n", append=True)

        node_counts_path = os.path.join(exp_dir, f"F{self.function}_node_counts.txt")
        utils.save_text(node_counts_path, f"{self.job + 1}\t{best_solution.nodeCount}\r\n", append=True)

        print(f"\nFinished Job {self.job} for F{self.function}. Generations: {self.generation}")
        print(f"Final Best Fitness: {best_solution.f:.6f}, Final Test Fitness: {best_solution.tf:.6f}")


# 修改 mldep.py 的 __main__ 部分
def run_all_benchmarks():
    """运行所有12个基准问题"""

    # 12个问题对应的函数ID映射
    problem_mapping = {
        # 1: "Septic",
        # 2: "Nguyen-5",
        # 3: "Keijzer-1",
        # 4: "Korns-11",
        # 5: "Keijzer-14",
         6: "Keijzer-15",
        # 7: "Vladislavleva-8",
        # 8: "Keijzer-5",
        # 9: "Vladislavleva-5",
        # 10: "Energycooling",  
        # 11: "Energyheating",
        # 12: "Winered"
    }

    for function_id, problem_name in problem_mapping.items():
        print(f"\n{'=' * 50}")
        print(f"正在测试: {problem_name} (F{function_id})")
        print(f"{'=' * 50}")

        exp_dir = f"ExperimentalData/F{function_id}"
        os.makedirs(exp_dir, exist_ok=True)

        avg_fbest = 0.0
        avg_testing_fbest = 0.0
        avg_node_count = 0.0
        succ_count = 0

        # 对每个问题运行多次实验
        for job_id in range(30):  # 30次独立运行
            print(f"  运行 {job_id + 1}/30")

            m = MLDEP(function_id=function_id, job_id=job_id)
            m.run_de_logged()
            best = m.population[C.POPSIZE]

            avg_fbest += best.f
            avg_testing_fbest += best.tf
            avg_node_count += getattr(best, "nodeCount", 0)
            if best.f < 1e-6:
                succ_count += 1

        # 计算统计结果
        num_runs = 30.0
        succ_rate = succ_count / num_runs
        avg_fbest /= num_runs
        avg_testing_fbest /= num_runs
        avg_node_count /= num_runs

        # 保存最终结果
        final_path = "ExperimentalData/final_results.txt"
        utils.save_text(
            final_path,
            f"{function_id}\t{succ_rate}\t{avg_fbest:.6f}\t{avg_testing_fbest:.6f}\t{avg_node_count:.1f}\r\n",
            append=True
        )

        print(f"完成 {problem_name}: 成功率={succ_rate:.1%}, 测试NRMSE={avg_testing_fbest:.6f}")


# 替换原来的main部分
if __name__ == '__main__':
    run_all_benchmarks()