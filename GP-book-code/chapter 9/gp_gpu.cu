/*
 * SPDX-License-Identifier: MIT
 *
 * Copyright (C) 2026 Jinghui Zhong
 *
 * Companion source code for Chapter 9 of the book
 * "Genetic Programming Algorithms and Applications" (《遗传编程算法及其应用》).
 */
 #include<iostream>
 #include<string>
 #include<assert.h>
 #include<vector>
 #include<stack>
 #include<math.h>
 #include<algorithm>
 #include<time.h>
 using namespace std;
 
 //栈中的结构应该允许自定义
 #define MIN_DEPTH 7
 #define MAX_DEPTH 11
 #define GENERATION 5000
 #define POPULATION 100
 #define DATASET_SIZE 1000000
 #define M 1000000
 #define MUTRATE 0.2
 #define DEPTH 5
 #define THREAD_NUM 1024
 #define DATA_SIZE 3
 #define STACK_MAX_GM 50                 // 在全局内存中的最大栈深度，具体值如何设置还有待商榷
 #define DATANUM_INSHARED 1              // 24 * 1024 B / 4 Byte(float) / 3(DATA_SIZE) / 128(THREAD_NUM),这个的数量设置为多少还有待商榷，需要考虑共享内存占用和寄存器占用等的资源平衡
 #define CASH_SIZE THREAD_NUM * DATA_SIZE * DATANUM_INSHARED
 #define STACK_SIZE 20 //这个的取值也有待商榷
 #define POST_SIZE_MAX 5000
 
 __device__ int stack_max_size = 0;
 /*************************************************************************
  *description: 函数标识符，用于记录操作符以及对应的输入个数
  *TODO:
     [ ] 貌似还需要记录 参数个数-操作符 的映射？
  *return {*}
  *************************************************************************/
 struct Func {
     string label;
     int numchild_;
 };
 
 vector<Func> function_set;
 vector<string> terminal_set;
 vector<vector<float> > input_v;
 vector<float> output;
 int* post_GPU[POPULATION];
 int* post_size[POPULATION];

 float input[DATASET_SIZE * DATA_SIZE];
 float* input_GPU;
 float* output_GPU[POPULATION];
 float* stack_cash;
 
 const int range[2] = { -10, 10 };
 
 /*************************************************************************
  *description: 生成均匀分布的随机值
  *TODO:
  *return {*}
  *************************************************************************/
 float RandUniform() {
     return float(rand() % 10000) / 10000.f;
 }
 
 /*************************************************************************
  *description: 树节点表示,基类
  *TODO:
     [ ] 如何记录树节点总个数
     [ ] 父节点到子节点的指针复制好像有点问题
     [ ] childs好像没有设置数组大小，不会出问题么
     [ ] 没有记录深度
  *return {*}
  *************************************************************************/
 struct treeNode {
     treeNode(string, int, int, float);
     //~treeNode();
     string label_;           //三种类型，Func, Input，Constant
     int type_;               //如果是Input或Func节点，表示下标，否则为-1
     float value_;            //如果是Constant节点，表示值
     int numchild_;           //子节点数量
     treeNode* Copy();        //复制包括该节点在内的子树
     string getLabel();
     int size();
     int depth();
     treeNode** childs;      //子节点
     float fitness;           //适应度值
 };
 
 string treeNode::getLabel() {
     if (label_ == "Func") {
         return function_set[type_].label;
     }
     if (label_ == "Input") {
         return terminal_set[type_];
     }
 }
 
 // treeNode::~treeNode(){
 //     delete[] childs;
 // }
 
 treeNode::treeNode(string label, int type, int numchild, float value = -1) {
     label_ = label;
     type_ = type;
     value_ = value;
     numchild_ = numchild;
     if (numchild > 0) {
         this->childs = new treeNode * [numchild];
     }
 }
 
 treeNode* treeNode::Copy() {
     treeNode* cloneNode = new treeNode(this->label_, this->type_, this->numchild_, this->value_);
     for (int i = 0; i < this->numchild_; ++i) {
         cloneNode->childs[i] = this->childs[i]->Copy();
     }
     return cloneNode;
 }
 
 int treeNode::size() {
     int size = 0;
     for (int i = 0; i < this->numchild_; ++i) {
         int sub_size = this->childs[i]->size();
         size += sub_size;
     }
     return size + 1;//自身算1个节点，故加1
 }
 
 int treeNode::depth(){
    int depth = 0;
    for(int i = 0; i < this->numchild_; ++i){
        int sub_depth = this->childs[i]->depth();
        if(sub_depth > depth){
            depth = sub_depth;
        }
    }
    return depth + 1;
 }
 void deleteTree(treeNode* subtree) {
     assert(subtree->label_ != "");
     for (int i = 0; i < subtree->numchild_; ++i) {
         deleteTree(subtree->childs[i]);
     }
     delete subtree;
 }
 
 /*************************************************************************
  *description: 寻找指定交叉点，执行交叉操作
  *param {treeNode*} subtree
  *param {treeNode*} denote
  *param {int&} count
  *TODO:
     [ ] 当denote为空时，为搜索目标子树，返回复制体即可；
  *return {*}
  *************************************************************************/
 treeNode* scanTree(treeNode** subtree, treeNode* denote, int& count, int depth) {
     count -= 1;
     if (count <= 1) {
         if (!denote) {
             return (*subtree)->Copy();
         }
         else {
             assert((*subtree)->label_ != "");
             if(depth + denote->depth() < MAX_DEPTH){
                deleteTree(*subtree);
                *subtree = denote->Copy();
             }
         }
     }
     else {
         treeNode* ret = NULL;
         for (int i = 0; i < (*subtree)->numchild_; ++i) {
             if (count > 1) {
                 ret = scanTree(&(*subtree)->childs[i], denote, count, depth + 1);
             }
             else {
                 return ret;
             }
         }
         //assert(count <= 1);当到达了叶节点时，有可能count不为0；因此这里不能做此判断
     }
     return *subtree;
 }
 
 /*************************************************************************
  *description: 生成随机子树，用于mutation阶段
  *param {int} depth
  *TODO:
     [ ] label的表示貌似只有三种，而不是用符号
  *return {*}
  *************************************************************************/
 void randSubtree(treeNode** self, int depth, int cur_depth) {
     float randuniform = RandUniform();
     if (depth == 0 || randuniform < 0.5 || cur_depth >= MAX_DEPTH) {
         //终点集
         int randval = rand() % terminal_set.size();
         *self = new treeNode("Input", randval, 0);
     }
     else {
         //函数集
         int randval = rand() % function_set.size();
         int numchild = function_set[randval].numchild_;
         *self = new treeNode("Func", randval, numchild);
         for (int i = 0; i < numchild; ++i) {
             randSubtree(&((*self)->childs[i]), depth - 1, cur_depth + 1);
         }
     }
 }
 
 /*************************************************************************
  *description: 交叉操作
  *param {treeNode*} self
  *param {treeNode*} other
  *TODO:
  *return {*}
  *************************************************************************/
 void CrossOver(treeNode** self, treeNode** other) {
     float randuniform = RandUniform();
     if (randuniform < 0.9) {
         int size1 = (*self)->size(), size2 = (*other)->size();
         int crosspoint1 = rand() % size1, crosspoint2 = rand() % size2;
         treeNode* denote = scanTree(other, NULL, crosspoint2, 0);
         scanTree(self, denote, crosspoint1, 0);
     }
 }
 
 /*************************************************************************
  *description: 变异，每个树节点有一定概率进行变异，被替换为随机子树
  *param {treeNode*} self
  *TODO:
     [ ] 目前只用于简单情况，后期需要匹配如输入参数个数、输出参数个数等
  *return {*}
  *************************************************************************/
 void Mutation(treeNode** self, int depth) {
     float randval = RandUniform();
     if (randval < MUTRATE) {
         deleteTree(*self);
         randSubtree(self, DEPTH, depth);
     }
     else {
         for (int i = 0; i < (*self)->numchild_; ++i) {
 
             Mutation(&((*self)->childs[i]), depth + 1);
         }
     }
 }
 
 /*************************************************************************
  *description: 寻找指定变异点进行变异
  *param {treeNode**} self
  *param {int&} count
  *TODO:
  *return {*}
  *************************************************************************/
 void scanTreeMutation(treeNode** self, int& count, int depth) {
     count -= 1;
     if (count == 0) {
         deleteTree(*self);
         randSubtree(self, DEPTH, depth);
     }
     else {
         for (int i = 0; i < (*self)->numchild_; ++i) {
             if (count >= 0) {
                 scanTreeMutation(&(*self)->childs[i], count, depth + 1);
             }
         }
     }
 }
 
 /*************************************************************************
  *description: 变异，在树中随机选取一个变异点，替换为随机子树
  *param {treeNode*} self
  *TODO:
  *return {*}
  *************************************************************************/
 void MutationRP(treeNode** self) {
     int size = (*self)->size();
     float randval = RandUniform();
     if (randval < MUTRATE) {
         int count = rand() % size;
         scanTreeMutation(self, count, 0);
     }
 }
 
 /*************************************************************************
  *description: 初始化，根据最小最大深度随机生成树
  *TODO:
     [ ] 需要支持常量生成
  *return {*}
  *************************************************************************/
 treeNode* InitTree() {
     int depth = 0;
     vector<treeNode*> tstack;
     vector<treeNode*> tstack_tmp;
     //根节点一定是操作符
     int randval = rand() % function_set.size();
     treeNode* tnode = new treeNode("Func", randval, function_set[randval].numchild_);
     tstack.push_back(tnode);
     while (!tstack.empty()) {
         treeNode* tnode_tmp = tstack.back();
         tstack.pop_back();
         for (int i = 0; i < tnode_tmp->numchild_; ++i) {
             float randuniform = RandUniform();
             if (depth < MIN_DEPTH || (depth < MAX_DEPTH && randuniform < 0.5)) {
                 randval = rand() % function_set.size();
                 tnode_tmp->childs[i] = new treeNode("Func", randval, function_set[randval].numchild_);
             }
             else if (depth >= MAX_DEPTH || randuniform >= 0.5) {
                 randval = rand() % terminal_set.size();
                 tnode_tmp->childs[i] = new treeNode("Input", randval, 0);
             }
             else {
                 exit(-1);
             }
             tstack_tmp.push_back(tnode_tmp->childs[i]);
         }
         if (tstack.empty() && !tstack_tmp.empty()) {
             tstack = tstack_tmp;
             depth += 1;
             tstack_tmp.clear();
         }
     }
     return tnode;
 }
 
 struct opers{
    int label_;     //用于登记是func还是input
    int type_;      //集合下标
 };

/************************************************************************* 
 *description: 从语法表达树转为后缀表达式，为了适合于在GPU中运行，将树节点转换为opers结构
  *param {treeNode*} candidate
  *param {vector<opers>&} post
 *TODO: 
 *return {*}
 *************************************************************************/ 
void tree2post(treeNode* candidate, vector<opers>& post){
    for(int i = 0; i < candidate->numchild_; ++i){
        tree2post(candidate->childs[i], post);
    }
    opers oper_tmp;
    oper_tmp.type_ = candidate->type_;
    if(candidate->label_ == "Func"){
        oper_tmp.label_ = 0;
    }
    else if(candidate->label_ == "Input"){
        oper_tmp.label_ = 1;
    }
    post.push_back(oper_tmp);
}

 /*************************************************************************
  *description: 从语法表达树转为后缀表达式
  *param {treeNode*} candidate
  *param {vector<treeNode*>&} post
  *TODO:
  *return {*}
  *************************************************************************/
 void tree2post(treeNode* candidate, vector<treeNode*>& post) {
     for (int i = 0; i < candidate->numchild_; ++i) {
         tree2post(candidate->childs[i], post);
     }
     post.push_back(candidate);
 }
 
/************************************************************************* 
 *description: 
  *param {oper*} program: 后缀表达式
  *param {int*} program_size: 表达式长度
  *param {float**} dataset: 数据集
  *param {int*} dataset_size: 数据集大小 
  *param {float**} stack_cash: 位于全局内存的缓冲栈 
 *TODO: 
 *return {*}
 *************************************************************************/ 
__global__ void execution_GPU(int* program, int* program_size, float* dataset, float* data_output, float* stack_cash){
    int tid = threadIdx.x;
    int t_n = blockDim.x;
    // 每个线程执行一个程序，不同线程执行不同程序
    int max_stack = 0.75 * STACK_SIZE, min_stack = 0.25 * STACK_SIZE; 
    int transfer_atime = 0.25 * STACK_SIZE;
    // 将程序加载到共享内存中
    extern __shared__ int post[];
    while(tid < *program_size){//如果多个线程块，则不能这么读取
        post[tid * 2] = program[tid * 2];
        post[tid * 2 + 1] = program[tid * 2 + 1];
        tid += t_n;
    }
    
    // 部分数据预加载到共享内存, 数据格式: posi + type * thread_num(假设thread_num是32的倍数)；
        // 方案1，每次线程读取数据时，将新的数据填充到原先的位置上，同时更新数据指针位置。
            // 数据ID更新：data_idx += thread_num; 
            // 初始指针(posi)更新：(thread_num * data_size * data_idx) % cash_size
        // 方案2，每轮指定固定线程进行额外的数据传输操作，将数据补充到缓冲指针区域。

    tid = threadIdx.x;
    t_n = blockDim.x;
    //首先将cash区域填充完成
    __shared__ float data_cash[CASH_SIZE];
    int data_idx = threadIdx.x;
    int data_group = data_idx / blockDim.x;
    while(data_group < DATANUM_INSHARED){//每个线程轮流负责一个数据
        for(int i = 0; i < DATA_SIZE; ++i){
            data_cash[data_group * blockDim.x * DATA_SIZE + blockDim.x * i + data_idx % blockDim.x] = dataset[data_idx * DATA_SIZE + i];
        }
        data_idx += blockDim.x;
        data_group = data_idx / blockDim.x;
    }
    __threadfence();
    int dcash_pointer = threadIdx.x;

    register float data_stack[STACK_SIZE];//固定栈的大小，这样才能放到寄存器中.
    data_idx = threadIdx.x;
    while(data_idx < DATASET_SIZE){
        int stack_pointer = 0, tail_pointer = 0, cash_pointer = 0;//设置栈指针以及缓冲栈指针;

        //计算栈当前数据量： stack_pointer - tail_pointer + stack_pointer > tail_pointer ? 0 : STACK_SIZE;
        //如果栈数据量大于75%，tail_pointer数据开始传输回全局内存；如果栈数据量小于25%，tail_pointer将数据从全局内存传输回栈

        int stack_size = 0;
        for (int j = 0; j < *program_size; ++j) {//每个线程对分配给自己的每个数据都遍历一次程序
            if (post[j * 2] == 0) {
                float val1, val2, val_tmp;
                switch (post[j * 2 + 1]) {
                case 0: //'+'
                    
                    val1 = data_stack[(stack_pointer - 1 + STACK_SIZE) % STACK_SIZE];
                    val2 = data_stack[(stack_pointer - 2 + STACK_SIZE) % STACK_SIZE];
                    val_tmp = val1 + val2;
                    data_stack[(stack_pointer - 2 + STACK_SIZE) % STACK_SIZE] = val_tmp;
                    stack_pointer = (STACK_SIZE + stack_pointer - 1) % STACK_SIZE;
                    stack_size -= 1;
                    break;
                case 1: //'-'

                    val1 = data_stack[(stack_pointer - 1 + STACK_SIZE) % STACK_SIZE];
                    val2 = data_stack[(stack_pointer - 2 + STACK_SIZE) % STACK_SIZE];
                    val_tmp = val1 - val2;
                    data_stack[(stack_pointer - 2 + STACK_SIZE) % STACK_SIZE] = val_tmp;
                    stack_pointer = (STACK_SIZE + stack_pointer - 1) % STACK_SIZE;
                    stack_size -= 1;
                    break;
                case 2: //'*'

                    val1 = data_stack[(stack_pointer - 1 + STACK_SIZE) % STACK_SIZE];
                    val2 = data_stack[(stack_pointer - 2 + STACK_SIZE) % STACK_SIZE];
                    val_tmp = val1 * val2;
                    data_stack[(stack_pointer - 2 + STACK_SIZE) % STACK_SIZE] = val_tmp;
                    stack_pointer = (STACK_SIZE + stack_pointer - 1) % STACK_SIZE;
                    stack_size -= 1;
                    break;
                case 3: //'/'检查是否可除

                    val1 = data_stack[(stack_pointer - 1 + STACK_SIZE) % STACK_SIZE];
                    val2 = data_stack[(stack_pointer - 2 + STACK_SIZE) % STACK_SIZE];
                    val_tmp = INT_MAX;
                    if (val2 != 0) {
                        val_tmp = val1 / val2;
                    }
                    data_stack[(stack_pointer - 2 + STACK_SIZE) % STACK_SIZE] = val_tmp;
                    stack_pointer = (STACK_SIZE + stack_pointer - 1) % STACK_SIZE;
                    stack_size -= 1;
                    break;
                case 4: //'sqrt'

                    val1 = data_stack[(stack_pointer - 1 + STACK_SIZE) % STACK_SIZE];
                    val_tmp = INT_MAX;
                    if (val1 >= 0) {
                        val_tmp = sqrt(val1);
                    }
                    data_stack[(stack_pointer - 1 + STACK_SIZE) % STACK_SIZE] = val_tmp;
                    break;
                default:
                    printf("can not match the operator, location: evaluation function..\n");
                }
                
                if(j % 10 == 5){
                    //如果栈内数据过少，则往tail_pointer-1位置传入缓冲数据
                    if(cash_pointer > 0 && stack_size < min_stack){
                        printf("here..\n");
                        for(int k = 0; k < transfer_atime; ++k){
                            cash_pointer -= 1;//cash_pointer指向未赋值位置，故应先减1
                            tail_pointer = (tail_pointer + STACK_SIZE - 1) % STACK_SIZE;
                            data_stack[tail_pointer] = stack_cash[threadIdx.x * STACK_MAX_GM + cash_pointer];
                        }
                        stack_size += transfer_atime;
                    }
                }
            }
            else {
                
                data_stack[stack_pointer] = data_cash[dcash_pointer + blockDim.x * post[j * 2 + 1]];//读入数据

                stack_pointer = (stack_pointer + 1) % STACK_SIZE;
                stack_size += 1;
                // if(threadIdx.x == 0){
                //     if(stack_size > stack_max_size){
                //         printf("stack size: %d", stack_size);
                //         stack_max_size = stack_size;
                //     }
                // }
                if(j % 10 == 0){//多少次检查一次，有待商榷
                    //如果栈数据过多，则将tail_pointer位置的一个或多个数据传回全局内存
                    if(stack_size > max_stack){
                        printf("---------%d, %d\n", stack_size, max_stack);
                        for(int k = 0; k < transfer_atime; ++k){
                            stack_cash[threadIdx.x * STACK_MAX_GM + cash_pointer] = data_stack[tail_pointer];
                            tail_pointer = (tail_pointer + 1) % STACK_SIZE;
                            cash_pointer += 1;
                        }
                        stack_size -= transfer_atime;
                        //return;
                    }
                }
            }
        }
        data_output[data_idx] = data_stack[stack_pointer];
        

        for(int j = 0; j < DATA_SIZE; ++j){
            //将新数据传入共享内存相应位置
            data_cash[dcash_pointer + blockDim.x * j] = dataset[(data_idx + DATANUM_INSHARED * blockDim.x)* DATA_SIZE + j];
        }
        //__threadfence();
        data_idx += blockDim.x;
        dcash_pointer = (dcash_pointer + blockDim.x * DATA_SIZE) % (DATANUM_INSHARED * DATA_SIZE * blockDim.x);
    }
} 
 
 /*************************************************************************
  *description: 适应度值计算
  *param {const vector<float>&} ret: 保存每组数据获得的预估值
  *TODO:
  *return {*}
  *************************************************************************/
 float calculFitness(const vector<float>& ret) {
     float fitness = 0;
     for (int i = 0; i < DATASET_SIZE; ++i) {
         // cout << ret[i] << ' ' << output[i] << endl;
         // fitness += fabs((ret[i] - output[i]) / output[i]) * 100.f;
         fitness += fabs(fabs(ret[i]) - fabs(output[i]));
     }
     // fitness = DATASET_SIZE * M - fitness;
     fitness /= (float)DATASET_SIZE;
     return fitness;
 }

  /*************************************************************************
  *description: 适应度值计算
  *param {const vector<float>&} ret: 保存每组数据获得的预估值
  *TODO:
  *return {*}
  *************************************************************************/
  float calculFitness_GPU(const float* ret) {
    float fitness = 0;
    for (int i = 0; i < DATASET_SIZE; ++i) {
        // cout << ret[i] << ' ' << output[i] << endl;
        // fitness += fabs((ret[i] - output[i]) / output[i]) * 100.f;
        fitness += fabs(fabs(ret[i]) - fabs(output[i]));
    }
    // fitness = DATASET_SIZE * M - fitness;
    fitness /= (float)DATASET_SIZE;
    return fitness;
}
 

/*************************************************************************
 *description: 对数据集中的一组数据执行表达式, 操作符包括+,-,*,/,sin,cos,sqrt,pow
 *param {const vector<treeNode*>&} post: 后缀表达式
 *TODO:
    [ ] 缺少数据集输入部分
 *return {*}
 *************************************************************************/
 float execution(const vector<treeNode*>& post, int input_id) {
    vector<float> data_stack;
    for (int j = 0; j < post.size(); ++j) {
        if (post[j]->label_ == "Func") {
            float val1, val2, val_tmp;
            switch (post[j]->type_) {
            case 0: //'+'
                if (data_stack.size() < 2) {
                    cout << "+ wrong..\n";
                    cin >> val1;
                }
                assert(data_stack.size() >= 2);

                val1 = data_stack.back();
                data_stack.pop_back();
                val2 = data_stack.back();
                data_stack.pop_back();
                val_tmp = val1 + val2;
                data_stack.push_back(val_tmp);
                break;
            case 1: //'-'
                if (data_stack.size() < 2) {
                    cout << "- wrong..\n";
                    cin >> val1;
                }
                assert(data_stack.size() >= 2);

                val1 = data_stack.back();
                data_stack.pop_back();
                val2 = data_stack.back();
                data_stack.pop_back();
                val_tmp = val1 - val2;
                data_stack.push_back(val_tmp);
                break;
            case 2: //'*'
                if (data_stack.size() < 2) {
                    cout << "* wrong..\n";
                    cin >> val1;
                }
                assert(data_stack.size() >= 2);

                val1 = data_stack.back();
                data_stack.pop_back();
                val2 = data_stack.back();
                data_stack.pop_back();
                val_tmp = val1 * val2;
                data_stack.push_back(val_tmp);
                break;
            case 3: //'/'检查是否可除
                if (data_stack.size() < 2) {
                    cout << "/ wrong..\n";
                    cin >> val1;
                }
                assert(data_stack.size() >= 2);

                val1 = data_stack.back();
                data_stack.pop_back();
                val2 = data_stack.back();
                data_stack.pop_back();
                val_tmp = INT_MAX;
                if (val2 != 0) {
                    val_tmp = val1 / val2;
                }
                data_stack.push_back(val_tmp);
                break;
                // case 4: //'sin'
                //     if (data_stack.size() < 1) {
                //         cout << "sin wrong..\n";
                //         cin >> val1;
                //     }
                //     assert(data_stack.size() >= 1);

                //     val1 = data_stack.back();
                //     data_stack.pop_back();
                //     val_tmp = sin(val1);
                //     data_stack.push_back(val_tmp);
                //     break;
                // case 5: //'cos'
                //     if (data_stack.size() < 1) {
                //         cout << "cos wrong..\n";
                //         cin >> val1;
                //     }
                //     assert(data_stack.size() >= 1);

                //     val1 = data_stack.back();
                //     data_stack.pop_back();
                //     val_tmp = cos(val1);
                //     data_stack.push_back(val_tmp);
                //     break;
                // case 6: //'pow'
                //     if (data_stack.size() < 2) {
                //         cout << "pow wrong..\n";
                //         cin >> val1;
                //     }
                //     assert(data_stack.size() >= 2);

                //     val1 = data_stack.back();
                //     data_stack.pop_back();
                //     val2 = data_stack.back();
                //     data_stack.pop_back();
                //     val_tmp = pow(val1, val2);
                //     data_stack.push_back(val_tmp);
                //     break;
            case 4: //'sqrt'
                if (data_stack.size() < 1) {
                    cout << "sqrt wrong..\n";
                    cin >> val1;
                }
                assert(data_stack.size() >= 1);

                val1 = data_stack.back();
                val_tmp = INT_MAX;
                data_stack.pop_back();
                if (val1 >= 0) {
                    val_tmp = sqrt(val1);
                }
                data_stack.push_back(val_tmp);
                break;
            default:
                cout << "can not match the operator, location: evaluation function.." << endl;
                exit(-1);
            }
        }
        else {
            data_stack.push_back(input_v[input_id][post[j]->type_]);//读入数据
        }
    }
    assert(data_stack.size() == 1);//最后data_stack应该只剩一个结果，如果data_stack内数据量大于1；证明语法树未执行完整
    return data_stack[0];
}

 /*************************************************************************
  *description: 适应度评估，包括表达式执行和适应度值获取两个部分
  *param {treeNode*} candidate
  *TODO:
  *return {*}
  *************************************************************************/
 float evaluation(treeNode* candidate) {
     //从表达树转为后缀表达式
     vector<treeNode*> post;
     tree2post(candidate, post);
    //  cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< post size: " << ' ' << post.size() << endl;
    //  for (int i = 0; i < post.size(); ++i) {
    //      cout << post[i]->getLabel() << ' ';
    //  }
    //  cout << endl << endl;
 
     //表达式执行
     vector<float> ret;
     for (int i = 0; i < DATASET_SIZE; ++i) {
         float result = execution(post, i);
         ret.push_back(result);//将所得结果压入结果栈中保存
     }
 
     //获取适应度值
     float fitness = calculFitness(ret);
     return fitness;
 }
 
 
 struct Statistic{
    float aver_size_post = 0;
 };

 void evaluation_GPU(treeNode* candidate, int indiv_id, cudaStream_t* stream, Statistic& aver_post) {
    //从表达树转为后缀表达式
    vector<treeNode*> post;
    tree2post(candidate, post);
    int size = post.size();
    aver_post.aver_size_post += size;
    // cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< post size: " << ' ' << post.size() << endl;
    //if(post.size() < 2){
    //  for (int i = 0; i < post.size(); ++i) {
    //      cout << post[i]->getLabel() << ' ';
    //  }
    //  cout << endl << endl;
    //}
    int* post_tmp = new int[post.size() * 2];
    for(int i = 0; i < post.size(); ++i){
        if(post[i]->label_ == "Func"){
            post_tmp[i * 2] = 0;
        }
        if(post[i]->label_ == "Input"){
            post_tmp[i * 2] = 1;
        }
        post_tmp[i * 2 + 1] = post[i]->type_;
    }

    int post_size_t = post.size();
    cudaMemcpyAsync(post_size[indiv_id], &post_size_t, sizeof(int), cudaMemcpyHostToDevice, stream[indiv_id]);
    cudaMemcpyAsync(post_GPU[indiv_id], post_tmp, sizeof(int) * post.size() * 2, cudaMemcpyHostToDevice, stream[indiv_id]);
    //__global__ void execution(oper* program, int* program_size, float** dataset, int* dataset_size, float* data_output, float** stack_cash){
    
    execution_GPU<<<1, THREAD_NUM, post_size_t * 2 * sizeof(int), stream[indiv_id]>>>(post_GPU[indiv_id], post_size[indiv_id], input_GPU, output_GPU[indiv_id], stack_cash + indiv_id * THREAD_NUM * STACK_MAX_GM);
    
    // cudaDeviceSynchronize();
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n%s\n", cudaGetErrorString(cudaStatus));
         exit(-1);
    }
    //执行结果传输
}

void transferDataset(){
    for(int i = 0; i < POPULATION; ++i){
        cudaMalloc((void**)&post_GPU[i], POST_SIZE_MAX * 2 * sizeof(int));
        //output
        cudaMalloc((void**)&output_GPU[i], DATASET_SIZE * sizeof(float));
    cudaMalloc((void**)&post_size[i], sizeof(int));
    }
    //dataset
    cudaMalloc((void**)&input_GPU, DATASET_SIZE * DATA_SIZE * sizeof(float));
    cudaMemcpy(input_GPU, input, DATASET_SIZE * DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    //stack_cash
    cudaMalloc((void**)&stack_cash, STACK_MAX_GM * THREAD_NUM * POPULATION * sizeof(float));
}

 /*************************************************************************
  *description: 操作符包括：+,-,*,/,sin,cos,sqrt,pow；输入包括：x1, x2, x4
  *TODO:
  *return {*}
  *************************************************************************/
 void Register() {
     //函数集
     Func func_register;
     func_register.label = "+";
     func_register.numchild_ = 2;
     function_set.push_back(func_register);
 
     func_register.label = "-";
     func_register.numchild_ = 2;
     function_set.push_back(func_register);
 
     func_register.label = "*";
     func_register.numchild_ = 2;
     function_set.push_back(func_register);
 
     func_register.label = "/";
     func_register.numchild_ = 2;
     function_set.push_back(func_register);
 
     // func_register.label = "sin";
     // func_register.numchild_ = 1;
     // function_set.push_back(func_register);
 
     // func_register.label = "cos";
     // func_register.numchild_ = 1;
     // function_set.push_back(func_register);
 
     // func_register.label = "pow";
     // func_register.numchild_ = 2;
     // function_set.push_back(func_register);
 
     func_register.label = "sqrt";
     func_register.numchild_ = 1;
     function_set.push_back(func_register);
 
     //终点集
     terminal_set.push_back("x1");
     terminal_set.push_back("x2");
     terminal_set.push_back("x4");
 }
 
 /*************************************************************************
  *description: 根据DATASET_SIZE生成数据集，公式为y = x^5 + x^2 + x + x_2 ^ 3 + x_2^4 + x_2 + x_4 * x
  *TODO:
  *return {*}
  *************************************************************************/
 void generateDataset() {
 
     //y = x^5 + x^2 + x - x_2 ^ 3 + x_2^4 + x_2 + x_4 * x
     for (int i = 0; i < DATASET_SIZE; ++i) {
 
         vector<float> input_tmp;
 
         //生成值
         float x1 = rand() % range[1] + RandUniform(), x2 = rand() % range[1] + RandUniform(), x4 = rand() % range[1] + RandUniform();
 
         //生成正负符号
         float randval = RandUniform();
         if (randval < 0.5) {
             x1 = -x1;
         }
         randval = RandUniform();
         if (randval < 0.5) {
             x2 = -x2;
         }
         randval = RandUniform();
         if (randval < 0.5) {
             x4 = -x4;
         }
 
         input_tmp.push_back(x1);
         input_tmp.push_back(x2);
         input_tmp.push_back(x4);
 
         //生成结果并保存
         float uns = pow(x1, 5) + pow(x1, 2) - pow(x2, 3) + pow(x2, 4) + x2 + x4 * x1 + x1;
 
         input_v.push_back(input_tmp);
         input[i * DATA_SIZE + 0] = x1;
         input[i * DATA_SIZE + 1] = x2;
         input[i * DATA_SIZE + 2] = x4;
         output.push_back(uns);
     }
 }
 
 struct Individual {
     Individual(treeNode*, float);
     treeNode* chrom_;
     float fitness_;
 };
 
 Individual::Individual(treeNode* chrom, float fitness) {
     chrom_ = chrom;
     fitness_ = fitness;
 }
 
 /*************************************************************************
  *description:
  *TODO:
     [ ] 数据集生成没做
  *return {*}
  *************************************************************************/
 int main() {

    
    cudaStream_t stream[POPULATION];
    for(int i = 0; i < POPULATION; ++i){
        cudaStreamCreate(&stream[i]);
    }

     //生成符号集
     Register();
 
     //生成数据集
     generateDataset();
    
     transferDataset();
     treeNode* pop[POPULATION];
     float fitness[POPULATION];
 
     clock_t start, end;
     clock_t tstart, tend;
     clock_t gstart, gend;
     start = clock();
     tstart = start;
     float time_genetic = 0, time_GPU = 0;

     Statistic aver_message;
     //种群初始化
     for (int i = 0; i < POPULATION; ++i) {
         pop[i] = InitTree();
         cout << "chromosome ID: " << i << endl;
         evaluation_GPU(pop[i], i, stream, aver_message);
     }
    float* ret_init = new float[DATASET_SIZE];
    for(int i = 0; i < POPULATION; ++i){
        cudaMemcpyAsync(ret_init, output_GPU[i], DATASET_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
            //获取适应度值
        fitness[i] = calculFitness_GPU(ret_init);
    }
     aver_message.aver_size_post /= POPULATION;
     end = clock();
     cout << "<<========================================Initialization: " << double(end - start) / CLOCKS_PER_SEC << ", average post size: " << aver_message.aver_size_post << endl;

     //种群迭代
     for (int i = 0; i < GENERATION; ++i) {
         vector<Individual> pop_tmp;

        treeNode* child[POPULATION];
        float* ret[POPULATION];
         start = clock();
        aver_message.aver_size_post = 0;
         for (int j = 0; j < POPULATION; ++j) {
             //交叉
             int donate_idx = rand() % POPULATION;
             child[j] = pop[j]->Copy();
             //cout << "Crossover ..\n";

             CrossOver(&child[j], &pop[donate_idx]);
             //变异
             //cout << "Mutation ..\n";
             MutationRP(&child[j]);
         }

        end = clock();
        time_genetic += double(end - start) / CLOCKS_PER_SEC;
        // cout << "<<========================================GeneticOperation: " << double(end - start) / CLOCKS_PER_SEC << "s " << endl;

        start = clock();
        gstart = start;
        for(int j = 0; j < POPULATION; ++j){

            //适应度评估
            //cout << "Fitness Evaluation ..\n";
            evaluation_GPU(child[j], j, stream, aver_message);
        }
        cout << "here..\n";


         for(int j = 0; j < POPULATION; ++j){
            ret[j] = new float[DATASET_SIZE];
            cudaMemcpyAsync(ret[j], output_GPU[j], DATASET_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream[j]);
         }
        end = clock();
        // cout << "<<========================================Evaluation: " << double(end - start) / CLOCKS_PER_SEC << "s " << endl;
        time_GPU = double(end - start) / CLOCKS_PER_SEC;
        start = clock();
         for(int j = 0; j < POPULATION; ++j){
            float fit_child = calculFitness_GPU(ret[j]);
             Individual indiv_tmp(child[j], fit_child);
             pop_tmp.push_back(indiv_tmp);
             // //选择，与父代个体对比
             // cout << "Selection ..\n";
             // if (fit_child < fitness[j]) {
             //     assert(pop[j]->label_ != "");
             //     deleteTree(pop[j]);
             //     pop[j] = child;
             //     fitness[j] = fit_child;
             // }
              //cout << "iteration: " << i << ' ' << fit_child << endl;
             delete ret[j];
         }
         end = clock();
         cout << "time: " << double(end - start) / CLOCKS_PER_SEC << endl;
        cout << "here..\n";

        //  // 选择，排序后选前POPULATION个
        //  for (int j = 0; j < POPULATION; ++j) {
        //      Individual indiv_tmp(pop[j], fitness[j]);
        //      pop_tmp.push_back(indiv_tmp);
        //  }
        //  sort(pop_tmp.begin(), pop_tmp.end(), [=](const Individual& x, const Individual& y) {
        //      return x.fitness_ < y.fitness_;
        //      });
        //  for (int j = 0; j < POPULATION; ++j) {
        //      pop[j] = pop_tmp[j].chrom_;
        //      fitness[j] = pop_tmp[j].fitness_;
        //      cout << "individual: " << j << ' ' << fitness[j] << endl;
        //  }

        // 选择, 锦标赛选择
          for (int j = 0; j < POPULATION; ++j) {
             Individual indiv_tmp(pop[j], fitness[j]);
             pop_tmp.push_back(indiv_tmp);
         }
         sort(pop_tmp.begin(), pop_tmp.end(), [=](const Individual& x, const Individual& y) {
             return x.fitness_ < y.fitness_;
             });

        int candidate_num = 5;
        pop[0] = pop_tmp[0].chrom_;
        fitness[0] = pop_tmp[0].fitness_;
        pop_tmp.erase(pop_tmp.begin());
        cout << "individual: " << 0 << ' ' << fitness[0] << endl;
        for(int j = 1; j < POPULATION; ++j){
            int min_rid = pop_tmp.size();
            for(int k = 0; k < candidate_num; ++k){
                int rid = rand() % pop_tmp.size();
                if(rid < min_rid){
                    min_rid = rid;//因为预先进行了排序，因此序号越前，适应度越小
                }
            }
            pop[j] = pop_tmp[min_rid].chrom_;
            fitness[j] = pop_tmp[min_rid].fitness_;
            pop_tmp.erase(pop_tmp.begin() + min_rid);
            cout << "individual: " << j << ' ' << fitness[j] << endl;
        }

         for (int j = pop_tmp.size() - 1; j >= POPULATION; --j) {
             deleteTree(pop_tmp[j].chrom_);//删除剩余个体
         }
         
         end = clock();

        //  cout << "<<========================================Remain: " << double(end - start) / CLOCKS_PER_SEC << "s " << endl;
         time_genetic += double(end - start) / CLOCKS_PER_SEC;

         aver_message.aver_size_post /= POPULATION;
         end = clock();
         tend = end;
         gend = end;
         cout << "<<========================================Iteration: " << i << ", time cost: " << double(gend - gstart) / CLOCKS_PER_SEC << "s, " << time_genetic <<  "s, " << time_GPU <<  "s, " << time_GPU / time_genetic << ", average post size: " << aver_message.aver_size_post << endl;

         time_genetic = 0;

     }
     
    vector<treeNode*> post;
    tree2post(pop[0], post);
    int size = post.size();
    // cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< post size: " << ' ' << post.size() << endl;
    //if(post.size() < 2){
      for (int i = 0; i < post.size(); ++i) {
          cout << post[i]->getLabel() << ' ';
      }
      cout << endl << endl;
    //}
    //  int finish;
    //  cin >> finish;
 }