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
 #include<vector>
 #include<stack>
 #include<math.h>
 #include<algorithm>
 #include<time.h>
 #include<fstream>
 #include<assert.h>
 #include<mpi.h>
 using namespace std;
 
 //栈中的结构应该允许自定义
 #define MIN_DEPTH 5
 #define MAX_DEPTH 9
 #define GENERATION 30
 #define POPULATION 100
 #define DATASET_SIZE 1000
 #define M 1000000
 #define MUTRATE 0.2
 #define DEPTH 5
 #define THREAD_NUM 1024
 #define DATA_SIZE 3
 #define STACK_MAX_GM 100                 // 在全局内存中的最大栈深度，具体值如何设置还有待商榷
 #define DATANUM_INSHARED 16              // 24 * 1024 KB / 4 Byte(float) / 3(DATA_SIZE) / 128(THREAD_NUM),这个的数量设置为多少还有待商榷，需要考虑共享内存占用和寄存器占用等的资源平衡
 #define CASH_SIZE THREAD_NUM * DATA_SIZE * DATANUM_INSHARED
 #define STACK_SIZE 100 //这个的取值也有待商榷
 #define POST_SIZE_MAX 5000
 
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
float output[DATASET_SIZE];
 int* post_GPU;
 int* post_size;

 float input[DATASET_SIZE * DATA_SIZE];
 float* output_onGPU;
 float* fitness_GPU;
 float* input_GPU;
 float* output_GPU;
 float* stack_cash;
 
 const int range[2] = { -100, 10 };
 
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
 void randSubtree(treeNode** self, int depth) {
     float randuniform = RandUniform();
     if (depth == 0 || randuniform < 0.5) {
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
             randSubtree(&((*self)->childs[i]), depth - 1);
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
     if (randuniform < 0.5) {
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
 void Mutation(treeNode** self) {
     float randval = RandUniform();
     if (randval < MUTRATE) {
         deleteTree(*self);
         randSubtree(self, DEPTH);
     }
     else {
         for (int i = 0; i < (*self)->numchild_; ++i) {
 
             Mutation(&((*self)->childs[i]));
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
 void scanTreeMutation(treeNode** self, int& count) {
     count -= 1;
     if (count == 0) {
         deleteTree(*self);
         randSubtree(self, DEPTH);
     }
     else {
         for (int i = 0; i < (*self)->numchild_; ++i) {
             if (count >= 0) {
                 scanTreeMutation(&(*self)->childs[i], count);
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
         scanTreeMutation(self, count);
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

int post_size_t[POPULATION];
int post_size_max = 0;


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
         output[i] = uns;
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
 int main(int argc, char** argv) {

    MPI_Status status;
    int myid, numprocs, buffer_size = 0, subpop_num, indiv_size;
    double* indiv_buf, * fitness_buf, *fitness_gather;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        
    int* sendcount = new int[numprocs], displs[numprocs];
    int* recvcounts = new int[numprocs], displs_recv[numprocs];
    if(myid == 0){
        fitness_gather = (double*)malloc(POP_SIZE * sizeof(double));
    }
    subpop_num = (POPULATION – 1) / numprocs + 1;
    indiv_size = POPULATION / numprocs + (POPULATION % numprocs > myid ? 1 : 0);
    indiv_buf = (double*)malloc((MAX_PROGSIZE + 1) * subpop_num * sizeof(double));
    fitness_buf = (double*)malloc(subpop_num * sizeof(double));
    displs[0] = 0, displs_recv[0] = 0;
    
    for(int i = 0; i < numprocs; ++i){
        sendcount[i] =indiv_size * (MAX_PROGSIZE + 1);
        recvcount[i] = indiv_size * (MAX_PROGSIZE + 1);
        if(i > 0){
            displs[i] = displs[i – 1] + sendcounts[i – 1];
            displs_recv[i] = displs_recv[i – 1] + recvcounts[i – 1];
        }
    }

    //生成符号集
    Register();
 
    srand(0);
     //生成数据集
     generateDataset();
    
     treeNode* pop[POPULATION];
     float fitness[POPULATION];
 
     clock_t start, end;
     clock_t tstart, tend;

     Statistic aver_message;
    srand(0);
     //种群初始化
     for (int i = 0; i < POPULATION; ++i) {
         pop[i] = InitTree();
         cout << "chromosome ID: " << i << ' ';
     }
     start = clock();
     tstart = start;
     post_size_max = 0;

    MPI_Scatterv(pop, sendcounts, MPI_DOUBLE, displs, indiv_buf, recvcounts, MPI_DOUBLE, 0, MPI_COMM_WORLD);

     for(int i = 0; i < POPULATION; ++i){
        evaluation(pop[i]);
     }

    MPI_Gatherv(fitness_buf, indiv_size, MPI_DOUBLE, fitness_gather, recvcounts, displs_recv, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    float* ret_init = new float[DATASET_SIZE];
     end = clock();

     delete ret_init;
     aver_message.aver_size_post /= POPULATION;
     cout << "<<========================================Initialization: " << double(end - start) / CLOCKS_PER_SEC << ", average post size: " << aver_message.aver_size_post << endl;
    
    
     //种群迭代
     for (int i = 0; i < GENERATION; ++i) {
         int average_size = 0;
         vector<Individual> pop_tmp;

        treeNode* child[POPULATION];
        float ret[POPULATION];
         start = clock();
        tstart = start;
         
        aver_message.aver_size_post = 0;
         for (int j = 0; j < POPULATION; ++j) {
             //交叉
             int donate_idx = rand() % POPULATION;
             child[j] = pop[j]->Copy();
             CrossOver(&child[j], &pop[donate_idx]);
         }

         for(int j = 0; j < POPULATION; ++j){
             //变异
             MutationRP(&child[j]);
         }
        
        MPI_Scatterv(pop, sendcounts, MPI_DOUBLE, displs, indiv_buf, recvcounts, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for(int j = 0; j < POPULATION; ++j){
            ret[j] = evaluation(child[j]);
        }
        
        MPI_Gatherv(fitness_buf, indiv_size, MPI_DOUBLE, fitness_gather, recvcounts, displs_recv, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int j = 0; j < POPULATION; ++j)
        {
             Individual indiv_tmp(child[j], ret[j]);
             pop_tmp.push_back(indiv_tmp);
        }

         start = clock();
         // 选择，排序后选前POPULATION个
         for (int j = 0; j < POPULATION; ++j) {
             Individual indiv_tmp(pop[j], fitness[j]);
             pop_tmp.push_back(indiv_tmp);
         }
         sort(pop_tmp.begin(), pop_tmp.end(), [=](const Individual& x, const Individual& y) {
             return x.fitness_ < y.fitness_;
             });
         for (int j = 0; j < POPULATION; ++j) {
             pop[j] = pop_tmp[j].chrom_;
             fitness[j] = pop_tmp[j].fitness_;
         }
         for (int j = pop_tmp.size() - 1; j >= POPULATION; --j) {
             deleteTree(pop_tmp[j].chrom_);//删除剩余个体
         }
         
         
        aver_message.aver_size_post /= POPULATION;
         end = clock();
         tend = end;
         cout << "<<========================================Iteration: " << i << ", time cost: " << double(end - start) / CLOCKS_PER_SEC <<  "s, " << double(tend - tstart) / CLOCKS_PER_SEC << ", average post size: " << aver_message.aver_size_post << endl;
        cout << "average size: " << average_size << endl;
         exit(0);
     }
    //  int finish;
    //  cin >> finish;
 }