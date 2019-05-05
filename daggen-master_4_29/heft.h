#ifndef HEFT_H_
#define HEFT_H_

typedef struct _Htask *Htask;

typedef struct{
    double *Ptime;         //处理器时间（暂未使用）
    double *availP;        //处理器就绪时间，计算eft使用
    int *index; 
}HEFT_g;
HEFT_g heft_g;

struct _Htask{
    Task task;
    double comp_mean;      //任务平均计算开销
    double *comm_means;    //边平均通信开销
    double ranku;          //upward rank
    double aft;            //actual finish time
    double *efts;          //任务在每个处理器上的最早开始时间
    int processor;         //任务在哪个处理器上运行
};

void heft(DAG dag);

#endif /*HEFT_H_*/