#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

typedef struct _Etask *Etask;

typedef struct{
    double *availP;        //处理器就绪时间，计算eft使用
    double prev_makespan;
    double makespan;
}ENV_g;
ENV_g env_g;

struct _Etask{
    Task task;
    double aft;            //actual finish time
    int processor;         //任务在哪个处理器上运行
};

void produce_trace(DAG dag);
#endif /*ENVIRONMENT_H_*/