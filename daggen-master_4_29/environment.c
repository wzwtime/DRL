#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>

#include "daggen_commons.h"
#include "mcts.h"
#include "heft.h"
#include "environment.h"

double get_est(Etask *etask, Task task, int processor){
    int p_id, t_id, pro_id;
    double max_pred=-1, temp_pred;
    Task parent;
    t_id = task->tag;
    max_pred = env_g.availP[processor];
    //nb_parents=0时，最早开始时间为max_pred
    for(int i=0;i<task->nb_parents;i++){
        parent = task->parents[i];
        p_id = parent->tag;
        pro_id = etask[p_id-1]->processor;
        if(mcts_g.bandwidth[pro_id][processor]!=0){
            temp_pred = etask[p_id-1]->aft +
                mcts_g.comm_costs[p_id][t_id]/mcts_g.bandwidth[pro_id][processor];
        }else{
            temp_pred = etask[p_id-1]->aft;
        }
        if(max_pred<temp_pred)
            max_pred = temp_pred;
    }
    return max_pred;
}

double get_current_makespan(){
    double max=0;
    for(int i=0;i<global.n_processors;i++){
        if(max < env_g.availP[i])
            max = env_g.availP[i];
    }
    return max;
}

void get_state(DAG dag, Queue queue, Queue res_queue, Etask *etask, FILE *fp){
    int i,j,nr;
    Task task;
    Element p;
    double est;
    j=0;
    p=queue->head;
    while(p){
        task = p->task;
        //printf("Task %d: ",task->tag-1);
        for(i=0;i<global.n_processors;i++){
            est = get_est(etask, task, i);
            //printf("%lf ",est);
            //打印est，w
            fprintf(fp, "%lf %lf ",est, task->comp_costs[i]);
        }
        fprintf(fp,"\n");
        p=p->next; j++;
    }
    //打印剩余队列的任务
    p=res_queue->head;
    while(p && j<global.capacity){
        task = p->task;
        for(i=0;i<global.n_processors;i++){
            fprintf(fp,"-1 %lf ",task->comp_costs[i]);
        }
        fprintf(fp,"\n");
        p=p->next; j++;
    }
    //剩余队列任务数量不足，补齐到30为止
    for(;j<global.capacity;j++){
        for(i=0;i<global.n_processors;i++){
            fprintf(fp, "-1 -1 ");
        }
        fprintf(fp,"\n");
    }
}
double get_reward(){
    return env_g.prev_makespan - env_g.makespan;
}

void env_update_scheduled_task(Etask *etask, Task task, int processor){
    double start_time;
    int t_id;
    start_time = get_est(etask, task, processor);
    env_g.availP[processor] = start_time + task->comp_costs[processor];
    t_id = task->tag - 1;
    etask[t_id]->processor = processor;
    etask[t_id]->aft = env_g.availP[processor];
}
//更新就绪队列的同时，更新剩余任务队列
void update_ready_task_res(DAG dag, Task task, Queue queue, Queue res_queue){
    int i,nb_ready=0;
    Task *task_ready;
    task_ready = NULL;
    for(i=0;i<task->nb_children;i++){
        task->children[i]->nb_parents_r--;
        if(task->children[i]->nb_parents_r == 0){
            task_ready = (Task *)realloc(task_ready,(nb_ready+1)*sizeof(Task));
            task_ready[nb_ready] = task->children[i];
            nb_ready++;
            delete_task_queue(res_queue,task->children[i]->tag);
        }
    }
    if(nb_ready!=0){
        update_queue(queue,task_ready,nb_ready);
    }
}

void action(DAG dag, Queue queue, Queue res_queue, Etask *etask, int task_id, int processor){
    Task task;
    task = delete_task_queue(queue,task_id);
    env_update_scheduled_task(etask, task, processor);
    update_ready_task_res(dag,task,queue,res_queue);
}

int get_random_ready_task(Queue queue){
    int rnum, i=0;
    Element p, prev;
    //printf("===========%d\n",queue->n);
    rnum = rand() % queue->n;
    //printf("===========ok\n");
    p = queue->head;
    while(i < rnum){
        prev=p; p=p->next;
        i++;
    }
    return p->task->tag;
}

void init_env(DAG dag, Queue queue, Etask *etask, Queue res_queue){
    int i,j,k;
    Task *task_ready;
    //calloc availP[]初始化为0
    env_g.availP = (double *)calloc(global.n_processors,sizeof(double));
    k=0;
    for(i=0;i<dag->nb_levels;i++){
        for(j=0;j<dag->nb_tasks_per_level[i];j++){
            etask[k]->task = dag->levels[i][j];
            k++;
            insert_queue(res_queue,dag->levels[i][j]);
        }
    }
    //设置就绪队列
    task_ready = (Task *)calloc(dag->nb_tasks_per_level[0],sizeof(Task));
    for(i=0;i<dag->nb_tasks_per_level[0];i++){
        task_ready[i] = dag->levels[0][i];
        delete_task_queue(res_queue,dag->levels[0][i]->tag);
    }
    update_queue(queue,task_ready,dag->nb_tasks_per_level[0]);
}

void reset_env(DAG dag, Queue queue, Queue res_queue, Etask *etask){
    int i,j,k;
    Task *task_ready;
    k=0;
    for(i=0;i<dag->nb_levels;i++)
        for (j=0; j<dag->nb_tasks_per_level[i]; j++){
            dag->levels[i][j]->nb_parents_r=global.nb_parents[k];
            k++;
            insert_queue(res_queue,dag->levels[i][j]);
        }
    for(i=0;i<global.n_processors;i++){
        env_g.availP[i]=0;    
        env_g.prev_makespan = 0;
        env_g.makespan = 0;
    }
    //重新设置就绪队列
    task_ready = (Task *)calloc(dag->nb_tasks_per_level[0],sizeof(Task));
    for(i=0;i<dag->nb_tasks_per_level[0];i++){
        task_ready[i] = dag->levels[0][i];
        delete_task_queue(res_queue,dag->levels[0][i]->tag);
    }
    update_queue(queue,task_ready,dag->nb_tasks_per_level[0]);
}
void print_label(DAG dag, Queue queue, FILE *fp,int task_id, int processor){
    int i,j;
    Element p, prev;
    p = queue->head;
    i=0;
    //打印选择的任务标签，就绪队列最大容量30
    while(p){
        if(p->task->tag == task_id){
            fprintf(fp,"1 ");
        }
        else{
            fprintf(fp,"0 ");
        }
        prev=p; p=p->next; i++;
    }
    for(j=i;j<global.capacity;j++){
        fprintf(fp, "0 ");
    }
    fprintf(fp,"\n");
    //打印选择的处理器标签

//    for(j=0;j<global.n_processors;j++){
//        if(j==processor)
//            fprintf(fp, "1 ");
//        else
//            fprintf(fp, "0 ");
//    }
//    fprintf(fp,"\n");
}
void produce_trace(DAG dag){
    int i,episode;
    Etask *etask;
    Queue queue,res_queue;  //res_queue 未进入就绪队列的剩余任务队列
    int task_id, processor;
    FILE *fp1,*fp2;
    fp1 = fopen("data_state.txt","w");
    fp2 = fopen("data_label.txt","w");
    //fp = fopen("data_state.txt","a+");

    queue = init_queue();
    res_queue = init_queue();

    etask = (Etask *)calloc(global.n,sizeof(Etask));
    for(i=0;i<global.n;i++)
        etask[i] = (Etask)calloc(1,sizeof(struct _Etask));
    
    init_env(dag,queue,etask,res_queue);
    //episode模拟次数
    episode=0;
    while(episode<global.episode){
        for(i=0;i<global.n;i++){
            //printf("state %d:\n",i);
            //打印就绪任务的个数
            //fprintf(fp,"%d\n",queue->n);
            //打印est,w
            //print_queue(queue);
//	    	print_queue(res_queue);
            get_state(dag,queue,res_queue,etask,fp1);
            //printf("makespan: %lf\n",get_current_makespan());
            task_id = get_random_ready_task(queue);
            processor = rand() % global.n_processors;
            //printf("schedule task %d to processor %d\n",task_id-1,processor);
            //打印选择任务、处理器的label
            print_label(dag,queue,fp2,task_id,processor);

            action(dag, queue, res_queue, etask, task_id, processor);
        }
        reset_env(dag, queue, res_queue, etask);
        episode++; 
    }
}