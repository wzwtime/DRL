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

void init_htask(DAG dag, Htask *htask){
    int i,j,k,p,q;
    double temp_comp,avg_bw;

    heft_g.Ptime = (double *)calloc(global.n_processors,sizeof(double));
    heft_g.availP = (double *)calloc(global.n_processors,sizeof(double));
    heft_g.index = (int *)calloc(global.n,sizeof(int));
    avg_bw=0;
    for(i=0;i<global.n_processors;i++)
        for(j=0;j<global.n_processors;j++){
            avg_bw += mcts_g.bandwidth[i][j];
        }
    avg_bw = avg_bw / (global.n_processors * global.n_processors);
    k=0;
    for(i=0;i<dag->nb_levels;i++){
        for(j=0;j<dag->nb_tasks_per_level[i];j++){
            htask[k]->task = dag->levels[i][j];
            temp_comp = 0;
            for(p=0;p<global.n_processors;p++){
                temp_comp += dag->levels[i][j]->comp_costs[p]; 
            }
            htask[k]->comp_mean = temp_comp / global.n_processors;
            htask[k]->comm_means = (double *)calloc(
                                dag->levels[i][j]->nb_children, sizeof(double));
            htask[k]->efts = (double *)calloc(global.n_processors,sizeof(double));
            for(p=0;p<dag->levels[i][j]->nb_children;p++){
                htask[k]->comm_means[p] = dag->levels[i][j]->comm_costs[p] / avg_bw;
            }    
            htask[k]->ranku = 0;        
            k++;
        }
    }
}

double max_succ(DAG dag, Htask *htask, Task task){
    int i,j;
    double temp,max;
    Task child;
    max = 0;
    for(i=0;i<task->nb_children;i++){
        child = task->children[i];
        temp = htask[task->tag-1]->comm_means[i] + htask[child->tag-1]->ranku;
        if(temp > max)
            max = temp;
    }
    return max;
}

void compute_ranku(DAG dag, Htask *htask){
    int i,j,k;
    Task task;
    //printf("compute_ranku 1: %d\n",dag->nb_levels-1);
    i = dag->nb_levels-1;
    for(k=0;k<dag->nb_tasks_per_level[i];k++){
        task = dag->levels[i][k];
        htask[task->tag-1]->ranku = htask[task->tag-1]->comp_mean;
    }
    //printf("compute_ranku 2\n");
    if(dag->nb_levels >= 2){
        for(i=dag->nb_levels-2;i>=0;i--){
            for(j=0;j<dag->nb_tasks_per_level[i];j++){
                task = dag->levels[i][j];
                htask[task->tag-1]->ranku = htask[task->tag-1]->comp_mean 
                                            + max_succ(dag,htask,task);
            }
        }
    }
}
void sort_ranku(Htask *htask){
    int i,j;
    Htask ptask;
    for(i=0;i<global.n-1;i++)
        for(j=0;j<global.n-i-1;j++){
            if(htask[j]->ranku < htask[j+1]->ranku){
                ptask = htask[j];
                htask[j] = htask[j+1];
                htask[j+1] = ptask;
            } 
        }
}

double max_availP(){
    double max=0;
    for(int i=0;i<global.n_processors;i++){
        if(max < heft_g.availP[i])
            max = heft_g.availP[i];
    }
    return max;
}

double compute_est(Htask *htask, Task task, int processor){
    int p_id, t_id, pro_id;
    double max_pred=-1, temp_pred;
    Task parent;
    t_id = task->tag;   //mcts_g.comm_costs的下标从1开始
    max_pred = heft_g.availP[processor];
    for(int i=0; i<task->nb_parents; i++){
        parent = task->parents[i];
        p_id = parent->tag;
        pro_id = htask[heft_g.index[p_id-1]]->processor;  //htask的下标从0开始
        //printf("p%d -> p%d bandwidth:%d\n",pro_id,processor,mcts_g.bandwidth[pro_id][processor]);
        //printf("t%d -> t%d comm_costs: %.2lf\n",p_id-1, t_id-1, mcts_g.comm_costs[p_id][t_id]);
        //printf("pid:%d aft:%.2f pro_id:%d\n",p_id-1, htask[heft_g.index[p_id-1]]->aft, pro_id);
        //printf("availP:%.2f\n",heft_g.availP[processor]);
        if(mcts_g.bandwidth[pro_id][processor]!=0){
            temp_pred = htask[heft_g.index[p_id-1]]->aft + 
                        mcts_g.comm_costs[p_id][t_id]/mcts_g.bandwidth[pro_id][processor];
            //printf("temp_pred:%.2lf\n",temp_pred);
        }else{
            temp_pred = htask[heft_g.index[p_id-1]]->aft;
            //printf("temp_pred:%.2lf\n",temp_pred);
        }
        if(max_pred < temp_pred)
            max_pred = temp_pred;
        //printf("max_pred:%.2lf\n",max_pred);
    }
    return max_pred;
}
double compute_eft(DAG dag, Htask *htask, Task task, int processor, FILE *fp){
    double est,eft;
    est = compute_est(htask,task,processor);
    //1） 打印就绪队列的任务
    //fprintf(fp, "%lf %lf ",est, task->comp_costs[processor]);
    
//    printf("est=%f comp=%f\n", est, task->comp_costs[processor]); 
    eft = est + task->comp_costs[processor];
    //printf("Task %d est: %.2lf, eft: %.2lf\n",task->tag-1,est,eft);
    return eft;
}

//void print_queue(Queue queue){
//    Element p;
//    printf("Queue length:%d\n",queue->n);
//    p=queue->head;
//    printf("ready queue tasks id:");
//    while(p){
//        printf("%d ", p->task->tag);
//        p=p->next;    
//    }
//    printf("\n");
//}

void init_heft_env(DAG dag, Queue queue, Queue res_queue){
    int i,j,k;
    Task *task_ready;                 
    for(i=0;i<dag->nb_levels;i++){
        for(j=0;j<dag->nb_tasks_per_level[i];j++){
            insert_queue(res_queue,dag->levels[i][j]);
        }
    }
//    printf("--1--res_queue:\n");
//    print_queue(res_queue);
    //设置就绪队列
    task_ready = (Task *)calloc(dag->nb_tasks_per_level[0],sizeof(Task));
    for(i=0;i<dag->nb_tasks_per_level[0];i++){
        task_ready[i] = dag->levels[0][i];
        delete_task_queue(res_queue,dag->levels[0][i]->tag);
    }
//    printf("--2--res_queue:\n");
//    print_queue(res_queue);
    update_queue(queue,task_ready,dag->nb_tasks_per_level[0]);
//    printf("--queue:\n");
//    print_queue(queue);
}
void reset_heft_env(DAG dag, Queue queue, Queue res_queue){
    int i,j,k;
    Task *task_ready;
    k=0;
    for(i=0;i<dag->nb_levels;i++)
        for (j=0; j<dag->nb_tasks_per_level[i]; j++){
        	/*此处很重要*/ 
            dag->levels[i][j]->nb_parents_r=global.nb_parents[k];
            k++;
            insert_queue(res_queue,dag->levels[i][j]);
        }

    //重新设置就绪队列
    task_ready = (Task *)calloc(dag->nb_tasks_per_level[0],sizeof(Task));
    for(i=0;i<dag->nb_tasks_per_level[0];i++){
        task_ready[i] = dag->levels[0][i];
        delete_task_queue(res_queue,dag->levels[0][i]->tag);
    }
    update_queue(queue,task_ready,dag->nb_tasks_per_level[0]);
}
//更新就绪队列的同时，更新剩余任务队列
void update_heft_ready_task_res(DAG dag, Task task, Queue queue, Queue res_queue){
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

void heft_action(DAG dag, Queue queue, Queue res_queue, Htask *htask, int task_id){
    Task task;
    task = delete_task_queue(queue,task_id);
    //printf("----------调度后-----------\n");
//    printf("--queue:\n");
//    print_queue(queue);
    //env_update_scheduled_task(htask, task, processor);
//   	printf("--1--res_queue:\n");
//    print_queue(res_queue);
    update_heft_ready_task_res(dag,task,queue,res_queue);
    //printf("--queue:\n");
    //print_queue(queue);
    //printf("--2--res_queue:\n");
    //print_queue(res_queue);
}
void get_heft_state(DAG dag, Queue queue, Queue res_queue, Htask *htask, FILE *fp){
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
            //printf("%lf ",est);
            //打印est，w
           	est = compute_est(htask,task,i);
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
void get_heft_label(DAG dag, Queue queue, FILE *fp,int task_id, int processor){
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

void heft(DAG dag){
    int i,j,k,min_p;
    double min_eft=INFINITY;
    Htask *htask;
    
    FILE *fp1,*fp2;
    fp1 = fopen("heft_data_state.txt","w");
    fp2 = fopen("heft_data_label.txt","w");
    Queue queue,res_queue;  //res_queue 未进入就绪队列的剩余任务队列
   	htask = (Htask *)calloc(global.n,sizeof(Htask));
   	for(i=0;i<global.n;i++)
   		htask[i] = (Htask)calloc(1,sizeof(struct _Htask));
   		
    /*修改*/
    int episode;
    for(episode=0;episode<global.episode;episode++)
    {
//    	printf("----------%d\n",episode);
	    //printf("heft 1: %lf\n", htask[0]->comp_mean);
		queue = init_queue();
    	res_queue = init_queue();
	    init_heft_env(dag,queue,res_queue);
	    
//	    printf("----------初始化前-----------\n");
	    //get_heft_state(dag,queue,res_queue,min_p,htask,fp1);
	    
	    init_htask(dag,htask);
	    //printf("heft 2\n");
	    compute_ranku(dag,htask);
	    //按ranku降序调度每个任务
	    sort_ranku(htask);
	    
	    //注意htask的下标与task->tag的转换关系，index存储该关系
	    
	    for(i=0;i<global.n;i++){
	        heft_g.index[htask[i]->task->tag-1]=i;
	    }
	    for(i=0;i<global.n;i++){
//	    	printf("===========================state=%d\n", i);
//	    	printf("tag=%d\n", htask[i]->task->tag);
//	    	print_queue(queue);
//	    	print_queue(res_queue);
	    	
	    	get_heft_state(dag,queue,res_queue,htask,fp1);
	    	
	    	/*wzw 获取状态[est1, wij1, est2, wij2,......,estp,wijp]*/ 
	    	//printf("state %d:\n", i);
	    	
	        min_eft=INFINITY;
	        //printf("$$$$$$task %d ranku:%.2lf\n",htask[i]->task->tag-1,htask[i]->ranku);
	        for(j=0;j<global.n_processors;j++){
	            htask[i]->efts[j] = compute_eft(dag,htask,htask[i]->task,j, fp1);
	            
	            if(min_eft > htask[i]->efts[j]){
	                min_eft = htask[i]->efts[j];
	                min_p = j;
	            }
	        }
	        
	        //printf("****task %d p:%d aft:%.2lf\n",htask[i]->task->tag-1,min_p,min_eft);
	        htask[i]->processor = min_p;
	        htask[i]->aft = min_eft;
	        heft_g.availP[min_p] = min_eft;
	        //printf("p=%d\n",min_p);
      
	        //打印label--选择任务 tag在就绪任务的位置 
            get_heft_label(dag, queue, fp2, htask[i]->task->tag, min_p); 
		    
		    // 更新两个队列 	        
	        heft_action(dag, queue, res_queue, htask, htask[i]->task->tag);

	        //printf("****task %d p:%d aft:%.2lf\n",htask[i]->task->tag-1,htask[i]->processor,htask[i]->aft);
	    }
	    printf("HEFT makespan: %lf\n", max_availP());
	    reset_heft_env(dag, queue, res_queue);
	    //init_heft_env(dag,queue,res_queue);
    }
    //for(i=0;i<global.n_processors;i++)
      //  printf("processor i:%lf\n",heft_g.availP[i]);
}