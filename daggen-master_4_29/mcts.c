#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>

#include "daggen_commons.h"
#include "mcts.h"

#define MAXtime 50
#define CP -10
//#define INFINITY 999999

//初始化处理器性能
void init_processor(){
    int i,j;
    FILE *fp_pc;
    fp_pc = fopen("processor.txt","w");
    for(i=0;i<global.n_processors;i++){
        fprintf(fp_pc,"%d ", (int)getRandomNumberBetween((double)MINP,(double)MAXP));
    }
    fprintf(fp_pc,"\n");
    fclose(fp_pc);
}
//初始化处理器间网络带宽
void init_bandwidth(){
    int i,j;
    FILE *fp_bw;
    fp_bw = fopen("bandwidth.txt","w");
    for(i=0;i<global.n_processors;i++){
        for(j=0;j<global.n_processors;j++){
            if(i!=j){
                fprintf(fp_bw,"%d ", 
                (int)getRandomNumberBetween((double)MINP*global.ccr,
                                    (double)MAXP*global.ccr));
            }else{
                fprintf(fp_bw,"0 ");
            }
        }
        fprintf(fp_bw, "\n");
    }
    fprintf(fp_bw,"\n");
    fclose(fp_bw);
}

// 分配mcts_g.prcossor_performance和bandwidth内存空间，并从文件读取相应数值
void assign_processor_bandwidth(){
    int i,j;
    FILE *fp_pc, *fp_bw;    
    fp_pc=fopen("processor.txt","r");
    fp_bw=fopen("bandwidth.txt","r"); 
    if(!fp_pc || !fp_bw){
        printf("processor.txt or bandwidth.txt not found!\n");
        init_processor();
        init_bandwidth();
        fp_pc=fopen("processor.txt","r");
        fp_bw=fopen("bandwidth.txt","r");
    }
    //分配内存空间
    mcts_g.processor_performance = (int *)calloc(global.n_processors,sizeof(int));
    mcts_g.bandwidth = (int **)calloc(global.n_processors,sizeof(int *));
    for(i=0;i<global.n_processors;i++)
        mcts_g.bandwidth[i]=(int *)calloc(global.n_processors,sizeof(int));
    mcts_g.comm_costs = (double **)calloc(global.n,sizeof(double *));
    for(i=0;i<global.n;i++)
        mcts_g.comm_costs[i]=(double *)calloc(global.n,sizeof(double));

    mcts_g.makespan = 0; mcts_g.nb_node = 0; mcts_g.node_id = 0;
    mcts_g.nodetrace = (Node *)calloc(global.n+1,sizeof(Node));
    mcts_g.scheduled = (Scheduled *)calloc(global.n+1,sizeof(Scheduled));
    mcts_g.availP = (double *)calloc(global.n_processors,sizeof(double));
    for(i=0;i<global.n+1;i++){
        mcts_g.scheduled[i] = (Scheduled)calloc(1,sizeof(struct _scheduled));
    }
    
    //读处理器性能
    for(i=0;i<global.n_processors;i++){
        fscanf(fp_pc,"%d ", &mcts_g.processor_performance[i]);
    }
    //读网络带宽
    for(i=0;i<global.n_processors;i++){
        for(j=0;j<global.n_processors;j++){
            fscanf(fp_bw, "%d ", &mcts_g.bandwidth[i][j]);
        }
        fscanf(fp_bw,"\n");
    }
    // print processor bandwidth
    // printf("processors performance:");
    // for(i=0;i<global.n_processors;i++){
    //     printf("%d ", mcts_g.processor_performance[i]);
    // }
    // printf("\n");
    // printf("bandwidth:\n");
    // for(i=0;i<global.n_processors;i++){
    //     for(j=0;j<global.n_processors;j++){
    //         printf("%d ", mcts_g.bandwidth[i][j]);
    //     }
    // printf("\n");
    // } 
}

//删除队列中的第n个element
Task delete_n_queue(Queue queue, int n){
    int i=0;
    Element p,prev;
    Task task;
    p = queue->head;
    if(n==0){
        queue->head = p->next;
        queue->n--;
        task=p->task; free(p); 
        return task; 
    }
    while(i<n){
        prev = p; p = p->next;
        i++;
    }
    prev->next = p->next;
    queue->n--;
    task = p->task; free(p);
    return task;
}
void print_queue(Queue queue){
    Element p;
    printf("Queue length:%d\n",queue->n);
    p=queue->head;
    printf("ready queue tasks id:");
    while(p){
        printf("%d ", p->task->tag);
        p=p->next;    
    }
    printf("\n");
}
Task delete_task_queue(Queue queue, int task_id){
    Element p,prev;
    Task task;
    p = queue->head;
    if(p->task->tag == task_id){
        queue->head = p->next;
        queue->n--;
        task = p->task; free(p);
        return task;
    }
    while(p){
        prev = p; p = p->next;
        if(p && p->task->tag == task_id){
            prev->next = p->next;
            queue->n--;
            task = p->task; free(p);
            return task;
        }
    }
    return NULL;
}

//队列头部增加一个任务
void insert_queue(Queue queue, Task task){
    Element element;
    element = (Element)calloc(1,sizeof(struct _Element));
    element->task = task;
    element->next = queue->head;
    queue->head = element;
    queue->n++;
}
void update_queue(Queue queue, Task *task_ready, int nb_tasks){
    int i;
    for(i=0;i<nb_tasks;i++){
        insert_queue(queue,task_ready[i]);
    }
}
Queue init_queue(){
    Queue queue;
    queue = (Queue)calloc(1,sizeof(struct _Queue));
    queue->n = 0;
    queue->head = NULL;
    return queue;
}
void init_children(Node node, Queue queue){
    int i,j,k;
    node->nb_children = queue->n * global.n_processors;
    if(node->nb_children == 0){
        node->children = NULL;
    }
    else{
        node->children = (Node *)calloc(node->nb_children,sizeof(Node));
        j=0;
        for(i=0;i<global.n_processors;i++){
            Element p;
            p=queue->head;
            while(p){
                node->children[j] = (Node)calloc(1,sizeof(struct _Node));
                node->children[j]->node_id = mcts_g.node_id++;
                node->children[j]->task_id = p->task->tag;
                node->children[j]->task = p->task;
                node->children[j]->processor = i;
                node->children[j]->Q = 0;
                node->children[j]->visits = 0;
                node->children[j]->nb_children = 0;
                node->children[j]->children = NULL;
                p=p->next;
                j++;
            }
        }
    }
}
void print_mcts_info(Tree tree, Queue queue){
    Element p;
    p=queue->head;
    while(p){printf("queue element %d\n",p->task->tag);p=p->next;}
    printf("tree %d:\n",tree->root->processor);
}

Task init_root_task(DAG dag){
    Task task;
    int i;
    task = (Task)calloc(1,sizeof(struct _Task));
    task->tag = 0; task->cost = 0; task->data_size = 0;
    task->nb_children = dag->nb_tasks_per_level[0];
    task->nb_parents = 0;
    task->nb_parents_r = 0;
    task->children = (Task *)calloc(task->nb_children,sizeof(Task));
    task->comm_costs = (double *)calloc(task->nb_children,sizeof(double));
    for(i=0;i<dag->nb_tasks_per_level[0];i++){
        task->children[i] = dag->levels[0][i];
        task->comm_costs[i] = 0;
    }
    return task;
}

void init_comm_costs(DAG dag){
    int i,j,k, p_id, c_id;
    for(i=0;i<dag->nb_levels;i++)
        for(j=0;j<dag->nb_tasks_per_level[i];j++){
            for(k=0;k<dag->levels[i][j]->nb_children;k++){
                p_id = dag->levels[i][j]->tag;
                c_id = dag->levels[i][j]->children[k]->tag;
                //printf("init_comm_costs:1 %d %d\n", p_id, c_id);
                //printf("init_comm_costs:2 %d\n", mcts_g.comm_costs[p_id][c_id]);
                mcts_g.comm_costs[p_id][c_id] = dag->levels[i][j]->comm_costs[k];
            }
        }
}

Tree init_tree(DAG dag, Queue queue){ 
    int i;
    Tree tree;
    Task *task_ready;
    Node rootNode;
    
    //printf("init_tree:1\n");
    init_comm_costs(dag);

    tree = (Tree)calloc(1,sizeof(struct _Tree));
    tree->depth = 1;
    tree->root = (Node)calloc(1,sizeof(struct _Node));
    rootNode = tree->root;
    rootNode->node_id = mcts_g.node_id++;
    rootNode->task_id = 0;
    rootNode->task = init_root_task(dag);
    rootNode->processor = 0;
    rootNode->Q = 0;
    rootNode->visits = 0;

    task_ready = (Task *)calloc(dag->nb_tasks_per_level[0],sizeof(Task));
    for(i=0;i<dag->nb_tasks_per_level[0];i++){
        task_ready[i] = dag->levels[0][i];
    }
    //printf("init_tree:2\n");    
    update_queue(queue,task_ready,dag->nb_tasks_per_level[0]);    
    //printf("init_tree:3\n");
    init_children(tree->root,queue);
    mcts_g.nodetrace[mcts_g.nb_node] = rootNode;
    mcts_g.nb_node++;    
    return tree;
}

Node best_child(Node node){
    int i;
    double uct,max,rp;
    Node bestNode;
    max = -9999999;
    rp = rand()/(RAND_MAX+1.0);
    //70%概率选择max uct节点，30%概率随机
    //printf("rp:%lf\n",rp);
    if(rp>0){
        for(i=0;i<node->nb_children;i++){
            int visits = node->children[i]->visits;
            if(visits != 0){
                uct = -node->children[i]->Q / visits
                    - global.cp * sqrt( 2*log(node->visits) / visits);
            }else{
                uct = INFINITY;
            }
            if(uct > max){
                bestNode = node->children[i];
                max = uct;
            }    
        }
    }else{
        i = rand()%node->nb_children;
        //printf("i:%d\n",i);
        bestNode = node->children[i];
    }
    return bestNode;
}

void update_ready_task(DAG dag, Task task, Queue queue){
    int i,nb_ready=0;
    Task *task_ready;
    task_ready = NULL;
    for(i=0;i<task->nb_children;i++){
        task->children[i]->nb_parents_r--;
        if(task->children[i]->nb_parents_r == 0){
            task_ready = (Task *)realloc(task_ready,(nb_ready+1)*sizeof(Task));
            task_ready[nb_ready] = task->children[i];
            nb_ready++;
        }
    }
    if(nb_ready!=0){
        update_queue(queue,task_ready,nb_ready);
    }
}

double get_start_time(Task task, int processor){
    int task_id,i,j,parent_id,parent_proc;
    double finish_time, comm_time, max_time = -1;
    //printf("get_start_time:1\n");
    task_id = task->tag;
    //printf("get_start_time:2\n");
    for(i=0;i<task->nb_parents;i++){
        //printf("get_start_time:3\n");
        parent_id = task->parents[i]->tag;
        //printf("get_start_time:4\n");
        parent_proc = mcts_g.scheduled[parent_id]->processor;
        if(mcts_g.bandwidth[parent_proc][processor] !=0 ){
            comm_time = mcts_g.comm_costs[parent_id][task_id] / mcts_g.bandwidth[parent_proc][processor];
        }else{
            comm_time = 0;
        }
        //printf("get_start_time:5\n");
        finish_time = mcts_g.scheduled[parent_id]->AFT + comm_time;
        //printf("get_start_time:6\n");
        if(finish_time > max_time){
            max_time = finish_time;
        }
    }
    //printf("get_start_time:7\n");
    //没有父节点
    return max_time > mcts_g.availP[processor]? max_time:mcts_g.availP[processor];
}

void update_scheduled_task(Task task, int processor){
    double start_time;
    mcts_g.count++;
    //printf("update_scheduled_task:1\n");    
    start_time = get_start_time(task, processor);
    //printf("update_scheduled_task:1-1\n");
    //printf("update_scheduled_task:2 %lf\n",mcts_g.scheduled[task->tag]->AFT);          
    mcts_g.availP[processor] = start_time + task->comp_costs[processor];  
    mcts_g.scheduled[task->tag]->processor = processor;
    mcts_g.scheduled[task->tag]->AFT = mcts_g.availP[processor];
    //printf("update_scheduled_task:3 %lf\n",mcts_g.scheduled[task->tag]->AFT); 
}

void expand(Node node, Queue queue){
    init_children(node,queue);
}

double get_makespan(){
    int i;
    double aft, max=-1;
    for(i=0;i<global.n_processors;i++){
        aft = mcts_g.availP[i];
        if(aft > max){
            max = aft;
        }
    }
    return max;
}

void backup(){
    int i;
    double makespan;
    makespan = get_makespan();
    for(i=0;i<mcts_g.nb_node;i++){
        mcts_g.nodetrace[i]->Q += makespan;
        mcts_g.nodetrace[i]->visits++;
    }
}
//
void output_subtree(Node node, FILE *fp){
    int p_nid, c_nid, p_tid, c_tid; //注意node_id 和 task_id的区别
    double uct;
    if(node!=NULL){
        fprintf(fp,"%d [label = \"Q=%.2lf\nN=%d\n Q/N=%.2lf\"]\n",
                node->node_id, node->Q, node->visits, node->Q/node->visits);
        p_nid = node->node_id; p_tid = node->task_id;
        for(int i=0;i<node->nb_children;i++){
            c_nid = node->children[i]->node_id; c_tid = node->children[i]->task_id;
            uct = -node->children[i]->Q / node->children[i]->visits
                - CP*sqrt(2*log(node->visits)/node->children[i]->visits);
            fprintf(fp,"%d -> %d ", p_nid, c_nid);
            fprintf(fp,"[label = \"t=%d, p=%d, uct=%.2lf\"]\n", c_tid-1, node->children[i]->processor, uct);
            output_subtree(node->children[i],fp);
        }
    }
}
//保存mcts树
void output_tree(Tree tree){
    FILE *fp;
    Node node;
    int p_id, c_id;
    fp=fopen("tree.dot","w");
    fprintf(fp,"digraph G {");
    node = tree->root;
    if(node!=NULL){
        output_subtree(node,fp);
    }
    fprintf(fp,"}\n");
    fclose(fp);
}

Node tree_policy(DAG dag, Tree tree, Queue queue){
    Node node,child;
    Task task; 
    //printf("tree_policy:1\n");
    node = tree->root;
    while(node->children != NULL){
        //printf("tree_policy:3 %d\n",node->nb_children);
        child = best_child(node);
        //printf("tree_policy:4 %d\n", child->task_id);
        task = delete_task_queue(queue,child->task_id);
        //printf("tree_policy:5 %d\n",task->tag);
        update_scheduled_task(task,child->processor);
        //printf("tree_policy:6\n");
        update_ready_task(dag, task, queue);
        //print_queue(queue);
        mcts_g.nodetrace[mcts_g.nb_node] = child;
        mcts_g.nb_node++;
        node = child;
    }
    return node;
}
int get_min_processor(){
    int i,min_pro;
    double min=INFINITY;
    for(i=0;i<global.n_processors;i++){
        if(min > mcts_g.availP[i]){
            min = mcts_g.availP[i];
            min_pro = i;
        }
    }
    return min_pro;
}
int get_best_processor(Task task){
    int i,bestp=0;
    double est,eft,min=INFINITY;
    for(i=0;i<global.n_processors;i++){
        est = get_start_time(task,i);
        eft = est + task->comp_costs[i];
        if(min > eft){
            min = eft;
            bestp=i;
        }
    }
    return bestp;
}
void default_random_policy(DAG dag, Node leafnode, Queue queue){
    Task task,child;
    int processor;
    double rp;
//    printf("default_random_policy:1\n");
    task = leafnode->task; 
    while(queue->n != 0){ 
        //printf("default_random_policy:5\n");       
        child = delete_n_queue(queue, rand() % queue->n);
        //printf("default_random_policy:6\n");
        rp = rand()/(RAND_MAX+1.0);
        if(rp > 0.2){
        //    update_scheduled_task(child, rand() % global.n_processors);
        update_scheduled_task(child, get_best_processor(child));
        }else{
        //    update_scheduled_task(child, get_min_processor());
        update_scheduled_task(child, rand() % global.n_processors);
        }
        //update_scheduled_task(child, get_best_processor(child));
        //printf("default_random_policy:7\n");
        update_ready_task(dag, child, queue);
        //printf("default_random_policy:8\n");
    }
}
void UCTsearch(DAG dag, Tree tree, Queue queue){
    int i,j,k, nb_simulation=0;
    Node leaf_node;
    Task *task_ready,task;
    double temp_makespan,min_makespan = INFINITY;
    FILE *fp_ms;
    fp_ms=fopen("makespan.txt","w");
    
    while(nb_simulation < global.n_simulations){
        //printf("UCT search 1\n");
        leaf_node = tree_policy(dag, tree, queue);
        if(leaf_node->visits == 0){
            //printf("UCT search 2\n");
            //print_queue(queue);
            default_random_policy(dag,leaf_node,queue);
        }else{
            //printf("===========UCT search 3\n");
            expand(leaf_node,queue);
            //printf("uct 5: %d\n", leaf_node->nb_children);
            if(leaf_node->nb_children!=0){
                task = delete_task_queue(queue,leaf_node->children[0]->task_id);
                //printf("tree_policy:5 %d\n",task->tag);
                update_scheduled_task(task,leaf_node->children[0]->processor);
                //printf("tree_policy:6\n");
                update_ready_task(dag, task, queue);
                mcts_g.nodetrace[mcts_g.nb_node] = leaf_node->children[0];
                mcts_g.nb_node++;
                //printf("uct 6\n");
                default_random_policy(dag,leaf_node->children[0],queue);
            }else{
                default_random_policy(dag,leaf_node,queue);    
            }           
        }
        //printf("UCT search 4\n");
        backup();
        temp_makespan = get_makespan();
        fprintf(fp_ms,"%lf %lf\n",temp_makespan,mcts_g.nodetrace[0]->Q/mcts_g.nodetrace[0]->visits);
        if(min_makespan > temp_makespan)
            min_makespan = temp_makespan;
        
        //恢复nb_parents_r
        k=0;
        for (i=0; i<dag->nb_levels;i++)
            for (j=0; j<dag->nb_tasks_per_level[i]; j++){
                dag->levels[i][j]->nb_parents_r=global.nb_parents[k];
                k++;
            }
        for(i=0;i<global.n_processors;i++){
            //printf("%lf ",mcts_g.availP[i]);
            mcts_g.availP[i] = 0;
        }
        //printf("\n");
        //rootnode仍保存在nodetrace[0]中，nodetrace从1开始记录
        mcts_g.nb_node = 1;
        //初始化就绪任务队列
        task_ready = (Task *)calloc(dag->nb_tasks_per_level[0],sizeof(Task));
        for(i=0;i<dag->nb_tasks_per_level[0];i++){
            task_ready[i] = dag->levels[0][i];
        }
        update_queue(queue,task_ready,dag->nb_tasks_per_level[0]);

        for(i=0;i<global.n+1;i++){
            mcts_g.scheduled[i]->processor = 0;
            mcts_g.scheduled[i]->AFT = 0;
        }

        nb_simulation++;

        //printf("***%d\n",nb_simulation);
    }
    output_tree(tree);
    printf("MCTS makespan:%lf\n",min_makespan);
    printf("count:%d\n",mcts_g.count);
    fclose(fp_ms);
}

