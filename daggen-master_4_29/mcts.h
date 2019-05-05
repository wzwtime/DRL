#ifndef MCTS_H_
#define MCTS_H_

#define MINP 50
#define MAXP 300
#define MAXtask 500

typedef struct _Node *Node;
typedef struct _Tree *Tree;
typedef struct _Queue *Queue;
typedef struct _Element *Element;
typedef struct _scheduled *Scheduled;

typedef struct{
    int *processor_performance;
    int **bandwidth;
    double **comm_costs;
    double makespan;
    int node_id;
    int nb_node;
    Node *nodetrace;
    Scheduled *scheduled;
    double *availP;
    int count;
//    int **comp_costs;
//    int **comm_costs;
} MCTS_g;
MCTS_g mcts_g;

struct _scheduled{
    int processor;
    double AFT;
};

struct _Element{
    Task task;
    Element next;
};

struct _Queue{
    int n;
    Element head;
};

struct _Node{
    int node_id;
    int task_id;
    Task task;
    int processor;
    double Q;
    int visits;
    int nb_children;
    Node *children;    
};

struct _Tree{
    int depth;
    Node root;
};

void create_root_node(DAG dag, Tree tree, Queue queue);
void assign_processor_bandwidth();
void UCTsearch(DAG dag, Tree tree, Queue queue);
Tree init_tree(DAG dag, Queue queue);
Queue init_queue();
void insert_queue(Queue queue, Task task);
void init_processor();
void init_bandwidth();
void update_queue(Queue queue, Task *task_ready, int nb_tasks);
Task delete_task_queue(Queue queue, int task_id);
void update_ready_task(DAG dag, Task task, Queue queue);

#endif /*MCTS_H_*/