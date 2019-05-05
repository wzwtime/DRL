/******************************************************************************
 * Copyright (c) 2007-2013. F. Suter, S. Hunold.
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the license (GNU LGPL 2.1) which comes with this package.
 *****************************************************************************/

#ifndef DAGGEN_COMMONS_H_
#define DAGGEN_COMMONS_H_

#ifndef MAX
#  define MAX(x,y) ((x) > (y) ? (x) : (y))
#endif
#ifndef MIN
#  define MIN(x,y) ((x) < (y) ? (x) : (y))
#endif

typedef enum {
  MIXED =0,
  N_2,
  N_LOG_N, /* (n2 log(n2) indeed */
  N_3
} complexity_t;


#define OUTPUT (global.output_file)
#define GIGA 1024*1024*1024

/*********************************/
/** Global variables            **/
/*********************************/

typedef struct {
  int n;          /* number of tasks in the graph       */
  int n_processors; /* number of processors in the system*/
  int n_simulations;
  int cp;
  double fat;     /* fatness parameter                    */
  double regular; /* regularity                         */
  double ccr;     /* Communication to computation ratio */
  double density;
  double mindata, maxdata;
  double minalpha, maxalpha; /* Amdahl's law parameter */
  int jump;
  int dot_output;
  int *nb_parents;
  FILE *output_file;
  int capacity;
  int episode;
} Global;
Global global;

typedef struct _Task *Task;
typedef struct _DAG *DAG;

struct _Task {
  int tag;                //任务编号
  double cost;            //任务的基准计算量，生成comp_costs时的基准
  double *comp_costs;    //在每个处理器上的计算开销  
  int data_size;         //生成comm_costs数据量时的参考基准
  double alpha;
  int nb_children;
  int nb_parents;
  int nb_parents_r;      //用于判断任务是否就绪，等于0时父亲任务全部调度，该任务就绪。
  Task *children;
  Task *parents;
  double *comm_costs;    //传给每个孩子节点的数据量（命名欠妥），实际数据通信开销调度时确定
  int *transfer_tags;    //边编号
  complexity_t complexity;
};

struct _DAG {
  int nb_levels;
  int *nb_tasks_per_level;
  Task **levels;
};


int parseOptions(int argc, char *const *argv);

void printUsage(void);

void outputDAG(DAG dag);
void outputDOT(DAG dag);
void print_dag_info(DAG dag);

int getIntRandomNumberAround(int x, double perc);

double getRandomNumberBetween(double x, double y);

void generate_randomID(int min, int max, int n, int a[]);

void IDtoIJ(int id, int *ij, DAG dag);

int get_nb_tasks(DAG dag, int n_level);

// void saveDAG(DAG dag);

// DAG readDAG();

#endif /*DAGGEN_COMMONS_H_*/
