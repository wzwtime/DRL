/******************************************************************************
 * Copyright (c) 2007-2013. F. Suter, S. Hunold.
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the license (GNU LGPL 2.1) which comes with this package.
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>

#include "daggen_commons.h"

/*********************************/
/** Command line option parsing **/
/*********************************/

/*
 * Parse the options and set default values
 *
 * returns -1 for usage
 * returns -2 for no usage
 * returns 0 when Ok
 */

int parseOptions(int argc, char *const *argv) {
  int ret_val = 0;
  int c;

  int oflag = 0;
  int nflag = 0;
  int pflag = 0;
  int sflag = 0;
  int cflag = 0;
  int rflag = 0;
  int eflag = 0;
  int fat_flag = 0;
  int density_flag = 0;
  int ccr_flag = 0;
  int mindata_flag = 0;
  int maxdata_flag = 0;
  int minalpha_flag = 0;
  int maxalpha_flag = 0;
  int regular_flag = 0;
  int jump_flag = 0;
  int dot_flag = 0;
  FILE *fp;
  fp = fopen("digraph.dot","w");
  global.output_file=fp;
  global.jump=1;
  global.mindata=2048;
  global.maxdata=11264;
  global.minalpha=0.0;
  global.maxalpha=0.2;
  global.ccr=0.2;
  global.density=0.5;
  global.fat=0.5;
  global.regular=0.9;
  global.n=100;
  global.n_processors=4;
  global.n_simulations=100;
  global.cp=-10;
  global.dot_output=0;
  global.capacity = 30;
  global.episode = 3;

  while (ret_val == 0) {

    static struct option long_options[] = {
        { "fat", 1, 0, 0 }, /* double  */
        { "density", 1, 0, 0 }, /* double */
        { "ccr", 1, 0, 0 }, /* int   */
        { "mindata", 1, 0, 0 }, /* int     */
        { "maxdata", 1, 0, 0 }, /* int     */
        { "minalpha", 1, 0, 0 }, /* double   */
        { "maxalpha", 1, 0, 0 }, /* double   */
        { "regular", 1, 0, 0 }, /* double   */
        { "jump", 1, 0, 0 }, /* int   */
        { "dot", 0, 0, 0},
        { 0, 0, 0, 0 } };

    /* getopt_long stores the option index here. */
    int option_index = 0;

    /*
		 parameters
		 h help
		 o string output
		 n int nodes
     p int processors
     s int simulation times
     c int cp value
     r int capacity
     e int episode
     */

    c = getopt_long(argc, argv, "ho:n:p:s:c:r:e:", long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c) {
    case 0:
      /* If this option set a flag, do nothing else now. */
      if (long_options[option_index].flag != 0)
        break;

      // we do nothing right now with the long opts
      const char *optname = long_options[option_index].name;

      if (!strcmp(optname, "fat")) {
        double fat=atof(optarg);

        fat_flag = 1;

        if ((fat < 0.0) || (fat > 1.0)) {
          fprintf(stderr,"Error: invalid fat value: %f\n",fat);
          ret_val = -1;
        }
        global.fat=fat;

      } else if (!strcmp(optname, "density")) {
        double density=atof(optarg);

        density_flag = 1;

        if ((density < 0.0) || (density > 1.0)) {
          fprintf(stderr,"Error: invalid density value: %f\n",density);
          ret_val = -1;
        }
        global.density=density;

      } else if (!strcmp(optname, "ccr")) {

        int ccr=atoi(optarg);
        ccr_flag = 1;

        if (ccr < 0) {
          fprintf(stderr,"Error: invalid ccr value: %d\n",ccr);
          ret_val = -1;
        }
        global.ccr=ccr;

      } else if (!strcmp(optname, "mindata")) {

        int mindata=atoi(optarg);

        mindata_flag = 1;

        if (mindata <= 0) {
          fprintf(stderr,"Error: invalid mindata value: %d\n",mindata);
          ret_val = -1;
        }
        global.mindata=mindata;

      } else if (!strcmp(optname, "maxdata")) {

        int maxdata=atoi(optarg);

        maxdata_flag = 1;

        if ((maxdata <= 0) || (maxdata < global.mindata)) {
          fprintf(stderr,"Error: invalid maxdata value: %d\n",maxdata);
          ret_val = -1;
        }
        global.maxdata=maxdata;

      } else if (!strcmp(optname, "minalpha")) {
        double minalpha=atof(optarg);

        minalpha_flag = 1;

        if (minalpha < 0.0) {
          fprintf(stderr,"Error: invalid minalpha value: %f\n",minalpha);
          ret_val = -1;
        }
        global.minalpha=minalpha;

      } else if (!strcmp(optname, "maxalpha")) {
        double maxalpha=atof(optarg);

        maxalpha_flag = 1;

        if ((maxalpha < 0.0) || (maxalpha < global.minalpha)) {
          fprintf(stderr,"Error: invalid maxalpha value: %f\n",maxalpha);
          ret_val = -1;
        }
        global.maxalpha=maxalpha;

      } else if (!strcmp(optname, "regular")) {

        double regular=atof(optarg);

        regular_flag = 1;

        if ((regular < 0.0) || (regular > 1.0)) {
          fprintf(stderr,"Error: invalid regular value: %f\n",regular);
          ret_val = -1;
        }
        global.regular=regular;

      } else if (!strcmp(optname, "jump")) {
        int jump = atoi(optarg);

        jump_flag = 1;

        if (jump < 0) {
          fprintf(stderr,"Error: invalud jump value: %d\n",jump);
          ret_val = -1;
        }
        global.jump=jump;

      } else if (!strcmp(optname, "dot")) {

        dot_flag = 1;

        global.dot_output=1;
      } else {
        fprintf(stderr, "unknown switch: %s\n", optname);
      }
      break;

    case 'n': {
      int n=atoi(optarg);
      nflag = 1;

      if (n <= 0) {
        fprintf(stderr,"Error: invalid n value: %d\n", n);
        ret_val = -1;
      }
      global.n=n;
    }
    break;
    case 'p': {
      int np=atoi(optarg);
      pflag = 1;
      
      if (np <= 0) {
        fprintf(stderr,"Error: invalid np value: %d\n", np);
        ret_val = -1;
      }
      global.n_processors=np;
    }
    break;
    
    case 's': {
      int sp=atoi(optarg);
      sflag = 1;
      
      if (sp <= 0) {
        fprintf(stderr,"Error: invalid sp value: %d\n", sp);
        ret_val = -1;
      }
      global.n_simulations=sp;
    }
    break;
    case 'r': {
      int ca=atoi(optarg);
      rflag = 1;
      
      if (ca <= 0) {
        fprintf(stderr,"Error: invalid ca value: %d\n", ca);
        ret_val = -1;
      }
      global.capacity=ca;
    }
    break;
    case 'e': {
      int ep=atoi(optarg);
      eflag = 1;
      
      if (ep <= 0) {
        fprintf(stderr,"Error: invalid ep value: %d\n",ep);
        ret_val = -1;
      }
      global.episode = ep;
    }
    break;
    case 'c': {
      int cp=atoi(optarg);
      cflag = 1;
      global.cp=cp;
    }
    break;
    case 'o':
      oflag = 1;
      const char *filename = optarg;
      if ((global.output_file=fopen(filename, "w")) == NULL) {
        fprintf(stderr,"Error: Cannot open file '%s' for output\n", filename);
        ret_val = -1;
      }
      break;
    case 'h':
      ret_val = -1; // show usage
      break;
    case '?':
      /* getopt_long already printed an error message. */
      break;

    default:
      abort();
      break;
    }
  }

  if( ret_val == 0 ) {

    if (!nflag) {
      fprintf(stderr,"Warning: using default n value (%d)\n", global.n);
    }
    if (!pflag) {
      fprintf(stderr,"Warning: using default np value (%d)\n", global.n_processors);
    }
    if (!sflag) {
      fprintf(stderr,"Warning: using default sp value (%d)\n", global.n_simulations);
    }
    if (!rflag) {
      fprintf(stderr,"Warning: using default ca value (%d)\n", global.capacity);
    }
    if (!eflag) {
      fprintf(stderr,"Warning: using default ep value (%d)\n", global.episode);
    }
    if (!cflag) {
      fprintf(stderr,"Warning: using default cp value (%d)\n", global.cp);
    }
    if (!oflag) {
      fprintf(stderr,"Warning: Sending output to stdout\n");
    }
    if (!jump_flag) {
      fprintf(stderr,"Warning: using default jump value (%d)\n", global.jump);
    }
    if (!fat_flag) {
      fprintf(stderr,"Warning: using default fat value (%g)\n", global.fat);
    }
    if (!density_flag) {
      fprintf(stderr,"Warning: using default density value (%g)\n",
          global.density);
    }
    if (!ccr_flag) {
      fprintf(stderr,"Warning: using default ccr value (%lf)\n", global.ccr);
    }
    if (!mindata_flag) {
      fprintf(stderr,"Warning: using default mindata value (%g)\n",
          global.mindata);
    }
    if (!maxdata_flag) {
      fprintf(stderr,"Warning: using default maxdata value (%g)\n",
          global.maxdata);
    }
    if (!minalpha_flag) {
      fprintf(stderr,"Warning: using default minalpha value (%g)\n",
          global.minalpha);
    }
    if (!maxalpha_flag) {
      fprintf(stderr,"Warning: using default maxalpha value (%g)\n",
          global.maxalpha);
    }
    if (!regular_flag) {
      fprintf(stderr,"Warning: using default regular value (%g)\n",
          global.regular);
    }

  }
  global.nb_parents = (int *)calloc(global.n,sizeof(int));
  return ret_val;
}


/*
 * printUsage()
 */
void printUsage(void)
{
  fprintf(stderr,"daggen [options] [-o <output file>]\n"
      "\t -n <number of tasks>\n"
      "\t --mindata <minimum data size>\n"
      "\t --maxdata <maximum data size>\n"
      "\t --minalpha <minimum Amdahl's law parameter value>\n"
      "\t --maxalpha <maximum Amdahl's law parameter value>\n");
  fprintf(stderr,
      "\t --fat <dag shape>\n"
      "\t	 fat=1.0: fat (maximum parallelism)\n"
      "\t	 fat=0.0: thin (minimum parallelism)\n"
      "\t --density <density>\n"
      "\t    density=0.0: minimum number of dependencies\n"
      "\t    density=1.0: full graph\n"
      "\t --regular <regularity for num tasks per level>\n"
      "\t    regular= 1.0: perfectly regular\n"
      "\t    regular= 0.0: irregular\n"
      "\t --ccr <communication(MBytes) to computation(sec) ratio>\n"
      "\t --jump <number of levels spanned by communications>\n"
      "\t    jump=1: perfectly synchronized levels\n"
      "\t --dot: output generated DAG in the DOT format\n"
      "\n");
  return;
}


/****************/
/** DAG output **/
/****************/

void outputDAG(DAG dag) {
  int i, j, k;
  /* starting at 1 for the root node */
  int node_count=1;

  /* count and tag the nodes */
  for (i=0; i<dag->nb_levels; i++) {
    for (j=0; j<dag->nb_tasks_per_level[i]; j++) {
      dag->levels[i][j]->tag = node_count++;
    }
    for (j=0; j<dag->nb_tasks_per_level[i]; j++) {
      for (k=0; k<dag->levels[i][j]->nb_children; k++) {
        dag->levels[i][j]->transfer_tags[k] = node_count++;
      }
    }
  }
  /* accounting for the END node */
  fprintf(OUTPUT,"NODE_COUNT %d\n",node_count+1);

  /* Create the root node */
  fprintf(OUTPUT,"NODE 0 ");
  for (i=0; i<dag->nb_tasks_per_level[0]-1; i++) {
    fprintf(OUTPUT,"%d,",dag->levels[0][i]->tag);
  }
  if (dag->nb_tasks_per_level[0])
    fprintf(OUTPUT,"%d ROOT 0.0 0.0\n",dag->levels[0][i]->tag);
  else
    fprintf(OUTPUT,"%d ROOT 0.0 0.0\n",node_count);

  /* Creating the regular nodes until next to last level */
  for (i=0; i<dag->nb_levels-1; i++) {
    for (j=0; j<dag->nb_tasks_per_level[i]; j++) {
      /* do the COMPUTATION */
      fprintf(OUTPUT,"NODE %d ",dag->levels[i][j]->tag);
      for (k=0; k<dag->levels[i][j]->nb_children-1; k++) {
        fprintf(OUTPUT,"%d,",dag->levels[i][j]->transfer_tags[k]);
      }
      if (dag->levels[i][j]->nb_children) {
        fprintf(OUTPUT,"%d COMPUTATION %.0f %.2f\n",
            dag->levels[i][j]->transfer_tags[k],
            dag->levels[i][j]->cost,
            dag->levels[i][j]->alpha);
      } else {
        fprintf(OUTPUT,"%d COMPUTATION %.0f %.2f\n",
            node_count,
            dag->levels[i][j]->cost,
            dag->levels[i][j]->alpha);
      }
      /* do the TRANSFER */
      for (k=0; k<dag->levels[i][j]->nb_children; k++) {
        fprintf(OUTPUT,"NODE %d ",dag->levels[i][j]->transfer_tags[k]);
        fprintf(OUTPUT,"%d TRANSFER %.0f 0.0\n",
            dag->levels[i][j]->children[k]->tag,
            dag->levels[i][j]->comm_costs[k]);
      }
    }
  }

  /* Do the last level */
  for (j=0; j<dag->nb_tasks_per_level[dag->nb_levels-1]; j++) {
    fprintf(OUTPUT,"NODE %d %d COMPUTATION %.0f %.2f\n",
        dag->levels[dag->nb_levels-1][j]->tag,
        node_count,
        dag->levels[dag->nb_levels-1][j]->cost,
        dag->levels[dag->nb_levels-1][j]->alpha);
  }

  /* Do the end node */
  fprintf(OUTPUT,"NODE %d - END 0.0 0.0\n",node_count);
}

void outputDOT(DAG dag) {
  int i, j, k;
  /* starting at 1 for the root node */
  int node_count=1;

  /* count and tag the nodes */
  for (i=0; i<dag->nb_levels; i++) {
    for (j=0; j<dag->nb_tasks_per_level[i]; j++) {
      dag->levels[i][j]->tag = node_count++;
    }
//    for (j=0; j<dag->nb_tasks_per_level[i]; j++) {
//      for (k=0; k<dag->levels[i][j]->nb_children; k++) {
//        dag->levels[i][j]->transfer_tags[k] = node_count++;
//      }
//    }
  }
  /* accounting for the END node */
  fprintf(OUTPUT,"digraph G {\n");

  /* Create the root node */
  fprintf(OUTPUT,"  0 [label=root,size=\"0\",alpha=\"0\"]\n");
  for (i=0; i<dag->nb_tasks_per_level[0]; i++) {
    fprintf(OUTPUT,"  0 -> %d [label = \"0\"]\n",dag->levels[0][i]->tag);
  }  

  /* Creating the regular nodes until next to last level */
  for (i=0; i<dag->nb_levels; i++) {
    for (j=0; j<dag->nb_tasks_per_level[i]; j++) {
      /* do the COMPUTATION */
      fprintf(OUTPUT,"  %d [label= \"%d C=%.3lf\",size=\"%.0f\", alpha=\"%.2f\"]\n",
          dag->levels[i][j]->tag,
          dag->levels[i][j]->tag-1,
          dag->levels[i][j]->cost,
          dag->levels[i][j]->cost,
          dag->levels[i][j]->alpha);

      /* do the TRANSFER */
      for (k=0; k<dag->levels[i][j]->nb_children; k++) {
        fprintf(OUTPUT,"  %d -> %d [label =\"%.f\"]\n",
            dag->levels[i][j]->tag,
            dag->levels[i][j]->children[k]->tag,
            dag->levels[i][j]->comm_costs[k]);
      }
    }
  }

//  /* Do the last level */
//  for (j=0; j<dag->nb_tasks_per_level[dag->nb_levels-1]; j++) {
//    fprintf(OUTPUT,"  %d -> %d [label = \"0\"]\n",
//        dag->levels[dag->nb_levels-1][j]->tag,
//        node_count);
//  }

  for (i=0; i < dag->nb_levels;i++)
    for (j=0; j< dag->nb_tasks_per_level[i]; j++){
      if (dag->levels[i][j]->nb_children == 0){
      //   fprintf(OUTPUT, "0 child：%d\n",dag->levels[i][j]->tag);
         fprintf(OUTPUT, "  %d -> %d [label = \"0\"]\n",
         dag->levels[i][j]->tag, node_count);
      }
    }
  fprintf(OUTPUT,"}\n");
  fclose(OUTPUT);
}

void print_dag_info(DAG dag){
  int i,j,k;

  printf("task costs:\n");
  for(i=0;i<dag->nb_levels;i++){
    for(j=0;j<dag->nb_tasks_per_level[i];j++){
      printf("%.2f ", dag->levels[i][j]->cost);
    }
    printf("\n");
  }
  printf("task comp_costs:\n");
  for(i=0;i<dag->nb_levels;i++)
    for(j=0;j<dag->nb_tasks_per_level[i];j++){
      for(k=0;k<global.n_processors;k++){
        printf("%.2f ", dag->levels[i][j]->comp_costs[k]);
      }
    printf("\n");
  }
  printf("task datasize:\n");
  for(i=0;i<dag->nb_levels;i++){
    for(j=0;j<dag->nb_tasks_per_level[i];j++){
      printf("%d ", dag->levels[i][j]->data_size);
    }
    printf("\n");
  }

  printf("task comm_costs:\n");
  for(i=0;i<dag->nb_levels;i++){
    for(j=0;j<dag->nb_tasks_per_level[i];j++){
      for(k=0;k<dag->levels[i][j]->nb_children;k++){
        printf("%.2f ", dag->levels[i][j]->comm_costs[k]);
      }
    printf("\n");
    }
  }
}

/***********************/
/** Random generators **/
/***********************/

/*
 * getIntRandomNumberAround()
 *
 * returns a STRICTLY POSITIVIE int around x, by perc percents.
 */
int getIntRandomNumberAround(int x, double perc) {
  double r;
  int new_int;

  r = -perc + (2*perc*rand()/(RAND_MAX+1.0));
  new_int = MAX(1,(int)((double)x * (1.0 + r/100.00)));
  return new_int;
}

/*
 * getRandomNumberBetween()
 *
 */
double getRandomNumberBetween(double x, double y) {
  double r;

  r = x + (y-x)*rand()/(RAND_MAX+1.0);
  return r;
}

// generate n number of non-repetitive random IDs between min and max
void generate_randomID(int min, int max, int n, int a[]){
	
	srand(time(NULL));
	
	int RandNum,i,j,flag=0,t=0;

	while(1){
		flag = 0;
		if(t == n)
			break;
		RandNum = ( rand() % (max-min) ) + min;
		for(i=0;i<t;i++){
			if(a[i]==RandNum)
				flag=1;
		}
		if(flag != 1){
			a[t]=RandNum;
			//printf("%d ",a[t]);
			t++;
		}
	}
  //printf("\n");
}
// given node ID, get level i, index j 
void IDtoIJ(int id, int *ij, DAG dag){
  int i,j;
  for(i=0; i < dag->nb_levels; i++){
     if (id < dag->nb_tasks_per_level[i]){
       break;
     }
     else{
       id = id - dag->nb_tasks_per_level[i];
     }  
  }
  ij[0]=i;
  ij[1]=id; 
//  printf("ij[]:%d,%d\n",ij[0],ij[1]);
}

// void saveDAG(DAG dag){
//   FILE *fp,*fp1;
//   int g_size;
//   fp = fopen("dag_struct.ds","w");
//   fp1 = fopen("dag_size.txt","w");
//   g_size = sizeof(dag);
//   fwrite(&g_size,sizeof(int),1,fp1);
//   fwrite(dag, sizeof(dag),1,fp);
//   fclose(fp);
// }

// DAG readDAG(){
//   FILE *fp,*fp1;
//   DAG *dag;
//   int g_size;
//   fp1 = fopen("dag_size.txt","r");
//   fp = fopen("dag_struct.ds","r");
//   fread(&g_size,sizeof(int),1,fp1);
//   printf("g_size:%d\n",g_size);
//   dag = (DAG)calloc(1,sizeof(g_size));
//   fread(dag, g_size, 1, fp);
//   return dag;
// }

// 获得第n_level层之上的任务数量
int get_nb_tasks(DAG dag, int n_level){
    int i;
    int n_tasks=0;
    for(i=0; i<n_level; i++){
       n_tasks += dag->nb_tasks_per_level[i];
    }
    return n_tasks;
}

void set_computation_cost(DAG dag, int n_proc, int W_dag, double beta){
     int i, j;
     int w_i;
     srand(time(NULL));
     for(i=0; i < dag->nb_levels; i++)
        for(j=0; j < dag->nb_tasks_per_level[i]; j++){
            
        }
}