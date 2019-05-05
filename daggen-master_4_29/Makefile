all: daggen

daggen_OBJECTS = daggen.o daggen_commons.o mcts.o heft.o environment.o
# Tool invocations
daggen: $(daggen_OBJECTS) 
	gcc $(CFLAGS) -o daggen $(daggen_OBJECTS) -lm

daggen.o:daggen.c
	gcc -c daggen.c
daggen_commons.o:daggen_commons.c
	gcc -c daggen_commons.c
mcts.o:mcts.c
	gcc -c mcts.c daggen_commons.c
heft.o:heft.c
	gcc -c heft.c mcts.c daggen_commons.c
environment.o:environment.c
	gcc -c environment.c heft.c mcts.c daggen_commons.c
.PHONY:clean
	rm *.o