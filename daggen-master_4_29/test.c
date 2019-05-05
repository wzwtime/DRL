#include<stdio.h>

#include<stdio.h>
void calculate(double a,double b, double *add, double *multiple);
int main()
{
	double f,g,add,multiple;
		scanf("%lf",&f);
		scanf("%lf",&g);
		calculate(f,g,&add,&multiple);
		printf("add=%g,multiple=%g\n",add,multiple);
         return 0;
}
void calculate(double a,double b,double *add, double *multiple)
{
		*add=a+b;
		*multiple=a*b;

}
