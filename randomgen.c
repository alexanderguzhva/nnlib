#include <math.h>

double myrandom(int * randseed)
{
	(*randseed) = (0x08088405 * (*randseed) + 1) & 0x7fffffff;
    return (((*randseed) / 2147483648.) * 1. - 0.);
}


int myintrandom(int * randseed, int themax)
{
	(*randseed) = (0x08088405 * (*randseed) + 1) & 0x7fffffff;
	return ((*randseed) % themax);
}
