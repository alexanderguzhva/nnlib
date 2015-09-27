#include "cache.h"
#include <math.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef SIGMOID_CACHE
void Mx_InitializeSigmoidCache(void)
{
	int i;

    for (i = 0; i < 2 * SIGMOID_N + 1; i++)
	{
        sigmoidCache[i] = SIGMOID_RANGE * (2 * (i / (double)(2 * SIGMOID_N)) - 1);
		sigmoidCache[i] = 1 / (1 + exp(-sigmoidCache[i]));
    }
}
#endif


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef TANH_CACHE
void Mx_InitializeTanhCache(void)
{
	int i;

    for (i = 0; i < 2 * TANH_N + 1; i++)
	{
        tanhCache[i] = TANH_RANGE * (2 * (i / (double)(2 * TANH_N)) - 1);
		tanhCache[i] = exp(-2 * tanhCache[i]);
		tanhCache[i] = 2 / (1 + tanhCache[i]) - 1;
    }
}
#endif
