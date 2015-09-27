#ifndef _CACHE_H_
#define _CACHE_H

#include "matrix.h"




#define SIGMOID_CACHE

#ifdef SIGMOID_CACHE
#define SIGMOID_N 10240
#define SIGMOID_RANGE 5.0
zMatrixElementType sigmoidCache[2 * SIGMOID_N + 1];
void Mx_InitializeSigmoidCache(void);
#endif

#define TANH_CACHE
#ifdef TANH_CACHE
#define TANH_N 10240
#define TANH_RANGE 5.0
zMatrixElementType tanhCache[2 * TANH_N + 1];
void Mx_InitializeTanhCache(void);
#endif



#endif
