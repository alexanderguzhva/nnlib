#include "matrix_nn.h"

#include <math.h>

#include "cache.h"


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_VectorProcess_Sigmoid(zVectorType * zSourceVector, zVectorType * zDestinationVector)
{

    //non-optimized
    zInt i, j;

#ifdef SIGMOID_CACHE
    zMatrixElementType q0;
#endif

    if (zSourceVector->dim != zDestinationVector->dim)
    {
        return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;
    }

#ifndef SIGMOID_CACHE
    for (i = 0; i < zSourceVector->dim; i++)
    {
        zDestinationVector->data[i] = 1/(1+exp(-zSourceVector->data[i]));
    }
#else
    for (i = 0; i < zSourceVector->dim; i++)
    {
        q0 = zSourceVector->data[i];
        if (fabs(q0) > SIGMOID_RANGE)
        {
            zDestinationVector->data[i] = 1/(1+exp(-zSourceVector->data[i]));
        }
        else
        {
            j = (int) (q0 * SIGMOID_N / SIGMOID_RANGE + SIGMOID_N);
            zDestinationVector->data[i] = sigmoidCache[j];
        }
    }
#endif

    return ZERROR_NO_ERROR;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_VectorProcess_SigmoidD(zVectorType * zSourceVector, zVectorType * zDestinationVector)
{
    zInt i;

    if (zSourceVector->dim != zDestinationVector->dim)
    {
        return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;
    }

    for (i = 0; i < zSourceVector->dim; i++)
    {
        zDestinationVector->data[i] = zSourceVector->data[i] * (1 - zSourceVector->data[i]);
    }

    return ZERROR_NO_ERROR;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_VectorProcess_Tanh(zVectorType * zSourceVector, zVectorType * zDestinationVector)
{
    zInt i;
    zDatatype q;

#ifdef TANH_CACHE
    zMatrixElementType q0;
    zInt j;
#endif

    if (zSourceVector->dim != zDestinationVector->dim)
    {
        return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;
    }

#ifndef TANH_CACHE
    for (i = 0; i < zSourceVector->dim; i++)
    {
        q = exp(-2*zSourceVector->data[i]);
        zDestinationVector->data[i] = 2 / (1 + q) - 1;
    }
#else
    for (i = 0; i < zSourceVector->dim; i++)
    {
        q0 = zSourceVector->data[i];
        if (fabs(q0) > TANH_RANGE)
        {
            q = exp(-2*q0);
            zDestinationVector->data[i] = 2 / (1 + q) - 1;
        }
        else
        {
            j = (int) (q0 * TANH_N / TANH_RANGE + TANH_N);
            zDestinationVector->data[i] = tanhCache[j];
        }
    }
#endif

    return ZERROR_NO_ERROR;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_VectorProcess_TanhD(zVectorType * zSourceVector, zVectorType * zDestinationVector)
{
    zInt i;
    if (zSourceVector->dim != zDestinationVector->dim)
    {
        return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;
    }

    for (i = 0; i < zSourceVector->dim; i++)
    {
        zDestinationVector->data[i] = 1. - zSourceVector->data[i] * zSourceVector->data[i];
    }

    return ZERROR_NO_ERROR;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_VectorProcess_LinearD(zVectorType * zSourceVector, zVectorType * zDestinationVector)
{
    zInt i;

    if (zSourceVector->dim != zDestinationVector->dim)
    {
        return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;
    }

    for (i = 0; i < zSourceVector->dim; i++)
    {
        zDestinationVector->data[i] = 1.;
    }

    return ZERROR_NO_ERROR;
}

zFunctionResult Mx_VectorProcess_Linear(zVectorType * zSourceVector, zVectorType * zDestinationVector)
{
    zInt i;
    if (zSourceVector->dim != zDestinationVector->dim)
    {
        return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;
    }


    for (i = 0; i < zSourceVector->dim; i++)
    {
        zDestinationVector->data[i] = zSourceVector->data[i];
    }

    return ZERROR_NO_ERROR;

}
