#ifndef _MATRIX_NN_H_
#define _MATRIX_NN_H_

#include "matrix.h"

zFunctionResult Mx_VectorProcess_Sigmoid(zVectorType * zSourceVector, zVectorType * zDestinationVector);
zFunctionResult Mx_VectorProcess_SigmoidD(zVectorType * zSourceVector, zVectorType * zDestinationVector);
zFunctionResult Mx_VectorProcess_Linear(zVectorType * zSourceVector, zVectorType * zDestinationVector);
zFunctionResult Mx_VectorProcess_LinearD(zVectorType * zSourceVector, zVectorType * zDestinationVector);
zFunctionResult Mx_VectorProcess_Tanh(zVectorType * zSourceVector, zVectorType * zDestinationVector);
zFunctionResult Mx_VectorProcess_TanhD(zVectorType * zSourceVector, zVectorType * zDestinationVector);

#endif
