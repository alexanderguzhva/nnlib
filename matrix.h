#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "errors.h"


#define MATRIX_MAX_DIMENSION 40000
#define VECTOR_MAX_DIMENSION 40000

typedef zDatatype zMatrixElementType;

typedef struct 
{
	zMatrixElementType ** data;
	zInt dimX;
	zInt dimY;
} zMatrixType;

zFunctionResult Mx_CreateMatrix(zMatrixType * zMatrix, zInt dimX, zInt dimY);
zFunctionResult Mx_ClearMatrix(zMatrixType * zMatrix, zMatrixElementType value);
zFunctionResult Mx_AddMatrix(zMatrixType * zMatrix1, zMatrixType * zMatrix2, zMatrixType * zMatrixOut);
zFunctionResult Mx_MultiplyMatrix(zMatrixType * zMatrix1, zMatrixType * zMatrix2, zMatrixType * zMatrixOut);
zFunctionResult Mx_FreeMatrix(zMatrixType * zMatrix);
zFunctionResult Mx_UpdateMatrix(zMatrixType * zMatrix, zInt dimX, zInt dimY);
zFunctionResult Mx_CopyMatrix(zMatrixType * zMatrix, zMatrixType * zOutMatrix);
zFunctionResult Mx_AddToMatrixWithAlpha(zMatrixType * zMatrix1, zMatrixType * zMatrix2, zMatrixElementType alpha);

typedef struct
{
	zMatrixElementType * data;
	zInt dim;
} zVectorType;

zFunctionResult Mx_CreateVector(zVectorType * zVector, zInt dim);
zFunctionResult Mx_FreeVector(zVectorType * zVector);
zFunctionResult Mx_UpdateVector(zVectorType * zVector, zInt dim);
zFunctionResult Mx_CopyVector(zVectorType * zSourceVector, zVectorType * zDestinationVector);
zFunctionResult Mx_CopyVectorWithTransparency(zVectorType * zSourceVector, zVectorType * zDestinationVector, zMatrixElementType alpha);
zFunctionResult Mx_CopyVectorWithShift(zVectorType * zSourceVector, zVectorType * zDestinationVector, zInt Shift);
zFunctionResult Mx_CopyVectorWithTransparencyAndShift(zVectorType * zSourceVector, zVectorType * zDestinationVector, zMatrixElementType alpha, zInt Shift);
zFunctionResult Mx_ClearVector(zVectorType * zVector, zMatrixElementType value);
 
zFunctionResult Mx_VectorProcess_Sigmoid(zVectorType * zSourceVector, zVectorType * zDestinationVector);
zFunctionResult Mx_VectorProcess_SigmoidD(zVectorType * zSourceVector, zVectorType * zDestinationVector);
zFunctionResult Mx_VectorProcess_Linear(zVectorType * zSourceVector, zVectorType * zDestinationVector);
zFunctionResult Mx_VectorProcess_LinearD(zVectorType * zSourceVector, zVectorType * zDestinationVector);
zFunctionResult Mx_VectorProcess_Tanh(zVectorType * zSourceVector, zVectorType * zDestinationVector);
zFunctionResult Mx_VectorProcess_TanhD(zVectorType * zSourceVector, zVectorType * zDestinationVector);
	
zFunctionResult Mx_VectorCorrelation(zVectorType * zVector1, zVectorType * zVector2, zDatatype * outR);

zFunctionResult Mx_MultiplyMatrixVector(zMatrixType * zMatrix, zVectorType * zVector, zVectorType * zOutVector);


typedef struct
{
	zInt * data;
	zInt dim;
} zIntVectorType;

zFunctionResult Mx_CreateIntVector(zIntVectorType * zVector, zInt dim);
zFunctionResult Mx_FreeIntVector(zIntVectorType * zVector);
zFunctionResult Mx_UpdateIntVector(zIntVectorType * zVector, zInt dim);



typedef struct 
{
	zMatrixElementType *** data;
	zInt dimX;
	zInt dimY;
	zInt dimZ;
} zMatrix3Type;


zFunctionResult Mx_CreateMatrix3(zMatrix3Type * zMatrix3, zInt dimX, zInt dimY, zInt dimZ);
zFunctionResult Mx_FreeMatrix3(zMatrix3Type * zMatrix3);
zFunctionResult Mx_UpdateMatrix3(zMatrix3Type * zMatrix3, zInt dimX, zInt dimY, zInt dimZ);



typedef struct 
{
	zMatrixElementType **** data;
	zInt dimX;
	zInt dimY;
	zInt dimZ;
	zInt dimW;
} zMatrix4Type;

zFunctionResult Mx_CreateMatrix4(zMatrix4Type * zMatrix4, zInt dimX, zInt dimY, zInt dimZ, zInt dimW);
zFunctionResult Mx_ClearMatrix4(zMatrix4Type * zMatrix4);
zFunctionResult Mx_FreeMatrix4(zMatrix4Type * zMatrix4);
zFunctionResult Mx_UpdateMatrix4(zMatrix4Type * zMatrix4, zInt dimX, zInt dimY, zInt dimZ, zInt dimW);

#endif



