#include "matrix.h"
#include "mem.h"
#include <math.h>


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_CreateMatrix(zMatrixType * zMatrix, zInt dimX, zInt dimY)
{
	zInt i;

	if (dimX < 0 || dimX >= MATRIX_MAX_DIMENSION ||
		dimY < 0 || dimY >= MATRIX_MAX_DIMENSION)
	{
		return ZERROR_MATRIX_BAD_DIMENSIONS;
	}

	//initially, all dimensions are empty
	zMatrix->dimX = 0;
	zMatrix->dimY = 0;


	zMatrix->data = (zMatrixElementType **) Mem_AllocateCleared(sizeof(zMatrixElementType *) * dimX);
	if (zMatrix->data == NULL && dimX != 0)
	{
		return ZERROR_MEMORY_ALLOCATION_ERROR;
	}

	//save dimX
	zMatrix->dimX = dimX;

	for (i=0; i<dimX; i++)
	{
		zMatrix->data[i] = (zMatrixElementType *) Mem_AllocateCleared(sizeof(zMatrixElementType) * dimY);
		if (zMatrix->data[i] == NULL && dimY != 0)
		{
			return ZERROR_MEMORY_ALLOCATION_ERROR;
		};

		//save dimY
		zMatrix->dimY = dimY;
    }


	zMatrix->dimX = dimX;
	zMatrix->dimY = dimY;

	return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_AddMatrix(zMatrixType * zMatrix1, zMatrixType * zMatrix2, zMatrixType * zMatrixOut)
{
	zInt i,j;

	if (zMatrix1 == NULL ||
		zMatrix2 == NULL ||
		zMatrixOut == NULL)
	{
		return ZERROR_MATRIX_IS_NULL;
	}


	if (zMatrix1->dimX != zMatrixOut->dimX ||
		zMatrix1->dimY != zMatrixOut->dimY ||
		zMatrix2->dimX != zMatrixOut->dimX ||
		zMatrix2->dimY != zMatrixOut->dimY )
	{
		return ZERROR_MATRICES_DIMENSIONS_DO_NOT_MATCH;
    }

	for (i=0; i<zMatrix1->dimX; i++)
		for (j=0; j<zMatrix1->dimY; j++)
			zMatrixOut->data[i][j]=zMatrix1->data[i][j] + zMatrix2->data[i][j];

	return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_FreeMatrix(zMatrixType * zMatrix)
{
	zInt i;

	if (zMatrix == NULL)
	{
		//nothing to do
		return ZERROR_NO_ERROR;
	}

	for (i=0; i < zMatrix->dimX; i++)
	{
		if (zMatrix->data[i] != NULL)
			Mem_Free(zMatrix->data[i]);
    }

	Mem_Free(zMatrix->data);

	zMatrix->data = 0;
	zMatrix->dimX = 0;
	zMatrix->dimY = 0;

	return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_UpdateMatrix(zMatrixType * zMatrix, zInt dimX, zInt dimY)
{
	zInt i;
	zInt newDimX;
	zInt newDimY;

	if (dimX < 0 || dimX >= MATRIX_MAX_DIMENSION ||
		dimY < 0 || dimY >= MATRIX_MAX_DIMENSION)
	{
		return ZERROR_MATRIX_BAD_DIMENSIONS;
	}

	if (zMatrix == NULL)
	{
		return ZERROR_MATRIX_IS_NULL;
	}

	if (zMatrix->dimX == dimX &&
		zMatrix->dimY == dimY)
	{
		//nothing to do	
		return ZERROR_NO_ERROR;
	}

	//initially, dimensions are 0
	newDimX = 0;
	newDimY = 0;

	zMatrix->data = (zMatrixElementType **) Mem_Reallocate(zMatrix->data, sizeof(zMatrixElementType *) * dimX);
	if (zMatrix->data == NULL && dimX != 0)
	{
		return ZERROR_MEMORY_ALLOCATION_ERROR;
	}

	if (zMatrix->dimX < dimX) 
	{
		//we have to clear newly allocated blocks
		for (i=zMatrix->dimX; i < dimX; i++)
			zMatrix->data[i] = NULL;
	}

	//ok, save newDimX
	newDimX = dimX;


	for (i=0; i<dimX; i++)
	{
		zMatrix->data[i] = (zMatrixElementType *) Mem_Reallocate(zMatrix->data[i],sizeof(zMatrixElementType) * dimY);
		if (zMatrix->data[i] == NULL && dimY != 0)
		{
			return ZERROR_MEMORY_ALLOCATION_ERROR;
		};

		//ok, save newDimY
		newDimY = dimY;
    }

	zMatrix->dimX = newDimX;
	zMatrix->dimY = newDimY;

	return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_ClearMatrix(zMatrixType * zMatrix, zMatrixElementType value)
{
	zInt i,j;

	if (zMatrix == NULL)
		//nothing to do	
		return ZERROR_NO_ERROR;

	for (i = 0; i < zMatrix->dimX; i++)
		for (j = 0; j < zMatrix->dimY; j++)
			zMatrix->data[i][j] = value;

	return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_CopyMatrix(zMatrixType * zMatrix, zMatrixType * zOutMatrix)
{
	zInt i,j;

	if (zMatrix == NULL ||
		zOutMatrix == NULL)
	{
		return ZERROR_MATRIX_IS_NULL;
	}


	if (zMatrix->dimX != zOutMatrix->dimX ||
		zMatrix->dimY != zOutMatrix->dimY)
	{
		return ZERROR_MATRICES_DIMENSIONS_DO_NOT_MATCH;
    }

	for (i = 0; i < zMatrix->dimX; i++)
		for (j = 0; j < zMatrix->dimY; j++)
			zOutMatrix->data[i][j] = zMatrix->data[i][j];

	return ZERROR_NO_ERROR;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_AddToMatrixWithAlpha(zMatrixType * zMatrix1, zMatrixType * zMatrix2, zMatrixElementType alpha)
{
	zInt i,j;

	if (zMatrix1 == NULL ||
		zMatrix2 == NULL)
	{
		return ZERROR_MATRIX_IS_NULL;
	}


	if (zMatrix1->dimX != zMatrix2->dimX ||
		zMatrix1->dimY != zMatrix2->dimY)
	{
		return ZERROR_MATRICES_DIMENSIONS_DO_NOT_MATCH;
    }


	for (i=0; i<zMatrix1->dimX; i++)
		for (j=0; j<zMatrix1->dimY; j++)
			zMatrix1->data[i][j] += zMatrix2->data[i][j] * alpha;

	return ZERROR_NO_ERROR;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_CreateVector(zVectorType * zVector, zInt dim)
{
	if (dim < 0 || dim >= VECTOR_MAX_DIMENSION)
	{
		return ZERROR_VECTOR_BAD_DIMENSION;
	}

	zVector->dim = dim;
	zVector->data = (zMatrixElementType *) Mem_AllocateCleared(sizeof(zMatrixElementType) * dim);
	if (zVector->data == NULL && dim != 0)
	{
		return ZERROR_MEMORY_ALLOCATION_ERROR;
    }

	return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_FreeVector(zVectorType * zVector)
{
	if (zVector == NULL)
	{
		//nothing to do
		return ZERROR_NO_ERROR;
	}

	Mem_Free(zVector->data);
	zVector->data = 0;
	zVector->dim = 0;

	return ZERROR_NO_ERROR;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_UpdateVector(zVectorType * zVector, zInt dim)
{
	if (dim < 0 || dim >= VECTOR_MAX_DIMENSION)
	{
		return ZERROR_VECTOR_BAD_DIMENSION;
	}

	zVector->dim = dim;
	zVector->data = (zMatrixElementType *) Mem_Reallocate(zVector->data, sizeof(zMatrixElementType) * dim);
	if (zVector->data == NULL && dim != 0)
	{
		return ZERROR_MEMORY_ALLOCATION_ERROR;
    }

	return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_CopyVector(zVectorType * zSourceVector, zVectorType * zDestinationVector)
{
	if (zSourceVector->dim != zDestinationVector->dim)
	{
		return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;
	}

	memcpy(zDestinationVector->data, zSourceVector->data, sizeof(zMatrixElementType) * zSourceVector->dim);

	return ZERROR_NO_ERROR;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_CopyVectorWithTransparency(zVectorType * zSourceVector, zVectorType * zDestinationVector, zMatrixElementType alpha)
{
	int i;

	if (zSourceVector->dim != zDestinationVector->dim)
	{
		return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;
	}

	for (i = 0; i < zSourceVector->dim; i++)
		zDestinationVector->data[i] = zDestinationVector->data[i] * (1 - alpha) + alpha * zSourceVector->data[i];

	return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_CopyVectorWithShift(zVectorType * zSourceVector, zVectorType * zDestinationVector, zInt Shift)
{
	int i;

	if (zSourceVector->dim + Shift!= zDestinationVector->dim)
	{
		return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;
	}

	for (i = 0; i < zSourceVector->dim; i++)
		zDestinationVector->data[i + Shift] = zSourceVector->data[i];

	return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_CopyVectorWithTransparencyAndShift(zVectorType * zSourceVector, zVectorType * zDestinationVector, zMatrixElementType alpha, zInt Shift)
{
	int i;

	if (zSourceVector->dim + Shift!= zDestinationVector->dim)
	{
		return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;
	}

	for (i = 0; i < zSourceVector->dim; i++)
		zDestinationVector->data[i + Shift] = zDestinationVector->data[i + Shift] * (1 - alpha) + alpha * zSourceVector->data[i];

	return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_ClearVector(zVectorType * zVector, zMatrixElementType value)
{
	zInt i;

	if (zVector == NULL)
		//nothing to do	
		return ZERROR_NO_ERROR;

	for (i = 0; i < zVector->dim; i++)
		zVector->data[i] = value;

	return ZERROR_NO_ERROR;

}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_VectorCorrelation(zVectorType * zVector1, zVectorType * zVector2, zDatatype * outR)
{
	zInt i;
	zMatrixElementType av,av2;
	zMatrixElementType q1,q2,q3;

	if (zVector1->dim != zVector2->dim)
		return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;

	av = 0;
	av2 = 0;
	q1 = 0;
	q2 = 0;
	q3 = 0;
	for (i=0; i < zVector1->dim; i++)
	{
		av += zVector1->data[i];
		av2 += zVector2->data[i];

		q1 += zVector1->data[i] * zVector2->data[i]; 
		q2 += zVector1->data[i] * zVector1->data[i];
		q3 += zVector2->data[i] * zVector2->data[i];
	}

	q1 = zVector1->dim * q1 - av * av2;

	q1 = q1 / sqrt(zVector1->dim * q2 - av*av) / sqrt(zVector1->dim * q3 - av2*av2);

    *outR = q1;

	return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_MultiplyMatrixVector(zMatrixType * zMatrix, zVectorType * zVector, zVectorType * zOutVector)
{
	zInt i,j;
	zMatrixElementType q;

	zMatrixElementType ** value1;
	zMatrixElementType * value11;
	zMatrixElementType * value2;
	zMatrixElementType * value3;

	if (zMatrix->dimX != zOutVector->dim ||
		zMatrix->dimY != zVector->dim)
	{
		return ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH;
	}


	//non-optimized. compiler optimizes than itself
	for (i = 0; i < zMatrix->dimX; i++)
	{
		q = 0;
		for (j = 0; j < zMatrix->dimY; j++)
		{
			q += zMatrix->data[i][j] * zVector->data[j];
		}

		zOutVector->data[i] = q;
	}


/*
	//optimized
	value1 = &(zMatrix->data[0]);
	value2 = &(zOutVector->data[0]);
	for (i = 0; i < zMatrix->dimX; i++)
	{
		value11 = *value1++;
		*value2 = 0;
		value3 = &(zVector->data[0]);
		for (j = 0; j < zMatrix->dimY; j++)
			*value2 += (*value11++) * (*value3++);

		value2++;
	}
*/


	return ZERROR_NO_ERROR;
}







//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_CreateIntVector(zIntVectorType * zIntVector, zInt dim)
{
	if (dim < 0 || dim >= VECTOR_MAX_DIMENSION)
	{
		return ZERROR_VECTOR_BAD_DIMENSION;
	}

	zIntVector->dim = dim;
	if (dim != 0)
	{
		zIntVector->data = (zInt *) Mem_AllocateCleared(sizeof(zInt) * dim);
	}
	else
	{
		zIntVector->data = NULL;
    }

	if (zIntVector->data == NULL && dim != 0)
	{
		return ZERROR_MEMORY_ALLOCATION_ERROR;
    }

	return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_FreeIntVector(zIntVectorType * zIntVector)
{
	if (zIntVector == NULL)
	{
		//nothing to do
		return ZERROR_NO_ERROR;
	}

	Mem_Free(zIntVector->data);
	zIntVector->data = 0;
	zIntVector->dim = 0;

	return ZERROR_NO_ERROR;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_UpdateIntVector(zIntVectorType * zIntVector, zInt dim)
{
	if (dim < 0 || dim >= VECTOR_MAX_DIMENSION)
	{
		return ZERROR_VECTOR_BAD_DIMENSION;
	}

	zIntVector->dim = dim;
	zIntVector->data = (zInt *) Mem_Reallocate(zIntVector->data, sizeof(zInt) * dim);
	if (zIntVector->data == NULL && dim != 0)
	{
		return ZERROR_MEMORY_ALLOCATION_ERROR;
    }

	return ZERROR_NO_ERROR;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_CreateMatrix3(zMatrix3Type * zMatrix3, zInt dimX, zInt dimY, zInt dimZ)
{
	zInt i,j;

	if (dimX < 0 || dimX >= MATRIX_MAX_DIMENSION ||
		dimY < 0 || dimY >= MATRIX_MAX_DIMENSION ||
		dimZ < 0 || dimZ >= MATRIX_MAX_DIMENSION)
	{
		return ZERROR_MATRIX_BAD_DIMENSIONS;
	}

	//initially, dimensions are 0
	zMatrix3->dimX = 0;
	zMatrix3->dimY = 0;
	zMatrix3->dimZ = 0;


	zMatrix3->data = (zMatrixElementType ***) Mem_AllocateCleared(sizeof(zMatrixElementType **) * dimX);
	if (zMatrix3->data == NULL && dimX != 0)
	{
		return ZERROR_MEMORY_ALLOCATION_ERROR;
	}

	//save dimx
	zMatrix3->dimX = dimX;

	for (i=0; i<dimX; i++)
	{
		zMatrix3->data[i] = (zMatrixElementType **) Mem_AllocateCleared(sizeof(zMatrixElementType *) * dimY);
		if (zMatrix3->data[i] == NULL && dimY != 0)
		{
			return ZERROR_MEMORY_ALLOCATION_ERROR;
		};
	
		//save dimy
		zMatrix3->dimY = dimY;


		for (j=0; j < dimY; j++)
		{
			zMatrix3->data[i][j] = (zMatrixElementType *) Mem_AllocateCleared(sizeof(zMatrixElementType) * dimZ);
			if (zMatrix3->data[i][j] == NULL && dimZ != 0)
			{
				return ZERROR_MEMORY_ALLOCATION_ERROR;
			};

			//save dimZ
			zMatrix3->dimZ = dimZ;
        }
    }



	return ZERROR_NO_ERROR;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_FreeMatrix3(zMatrix3Type * zMatrix3)
{
	zInt i,j;

	if (zMatrix3 == NULL)
	{
		//nothing to do
		return ZERROR_NO_ERROR;
	}

	for (i=0; i < zMatrix3->dimX; i++)
	{
		if (zMatrix3->data[i] != NULL)
		{
			for (j=0; j < zMatrix3->dimY; j++)
			{
				if (zMatrix3->data[i][j] != NULL)
					Mem_Free(zMatrix3->data[i][j]);
			}
			
			Mem_Free(zMatrix3->data[i]);
		}
    }

	Mem_Free(zMatrix3->data);

	zMatrix3->data = 0;
	zMatrix3->dimX = 0;
	zMatrix3->dimY = 0;
	zMatrix3->dimZ = 0;

	return ZERROR_NO_ERROR;

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_UpdateMatrix3(zMatrix3Type * zMatrix3, zInt dimX, zInt dimY, zInt dimZ)
{
	zInt i,j;
	zInt newDimX;
	zInt newDimY;
	zInt newDimZ;

	if (dimX < 0 || dimX >= MATRIX_MAX_DIMENSION ||
		dimY < 0 || dimY >= MATRIX_MAX_DIMENSION ||
		dimZ < 0 || dimZ >= MATRIX_MAX_DIMENSION)
	{
		return ZERROR_MATRIX_BAD_DIMENSIONS;
	}

	if (zMatrix3 == NULL)
	{
		return ZERROR_MATRIX_IS_NULL;
	}

	if (zMatrix3->dimX == dimX &&
		zMatrix3->dimY == dimY &&
		zMatrix3->dimZ == dimZ)
	{
		//nothing to do	
		return ZERROR_NO_ERROR;
	}

	newDimX = 0;
	newDimY = 0;
	newDimZ = 0;

	zMatrix3->data = (zMatrixElementType ***) Mem_Reallocate(zMatrix3->data, sizeof(zMatrixElementType **) * dimX);
	if (zMatrix3->data == NULL && dimX != 0)
	{
		return ZERROR_MEMORY_ALLOCATION_ERROR;
	}

	if (zMatrix3->dimX < dimX) 
	{
		//we have to clear newly allocated blocks
		for (i=zMatrix3->dimX; i < dimX; i++)
			zMatrix3->data[i] = NULL;
	}

	newDimX = dimX;

	for (i=0; i<dimX; i++)
	{
		zMatrix3->data[i] = (zMatrixElementType **) Mem_Reallocate(zMatrix3->data[i],sizeof(zMatrixElementType *) * dimY);
		if (zMatrix3->data[i] == NULL && dimY != 0)
		{
			return ZERROR_MEMORY_ALLOCATION_ERROR;
		};

		if (zMatrix3->dimY < dimY) 
		{
			//we have to clear newly allocated blocks
			for (j=zMatrix3->dimY; j < dimY; j++)
				zMatrix3->data[i][j] = NULL;
		}

		newDimY = dimY;

		for (j=0;j<dimY;j++)
		{
			if (zMatrix3->dimX < dimX)
			{
				zMatrix3->data[i][j] = (zMatrixElementType *) Mem_AllocateCleared(sizeof(zMatrixElementType) * dimZ);
			}
			else
			{
				zMatrix3->data[i][j] = (zMatrixElementType *) Mem_Reallocate(zMatrix3->data[i][j],sizeof(zMatrixElementType) * dimZ);
			};

			newDimZ = dimZ;
		}
	};

	zMatrix3->dimX = newDimX;
	zMatrix3->dimY = newDimY;
	zMatrix3->dimZ = newDimZ;


	return ZERROR_NO_ERROR;

}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_CreateMatrix4(zMatrix4Type * zMatrix4, zInt dimX, zInt dimY, zInt dimZ, zInt dimW)
{
	zInt i,j,k;

	if (dimX < 0 || dimX >= MATRIX_MAX_DIMENSION ||
		dimY < 0 || dimY >= MATRIX_MAX_DIMENSION ||
		dimZ < 0 || dimZ >= MATRIX_MAX_DIMENSION ||
		dimW < 0 || dimW >= MATRIX_MAX_DIMENSION)
	{
		return ZERROR_MATRIX_BAD_DIMENSIONS;
	}

	//initially
	zMatrix4->dimX = 0;
	zMatrix4->dimY = 0;
	zMatrix4->dimZ = 0;
	zMatrix4->dimW = 0;


	zMatrix4->data = (zMatrixElementType ****) Mem_AllocateCleared(sizeof(zMatrixElementType ***) * dimX);
	if (zMatrix4->data == NULL && dimX != 0)
	{
		return ZERROR_MEMORY_ALLOCATION_ERROR;
	}

	//save dimX
	zMatrix4->dimX = dimX;


	for (i=0; i<dimX; i++)
	{
		zMatrix4->data[i] = (zMatrixElementType ***) Mem_AllocateCleared(sizeof(zMatrixElementType **) * dimY);
		if (zMatrix4->data[i] == NULL && dimY != 0)
		{
			return ZERROR_MEMORY_ALLOCATION_ERROR;
        }

		//save dimY
		zMatrix4->dimY = dimY;


		for (j=0; j < dimY; j++)
		{
			zMatrix4->data[i][j] = (zMatrixElementType **) Mem_AllocateCleared(sizeof(zMatrixElementType *) * dimZ);
			if (zMatrix4->data[i][j] == NULL && dimZ != 0)
			{
				return ZERROR_MEMORY_ALLOCATION_ERROR;
            }

			//save dimZ
			zMatrix4->dimZ = dimZ;


			for (k=0; k < dimZ; k++)
			{
				zMatrix4->data[i][j][j] = (zMatrixElementType *) Mem_AllocateCleared(sizeof(zMatrixElementType) * dimW);
				if (zMatrix4->data[i][j][k] == NULL && dimW != 0)
				{
					return ZERROR_MEMORY_ALLOCATION_ERROR;
                }

				//save dimW
				zMatrix4->dimW = dimW;
			}
		}
    }



	return ZERROR_NO_ERROR;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_FreeMatrix4(zMatrix4Type * zMatrix4)
{
	zInt i,j,k;

	if (zMatrix4 == NULL)
	{
		//nothing to do
		return ZERROR_NO_ERROR;
	}

	for (i=0; i < zMatrix4->dimX; i++)
	{
		if (zMatrix4->data[i] != NULL)
		{
			for (j=0; j < zMatrix4->dimY; j++)
			{
				if (zMatrix4->data[i][j] != NULL)
				{
					for (k=0; k < zMatrix4->dimZ; k++)
					{
						if (zMatrix4->data[i][j][k] != NULL)
							Mem_Free(zMatrix4->data[i][j][k]);
                    }
					
					Mem_Free(zMatrix4->data[i][j]);
                }
            }

			Mem_Free(zMatrix4->data[i]);
        }
    }

	Mem_Free(zMatrix4->data);

	zMatrix4->data = 0;
	zMatrix4->dimX = 0;
	zMatrix4->dimY = 0;
	zMatrix4->dimZ = 0;
	zMatrix4->dimW = 0;

	return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_UpdateMatrix4(zMatrix4Type * zMatrix4, zInt dimX, zInt dimY, zInt dimZ, zInt dimW)
{
	zInt i,j,k;
	zInt newDimX;
	zInt newDimY;
	zInt newDimZ;
	zInt newDimW;

	if (dimX < 0 || dimX >= MATRIX_MAX_DIMENSION ||
		dimY < 0 || dimY >= MATRIX_MAX_DIMENSION ||
		dimZ < 0 || dimZ >= MATRIX_MAX_DIMENSION ||
		dimW < 0 || dimW >= MATRIX_MAX_DIMENSION)
	{
		return ZERROR_MATRIX_BAD_DIMENSIONS;
	}

	if (zMatrix4 == NULL)
	{
		return ZERROR_MATRIX_IS_NULL;
	}

	if (zMatrix4->dimX == dimX &&
		zMatrix4->dimY == dimY &&
		zMatrix4->dimZ == dimZ &&
		zMatrix4->dimW == dimW)
	{
		//nothing to do	
		return ZERROR_NO_ERROR;
	}

	newDimX = 0;
	newDimY = 0;
	newDimZ = 0;
	newDimW = 0;

	zMatrix4->data = (zMatrixElementType ****) Mem_Reallocate(zMatrix4->data, sizeof(zMatrixElementType ***) * dimX);
	if (zMatrix4->data == NULL && dimX != 0)
	{
		return ZERROR_MEMORY_ALLOCATION_ERROR;
	}

	if (zMatrix4->dimX < dimX) 
	{
		//we have to clear newly allocated blocks
		for (i=zMatrix4->dimX; i < dimX; i++)
			zMatrix4->data[i] = NULL;
	}

	newDimX = dimX;

	for (i=0; i<dimX; i++)
	{
		zMatrix4->data[i] = (zMatrixElementType ***) Mem_Reallocate(zMatrix4->data[i],sizeof(zMatrixElementType **) * dimY);
		if (zMatrix4->data[i] == NULL && dimY != 0)
		{
			return ZERROR_MEMORY_ALLOCATION_ERROR;
        }

		if (zMatrix4->dimY < dimY) 
		{
			//we have to clear newly allocated blocks
			for (j=zMatrix4->dimY; j < dimY; j++)
				zMatrix4->data[i][j] = NULL;
		}

		newDimY = dimY;

		for (j=0;j<dimY;j++)
		{
			if (zMatrix4->dimX < dimX)
			{
				zMatrix4->data[i][j] = (zMatrixElementType **) Mem_Allocate(sizeof(zMatrixElementType *) * dimZ);
			}
			else
			{
				zMatrix4->data[i][j] = (zMatrixElementType **) Mem_Reallocate(zMatrix4->data[i][j],sizeof(zMatrixElementType *) * dimZ);
            }

			if (zMatrix4->data[i][j] == NULL && dimZ != 0)
			{
				return ZERROR_MEMORY_ALLOCATION_ERROR;
            }

			if (zMatrix4->dimZ < dimZ) 
			{
				//we have to clear newly allocated blocks
				for (k=zMatrix4->dimZ; k < dimZ; k++)
					zMatrix4->data[i][j][k] = NULL;
			}

			newDimZ = dimZ;

			for (k=0;k<dimZ;k++)
			{
				if (zMatrix4->dimX < dimX ||
					zMatrix4->dimY < dimY)
				{
					zMatrix4->data[i][j][k] = (zMatrixElementType *) Mem_AllocateCleared(sizeof(zMatrixElementType) * dimW);
				}
				else
				{
					zMatrix4->data[i][j][k] = (zMatrixElementType *) Mem_Reallocate(zMatrix4->data[i][j][k],sizeof(zMatrixElementType) * dimW);
                }

				newDimW = dimW;
            }
        }
    }

	zMatrix4->dimX = newDimX;
	zMatrix4->dimY = newDimY;
	zMatrix4->dimZ = newDimZ;
	zMatrix4->dimW = newDimW;

	return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult Mx_ClearMatrix4(zMatrix4Type * zMatrix4)
{
	zInt i,j,k,l;

	if (zMatrix4 == NULL)
		//nothing to do	
		return ZERROR_NO_ERROR;

	for (i = 0; i < zMatrix4->dimX; i++)
		for (j = 0; j < zMatrix4->dimY; j++)
			for (k = 0; k < zMatrix4->dimZ; k++)
				for (l = 0; l < zMatrix4->dimW; l++)
					zMatrix4->data[i][j][k][l] = 0;

	return ZERROR_NO_ERROR;
}



