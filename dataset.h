#ifndef _DATASET_H_
#define _DATASET_H_

#include "matrix.h"
#include "types.h"


#define DS_DATASET_IS_DEAD 1
#define DS_DATASET_IS_NOT_DEAD 0

#define DS_DATASET_IS_ACQUIRED 1
#define DS_DATASET_IS_NOT_ACQUIRED 0



typedef struct
{
    zMatrixType data;
    zMatrixType dataOut;

    zInt datasetIsDead;
    zInt datasetIsAcquired;

} DatasetStruct;

DatasetStruct ** DS_datasetVault;
zInt DS_datasetVaultMaxDatasetsAllocated;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///  user functions
zFunctionResult DS_CreateDataset(void);
zFunctionResult DS_DeleteDataset(zInt datasetID);

zFunctionResult DS_SetDataset1DFloat(zInt datasetID, zInt dimX, zInt dimYData, zInt dimYDataOut, float * data, float * dataOut);
zFunctionResult DS_SetDatasetInput1DFloat(zInt datasetID, zInt dimX, zInt dimYData, float * data);
zFunctionResult DS_SetDatasetOutput1DFloat(zInt datasetID, zInt dimX, zInt dimYDataOut, float * dataOut);

zFunctionResult DS_GetDataset1DFloat(zInt datasetID, float * data, float * dataOut);
zFunctionResult DS_GetDatasetInput1DFloat(zInt datasetID, float * data);
zFunctionResult DS_GetDatasetOutput1DFloat(zInt datasetID, float * dataOut);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///  internal functions
zFunctionResult DS_InitializeDatasetVault(void);
zFunctionResult DS_ReleaseDatasetVault(void);

zFunctionResult DS_TryToAcquireDataset(zInt datasetID, DatasetStruct ** dataset);
zFunctionResult DS_ReleaseAcquiredDataset(zInt datasetID);

zFunctionResult DSi_DeleteDataset(zInt datasetID);
zFunctionResult DSi_AcquireDataset(zInt datasetID, DatasetStruct ** dataset);
zFunctionResult DSi_ReleaseDataset(zInt datasetID);


zFunctionResult DS_CheckDatasetIsntDeadNAcquired(zInt datasetID);
zFunctionResult DS_CheckDatasetIsntAcquired(zInt datasetID);
zFunctionResult DS_CheckDatasetIsntDead(zInt datasetID);





#endif
