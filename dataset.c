#include "dataset.h"

#include "mem.h"




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_CheckDatasetIsntDeadNAcquired(zInt datasetID)
{
    if (datasetID < 0 || datasetID >= DS_datasetVaultMaxDatasetsAllocated)
    {
        return ZERROR_BAD_DATASET_ID;
    }


    if ((*DS_datasetVault[datasetID]).datasetIsDead == DS_DATASET_IS_DEAD)
    {
        return ZERROR_DATASET_IS_DEAD;
    }


    if ((*DS_datasetVault[datasetID]).datasetIsAcquired == DS_DATASET_IS_ACQUIRED)
    {
        //nothing to do
        return ZERROR_DATASET_IS_ACQUIRED;
    }

    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_CheckDatasetIsntDead(zInt datasetID)
{
    if (datasetID < 0 || datasetID >= DS_datasetVaultMaxDatasetsAllocated)
    {
        return ZERROR_BAD_DATASET_ID;
    }


    if ((*DS_datasetVault[datasetID]).datasetIsDead == DS_DATASET_IS_DEAD)
    {
        return ZERROR_DATASET_IS_DEAD;
    }

    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_CheckDatasetIsntAcquired(zInt datasetID)
{
    if (datasetID < 0 || datasetID >= DS_datasetVaultMaxDatasetsAllocated)
    {
        return ZERROR_BAD_DATASET_ID;
    }


    if ((*DS_datasetVault[datasetID]).datasetIsAcquired == DS_DATASET_IS_ACQUIRED)
    {
        //nothing to do
        return ZERROR_DATASET_IS_ACQUIRED;
    }

    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_InitializeDatasetVault(void)
{
    DS_datasetVaultMaxDatasetsAllocated = 0;
    DS_datasetVault = NULL;


    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_ReleaseDatasetVault(void)
{
    zFunctionResult resultID;
    zInt i;

    //first, check all available datasets
    for (i = 0; i < DS_datasetVaultMaxDatasetsAllocated; i++)
    {
        if ((*DS_datasetVault[i]).datasetIsAcquired == DS_DATASET_IS_ACQUIRED)
        {
            //wow, cannot release vault, some datasets are still in use
            return ZERROR_SOME_DATASETS_ARE_STILL_ACQUIRED;
        }
    }


    while (DS_datasetVaultMaxDatasetsAllocated > 0)
    {
        resultID = DSi_DeleteDataset(DS_datasetVaultMaxDatasetsAllocated - 1);
        if (resultID != ZERROR_NO_ERROR)
        {
            return resultID;
        }
    }

    if (DS_datasetVault != NULL)
    {
        Mem_Free(DS_datasetVault);
        DS_datasetVault = NULL;
    }

    DS_datasetVaultMaxDatasetsAllocated = 0;

    return ZERROR_NO_ERROR;
}








//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DSi_DeleteDataset(zInt datasetID)
{
    DatasetStruct * datasetToDie;
    DatasetStruct * dataset;
    zFunctionResult resultID;

    //assuming CS is locked and dataID is correct and datasetID is not acquired

    if ((*DS_datasetVault[datasetID]).datasetIsDead == DS_DATASET_IS_DEAD)
    {
        //nothing to do
        return ZERROR_NO_ERROR;
    }

    DSi_AcquireDataset(datasetID, &dataset);


    resultID = Mx_FreeMatrix(&(dataset->data));
    if (resultID != ZERROR_NO_ERROR)
    {
        DSi_ReleaseDataset(datasetID);
        return resultID;
    }


    resultID = Mx_FreeMatrix(&(dataset->dataOut));
    if (resultID != ZERROR_NO_ERROR)
    {
        DSi_ReleaseDataset(datasetID);
        return resultID;
    }



    dataset->datasetIsDead = DS_DATASET_IS_DEAD;


    //now try to shrink DS_datasetVault size
    while (DS_datasetVaultMaxDatasetsAllocated > 0 &&
           (*DS_datasetVault[DS_datasetVaultMaxDatasetsAllocated - 1]).datasetIsDead == DS_DATASET_IS_DEAD)
    {
        //shrink memory

        //step 1
        //kill old net
        datasetToDie = &((*DS_datasetVault[DS_datasetVaultMaxDatasetsAllocated - 1]));
        Mem_Free(datasetToDie);

        //step 2
        //realloc netvault
        DS_datasetVault = (DatasetStruct **) Mem_Reallocate(DS_datasetVault, (DS_datasetVaultMaxDatasetsAllocated - 1) * sizeof(DatasetStruct*));
        DS_datasetVaultMaxDatasetsAllocated--;
    }


    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_DeleteDataset(zInt datasetID)
{
    zFunctionResult resultID;

    resultID = DS_CheckDatasetIsntAcquired(datasetID);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    //internal function call
    resultID = DSi_DeleteDataset(datasetID);


    return resultID;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_TryToAcquireDataset(zInt datasetID, DatasetStruct ** dataset)
{
    zFunctionResult resultID;

    resultID = DS_CheckDatasetIsntDeadNAcquired(datasetID);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    resultID = DSi_AcquireDataset(datasetID, dataset);

    return resultID;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DSi_AcquireDataset(zInt datasetID, DatasetStruct ** dataset)
{
    //acquire dataset
    (*DS_datasetVault[datasetID]).datasetIsAcquired = DS_DATASET_IS_ACQUIRED;
    if (dataset != NULL)
        *dataset = &((*DS_datasetVault[datasetID]));

    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_ReleaseAcquiredDataset(zInt datasetID)
{
    zFunctionResult resultID;

    resultID = DS_CheckDatasetIsntDead(datasetID);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    resultID = DSi_ReleaseDataset(datasetID);

    return resultID;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DSi_ReleaseDataset(zInt datasetID)
{
    //release dataset
    (*DS_datasetVault[datasetID]).datasetIsAcquired = DS_DATASET_IS_NOT_ACQUIRED;

    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_CreateDataset(void)
{
    zInt i,j;
    zInt flag;
    DatasetStruct * newDataset;

    j = 0;
    flag = 0;
    if (DS_datasetVault != NULL)
    {
        //some nets exist, check them

        //are there dead nets among them?
        for (i = 0; i < DS_datasetVaultMaxDatasetsAllocated; i++)
        {
            if ((*DS_datasetVault[i]).datasetIsDead == DS_DATASET_IS_DEAD)
            {
                //ok, found dead net
                //reinitialize it
                flag = 1;
                j = i;
                break;
            }
        }
    }

    if (flag == 0)
    {
        //no dead datasets
        //so allocating new one

        //step 1. alloc mem for new net
        newDataset = (DatasetStruct *) Mem_Allocate(sizeof(DatasetStruct));
        if (newDataset == NULL)
        {
            //oops, realloc failed
            return ZERROR_MEMORY_ALLOCATION_ERROR;
        }

        //step 2. realloc NetVault
        DS_datasetVault = (DatasetStruct **) Mem_Reallocate(DS_datasetVault, (DS_datasetVaultMaxDatasetsAllocated + 1) * sizeof(DatasetStruct *));
        if (DS_datasetVault == NULL)
        {
            //oops, realloc failed
            return ZERROR_MEMORY_ALLOCATION_ERROR;
        }

        DS_datasetVault[DS_datasetVaultMaxDatasetsAllocated] = newDataset;


        j = DS_datasetVaultMaxDatasetsAllocated;

        DS_datasetVaultMaxDatasetsAllocated++;
    }

    //initialize net
    //j points to net
    (*DS_datasetVault[j]).datasetIsAcquired = DS_DATASET_IS_NOT_ACQUIRED;
    (*DS_datasetVault[j]).datasetIsDead = DS_DATASET_IS_NOT_DEAD;
    Mx_CreateMatrix(&((*DS_datasetVault[j]).data), 0, 0);
    Mx_CreateMatrix(&((*DS_datasetVault[j]).dataOut), 0, 0);

    return j;

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_SetDataset2DFloat(zInt datasetID, zInt dimX, zInt dimYData, zInt dimYDataOut, zMatrixType * data, zMatrixType * dataOut)
{
    zFunctionResult resultID;
    DatasetStruct * dataset;
    zInt i,j,index;

    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    resultID = Mx_UpdateMatrix(&(dataset->data),dimX,dimYData);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    resultID = Mx_UpdateMatrix(&(dataset->dataOut),dimX,dimYDataOut);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    index = 0;
    for (i = 0; i < dimX; i++)
        for (j = 0; j < dimYData; j++)
        {
            dataset->data.data[i][j] = data->data[i][j];
            index++;
        }


    index = 0;
    for (i = 0; i < dimX; i++)
        for (j = 0; j < dimYDataOut; j++)
        {
            dataset->dataOut.data[i][j] = dataOut->data[i][j];
            index++;
        }

    return DS_ReleaseAcquiredDataset(datasetID);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_SetDataset1DFloat(zInt datasetID, zInt dimX, zInt dimYData, zInt dimYDataOut, float * data, float * dataOut)
{
    zFunctionResult resultID;
    DatasetStruct * dataset;
    zInt i,j,index;

    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    resultID = Mx_UpdateMatrix(&(dataset->data),dimX,dimYData);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    resultID = Mx_UpdateMatrix(&(dataset->dataOut),dimX,dimYDataOut);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    index = 0;
    for (i = 0; i < dimX; i++)
        for (j = 0; j < dimYData; j++)
        {
            dataset->data.data[i][j] = data[index];
            index++;
        }


    index = 0;
    for (i = 0; i < dimX; i++)
        for (j = 0; j < dimYDataOut; j++)
        {
            dataset->dataOut.data[i][j] = dataOut[index];
            index++;
        }

    return DS_ReleaseAcquiredDataset(datasetID);
}





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_SetDatasetInput1DFloat(zInt datasetID, zInt dimX, zInt dimYData, float * data)
{
    zFunctionResult resultID;
    DatasetStruct * dataset;
    zInt i,j,index;

    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    resultID = Mx_UpdateMatrix(&(dataset->data),dimX,dimYData);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    index = 0;
    for (i = 0; i < dimX; i++)
        for (j = 0; j < dimYData; j++)
        {
            dataset->data.data[i][j] = data[index];
            index++;
        }


    return DS_ReleaseAcquiredDataset(datasetID);
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_SetDatasetOutput1DFloat(zInt datasetID, zInt dimX, zInt dimYDataOut, float * dataOut)
{
    zFunctionResult resultID;
    DatasetStruct * dataset;
    zInt i,j,index;

    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    resultID = Mx_UpdateMatrix(&(dataset->dataOut),dimX,dimYDataOut);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    index = 0;
    for (i = 0; i < dimX; i++)
        for (j = 0; j < dimYDataOut; j++)
        {
            dataset->dataOut.data[i][j] = dataOut[index];
            index++;
        }

    return DS_ReleaseAcquiredDataset(datasetID);
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_GetDataset1DFloat(zInt datasetID, float * data, float * dataOut)
{
    zFunctionResult resultID;
    DatasetStruct * dataset;
    zInt i,j,index;

    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
        for (j = 0; j < dataset->data.dimY; j++)
        {
            data[index] = dataset->data.data[i][j];
            index++;
        }


    index = 0;
    for (i = 0; i < dataset->dataOut.dimX; i++)
        for (j = 0; j < dataset->dataOut.dimY; j++)
        {
            dataOut[index] = dataset->dataOut.data[i][j];
            index++;
        }

    return DS_ReleaseAcquiredDataset(datasetID);
}





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_GetDatasetInput1DFloat(zInt datasetID, float * data)
{
    zFunctionResult resultID;
    DatasetStruct * dataset;
    zInt i,j,index;

    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
        for (j = 0; j < dataset->data.dimY; j++)
        {
            data[index] = dataset->data.data[i][j];
            index++;
        }

    return DS_ReleaseAcquiredDataset(datasetID);
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult DS_GetDatasetOutput1DFloat(zInt datasetID, float * dataOut)
{
    zFunctionResult resultID;
    DatasetStruct * dataset;
    zInt i,j,index;

    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    index = 0;
    for (i = 0; i < dataset->dataOut.dimX; i++)
        for (j = 0; j < dataset->dataOut.dimY; j++)
        {
            dataOut[index] = dataset->dataOut.data[i][j];
            index++;
        }

    return DS_ReleaseAcquiredDataset(datasetID);
}


