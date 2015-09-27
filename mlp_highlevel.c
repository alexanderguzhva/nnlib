#include "mlp_highlevel.h"

#include <math.h>

#include "types.h"
#include "mlp.h"
#include "randomgen.h"





/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Create3SlabsPerc(int nInput, int nHidden, int nOutput, int hiddenFunc, int outputFunc, float w, int randseed)
{
    zInt netID;
    zInt inputSlabID;
    zInt hiddenSlabID;
    zInt outputSlabID;
    zFunctionResult resultID;
    zInt IH1LinkID;
    zInt H1OLinkID;
    NeuroNetStruct * net;
    zInt myRandseed;

    netID = NN_CreateNet();
    if (netID < 0)
    {
        return netID;
    }



    inputSlabID = NN_CreateSlab(netID, NN_SLAB_INPUT, nInput);
    if (inputSlabID < 0)
        return inputSlabID;

    hiddenSlabID = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden);
    if (hiddenSlabID < 0)
        return hiddenSlabID;

    outputSlabID = NN_CreateSlab(netID, NN_SLAB_OUTPUT, nOutput);
    if (outputSlabID < 0)
        return outputSlabID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID, hiddenFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, outputSlabID, outputFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    IH1LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID);
    if (IH1LinkID < 0)
        return IH1LinkID;

    H1OLinkID = NN_CreateLink(netID, hiddenSlabID, outputSlabID);
    if (H1OLinkID < 0)
        return H1OLinkID;


    myRandseed = randseed;
    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, outputSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    Mx_ClearMatrix(&(net->slabs[hiddenSlabID].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[outputSlabID].supp.old_dweights), 0);

    NN_ReleaseAcquiredNet(netID);


    return netID;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Train3SlabsPerc(int netID, float rHid, float rOut, float mHid, float mOut, int datasetID, int isRandseed, int * myrandseed, float * error)
{
    int i,j,k;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }



    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        if (isRandseed != 0)
        {
            k = myintrandom(myrandseed, dataset->data.dimX);
        }
        else
        {
            k = i;
        }


        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[k][j];
        }

        NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }


        Mx_ClearMatrix(&(net->slabs[1].supp.dweights), 0);
        Mx_ClearMatrix(&(net->slabs[2].supp.dweights), 0);

        NNi_SlabBPCalcD(&net, 2);
        NNi_SlabAdjustWeightsErr(&net, 2);
        NNi_SlabApplyChanges(&net, 2, &((*net).slabs[2]), rOut, mOut);



        NNi_SlabBPCalcD(&net, 1);
        NNi_SlabAdjustWeightsErr(&net, 1);
        NNi_SlabApplyChanges(&net, 1, &((*net).slabs[1]), rHid, mHid);
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));



    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}





/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Evaluate3SlabsPerc(int netID, int datasetID, float * error)
{
    int i,j,k;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[i][j];
        }


        NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }

    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputData3SlabsPerc(int netID, int datasetID, float * error, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 2, NULL, &(output.data[0]));
        }


        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            dataOut[index] = output.data[j];
            index++;
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputDataWithoutDead3SlabsPerc(int netID, int datasetID, int * hid1Enabled, float * error, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagateWithoutTransfer(&net, 1);
        for (j = 0; j < net->slabs[1].nNeurons; j++)
        {
            if (hid1Enabled[j] == 0)
            {
                net->slabs[1].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 1);

        NNi_SlabPropagate(&net, 2);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 2, NULL, &(output.data[0]));
        }


        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            dataOut[index] = output.data[j];
            index++;
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputDataEnh3SlabsPerc(int netID, int datasetID, float * error, float * dataOut, float * r, float * r2)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    zVectorType rav, rav2, rq1, rq2, rq3;
    zMatrixElementType rqHelp1, rqHelp2;
    zVectorType r2av, r2q1, r2q2;
    zMatrixElementType r2qHelp1;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);


    Mx_CreateVector(&rav, net->totalOutputNeurons);
    Mx_CreateVector(&rav2, net->totalOutputNeurons);
    Mx_CreateVector(&rq1, net->totalOutputNeurons);
    Mx_CreateVector(&rq2, net->totalOutputNeurons);
    Mx_CreateVector(&rq3, net->totalOutputNeurons);

    Mx_CreateVector(&r2av, net->totalOutputNeurons);
    Mx_CreateVector(&r2q1, net->totalOutputNeurons);
    Mx_CreateVector(&r2q2, net->totalOutputNeurons);


    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);


        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 2, NULL, &(output.data[0]));
        }


        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            dataOut[index] = output.data[j];
            index++;
        }

        //for r
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            rav.data[j] += output.data[j];
            rav2.data[j] += outputD.data[j];
            rq1.data[j] += output.data[j] * outputD.data[j];
            rq2.data[j] += output.data[j] * output.data[j];
            rq3.data[j] += outputD.data[j] * outputD.data[j];
        }

        //for r2
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            r2av.data[j] += (outputD.data[j] - output.data[j]) * (outputD.data[j] - output.data[j]);

            r2q1.data[j] += outputD.data[j] * outputD.data[j];
            r2q2.data[j] += outputD.data[j];
        }

    }

    //save error
    *error = Error;

    //calc r
    for (j = 0; j < net->totalOutputNeurons; j++)
    {
        rq1.data[j] = rq1.data[j] * dataset->data.dimX - rav.data[j] * rav2.data[j];

        rqHelp1 = dataset->data.dimX * rq2.data[j] - rav.data[j] * rav.data[j];
        rqHelp2 = dataset->data.dimX * rq3.data[j] - rav2.data[j] * rav2.data[j];

        if (rqHelp1 <= 0 || rqHelp2 <= 0)
        {
            r[j] = 0;
        }
        else
        {
            r[j] = rq1.data[j] / sqrt(rqHelp1) / sqrt(rqHelp2);
        }
    }


    //calc r2
    for (j = 0; j < net->totalOutputNeurons; j++)
    {

        r2qHelp1 = r2q1.data[j] - r2q2.data[j] * r2q2.data[j] / dataset->data.dimX;

        if (r2qHelp1 == 0)
        {
            r2[j] = 0;
        }
        else
        {
            r2[j] = 1 - r2av.data[j] / r2qHelp1;
        }
    }

    //free r2
    Mx_FreeVector(&r2av);
    Mx_FreeVector(&r2q1);
    Mx_FreeVector(&r2q2);


    //free r
    Mx_FreeVector(&rav);
    Mx_FreeVector(&rav2);
    Mx_FreeVector(&rq1);
    Mx_FreeVector(&rq2);
    Mx_FreeVector(&rq3);


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputDataMatrix3SlabsPerc(int netID, int datasetID, float * error, zMatrixType * matrixOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 2, NULL, &(output.data[0]));
        }


        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            matrixOut->data[i][j] = output.data[j];
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputDataWithoutDeadMatrix3SlabsPerc(int netID, int datasetID, int * hid1Enabled, float * error, zMatrixType * matrixOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagateWithoutTransfer(&net, 1);
        for (j = 0; j < net->slabs[1].nNeurons; j++)
        {
            if (hid1Enabled[j] == 0)
            {
                net->slabs[1].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 1);

        NNi_SlabPropagate(&net, 2);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 2, NULL, &(output.data[0]));
        }


        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            matrixOut->data[i][j] = output.data[j];
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputDataEnhMatrix3SlabsPerc(int netID, int datasetID, float * error, zMatrixType * matrixOut, float * r, float * r2)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    zVectorType rav, rav2, rq1, rq2, rq3;
    zMatrixElementType rqHelp1, rqHelp2;
    zVectorType r2av, r2q1, r2q2;
    zMatrixElementType r2qHelp1;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);


    Mx_CreateVector(&rav, net->totalOutputNeurons);
    Mx_CreateVector(&rav2, net->totalOutputNeurons);
    Mx_CreateVector(&rq1, net->totalOutputNeurons);
    Mx_CreateVector(&rq2, net->totalOutputNeurons);
    Mx_CreateVector(&rq3, net->totalOutputNeurons);

    Mx_CreateVector(&r2av, net->totalOutputNeurons);
    Mx_CreateVector(&r2q1, net->totalOutputNeurons);
    Mx_CreateVector(&r2q2, net->totalOutputNeurons);


    Error = 0;
//	index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);


        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 2, NULL, &(output.data[0]));
        }


        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            matrixOut->data[i][j] = output.data[j];
        }

        //for r
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            rav.data[j] += output.data[j];
            rav2.data[j] += outputD.data[j];
            rq1.data[j] += output.data[j] * outputD.data[j];
            rq2.data[j] += output.data[j] * output.data[j];
            rq3.data[j] += outputD.data[j] * outputD.data[j];
        }

        //for r2
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            r2av.data[j] += (outputD.data[j] - output.data[j]) * (outputD.data[j] - output.data[j]);

            r2q1.data[j] += outputD.data[j] * outputD.data[j];
            r2q2.data[j] += outputD.data[j];
        }

    }

    //save error
    *error = Error;

    //calc r
    for (j = 0; j < net->totalOutputNeurons; j++)
    {
        rq1.data[j] = rq1.data[j] * dataset->data.dimX - rav.data[j] * rav2.data[j];

        rqHelp1 = dataset->data.dimX * rq2.data[j] - rav.data[j] * rav.data[j];
        rqHelp2 = dataset->data.dimX * rq3.data[j] - rav2.data[j] * rav2.data[j];

        if (rqHelp1 <= 0 || rqHelp2 <= 0)
        {
            r[j] = 0;
        }
        else
        {
            r[j] = rq1.data[j] / sqrt(rqHelp1) / sqrt(rqHelp2);
        }
    }


    //calc r2
    for (j = 0; j < net->totalOutputNeurons; j++)
    {

        r2qHelp1 = r2q1.data[j] - r2q2.data[j] * r2q2.data[j] / dataset->data.dimX;

        if (r2qHelp1 == 0)
        {
            r2[j] = 0;
        }
        else
        {
            r2[j] = 1 - r2av.data[j] / r2qHelp1;
        }
    }

    //free r2
    Mx_FreeVector(&r2av);
    Mx_FreeVector(&r2q1);
    Mx_FreeVector(&r2q2);


    //free r
    Mx_FreeVector(&rav);
    Mx_FreeVector(&rav2);
    Mx_FreeVector(&rq1);
    Mx_FreeVector(&rq2);
    Mx_FreeVector(&rq3);


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int EvaluateR3SlabsPerc(int netID, int datasetID, float * r)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    zVectorType av, av2, q1, q2, q3;
    zMatrixElementType qHelp1, qHelp2;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;

    //coeffs for calculating r
    Mx_CreateVector(&av, net->totalOutputNeurons);
    Mx_CreateVector(&av2, net->totalOutputNeurons);
    Mx_CreateVector(&q1, net->totalOutputNeurons);
    Mx_CreateVector(&q2, net->totalOutputNeurons);
    Mx_CreateVector(&q3, net->totalOutputNeurons);


    //
    for (i = 0; i < dataset->data.dimX; i++)
    {
        k = i;


        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[i][j];
        }

        NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            av.data[j] += output.data[j];
            av2.data[j] += outputD.data[j];
            q1.data[j] += output.data[j] * outputD.data[j];
            q2.data[j] += output.data[j] * output.data[j];
            q3.data[j] += outputD.data[j] * outputD.data[j];
        }
    }


    //
    for (j = 0; j < net->totalOutputNeurons; j++)
    {
        q1.data[j] = q1.data[j] * dataset->data.dimX - av.data[j] * av2.data[j];

        qHelp1 = dataset->data.dimX * q2.data[j] - av.data[j] * av.data[j];
        qHelp2 = dataset->data.dimX * q3.data[j] - av2.data[j] * av2.data[j];

        if (qHelp1 <= 0 || qHelp2 <= 0)
        {
            r[j] = 0;
        }
        else
        {
            r[j] = q1.data[j] / sqrt(qHelp1) / sqrt(qHelp2);
        }
    }

    //
    Mx_FreeVector(&av);
    Mx_FreeVector(&av2);
    Mx_FreeVector(&q1);
    Mx_FreeVector(&q2);
    Mx_FreeVector(&q3);

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int EvaluateR23SlabsPerc(int netID, int datasetID, float * r2)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    zVectorType av, q1, q2;
    zMatrixElementType qHelp1;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;

    //coeffs for calculating r
    Mx_CreateVector(&av, net->totalOutputNeurons);
    Mx_CreateVector(&q1, net->totalOutputNeurons);
    Mx_CreateVector(&q2, net->totalOutputNeurons);

    //
    for (i = 0; i < dataset->data.dimX; i++)
    {
        k = i;


        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[i][j];
        }

        NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            av.data[j] += (outputD.data[j] - output.data[j]) * (outputD.data[j] - output.data[j]);

            q1.data[j] += outputD.data[j] * outputD.data[j];
            q2.data[j] += outputD.data[j];
        }
    }


    //
    for (j = 0; j < net->totalOutputNeurons; j++)
    {

        qHelp1 = q1.data[j] - q2.data[j] * q2.data[j] / dataset->data.dimX;

        if (qHelp1 == 0)
        {
            r2[j] = 0;
        }
        else
        {
            r2[j] = 1 - av.data[j] / qHelp1;
        }
    }

    //
    Mx_FreeVector(&av);
    Mx_FreeVector(&q1);
    Mx_FreeVector(&q2);

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



int GetLayerWeightsSize(int netID, int layerID)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;
    int res;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    //
    res = net->slabs[layerID].weights.dimX * net->slabs[layerID].weights.dimY;

    NN_ReleaseAcquiredNet(netID);

    return res;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetWeightsPerc(int netID, int layerID, float * weightsHidden)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;
    int index, i, j;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    //
    index = 0;
    for (i = 0; i < net->slabs[layerID].weights.dimX; i++)
        for (j = 0; j < net->slabs[layerID].weights.dimY; j++)
        {
            weightsHidden[index] = net->slabs[layerID].weights.data[i][j];
            index++;
        }


    NN_ReleaseAcquiredNet(netID);

    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetWeightsDimPerc(int netID, int layerID, int * dimX, int * dimY)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    //
    *dimX = net->slabs[layerID].weights.dimX;
    *dimY = net->slabs[layerID].weights.dimY;


    NN_ReleaseAcquiredNet(netID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int SetWeightsPerc(int netID, int layerID, float * weightsHidden)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;
    int index, i, j;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    //
    index = 0;
    for (i = 0; i < net->slabs[layerID].weights.dimX; i++)
        for (j = 0; j < net->slabs[layerID].weights.dimY; j++)
        {
            net->slabs[layerID].weights.data[i][j] = weightsHidden[index];
            index++;
        }


    NN_ReleaseAcquiredNet(netID);

    return 0;
}







/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetHiddenSlabData3SlabsPerc(int netID, int datasetID, int slabID, int neuronID, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    if (slabID < 1 || slabID > 3)
        return ZERROR_BAD_SLAB_ID;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    if (neuronID < 0 || neuronID >= net->slabs[slabID].nNeurons)
    {
        NN_ReleaseAcquiredNet(netID);
        DS_ReleaseAcquiredDataset(datasetID);
        return ZERROR_BAD_NEURON_ID;
    }


//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, slabID);

        //transfer back
        dataOut[index] = net->slabs[slabID].outputs.data[neuronID];
        index++;
    }


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetHiddenSlabDataWithoutDead3SlabsPerc(int netID, int datasetID, int slabID, int neuronID, int * hid1Enabled, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    if (slabID < 1 || slabID > 3)
        return ZERROR_BAD_SLAB_ID;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    if (neuronID < 0 || neuronID >= net->slabs[slabID].nNeurons)
    {
        NN_ReleaseAcquiredNet(netID);
        DS_ReleaseAcquiredDataset(datasetID);
        return ZERROR_BAD_NEURON_ID;
    }


//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);

        NNi_SlabPropagateWithoutTransfer(&net, slabID);
        for (j = 0; j < net->slabs[slabID].nNeurons; j++)
        {
            if (hid1Enabled[j] == 0)
            {
                net->slabs[slabID].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, slabID);

        //transfer back
        dataOut[index] = net->slabs[slabID].outputs.data[neuronID];
        index++;
    }


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}










/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Create31LayerPerc(int nInput, int nHidden, int nOutput, int hiddenFunc, int outputFunc, float w, int randseed)
{
    zInt netID;
    zInt inputSlabID;
    zInt hiddenSlabID;
    zInt outputSlabID;
    zFunctionResult resultID;
    zInt IH1LinkID;
    zInt H1OLinkID;
    zInt IOLinkID;
    NeuroNetStruct * net;
    zInt myRandseed;

    netID = NN_CreateNet();
    if (netID < 0)
    {
        return netID;
    }

    inputSlabID = NN_CreateSlab(netID, NN_SLAB_INPUT, nInput);
    if (inputSlabID < 0)
        return inputSlabID;

    hiddenSlabID = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden);
    if (hiddenSlabID < 0)
        return hiddenSlabID;

    outputSlabID = NN_CreateSlab(netID, NN_SLAB_OUTPUT, nOutput);
    if (outputSlabID < 0)
        return outputSlabID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID, hiddenFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, outputSlabID, outputFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    IH1LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID);
    if (IH1LinkID < 0)
        return IH1LinkID;

    H1OLinkID = NN_CreateLink(netID, hiddenSlabID, outputSlabID);
    if (H1OLinkID < 0)
        return H1OLinkID;

    IOLinkID = NN_CreateLink(netID, inputSlabID, outputSlabID);
    if (IOLinkID < 0)
        return IOLinkID;

    myRandseed = randseed;
    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, outputSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    Mx_ClearMatrix(&(net->slabs[hiddenSlabID].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[outputSlabID].supp.old_dweights), 0);

    NN_ReleaseAcquiredNet(netID);

    return ZERROR_NO_ERROR;
}





















































/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Create3SlabsRecurrentHiddenPerc(int nInput, int nHidden, int nOutput, int hiddenFunc, int outputFunc, float w, int randseed, int delay)
{
    zInt netID;
    zInt inputSlabID;
    zInt hiddenSlabID;
    zInt outputSlabID;
    zInt recurrentSlabID;
    zFunctionResult resultID;
    zInt IH1LinkID;
    zInt H1OLinkID;
    NeuroNetStruct * net;
    zInt myRandseed;
    zInt iDelay;

    //
    if (delay < 0)
    {
        return ZERROR_LIB_BAD_DELAY;
    }


    //
    netID = NN_CreateNet();
    if (netID < 0)
    {
        return netID;
    }


    inputSlabID = NN_CreateSlab(netID, NN_SLAB_INPUT, nInput);
    if (inputSlabID < 0)
        return inputSlabID;

    hiddenSlabID = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden);
    if (hiddenSlabID < 0)
        return hiddenSlabID;

    outputSlabID = NN_CreateSlab(netID, NN_SLAB_OUTPUT, nOutput);
    if (outputSlabID < 0)
        return outputSlabID;


    //create recurrent slab
    recurrentSlabID = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden);
    if (recurrentSlabID < 0)
        return recurrentSlabID;

    //create delays
    for (iDelay = 0; iDelay < delay; iDelay++)
    {
        //
        resultID = NN_CreateSlab(netID, NN_SLAB_INPUT, nHidden);
        if (resultID < 0)
            return resultID;
    }


    //
    resultID = NN_SetSlabFunc(netID, hiddenSlabID, hiddenFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, outputSlabID, outputFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, recurrentSlabID, hiddenFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    //delay Slab's function
    for (iDelay = 0; iDelay < delay; iDelay++)
    {
        //
        resultID = NN_SetSlabFunc(netID, 4 + iDelay, NN_SLAB_FUNCTION_IDENTITY);
        if (resultID < 0)
            return resultID;
    }


    //
    IH1LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID);
    if (IH1LinkID < 0)
        return IH1LinkID;

    H1OLinkID = NN_CreateLink(netID, hiddenSlabID, outputSlabID);
    if (H1OLinkID < 0)
        return H1OLinkID;

    //link recurrent slab to hidden
    resultID = NN_CreateLink(netID, recurrentSlabID, hiddenSlabID);


    if (iDelay > 0)
    {

        resultID = NN_CreateLink(netID, hiddenSlabID, 3 + delay);
        if (resultID < 0)
            return resultID;

        for (iDelay = delay; iDelay >= 1; iDelay--)
        {
            //
            resultID = NN_CreateLink(netID, 3 + iDelay, 3 + iDelay - 1);
            if (resultID < 0)
                return resultID;
        }
    }



    //
    myRandseed = randseed;
    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, outputSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;


/*	resultID = NN_SlabGenerateRandomWeights(netID, recurrentSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;
*/

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    Mx_ClearMatrix(&(net->slabs[hiddenSlabID].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[outputSlabID].supp.old_dweights), 0);

    Mx_ClearMatrix(&(net->slabs[recurrentSlabID].weights), 0);
    Mx_ClearMatrix(&(net->slabs[recurrentSlabID].supp.old_dweights), 0);
    Mx_ClearVector(&(net->slabs[recurrentSlabID].outputs), 0);
    Mx_ClearVector(&(net->slabs[recurrentSlabID].inputs), 0);

    for (iDelay = 0; iDelay < delay; iDelay++)
    {
        Mx_ClearMatrix(&(net->slabs[4 + iDelay].weights), 0);
        Mx_ClearMatrix(&(net->slabs[4 + iDelay].supp.old_dweights), 0);
        Mx_ClearVector(&(net->slabs[4 + iDelay].outputs), 0);
        Mx_ClearVector(&(net->slabs[4 + iDelay].inputs), 0);
    }

    NN_ReleaseAcquiredNet(netID);


    return netID;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Train3SlabsRecurrentPerc(int netID, float rHid, float rOutput, float rRecurrent, float mHid, float mOutput, float mRecurrent, int datasetID, float transferCoeff, float * error)
{
    int i, j, k, iD;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    int iDelay;

    int backIndex;
    backIndex = 0;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }



    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Mx_ClearVector(&(net->slabs[3].outputs), 0);
    Mx_ClearVector(&(net->slabs[3].inputs), 0);

    for (iDelay = 4; iDelay < net->maxSlabsAllocated; iDelay++)
    {
        Mx_ClearVector(&(net->slabs[iDelay].outputs), 0);
        Mx_ClearVector(&(net->slabs[iDelay].inputs), 0);
    }


    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {

        //
        k = i;

        //
        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);		//input

        NNi_SlabPropagate(&net, 0);					//this is input slab
        NNi_SlabPropagate(&net, 3);					//this is recurrent slab
        NNi_SlabPropagate(&net, 1);					//this is hidden slab
        NNi_SlabPropagate(&net, 2);

        //now modify state of 3
        if (net->maxSlabsAllocated > 4)
        {
//			for (j = net->maxSlabsAllocated - 1; j >= 4; j--)
//				NNi_SlabPropagate(&net, j);


            //at least one delay block is present
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[4].outputs), &(net->slabs[3].inputs), transferCoeff, 1);

            for (iD = 4; iD < net->maxSlabsAllocated - 1; iD++)
            {
                //NNi_SlabPropagate(&net, iD);
                //set outputs
                Mx_CopyVector(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].outputs));
                //Mx_CopyVectorWithTransparency(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].inputs), transferCoeff);
            }

            //Mx_CopyVectorWithTransparency(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].inputs), transferCoeff);
            Mx_CopyVector(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].outputs));

        }
        else
        {
            //copy right from hidden
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[1].outputs), &(net->slabs[3].inputs), transferCoeff, 1);
        }




        //
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[k][j];
        }

        NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }

/*		if (backData != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                backData[backIndex] = output.data[j];
                backIndex++;
            }
        }
*/

        Mx_ClearMatrix(&(net->slabs[1].supp.dweights), 0);				//hidden
        Mx_ClearMatrix(&(net->slabs[2].supp.dweights), 0);				//out
        Mx_ClearMatrix(&(net->slabs[3].supp.dweights), 0);				//recurr


        NNi_SlabBPCalcD(&net, 2);
        NNi_SlabAdjustWeightsErr(&net, 2);
        NNi_SlabApplyChanges(&net, 2, &((*net).slabs[2]), rOutput, mOutput);


        NNi_SlabBPCalcD(&net, 1);
        NNi_SlabAdjustWeightsErr(&net, 1);
        NNi_SlabApplyChanges(&net, 1, &((*net).slabs[1]), rHid, mHid);

        NNi_SlabBPCalcD(&net, 3);
        NNi_SlabAdjustWeightsErr(&net, 3);
        NNi_SlabApplyChanges(&net, 3, &((*net).slabs[3]), rRecurrent, mRecurrent);
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));



    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Train3SlabsRecurrentIterativePerc(int netID, float rHid, float rOut, float rRecurrent, float mHid, float mOut, float mRecurrent, int datasetID, float transferCoeff, float * error)
{
    int i, j, k, iD;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD, currentPoints;

    int iDelay;

    int backIndex;
    backIndex = 0;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }



    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);
    Mx_CreateVector(&(currentPoints), net->totalOutputNeurons);

    for (i = 0; i < net->totalOutputNeurons; i++)
        currentPoints.data[i] = dataset->data.data[0][i];

    Mx_ClearVector(&(net->slabs[3].outputs), 0);
    Mx_ClearVector(&(net->slabs[3].inputs), 0);

    for (iDelay = 4; iDelay < net->maxSlabsAllocated; iDelay++)
    {
        Mx_ClearVector(&(net->slabs[iDelay].outputs), 0);
        Mx_ClearVector(&(net->slabs[iDelay].inputs), 0);
    }


    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {

        NNi_SetSlabInputs(&net, 0, &(currentPoints.data[0]));		//input

        //
        k = i;

        //

        NNi_SlabPropagate(&net, 0);					//this is input slab
        NNi_SlabPropagate(&net, 3);					//this is recurrent slab
        NNi_SlabPropagate(&net, 1);					//this is hidden slab
        NNi_SlabPropagate(&net, 2);

        //now modify state of 3
        if (net->maxSlabsAllocated > 4)
        {
//			for (j = net->maxSlabsAllocated - 1; j >= 4; j--)
//				NNi_SlabPropagate(&net, j);


            //at least one delay block is present
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[4].outputs), &(net->slabs[3].inputs), transferCoeff, 1);

            for (iD = 4; iD < net->maxSlabsAllocated - 1; iD++)
            {
                //NNi_SlabPropagate(&net, iD);
                //set outputs
                Mx_CopyVector(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].outputs));
                //Mx_CopyVectorWithTransparency(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].inputs), transferCoeff);
            }

            //Mx_CopyVectorWithTransparency(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].inputs), transferCoeff);
            Mx_CopyVector(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].outputs));

        }
        else
        {
            //copy right from hidden
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[1].outputs), &(net->slabs[3].inputs), transferCoeff, 1);
        }




        //
/*
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[k][j];
        }


        NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));
*/

        //
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[k][j];
        }

        NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }


//		NNi_SlabEvaluate(&net, 2, &(dataset->dataOut.data[k][0]), &(output.data[0]));

/*		for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[k][j];
        }
*/


/*		if (backData != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                backData[backIndex] = output.data[j];
                backIndex++;
            }
        }
*/

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            currentPoints.data[j] = output.data[j];
        }


        Mx_ClearMatrix(&(net->slabs[1].supp.dweights), 0);				//hidden
        Mx_ClearMatrix(&(net->slabs[2].supp.dweights), 0);				//out
        Mx_ClearMatrix(&(net->slabs[3].supp.dweights), 0);				//recurr


        NNi_SlabBPCalcD(&net, 2);
        NNi_SlabAdjustWeightsErr(&net, 2);
        NNi_SlabApplyChanges(&net, 2, &((*net).slabs[2]), rOut, mOut);

        NNi_SlabBPCalcD(&net, 1);
        NNi_SlabAdjustWeightsErr(&net, 1);
        NNi_SlabApplyChanges(&net, 1, &((*net).slabs[1]), rHid, mHid);

        NNi_SlabBPCalcD(&net, 3);
        NNi_SlabAdjustWeightsErr(&net, 3);
        NNi_SlabApplyChanges(&net, 3, &((*net).slabs[3]), rRecurrent, mRecurrent);
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));
    Mx_FreeVector(&(currentPoints));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputData3SlabsRecurrentPerc(int netID, int datasetID, float transferCoeff, float * error, float * backData)
{
    int i, j, k, iD;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    int iDelay;

    int backIndex;
    backIndex = 0;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }



    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Mx_ClearVector(&(net->slabs[3].outputs), 0);
    Mx_ClearVector(&(net->slabs[3].inputs), 0);

    for (iDelay = 4; iDelay < net->maxSlabsAllocated; iDelay++)
    {
        Mx_ClearVector(&(net->slabs[iDelay].outputs), 0);
        Mx_ClearVector(&(net->slabs[iDelay].inputs), 0);
    }


    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {

        //
        k = i;

        //
        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);		//input

        NNi_SlabPropagate(&net, 0);					//this is input slab
        NNi_SlabPropagate(&net, 3);					//this is recurrent slab
        NNi_SlabPropagate(&net, 1);					//this is hidden slab
        NNi_SlabPropagate(&net, 2);

        //now modify state of 3
        if (net->maxSlabsAllocated > 4)
        {
//			for (j = net->maxSlabsAllocated - 1; j >= 4; j--)
//				NNi_SlabPropagate(&net, j);


            //at least one delay block is present
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[4].outputs), &(net->slabs[3].inputs), transferCoeff, 1);

            for (iD = 4; iD < net->maxSlabsAllocated - 1; iD++)
            {
                //NNi_SlabPropagate(&net, iD);
                //set outputs
                Mx_CopyVector(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].outputs));
                //Mx_CopyVectorWithTransparency(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].inputs), transferCoeff);
            }

            //Mx_CopyVectorWithTransparency(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].inputs), transferCoeff);
            Mx_CopyVector(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].outputs));

        }
        else
        {
            //copy right from hidden
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[1].outputs), &(net->slabs[3].inputs), transferCoeff, 1);
        }




        //
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[k][j];
        }

        NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }

        if (backData != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                backData[backIndex] = output.data[j];
                backIndex++;
            }
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));



    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputData3SlabsRecurrentIterativePerc(int netID, int datasetID, float transferCoeff, float * error, float * backData)
{
    int i, j, k, iD;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD, currentPoints;

    int iDelay;

    int backIndex;
    backIndex = 0;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }



    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);
    Mx_CreateVector(&(currentPoints), net->totalOutputNeurons);

    for (i = 0; i < net->totalOutputNeurons; i++)
        currentPoints.data[i] = dataset->data.data[0][i];

    Mx_ClearVector(&(net->slabs[3].outputs), 0);
    Mx_ClearVector(&(net->slabs[3].inputs), 0);

    for (iDelay = 4; iDelay < net->maxSlabsAllocated; iDelay++)
    {
        Mx_ClearVector(&(net->slabs[iDelay].outputs), 0);
        Mx_ClearVector(&(net->slabs[iDelay].inputs), 0);
    }

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {

        //
        k = i;

        //
        NNi_SetSlabInputs(&net, 0, currentPoints.data);		//input

        NNi_SlabPropagate(&net, 0);					//this is input slab
        NNi_SlabPropagate(&net, 3);					//this is recurrent slab
        NNi_SlabPropagate(&net, 1);					//this is hidden slab
        NNi_SlabPropagate(&net, 2);

        //now modify state of 3
        if (net->maxSlabsAllocated > 4)
        {
//			for (j = net->maxSlabsAllocated - 1; j >= 4; j--)
//				NNi_SlabPropagate(&net, j);


            //at least one delay block is present
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[4].outputs), &(net->slabs[3].inputs), transferCoeff, 1);

            for (iD = 4; iD < net->maxSlabsAllocated - 1; iD++)
            {
                //NNi_SlabPropagate(&net, iD);
                //set outputs
                Mx_CopyVector(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].outputs));
                //Mx_CopyVectorWithTransparency(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].inputs), transferCoeff);
            }

            //Mx_CopyVectorWithTransparency(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].inputs), transferCoeff);
            Mx_CopyVector(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].outputs));

        }
        else
        {
            //copy right from hidden
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[1].outputs), &(net->slabs[3].inputs), transferCoeff, 1);
        }




        //
/*
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[k][j];
        }


        NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));
*/

        NNi_SlabEvaluate(&net, 2, &(dataset->dataOut.data[k][0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }


        if (backData != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                backData[backIndex] = output.data[j];
                backIndex++;
            }
        }


        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            currentPoints.data[j] = output.data[j];
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));
    Mx_FreeVector(&(currentPoints));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Evaluate3SlabsRecurrentPerc(int netID, int datasetID, float transferCoeff, float * error)
{
    int i, j, k, iD;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    int iDelay;

    int backIndex;
    backIndex = 0;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }



    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Mx_ClearVector(&(net->slabs[3].outputs), 0);
    Mx_ClearVector(&(net->slabs[3].inputs), 0);

    for (iDelay = 4; iDelay < net->maxSlabsAllocated; iDelay++)
    {
        Mx_ClearVector(&(net->slabs[iDelay].outputs), 0);
        Mx_ClearVector(&(net->slabs[iDelay].inputs), 0);
    }


    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {

        //
        k = i;

        //
        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);		//input

        NNi_SlabPropagate(&net, 0);					//this is input slab
        NNi_SlabPropagate(&net, 3);					//this is recurrent slab
        NNi_SlabPropagate(&net, 1);					//this is hidden slab
        NNi_SlabPropagate(&net, 2);

        //now modify state of 3
        if (net->maxSlabsAllocated > 4)
        {
//			for (j = net->maxSlabsAllocated - 1; j >= 4; j--)
//				NNi_SlabPropagate(&net, j);


            //at least one delay block is present
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[4].outputs), &(net->slabs[3].inputs), transferCoeff, 1);

            for (iD = 4; iD < net->maxSlabsAllocated - 1; iD++)
            {
                //NNi_SlabPropagate(&net, iD);
                //set outputs
                Mx_CopyVector(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].outputs));
                //Mx_CopyVectorWithTransparency(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].inputs), transferCoeff);
            }

            //Mx_CopyVectorWithTransparency(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].inputs), transferCoeff);
            Mx_CopyVector(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].outputs));

        }
        else
        {
            //copy right from hidden
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[1].outputs), &(net->slabs[3].inputs), transferCoeff, 1);
        }




        //
        if (dataset->dataOut.dimX != 0 &&
            dataset->dataOut.dimY != 0)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[k][j];
            }

            NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 2, NULL, &(output.data[0]));
        }

/*		for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[k][j];
        }

        NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }
*/
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));



    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Evaluate3SlabsRecurrentIterativePerc(int netID, int datasetID, float transferCoeff, float * error)
{
    int i, j, k, iD;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD, currentPoints;

    int iDelay;
    int backIndex;
    backIndex = 0;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }



    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);
    Mx_CreateVector(&(currentPoints), net->totalOutputNeurons);

    for (i = 0; i < net->totalOutputNeurons; i++)
        currentPoints.data[i] = dataset->data.data[0][i];

    Mx_ClearVector(&(net->slabs[3].outputs), 0);
    Mx_ClearVector(&(net->slabs[3].inputs), 0);

    for (iDelay = 4; iDelay < net->maxSlabsAllocated; iDelay++)
    {
        Mx_ClearVector(&(net->slabs[iDelay].outputs), 0);
        Mx_ClearVector(&(net->slabs[iDelay].inputs), 0);
    }

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {

        //
        k = i;

        //
        NNi_SetSlabInputs(&net, 0, currentPoints.data);		//input

        NNi_SlabPropagate(&net, 0);					//this is input slab
        NNi_SlabPropagate(&net, 3);					//this is recurrent slab
        NNi_SlabPropagate(&net, 1);					//this is hidden slab
        NNi_SlabPropagate(&net, 2);

        //now modify state of 3
        if (net->maxSlabsAllocated > 4)
        {
//			for (j = net->maxSlabsAllocated - 1; j >= 4; j--)
//				NNi_SlabPropagate(&net, j);


            //at least one delay block is present
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[4].outputs), &(net->slabs[3].inputs), transferCoeff, 1);

            for (iD = 4; iD < net->maxSlabsAllocated - 1; iD++)
            {
                //NNi_SlabPropagate(&net, iD);
                //set outputs
                Mx_CopyVector(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].outputs));
                //Mx_CopyVectorWithTransparency(&(net->slabs[iD + 1].outputs), &(net->slabs[iD].inputs), transferCoeff);
            }

            //Mx_CopyVectorWithTransparency(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].inputs), transferCoeff);
            Mx_CopyVector(&(net->slabs[1].outputs), &(net->slabs[net->maxSlabsAllocated - 1].outputs));

        }
        else
        {
            //copy right from hidden
            Mx_CopyVectorWithTransparencyAndShift(&(net->slabs[1].outputs), &(net->slabs[3].inputs), transferCoeff, 1);
        }




        //
        if (dataset->dataOut.dimX != 0 &&
            dataset->dataOut.dimY != 0)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[k][j];
            }

            NNi_SlabEvaluate(&net, 2, &(dataset->dataOut.data[k][0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 2, NULL, &(output.data[0]));
        }

//		NNi_SlabEvaluate(&net, 2, &(outputD.data[0]), &(output.data[0]));


        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            currentPoints.data[j] = output.data[j];
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));
    Mx_FreeVector(&(currentPoints));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}






























































/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Create4SlabsPerc(int nInput, int nHidden, int nHidden2, int nOutput, int hiddenFunc, int hiddenFunc2, int outputFunc, float w, int randseed)
{
    zInt netID;
    zInt inputSlabID;
    zInt hiddenSlabID;
    zInt hiddenSlabID2;
    zInt outputSlabID;
    zFunctionResult resultID;
    zInt IH1LinkID;
    zInt H1H2LinkID;
    zInt H2OLinkID;
    NeuroNetStruct * net;
    zInt myRandseed;


    netID = NN_CreateNet();
    if (netID < 0)
    {
        return netID;
    }

    inputSlabID = NN_CreateSlab(netID, NN_SLAB_INPUT, nInput);
    if (inputSlabID < 0)
        return inputSlabID;

    hiddenSlabID = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden);
    if (hiddenSlabID < 0)
        return hiddenSlabID;

    hiddenSlabID2 = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden2);
    if (hiddenSlabID2 < 0)
        return hiddenSlabID2;

    outputSlabID = NN_CreateSlab(netID, NN_SLAB_OUTPUT, nOutput);
    if (outputSlabID < 0)
        return outputSlabID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID, hiddenFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID2, hiddenFunc2);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, outputSlabID, outputFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    IH1LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID);
    if (IH1LinkID < 0)
        return IH1LinkID;

    H1H2LinkID = NN_CreateLink(netID, hiddenSlabID, hiddenSlabID2);
    if (H1H2LinkID < 0)
        return H1H2LinkID;

    H2OLinkID = NN_CreateLink(netID, hiddenSlabID2, outputSlabID);
    if (H2OLinkID < 0)
        return H2OLinkID;


    myRandseed = randseed;
    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID2, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, outputSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    Mx_ClearMatrix(&(net->slabs[hiddenSlabID].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[hiddenSlabID2].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[outputSlabID].supp.old_dweights), 0);

    NN_ReleaseAcquiredNet(netID);

    return ZERROR_NO_ERROR;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Create41SlabsPerc(int nInput, int nHidden, int nHidden2, int nOutput, int hiddenFunc, int hiddenFunc2, int outputFunc, float w, int randseed)
{
    zInt netID;
    zInt inputSlabID;
    zInt hiddenSlabID;
    zInt hiddenSlabID2;
    zInt outputSlabID;
    zFunctionResult resultID;
    zInt IH1LinkID;
    zInt IH2LinkID;
    zInt H1OLinkID;
    zInt H2OLinkID;
    NeuroNetStruct * net;
    zInt myRandseed;


    netID = NN_CreateNet();
    if (netID < 0)
    {
        return netID;
    }

    inputSlabID = NN_CreateSlab(netID, NN_SLAB_INPUT, nInput);
    if (inputSlabID < 0)
        return inputSlabID;

    hiddenSlabID = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden);
    if (hiddenSlabID < 0)
        return hiddenSlabID;

    hiddenSlabID2 = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden2);
    if (hiddenSlabID2 < 0)
        return hiddenSlabID2;

    outputSlabID = NN_CreateSlab(netID, NN_SLAB_OUTPUT, nOutput);
    if (outputSlabID < 0)
        return outputSlabID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID, hiddenFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID2, hiddenFunc2);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, outputSlabID, outputFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    IH1LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID);
    if (IH1LinkID < 0)
        return IH1LinkID;

    IH2LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID2);
    if (IH2LinkID < 0)
        return IH2LinkID;

    H1OLinkID = NN_CreateLink(netID, hiddenSlabID, outputSlabID);
    if (H1OLinkID < 0)
        return H1OLinkID;

    H2OLinkID = NN_CreateLink(netID, hiddenSlabID2, outputSlabID);
    if (H2OLinkID < 0)
        return H2OLinkID;


    myRandseed = randseed;
    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID2, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, outputSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    Mx_ClearMatrix(&(net->slabs[hiddenSlabID].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[hiddenSlabID2].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[outputSlabID].supp.old_dweights), 0);

    NN_ReleaseAcquiredNet(netID);

    return ZERROR_NO_ERROR;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Create42SlabsPerc(int nInput, int nHidden, int nHidden2, int nOutput, int hiddenFunc, int hiddenFunc2, int outputFunc, float w, int randseed)
{
    zInt netID;
    zInt inputSlabID;
    zInt hiddenSlabID;
    zInt hiddenSlabID2;
    zInt outputSlabID;
    zFunctionResult resultID;
    zInt IH1LinkID;
    zInt IH2LinkID;
    zInt H1OLinkID;
    zInt H2OLinkID;
    zInt IOLinkID;
    NeuroNetStruct * net;
    zInt myRandseed;


    netID = NN_CreateNet();
    if (netID < 0)
    {
        return netID;
    }

    inputSlabID = NN_CreateSlab(netID, NN_SLAB_INPUT, nInput);
    if (inputSlabID < 0)
        return inputSlabID;

    hiddenSlabID = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden);
    if (hiddenSlabID < 0)
        return hiddenSlabID;

    hiddenSlabID2 = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden2);
    if (hiddenSlabID2 < 0)
        return hiddenSlabID2;

    outputSlabID = NN_CreateSlab(netID, NN_SLAB_OUTPUT, nOutput);
    if (outputSlabID < 0)
        return outputSlabID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID, hiddenFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID2, hiddenFunc2);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, outputSlabID, outputFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    IH1LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID);
    if (IH1LinkID < 0)
        return IH1LinkID;

    IH2LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID2);
    if (IH2LinkID < 0)
        return IH2LinkID;

    H1OLinkID = NN_CreateLink(netID, hiddenSlabID, outputSlabID);
    if (H1OLinkID < 0)
        return H1OLinkID;

    H2OLinkID = NN_CreateLink(netID, hiddenSlabID2, outputSlabID);
    if (H2OLinkID < 0)
        return H2OLinkID;

    IOLinkID = NN_CreateLink(netID, inputSlabID, outputSlabID);
    if (IOLinkID < 0)
        return IOLinkID;

    myRandseed = randseed;
    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID2, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, outputSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    Mx_ClearMatrix(&(net->slabs[hiddenSlabID].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[hiddenSlabID2].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[outputSlabID].supp.old_dweights), 0);

    NN_ReleaseAcquiredNet(netID);

    return ZERROR_NO_ERROR;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Create43SlabsPerc(int nInput, int nHidden, int nHidden2, int nOutput, int hiddenFunc, int hiddenFunc2, int outputFunc, float w, int randseed)
{
    zInt netID;
    zInt inputSlabID;
    zInt hiddenSlabID;
    zInt hiddenSlabID2;
    zInt outputSlabID;
    zFunctionResult resultID;
    zInt IH1LinkID;
    zInt H1H2LinkID;
    zInt H2OLinkID;
    zInt link1, link2, link3;
    NeuroNetStruct * net;
    zInt myRandseed;


    netID = NN_CreateNet();
    if (netID < 0)
    {
        return netID;
    }

    inputSlabID = NN_CreateSlab(netID, NN_SLAB_INPUT, nInput);
    if (inputSlabID < 0)
        return inputSlabID;

    hiddenSlabID = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden);
    if (hiddenSlabID < 0)
        return hiddenSlabID;

    hiddenSlabID2 = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden2);
    if (hiddenSlabID2 < 0)
        return hiddenSlabID2;

    outputSlabID = NN_CreateSlab(netID, NN_SLAB_OUTPUT, nOutput);
    if (outputSlabID < 0)
        return outputSlabID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID, hiddenFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID2, hiddenFunc2);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, outputSlabID, outputFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    IH1LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID);
    if (IH1LinkID < 0)
        return IH1LinkID;

    H1H2LinkID = NN_CreateLink(netID, hiddenSlabID, hiddenSlabID2);
    if (H1H2LinkID < 0)
        return H1H2LinkID;

    H2OLinkID = NN_CreateLink(netID, hiddenSlabID2, outputSlabID);
    if (H2OLinkID < 0)
        return H2OLinkID;

    link1 = NN_CreateLink(netID, inputSlabID, outputSlabID);
    if (link1 < 0)
        return link1;

    link2 = NN_CreateLink(netID, inputSlabID, hiddenSlabID2);
    if (link2 < 0)
        return link2;

    link3 = NN_CreateLink(netID, hiddenSlabID, outputSlabID);
    if (link3 < 0)
        return link3;


    //
    myRandseed = randseed;
    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID2, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, outputSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    Mx_ClearMatrix(&(net->slabs[hiddenSlabID].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[hiddenSlabID2].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[outputSlabID].supp.old_dweights), 0);

    NN_ReleaseAcquiredNet(netID);

    return ZERROR_NO_ERROR;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Train4SlabsPerc(int netID, float rHid1, float rHid2, float rOut, float mHid1, float mHid2, float mOut, int datasetID, int isRandseed, int * myrandseed, float * error)
{
    int i,j,k;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        if (isRandseed != 0)
        {
            k = myintrandom(myrandseed, dataset->data.dimX);
        }
        else
        {
            k = i;
        }


        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[k][j];
        }

        NNi_SlabEvaluate(&net, 3, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }


        Mx_ClearMatrix(&(net->slabs[1].supp.dweights), 0);
        Mx_ClearMatrix(&(net->slabs[2].supp.dweights), 0);
        Mx_ClearMatrix(&(net->slabs[3].supp.dweights), 0);


        NNi_SlabBPCalcD(&net, 3);
        NNi_SlabAdjustWeightsErr(&net, 3);
        NNi_SlabApplyChanges(&net, 3, &((*net).slabs[3]), rOut, mOut);


        NNi_SlabBPCalcD(&net, 2);
        NNi_SlabAdjustWeightsErr(&net, 2);
        NNi_SlabApplyChanges(&net, 2, &((*net).slabs[2]), rHid2, mHid2);


        NNi_SlabBPCalcD(&net, 1);
        NNi_SlabAdjustWeightsErr(&net, 1);
        NNi_SlabApplyChanges(&net, 1, &((*net).slabs[1]), rHid1, mHid1);
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Evaluate4SlabsPerc(int netID, int datasetID, float * error)
{
    int i,j,k;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;




    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }
//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[i][j];
        }

        NNi_SlabEvaluate(&net, 3, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }

    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int EvaluateR4SlabsPerc(int netID, int datasetID, float * r)
{
    int i,j,k;
    zFunctionResult resultID;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    zVectorType av, av2, q1, q2, q3;
    double qHelp1, qHelp2;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }
//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    //coeffs for calculating r
    Mx_CreateVector(&av, net->totalOutputNeurons);
    Mx_CreateVector(&av2, net->totalOutputNeurons);
    Mx_CreateVector(&q1, net->totalOutputNeurons);
    Mx_CreateVector(&q2, net->totalOutputNeurons);
    Mx_CreateVector(&q3, net->totalOutputNeurons);

    //
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[i][j];
        }

        NNi_SlabEvaluate(&net, 3, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            av.data[j] += output.data[j];
            av2.data[j] += outputD.data[j];
            q1.data[j] += output.data[j] * outputD.data[j];
            q2.data[j] += output.data[j] * output.data[j];
            q3.data[j] += outputD.data[j] * outputD.data[j];
        }

    }

    //
    for (j = 0; j < net->totalOutputNeurons; j++)
    {
        q1.data[j] = q1.data[j] * dataset->data.dimX - av.data[j] * av2.data[j];

        qHelp1 = dataset->data.dimX * q2.data[j] - av.data[j] * av.data[j];
        qHelp2 = dataset->data.dimX * q3.data[j] - av2.data[j] * av2.data[j];

        if (qHelp1 <= 0 || qHelp2 <= 0)
        {
            r[j] = 0;
        }
        else
        {
            r[j] = q1.data[j] / sqrt(qHelp1) / sqrt(qHelp2);
        }
    }


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));

    Mx_FreeVector(&av);
    Mx_FreeVector(&av2);
    Mx_FreeVector(&q1);
    Mx_FreeVector(&q2);
    Mx_FreeVector(&q3);


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int EvaluateR24SlabsPerc(int netID, int datasetID, float * r2)
{
    int i,j,k;
    zFunctionResult resultID;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    zVectorType av, q1, q2;
    zMatrixElementType qHelp1;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }
//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    //coeffs for calculating r
    Mx_CreateVector(&av, net->totalOutputNeurons);
    Mx_CreateVector(&q1, net->totalOutputNeurons);
    Mx_CreateVector(&q2, net->totalOutputNeurons);

    //
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[i][j];
        }

        NNi_SlabEvaluate(&net, 3, &(outputD.data[0]), &(output.data[0]));


        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            av.data[j] += (outputD.data[j] - output.data[j]) * (outputD.data[j] - output.data[j]);

            q1.data[j] += outputD.data[j] * outputD.data[j];
            q2.data[j] += outputD.data[j];
        }

    }

    //
    for (j = 0; j < net->totalOutputNeurons; j++)
    {

        qHelp1 = q1.data[j] - q2.data[j] * q2.data[j] / dataset->data.dimX;

        if (qHelp1 == 0)
        {
            r2[j] = 0;
        }
        else
        {
            r2[j] = 1 - av.data[j] / qHelp1;
        }
    }



    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    Mx_FreeVector(&av);
    Mx_FreeVector(&q1);
    Mx_FreeVector(&q2);


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}





/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputData4SlabsPerc(int netID, int datasetID, float * error, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }
//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 3, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 3, NULL, &(output.data[0]));
        }

        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            dataOut[index] = output.data[j];
            index++;
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputDataMatrix4SlabsPerc(int netID, int datasetID, float * error, zMatrixType * matrixOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }
//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 3, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 3, NULL, &(output.data[0]));
        }

        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            matrixOut->data[i][j] = output.data[j];
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputDataWithoutDead4SlabsPerc(int netID, int datasetID, int * hid1Enabled, int * hid2Enabled, float * error, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }
//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);

        NNi_SlabPropagateWithoutTransfer(&net, 1);
        for (j = 0; j < net->slabs[1].nNeurons; j++)
        {
            if (hid1Enabled[j] == 0)
            {
                net->slabs[1].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 1);

        NNi_SlabPropagateWithoutTransfer(&net, 2);
        for (j = 0; j < net->slabs[2].nNeurons; j++)
        {
            if (hid2Enabled[j] == 0)
            {
                net->slabs[2].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 2);

        NNi_SlabPropagate(&net, 3);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 3, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 3, NULL, &(output.data[0]));
        }

        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            dataOut[index] = output.data[j];
            index++;
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}






/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputDataEnh4SlabsPerc(int netID, int datasetID, float * error, float * dataOut, float * r, float * r2)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    zVectorType rav, rav2, rq1, rq2, rq3;
    zMatrixElementType rqHelp1, rqHelp2;
    zVectorType r2av, r2q1, r2q2;
    zMatrixElementType r2qHelp1;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);


    Mx_CreateVector(&rav, net->totalOutputNeurons);
    Mx_CreateVector(&rav2, net->totalOutputNeurons);
    Mx_CreateVector(&rq1, net->totalOutputNeurons);
    Mx_CreateVector(&rq2, net->totalOutputNeurons);
    Mx_CreateVector(&rq3, net->totalOutputNeurons);

    Mx_CreateVector(&r2av, net->totalOutputNeurons);
    Mx_CreateVector(&r2q1, net->totalOutputNeurons);
    Mx_CreateVector(&r2q2, net->totalOutputNeurons);


    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 3, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 3, NULL, &(output.data[0]));
        }

        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            dataOut[index] = output.data[j];
            index++;
        }

        //for r
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            rav.data[j] += output.data[j];
            rav2.data[j] += outputD.data[j];
            rq1.data[j] += output.data[j] * outputD.data[j];
            rq2.data[j] += output.data[j] * output.data[j];
            rq3.data[j] += outputD.data[j] * outputD.data[j];
        }

        //for r2
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            r2av.data[j] += (outputD.data[j] - output.data[j]) * (outputD.data[j] - output.data[j]);

            r2q1.data[j] += outputD.data[j] * outputD.data[j];
            r2q2.data[j] += outputD.data[j];
        }

    }

    //save error
    *error = Error;

    //calc r
    for (j = 0; j < net->totalOutputNeurons; j++)
    {
        rq1.data[j] = rq1.data[j] * dataset->data.dimX - rav.data[j] * rav2.data[j];

        rqHelp1 = dataset->data.dimX * rq2.data[j] - rav.data[j] * rav.data[j];
        rqHelp2 = dataset->data.dimX * rq3.data[j] - rav2.data[j] * rav2.data[j];

        if (rqHelp1 <= 0 || rqHelp2 <= 0)
        {
            r[j] = 0;
        }
        else
        {
            r[j] = rq1.data[j] / sqrt(rqHelp1) / sqrt(rqHelp2);
        }
    }


    //calc r2
    for (j = 0; j < net->totalOutputNeurons; j++)
    {

        r2qHelp1 = r2q1.data[j] - r2q2.data[j] * r2q2.data[j] / dataset->data.dimX;

        if (r2qHelp1 == 0)
        {
            r2[j] = 0;
        }
        else
        {
            r2[j] = 1 - r2av.data[j] / r2qHelp1;
        }
    }

    //free r2
    Mx_FreeVector(&r2av);
    Mx_FreeVector(&r2q1);
    Mx_FreeVector(&r2q2);


    //free r
    Mx_FreeVector(&rav);
    Mx_FreeVector(&rav2);
    Mx_FreeVector(&rq1);
    Mx_FreeVector(&rq2);
    Mx_FreeVector(&rq3);


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetHiddenSlabData4SlabsPerc(int netID, int datasetID, int neuronID, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    if (neuronID < 0 || neuronID >= net->slabs[2].nNeurons)
    {
        NN_ReleaseAcquiredNet(netID);
        DS_ReleaseAcquiredDataset(datasetID);
        return ZERROR_BAD_NEURON_ID;
    }


//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);

        //transfer back
        dataOut[index] = net->slabs[2].outputs.data[neuronID];
        index++;
    }


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetHiddenSlabDataWithoutDead4SlabsPerc(int netID, int datasetID, int neuronID, int * hid1Enabled, int * hid2Enabled, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    if (neuronID < 0 || neuronID >= net->slabs[2].nNeurons)
    {
        NN_ReleaseAcquiredNet(netID);
        DS_ReleaseAcquiredDataset(datasetID);
        return ZERROR_BAD_NEURON_ID;
    }


//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);

        NNi_SlabPropagateWithoutTransfer(&net, 1);
        for (j = 0; j < net->slabs[1].nNeurons; j++)
        {
            if (hid1Enabled[j] == 0)
            {
                net->slabs[1].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 1);

        NNi_SlabPropagateWithoutTransfer(&net, 2);
        for (j = 0; j < net->slabs[2].nNeurons; j++)
        {
            if (hid2Enabled[j] == 0)
            {
                net->slabs[2].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 2);

        //transfer back
        dataOut[index] = net->slabs[2].outputs.data[neuronID];
        index++;
    }


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}
















/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Create5SlabsPerc(int nInput, int nHidden, int nHidden2, int nHidden3, int nOutput, int hiddenFunc, int hiddenFunc2, int hiddenFunc3, int outputFunc, float w, int randseed)
{
    zInt netID;
    zInt inputSlabID;
    zInt hiddenSlabID;
    zInt hiddenSlabID2;
    zInt hiddenSlabID3;
    zInt outputSlabID;
    zFunctionResult resultID;
    zInt IH1LinkID;
    zInt H1H2LinkID;
    zInt H2H3LinkID;
    zInt H3OLinkID;
    NeuroNetStruct * net;
    zInt myRandseed;


    netID = NN_CreateNet();
    if (netID < 0)
    {
        return netID;
    }

    inputSlabID = NN_CreateSlab(netID, NN_SLAB_INPUT, nInput);
    if (inputSlabID < 0)
        return inputSlabID;

    hiddenSlabID = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden);
    if (hiddenSlabID < 0)
        return hiddenSlabID;

    hiddenSlabID2 = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden2);
    if (hiddenSlabID2 < 0)
        return hiddenSlabID2;

    hiddenSlabID3 = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden3);
    if (hiddenSlabID3 < 0)
        return hiddenSlabID3;

    outputSlabID = NN_CreateSlab(netID, NN_SLAB_OUTPUT, nOutput);
    if (outputSlabID < 0)
        return outputSlabID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID, hiddenFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID2, hiddenFunc2);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID3, hiddenFunc3);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, outputSlabID, outputFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    IH1LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID);
    if (IH1LinkID < 0)
        return IH1LinkID;

    H1H2LinkID = NN_CreateLink(netID, hiddenSlabID, hiddenSlabID2);
    if (H1H2LinkID < 0)
        return H1H2LinkID;

    H2H3LinkID = NN_CreateLink(netID, hiddenSlabID2, hiddenSlabID3);
    if (H2H3LinkID < 0)
        return H1H2LinkID;

    H3OLinkID = NN_CreateLink(netID, hiddenSlabID3, outputSlabID);
    if (H3OLinkID < 0)
        return H3OLinkID;

    myRandseed = randseed;
    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID2, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID3, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, outputSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    Mx_ClearMatrix(&(net->slabs[hiddenSlabID].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[hiddenSlabID2].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[hiddenSlabID3].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[outputSlabID].supp.old_dweights), 0);

    NN_ReleaseAcquiredNet(netID);

    return ZERROR_NO_ERROR;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Create51SlabsPerc(int nInput, int nHidden, int nHidden2, int nHidden3, int nOutput, int hiddenFunc, int hiddenFunc2, int hiddenFunc3, int outputFunc, float w, int randseed)
{
    zInt netID;
    zInt inputSlabID;
    zInt hiddenSlabID;
    zInt hiddenSlabID2;
    zInt hiddenSlabID3;
    zInt outputSlabID;
    zFunctionResult resultID;
    zInt IH1LinkID;
    zInt H1H2LinkID;
    zInt H2H3LinkID;
    zInt H3OLinkID;
    NeuroNetStruct * net;
    zInt myRandseed;


    netID = NN_CreateNet();
    if (netID < 0)
    {
        return netID;
    }

    inputSlabID = NN_CreateSlab(netID, NN_SLAB_INPUT, nInput);
    if (inputSlabID < 0)
        return inputSlabID;

    hiddenSlabID = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden);
    if (hiddenSlabID < 0)
        return hiddenSlabID;

    hiddenSlabID2 = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden2);
    if (hiddenSlabID2 < 0)
        return hiddenSlabID2;

    hiddenSlabID3 = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden3);
    if (hiddenSlabID3 < 0)
        return hiddenSlabID3;

    outputSlabID = NN_CreateSlab(netID, NN_SLAB_OUTPUT, nOutput);
    if (outputSlabID < 0)
        return outputSlabID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID, hiddenFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID2, hiddenFunc2);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID3, hiddenFunc3);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, outputSlabID, outputFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    IH1LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID);
    if (IH1LinkID < 0)
        return IH1LinkID;

    H1H2LinkID = NN_CreateLink(netID, hiddenSlabID, hiddenSlabID2);
    if (H1H2LinkID < 0)
        return H1H2LinkID;

    H2H3LinkID = NN_CreateLink(netID, hiddenSlabID2, hiddenSlabID3);
    if (H2H3LinkID < 0)
        return H1H2LinkID;

    H3OLinkID = NN_CreateLink(netID, hiddenSlabID3, outputSlabID);
    if (H3OLinkID < 0)
        return H3OLinkID;


    resultID = NN_CreateLink(netID, inputSlabID, outputSlabID);
    if (resultID < 0)
        return resultID;

    resultID = NN_CreateLink(netID, inputSlabID, hiddenSlabID3);
    if (resultID < 0)
        return resultID;

    resultID = NN_CreateLink(netID, inputSlabID, hiddenSlabID2);
    if (resultID < 0)
        return resultID;

    resultID = NN_CreateLink(netID, hiddenSlabID, outputSlabID);
    if (resultID < 0)
        return resultID;

    resultID = NN_CreateLink(netID, hiddenSlabID, hiddenSlabID3);
    if (resultID < 0)
        return resultID;

    resultID = NN_CreateLink(netID, hiddenSlabID2, outputSlabID);
    if (resultID < 0)
        return resultID;


    myRandseed = randseed;
    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID2, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID3, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, outputSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    Mx_ClearMatrix(&(net->slabs[hiddenSlabID].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[hiddenSlabID2].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[hiddenSlabID3].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[outputSlabID].supp.old_dweights), 0);

    NN_ReleaseAcquiredNet(netID);

    return ZERROR_NO_ERROR;
}






/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Create52SlabsPerc(int nInput, int nHidden, int nHidden2, int nHidden3, int nOutput, int hiddenFunc, int hiddenFunc2, int hiddenFunc3, int outputFunc, float w, int randseed)
{
    zInt netID;
    zInt inputSlabID;
    zInt hiddenSlabID;
    zInt hiddenSlabID2;
    zInt hiddenSlabID3;
    zInt outputSlabID;
    zFunctionResult resultID;
    zInt IH1LinkID;
    zInt H1H2LinkID;
    zInt H2H3LinkID;
    zInt H3OLinkID;
    NeuroNetStruct * net;
    zInt myRandseed;


    netID = NN_CreateNet();
    if (netID < 0)
    {
        return netID;
    }

    inputSlabID = NN_CreateSlab(netID, NN_SLAB_INPUT, nInput);
    if (inputSlabID < 0)
        return inputSlabID;

    hiddenSlabID = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden);
    if (hiddenSlabID < 0)
        return hiddenSlabID;

    hiddenSlabID2 = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden2);
    if (hiddenSlabID2 < 0)
        return hiddenSlabID2;

    hiddenSlabID3 = NN_CreateSlab(netID, NN_SLAB_HIDDEN, nHidden3);
    if (hiddenSlabID3 < 0)
        return hiddenSlabID3;

    outputSlabID = NN_CreateSlab(netID, NN_SLAB_OUTPUT, nOutput);
    if (outputSlabID < 0)
        return outputSlabID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID, hiddenFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID2, hiddenFunc2);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, hiddenSlabID3, hiddenFunc3);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SetSlabFunc(netID, outputSlabID, outputFunc);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;


    IH1LinkID = NN_CreateLink(netID, inputSlabID, hiddenSlabID);
    if (IH1LinkID < 0)
        return IH1LinkID;

    resultID = NN_CreateLink(netID, inputSlabID, hiddenSlabID2);
    if (resultID < 0)
        return resultID;

    resultID = NN_CreateLink(netID, inputSlabID, hiddenSlabID3);
    if (resultID < 0)
        return resultID;

    resultID = NN_CreateLink(netID, hiddenSlabID, outputSlabID);
    if (resultID < 0)
        return resultID;

    resultID = NN_CreateLink(netID, hiddenSlabID2, outputSlabID);
    if (resultID < 0)
        return resultID;

    H3OLinkID = NN_CreateLink(netID, hiddenSlabID3, outputSlabID);
    if (H3OLinkID < 0)
        return H3OLinkID;



    myRandseed = randseed;
    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID2, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, hiddenSlabID3, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_SlabGenerateRandomWeights(netID, outputSlabID, w, &myRandseed);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    Mx_ClearMatrix(&(net->slabs[hiddenSlabID].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[hiddenSlabID2].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[hiddenSlabID3].supp.old_dweights), 0);
    Mx_ClearMatrix(&(net->slabs[outputSlabID].supp.old_dweights), 0);

    NN_ReleaseAcquiredNet(netID);

    return ZERROR_NO_ERROR;
}






/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Train5SlabsPerc(int netID, float rHid1, float rHid2, float rHid3, float rOut, float mHid1, float mHid2, float mHid3, float mOut, int datasetID, int isRandseed, int * myrandseed, float * error)
{
    int i,j,k;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }
//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        if (isRandseed != 0)
        {
            k = myintrandom(myrandseed, dataset->data.dimX);
        }
        else
        {
            k = i;
        }


        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);
        NNi_SlabPropagate(&net, 4);


        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[k][j];
        }

//		NN_SlabEvaluate(netID, 4, &(outputD.data[0]), &(output.data[0]));
        NNi_SlabEvaluate(&net, 4, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }


        Mx_ClearMatrix(&(net->slabs[1].supp.dweights), 0);
        Mx_ClearMatrix(&(net->slabs[2].supp.dweights), 0);
        Mx_ClearMatrix(&(net->slabs[3].supp.dweights), 0);
        Mx_ClearMatrix(&(net->slabs[4].supp.dweights), 0);


        NNi_SlabBPCalcD(&net, 4);
        NNi_SlabAdjustWeightsErr(&net, 4);
        NNi_SlabApplyChanges(&net, 4, &((*net).slabs[4]), rOut, mOut);


        NNi_SlabBPCalcD(&net, 3);
        NNi_SlabAdjustWeightsErr(&net, 3);
        NNi_SlabApplyChanges(&net, 3, &((*net).slabs[3]), rHid3, mHid3);


        NNi_SlabBPCalcD(&net, 2);
        NNi_SlabAdjustWeightsErr(&net, 2);
        NNi_SlabApplyChanges(&net, 2, &((*net).slabs[2]), rHid2, mHid2);


        NNi_SlabBPCalcD(&net, 1);
        NNi_SlabAdjustWeightsErr(&net, 1);
        NNi_SlabApplyChanges(&net, 1, &((*net).slabs[1]), rHid1, mHid1);
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}






/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Evaluate5SlabsPerc(int netID, int datasetID, float * error)
{
    int i,j,k;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }
//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

//		NN_SetSlabInputs(netID, 0, dataset->data.data[k]);
        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);
        NNi_SlabPropagate(&net, 4);

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[i][j];
        }

        NNi_SlabEvaluate(&net, 4, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            Error += (outputD.data[j] - output.data[j]) *
                    (outputD.data[j] - output.data[j]);
        }

    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}





/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int EvaluateR5SlabsPerc(int netID, int datasetID, float * r)
{
    int i,j,k;
    zFunctionResult resultID;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    zVectorType av, av2, q1, q2, q3;
    double qHelp1, qHelp2;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }
//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);


    //coeffs for calculating r
    Mx_CreateVector(&av, net->totalOutputNeurons);
    Mx_CreateVector(&av2, net->totalOutputNeurons);
    Mx_CreateVector(&q1, net->totalOutputNeurons);
    Mx_CreateVector(&q2, net->totalOutputNeurons);
    Mx_CreateVector(&q3, net->totalOutputNeurons);


    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

//		NN_SetSlabInputs(netID, 0, dataset->data.data[k]);
        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);
        NNi_SlabPropagate(&net, 4);

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[i][j];
        }

        NNi_SlabEvaluate(&net, 4, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            av.data[j] += output.data[j];
            av2.data[j] += outputD.data[j];
            q1.data[j] += output.data[j] * outputD.data[j];
            q2.data[j] += output.data[j] * output.data[j];
            q3.data[j] += outputD.data[j] * outputD.data[j];
        }

    }

    //
    for (j = 0; j < net->totalOutputNeurons; j++)
    {
        q1.data[j] = q1.data[j] * dataset->data.dimX - av.data[j] * av2.data[j];

        qHelp1 = dataset->data.dimX * q2.data[j] - av.data[j] * av.data[j];
        qHelp2 = dataset->data.dimX * q3.data[j] - av2.data[j] * av2.data[j];

        if (qHelp1 <= 0 || qHelp2 <= 0)
        {
            r[j] = 0;
        }
        else
        {
            r[j] = q1.data[j] / sqrt(qHelp1) / sqrt(qHelp2);
        }
    }


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));

    Mx_FreeVector(&av);
    Mx_FreeVector(&av2);
    Mx_FreeVector(&q1);
    Mx_FreeVector(&q2);
    Mx_FreeVector(&q3);


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}







/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int EvaluateR25SlabsPerc(int netID, int datasetID, float * r2)
{
    int i,j,k;
    zFunctionResult resultID;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    zVectorType av, q1, q2;
    zMatrixElementType qHelp1;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }
//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);


    //coeffs for calculating r
    Mx_CreateVector(&av, net->totalOutputNeurons);
    Mx_CreateVector(&q1, net->totalOutputNeurons);
    Mx_CreateVector(&q2, net->totalOutputNeurons);


    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

//		NN_SetSlabInputs(netID, 0, dataset->data.data[k]);
        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);
        NNi_SlabPropagate(&net, 4);

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            outputD.data[j] = dataset->dataOut.data[i][j];
        }

        NNi_SlabEvaluate(&net, 4, &(outputD.data[0]), &(output.data[0]));

        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            av.data[j] += (outputD.data[j] - output.data[j]) * (outputD.data[j] - output.data[j]);

            q1.data[j] += outputD.data[j] * outputD.data[j];
            q2.data[j] += outputD.data[j];
        }

    }

    //
    for (j = 0; j < net->totalOutputNeurons; j++)
    {

        qHelp1 = q1.data[j] - q2.data[j] * q2.data[j] / dataset->data.dimX;

        if (qHelp1 == 0)
        {
            r2[j] = 0;
        }
        else
        {
            r2[j] = 1 - av.data[j] / qHelp1;
        }
    }



    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    Mx_FreeVector(&av);
    Mx_FreeVector(&q1);
    Mx_FreeVector(&q2);



    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputData5SlabsPerc(int netID, int datasetID, float * error, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);
        NNi_SlabPropagate(&net, 4);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 4, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 4, NULL, &(output.data[0]));
        }

        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            dataOut[index] = output.data[j];
            index++;
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputDataMatrix5SlabsPerc(int netID, int datasetID, float * error, zMatrixType * matrixOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);
        NNi_SlabPropagate(&net, 4);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 4, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 4, NULL, &(output.data[0]));
        }

        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            matrixOut->data[i][j] = output.data[j];
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputDataWithoutDead5SlabsPerc(int netID, int datasetID, int * hid1Enabled, int * hid2Enabled, int * hid3Enabled, float * error, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagateWithoutTransfer(&net, 1);
        for (j = 0; j < net->slabs[1].nNeurons; j++)
        {
            if (hid1Enabled[j] == 0)
            {
                net->slabs[1].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 1);

        NNi_SlabPropagateWithoutTransfer(&net, 2);
        for (j = 0; j < net->slabs[2].nNeurons; j++)
        {
            if (hid2Enabled[j] == 0)
            {
                net->slabs[2].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 2);

        NNi_SlabPropagateWithoutTransfer(&net, 3);
        for (j = 0; j < net->slabs[3].nNeurons; j++)
        {
            if (hid3Enabled[j] == 0)
            {
                net->slabs[3].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 3);

        NNi_SlabPropagate(&net, 4);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 4, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 4, NULL, &(output.data[0]));
        }

        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            dataOut[index] = output.data[j];
            index++;
        }
    }

    *error = Error;

    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetOutputDataEnh5SlabsPerc(int netID, int datasetID, float * error, float * dataOut, float * r, float * r2)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;

    zVectorType rav, rav2, rq1, rq2, rq3;
    zMatrixElementType rqHelp1, rqHelp2;
    zVectorType r2av, r2q1, r2q2;
    zMatrixElementType r2qHelp1;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }

//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);


    Mx_CreateVector(&rav, net->totalOutputNeurons);
    Mx_CreateVector(&rav2, net->totalOutputNeurons);
    Mx_CreateVector(&rq1, net->totalOutputNeurons);
    Mx_CreateVector(&rq2, net->totalOutputNeurons);
    Mx_CreateVector(&rq3, net->totalOutputNeurons);

    Mx_CreateVector(&r2av, net->totalOutputNeurons);
    Mx_CreateVector(&r2q1, net->totalOutputNeurons);
    Mx_CreateVector(&r2q2, net->totalOutputNeurons);


    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);
        NNi_SlabPropagate(&net, 4);

        if (dataset->dataOut.data != NULL)
        {
            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                outputD.data[j] = dataset->dataOut.data[i][j];
            }

            NNi_SlabEvaluate(&net, 4, &(outputD.data[0]), &(output.data[0]));

            for (j = 0; j < net->totalOutputNeurons; j++)
            {
                Error += (outputD.data[j] - output.data[j]) *
                        (outputD.data[j] - output.data[j]);
            }
        }
        else
        {
            NNi_SlabEvaluate(&net, 4, NULL, &(output.data[0]));
        }

        //transfer back
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            dataOut[index] = output.data[j];
            index++;
        }

        //for r
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            rav.data[j] += output.data[j];
            rav2.data[j] += outputD.data[j];
            rq1.data[j] += output.data[j] * outputD.data[j];
            rq2.data[j] += output.data[j] * output.data[j];
            rq3.data[j] += outputD.data[j] * outputD.data[j];
        }

        //for r2
        for (j = 0; j < net->totalOutputNeurons; j++)
        {
            r2av.data[j] += (outputD.data[j] - output.data[j]) * (outputD.data[j] - output.data[j]);

            r2q1.data[j] += outputD.data[j] * outputD.data[j];
            r2q2.data[j] += outputD.data[j];
        }

    }

    //save error
    *error = Error;

    //calc r
    for (j = 0; j < net->totalOutputNeurons; j++)
    {
        rq1.data[j] = rq1.data[j] * dataset->data.dimX - rav.data[j] * rav2.data[j];

        rqHelp1 = dataset->data.dimX * rq2.data[j] - rav.data[j] * rav.data[j];
        rqHelp2 = dataset->data.dimX * rq3.data[j] - rav2.data[j] * rav2.data[j];

        if (rqHelp1 <= 0 || rqHelp2 <= 0)
        {
            r[j] = 0;
        }
        else
        {
            r[j] = rq1.data[j] / sqrt(rqHelp1) / sqrt(rqHelp2);
        }
    }


    //calc r2
    for (j = 0; j < net->totalOutputNeurons; j++)
    {

        r2qHelp1 = r2q1.data[j] - r2q2.data[j] * r2q2.data[j] / dataset->data.dimX;

        if (r2qHelp1 == 0)
        {
            r2[j] = 0;
        }
        else
        {
            r2[j] = 1 - r2av.data[j] / r2qHelp1;
        }
    }

    //free r2
    Mx_FreeVector(&r2av);
    Mx_FreeVector(&r2q1);
    Mx_FreeVector(&r2q2);


    //free r
    Mx_FreeVector(&rav);
    Mx_FreeVector(&rav2);
    Mx_FreeVector(&rq1);
    Mx_FreeVector(&rq2);
    Mx_FreeVector(&rq3);


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}








/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetHiddenSlabData5SlabsPerc(int netID, int datasetID, int neuronID, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    if (neuronID < 0 || neuronID >= net->slabs[3].nNeurons)
    {
        NN_ReleaseAcquiredNet(netID);
        DS_ReleaseAcquiredDataset(datasetID);
        return ZERROR_BAD_NEURON_ID;
    }


//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);
        NNi_SlabPropagate(&net, 1);
        NNi_SlabPropagate(&net, 2);
        NNi_SlabPropagate(&net, 3);

        //transfer back
        dataOut[index] = net->slabs[3].outputs.data[neuronID];
        index++;
    }


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GetHiddenSlabDataWithoutDead5SlabsPerc(int netID, int datasetID, int neuronID, int * hid1Enabled, int * hid2Enabled, int * hid3Enabled, float * dataOut)
{
    int i,j,k;
    int index;
    zFunctionResult resultID;
    zDatatype Error;

    DatasetStruct * dataset;
    NeuroNetStruct * net;

    zVectorType output, outputD;


    resultID = DS_TryToAcquireDataset(datasetID, &dataset);
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        DS_ReleaseAcquiredDataset(datasetID);
        return resultID;
    }


    if (neuronID < 0 || neuronID >= net->slabs[3].nNeurons)
    {
        NN_ReleaseAcquiredNet(netID);
        DS_ReleaseAcquiredDataset(datasetID);
        return ZERROR_BAD_NEURON_ID;
    }


//	net = &NN_netVault[netID];

    Mx_CreateVector(&(output), net->totalOutputNeurons);
    Mx_CreateVector(&(outputD), net->totalOutputNeurons);

    Error = 0;
    index = 0;
    for (i = 0; i < dataset->data.dimX; i++)
    {
        //k = myintrandom(dataset->data.dimX);
        k = i;

        NNi_SetSlabInputs(&net, 0, dataset->data.data[k]);

        NNi_SlabPropagate(&net, 0);


        NNi_SlabPropagateWithoutTransfer(&net, 1);
        for (j = 0; j < net->slabs[1].nNeurons; j++)
        {
            if (hid1Enabled[j] == 0)
            {
                net->slabs[1].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 1);

        NNi_SlabPropagateWithoutTransfer(&net, 2);
        for (j = 0; j < net->slabs[2].nNeurons; j++)
        {
            if (hid2Enabled[j] == 0)
            {
                net->slabs[2].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 2);

        NNi_SlabPropagateWithoutTransfer(&net, 3);
        for (j = 0; j < net->slabs[3].nNeurons; j++)
        {
            if (hid3Enabled[j] == 0)
            {
                net->slabs[3].outputs.data[j] = 0;
            }
        }
        NNi_SlabTransferOutput(&net, 3);


        //transfer back
        dataOut[index] = net->slabs[3].outputs.data[neuronID];
        index++;
    }


    Mx_FreeVector(&(output));
    Mx_FreeVector(&(outputD));


    NN_ReleaseAcquiredNet(netID);
    DS_ReleaseAcquiredDataset(datasetID);

    return 0;
}
