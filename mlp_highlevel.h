#ifndef _MLP_HIGHLEVEL_H_
#define _MLP_HIGHLEVEL_H_

#include "matrix.h"
#include "dataset.h"


// functions that can be applied to every high-level net
int GetLayerWeightsSize(int netID, int layerID);
int GetWeightsPerc(int netID, int layerID, float * weightsHidden);

int GetWeightsDimPerc(int netID, int layerID, int * dimX, int * dimY);
int SetWeightsPerc(int netID, int layerID, float * weightsHidden);

int GetHiddenSlabData3SlabsPerc(int netID, int datasetID, int slabID, int neuronID, float * dataOut);
int GetHiddenSlabDataWithoutDead3SlabsPerc(int netID, int datasetID, int slabID, int neuronID, int * hid1Enabled, float * dataOut);


// 3.0 and 3.1
int Create3SlabsPerc(int nInput, int nHidden, int nOutput, int hiddenFunc, int outputFunc, float w, int randseed);
int Create31LayerPerc(int nInput, int nHidden, int nOutput, int hiddenFunc, int outputFunc, float w, int randseed);

int Train3SlabsPerc(int netID, float rHid, float rOut, float mHid, float mOut, int datasetID, int isRandseed, int * myrandseed, float * error);

int Evaluate3SlabsPerc(int netID, int datasetID, float * error);

int GetOutputData3SlabsPerc(int netID, int datasetID, float * error, float * dataOut);
int GetOutputDataWithoutDead3SlabsPerc(int netID, int datasetID, int * hid1Enabled, float * error, float * dataOut);
int GetOutputDataEnh3SlabsPerc(int netID, int datasetID, float * error, float * dataOut, float * r, float * r2);
int GetOutputDataMatrix3SlabsPerc(int netID, int datasetID, float * error, zMatrixType * matrixOut);
int GetOutputDataWithoutDeadMatrix3SlabsPerc(int netID, int datasetID, int * hid1Enabled, float * error, zMatrixType * matrixOut);
int GetOutputDataEnhMatrix3SlabsPerc(int netID, int datasetID, float * error, zMatrixType * matrixOut, float * r, float * r2);

int EvaluateR3SlabsPerc(int netID, int datasetID, float * r);
int EvaluateR23SlabsPerc(int netID, int datasetID, float * r2);





// 3 with 1 recurrent layer
int Create3SlabsRecurrentHiddenPerc(int nInput, int nHidden, int nOutput, int hiddenFunc, int outputFunc, float w, int randseed, int delay);

int Train3SlabsRecurrentPerc(int netID, float rHid, float rOutput, float rRecurrent, float mHid, float mOutput, float mRecurrent, int datasetID, float transferCoeff, float * error);
int Train3SlabsRecurrentIterativePerc(int netID, float rHid, float rOut, float rRecurrent, float mHid, float mOut, float mRecurrent, int datasetID, float transferCoeff, float * error);

int GetOutputData3SlabsRecurrentPerc(int netID, int datasetID, float transferCoeff, float * error, float * backData);
int GetOutputData3SlabsRecurrentIterativePerc(int netID, int datasetID, float transferCoeff, float * error, float * backData);

int Evaluate3SlabsRecurrentPerc(int netID, int datasetID, float transferCoeff, float * error);
int Evaluate3SlabsRecurrentIterativePerc(int netID, int datasetID, float transferCoeff, float * error);


// 4.0, 4.1, 4.2, 4.3 nets
int Create4SlabsPerc(int nInput, int nHidden, int nHidden2, int nOutput, int hiddenFunc, int hiddenFunc2, int outputFunc, float w, int randseed);
int Create41SlabsPerc(int nInput, int nHidden, int nHidden2, int nOutput, int hiddenFunc, int hiddenFunc2, int outputFunc, float w, int randseed);
int Create42SlabsPerc(int nInput, int nHidden, int nHidden2, int nOutput, int hiddenFunc, int hiddenFunc2, int outputFunc, float w, int randseed);
int Create43SlabsPerc(int nInput, int nHidden, int nHidden2, int nOutput, int hiddenFunc, int hiddenFunc2, int outputFunc, float w, int randseed);

int Train4SlabsPerc(int netID, float rHid1, float rHid2, float rOut, float mHid1, float mHid2, float mOut, int datasetID, int isRandseed, int * myrandseed, float * error);
int Evaluate4SlabsPerc(int netID, int datasetID, float * error);
int EvaluateR4SlabsPerc(int netID, int datasetID, float * r);
int EvaluateR24SlabsPerc(int netID, int datasetID, float * r2);

int GetOutputData4SlabsPerc(int netID, int datasetID, float * error, float * dataOut);
int GetOutputDataMatrix4SlabsPerc(int netID, int datasetID, float * error, zMatrixType * matrixOut);
int GetOutputDataWithoutDead4SlabsPerc(int netID, int datasetID, int * hid1Enabled, int * hid2Enabled, float * error, float * dataOut);
int GetOutputDataEnh4SlabsPerc(int netID, int datasetID, float * error, float * dataOut, float * r, float * r2);
int GetHiddenSlabData4SlabsPerc(int netID, int datasetID, int neuronID, float * dataOut);
int GetHiddenSlabDataWithoutDead4SlabsPerc(int netID, int datasetID, int neuronID, int * hid1Enabled, int * hid2Enabled, float * dataOut);



// 5.0, 5.1 and 5.2 nets
int Create5SlabsPerc(int nInput, int nHidden, int nHidden2, int nHidden3, int nOutput, int hiddenFunc, int hiddenFunc2, int hiddenFunc3, int outputFunc, float w, int randseed);
int Create51SlabsPerc(int nInput, int nHidden, int nHidden2, int nHidden3, int nOutput, int hiddenFunc, int hiddenFunc2, int hiddenFunc3, int outputFunc, float w, int randseed);
int Create52SlabsPerc(int nInput, int nHidden, int nHidden2, int nHidden3, int nOutput, int hiddenFunc, int hiddenFunc2, int hiddenFunc3, int outputFunc, float w, int randseed);

int Train5SlabsPerc(int netID, float rHid1, float rHid2, float rHid3, float rOut, float mHid1, float mHid2, float mHid3, float mOut, int datasetID, int isRandseed, int * myrandseed, float * error);
int Evaluate5SlabsPerc(int netID, int datasetID, float * error);
int EvaluateR5SlabsPerc(int netID, int datasetID, float * r);
int EvaluateR25SlabsPerc(int netID, int datasetID, float * r2);


int GetOutputData5SlabsPerc(int netID, int datasetID, float * error, float * dataOut);
int GetOutputDataMatrix5SlabsPerc(int netID, int datasetID, float * error, zMatrixType * matrixOut);
int GetOutputDataWithoutDead5SlabsPerc(int netID, int datasetID, int * hid1Enabled, int * hid2Enabled, int * hid3Enabled, float * error, float * dataOut);
int GetOutputDataEnh5SlabsPerc(int netID, int datasetID, float * error, float * dataOut, float * r, float * r2);

int GetHiddenSlabData5SlabsPerc(int netID, int datasetID, int neuronID, float * dataOut);
int GetHiddenSlabDataWithoutDead5SlabsPerc(int netID, int datasetID, int neuronID, int * hid1Enabled, int * hid2Enabled, int * hid3Enabled, float * dataOut);



#endif

