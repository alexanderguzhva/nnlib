#ifndef _MLP_H_
#define _MLP_H_


#include "types.h"
#include "matrix.h"


#define NN_MINIMAL_R 0
#define NN_MAXIMAL_R 100

#define NN_MINIMAL_M 0
#define NN_MAXIMAL_M 100

#define NN_NETVAULT_ALLOCBY 10

#define NN_SLAB_IS_DEAD 1
#define NN_SLAB_IS_NOT_DEAD 0

#define NN_LINK_IS_DEAD 1
#define NN_LINK_IS_NOT_DEAD 0

#define NN_NET_IS_DEAD 1
#define NN_NET_IS_NOT_DEAD 0

#define NN_NET_IS_ACQUIRED 1
#define NN_NET_IS_NOT_ACQUIRED 0

#define NN_SLAB_INPUT 0
#define NN_SLAB_HIDDEN 1
#define NN_SLAB_OUTPUT 2

#define NN_SLAB_FUNCTION_IDENTITY -1
#define NN_SLAB_FUNCTION_SIGMOID 0
#define NN_SLAB_FUNCTION_LINEAR 1
#define NN_SLAB_FUNCTION_TANH 2

#define NN_SLAB_MAX_NEURONS 8192

#define NN_NET_MAX_LINKS 128
#define NN_NET_MAX_SLABS 128

#define NN_ORDER_UNDEFINED -1


//struct, containing different 'temporary' values, that are necessary for CalcD
typedef struct
{
    zVectorType neti;		//number of outputs is nNeurons. This is 'unprocessed' copy of outputs, just a sum
    zVectorType df_dneti;	//dF/dNet[i], i is among nNeurons
    zVectorType GVector;	//for output layer only. G-vector. dim is nNeurons
    zMatrixType GMatrix;	//for hidden layer only. G-matrix. dimX is net.totalOutputNeurons, dimY is nNeurons

    zMatrixType dyi_duc;	//for output layer only. dY[i]/dU[c], i is among nNeurons, c is among neuronNWeights+1

    zMatrix4Type dyi_dwkl;	//for hidden layer only. dY[i]/dW[k,l]. dimX is among one of output-to-input connected layers, dimY is among net.totalOutputNeurons, dimZ is among nNeurons, dimW is among neuronNWeights+1

    zMatrixType dweights;
    zMatrixType old_dweights;

} SlabSupportStruct;



//single slab
typedef struct
{
    zInt slabType;		//input, hidden or output
    zInt slabIsDead;	//slab is dead?

    zInt nNeurons;		//number of neurons
    zInt func;			//assuming all neurons has the same func

    zInt neuronNWeights;	//each neuron has neuronNWeight+1 inputs

    zMatrixType weights;	//weights of neurons. dimX=nNeurons, dimY=neuronNWeights+1


    zVectorType inputs;			//number of inputs is neuronNWeights+1
    zVectorType outputs;		//number of outputs is nNeurons

    zVectorType desiredOutputs;	//the outputs that should be gained after propagate

    zVectorType outputDeltas;	//outputs[i] - desiredOutputs[i]

    SlabSupportStruct supp;		//struct, containing different 'temporary' values, that are necessary for CalcD

    zIntVectorType inputsWhoPoints;		//what slabs supply us with input data

    zInt numberOfSlabsConnectedWithAsO;	//number of slabs, that get outputs from out slab

    zInt propagateOrder;
    zInt trainOrder;
} SlabStruct;



typedef struct
{
    zInt linkIsDead;		//link is dead?

    zInt fromSlab;			//source link
    zInt toSlab;			//destination link

} LinkStruct;


typedef struct
{
    SlabStruct * slabs;				//slabs of this net
    zInt maxSlabsAllocated;			//max number of slabs allocated

    LinkStruct * links;				//links of this net
    zInt maxLinksAllocated;			//max number of links allocated

    zInt netIsDead;					//net is dead?

    zInt netIsAcquired;				//net is used by someone?

    zInt totalOutputNeurons;		//total number of output neurons
} NeuroNetStruct;





NeuroNetStruct ** NN_netVault;
zInt NN_netVaultMaxNetsAllocated;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///  user functions

zFunctionResult NN_CreateNet(void);
zFunctionResult NN_DeleteNet(zInt netID);

zFunctionResult NN_CreateSlab(zInt netID, zInt slabType, zInt nNeurons);
zFunctionResult NN_DeleteSlab(zInt netID, zInt slabID);

zFunctionResult NN_CreateLink(zInt netID, zInt fromSlab, zInt toSlab);
zFunctionResult NN_DeleteLink(zInt netID, zInt linkID);

zFunctionResult NN_SetSlabFunc(zInt netID, zInt slabID, zInt func);
zFunctionResult NN_GetSlabWeights(zInt netID, zInt slabID, zInt neuronID, zDatatype * outputValues);
zFunctionResult NN_SetSlabWeights(zInt netID, zInt slabID, zInt neuronID, zDatatype * inputValues);
zFunctionResult NN_DeleteSlab(zInt netID, zInt slabID);
zFunctionResult NN_SlabGenerateRandomWeights(zInt netID, zInt slabID, zDatatype w, zInt * randseed);

zFunctionResult NN_SlabPropagate(zInt netID, zInt slabID);

//netOutputValues can be null, correctValues can be also null
zFunctionResult NN_SlabEvaluate(zInt netID, zInt slabID, zMatrixElementType * correctValues, zMatrixElementType * netOutputValues);

zFunctionResult NN_SlabBPCalcD(zInt netID, zInt slabID);

zFunctionResult NN_SetSlabInputs(zInt netID, zInt slabID, zMatrixElementType * inputValues);

zFunctionResult NN_SlabAdjustWeightsErr(zInt netID, zInt slabID);

zFunctionResult NN_SlabSetPropagateOrder(zInt netID, zInt slabID, zInt order);
zFunctionResult NN_SlabSetTrainOrder(zInt netID, zInt slabID, zInt order);

zFunctionResult NN_SlabApplyChanges(zInt netID, zInt slabID, zDatatype r, zDatatype m);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///  internal functions
zFunctionResult NN_InitializeNeuroNetVault(void);
zFunctionResult NN_ReleaseNeuroNetVault(void);

zFunctionResult NN_TryToAcquireNet(zInt netID, NeuroNetStruct ** net);
zFunctionResult NN_ReleaseAcquiredNet(zInt netID);

zFunctionResult NNi_DeleteNet(zInt netID);
zFunctionResult NNi_AcquireNet(zInt netID, NeuroNetStruct ** net);

zFunctionResult NN_CheckNetIsntDeadNAcquired(zInt netID);
zFunctionResult NN_CheckNetIsntAcquired(zInt netID);
zFunctionResult NN_CheckNetIsntDead(zInt netID);

zFunctionResult NNi_DeleteSlab(NeuroNetStruct ** net, zInt slabID);
zFunctionResult NNi_UpdateSlab(NeuroNetStruct ** net, zInt slabID);

zFunctionResult NNi_DeleteLink(NeuroNetStruct ** net, zInt linkID);

zFunctionResult NNi_UpdateNumberOfNeurons(NeuroNetStruct ** net);

zFunctionResult NNi_SlabPropagate(NeuroNetStruct ** net, zInt slabID);
zFunctionResult NNi_SlabPropagateWithoutTransfer(NeuroNetStruct ** net, zInt slabID);

zFunctionResult NNi_SlabPropagateSigmoid(NeuroNetStruct ** net, zInt slabID);
zFunctionResult NNi_SlabPropagateLinear(NeuroNetStruct ** net, zInt slabID);
zFunctionResult NNi_SlabPropagateTanh(NeuroNetStruct ** net, zInt slabID);
zFunctionResult NNi_SlabPropagateIdentity(NeuroNetStruct ** net, zInt slabID);

zFunctionResult NNi_SlabTransferOutput(NeuroNetStruct ** net, zInt slabID);

zFunctionResult NNi_SlabBPCalcD(NeuroNetStruct ** net, zInt slabID);
zFunctionResult NNi_SlabBPCalcDHiddenToHidden(NeuroNetStruct ** net, zInt slabID, zInt linkID, zInt linkOrderID);
zFunctionResult NNi_SlabBPCalcDHidden(NeuroNetStruct ** net, zInt slabID);

zFunctionResult NNi_SlabBPCalcDOutputSigmoid(NeuroNetStruct ** net, zInt slabID);
zFunctionResult NNi_SlabBPCalcDOutputLinear(NeuroNetStruct ** net, zInt slabID);
zFunctionResult NNi_SlabBPCalcDOutputTanh(NeuroNetStruct ** net, zInt slabID);

zFunctionResult NNi_SlabAdjustWeightsErrOutput(NeuroNetStruct ** net, zInt slabID, SlabStruct * slab);
zFunctionResult NNi_SlabAdjustWeightsErrHidden(NeuroNetStruct ** net, zInt slabID, SlabStruct * slab);
zFunctionResult NNi_SlabAdjustWeightsErrHiddenToHidden(NeuroNetStruct ** net, zInt slabID, SlabStruct * slab, zInt linkID, zInt linkOrderID);

zFunctionResult NNi_SetSlabInputs(NeuroNetStruct ** net, zInt slabID, zMatrixElementType * inputValues);
zFunctionResult NNi_SetSlabInputsDoubleDelphi(NeuroNetStruct ** net, zInt slabID, double ** inputValues, int index);

zFunctionResult NNi_SlabEvaluate(NeuroNetStruct ** net, zInt slabID, zMatrixElementType * correctValues, zMatrixElementType * netOutputValues);

zFunctionResult NNi_SlabAdjustWeightsErr(NeuroNetStruct ** net, zInt slabID);

zFunctionResult NNi_SlabApplyChanges(NeuroNetStruct ** net, zInt slabID, SlabStruct * slab, zDatatype r, zDatatype m);


#endif // _MLP_H_

