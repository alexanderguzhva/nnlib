//errors
#ifndef _ERRORS_H_
#define _ERRORS_H_


#include "types.h"

typedef zInt zFunctionResult;

//error codes
//no error
#define ZERROR_NO_ERROR 0

//error has occured in the "unhandled" place
#define ZERROR_UNKNOWN_ERROR -1

//this code is not written yet
#define ZERROR_NOT_IMPLEMENTED_YET -2

//debug situation
#define ZERROR_DEBUG_SITUATION -3

//should-never-happen error
//i=2; if i=3 then raise(ZERROR_SHOULD_NEVER_HAPPEN);
#define ZERROR_SHOULD_NEVER_HAPPEN -4


//mem alloc error
#define ZERROR_MEMORY_ALLOCATION_ERROR -101


//matrix
#define ZERROR_MATRICES_DIMENSIONS_DO_NOT_MATCH -201

//matrix dimensions are strange, ex. < 0
#define ZERROR_MATRIX_BAD_DIMENSIONS -202
	
//null is among function parameters
#define ZERROR_MATRIX_IS_NULL -203

//vector dimension is strange, ex. < 0
#define ZERROR_VECTOR_BAD_DIMENSION -204

//
#define ZERROR_VECTORS_DIMENSIONS_DO_NOT_MATCH -205



//specific errors
//given slab is dead
#define ZERROR_NN_SLABISDEAD -1001

//bad r constant
#define ZERROR_NN_BAD_R -1002

//bad m constant
#define ZERROR_NN_BAD_M -1003


//seems that vault is not initialized
#define ZERROR_VAULT_NOT_INITIALIZED -1051

//bad net ID
#define ZERROR_BAD_NET_ID -1052

//net is dead
#define ZERROR_NET_IS_DEAD -1053

//net is acquired
#define ZERROR_NET_IS_ACQUIRED -1054

//some nets are still acquired!
#define ZERROR_SOME_NETS_ARE_STILL_ACQUIRED -1055

//bad neurons number
#define ZERROR_BAD_NEURONS_NUMBER -1056

//bad slab type number
#define ZERROR_BAD_SLAB_TYPE -1057

//bad slab ID
#define ZERROR_BAD_SLAB_ID -1058

//too many slabs
#define ZERROR_NET_TOO_MANY_SLABS -1059

//too many links
#define ZERROR_NET_TOO_MANY_LINKS -1060

//bad slab ID
#define ZERROR_BAD_LINK_ID -1061

//bad or undefined transfer function
#define ZERROR_BAD_TRANSFER_FUNCTION -1062

//something is wrong with slab with identity transfer function
#define ZERROR_BAD_IDENTITY_SLAB -1063

//bad neuron number
#define ZERROR_BAD_NEURON_ID -1064



//seems that dataset vault is not initialized
#define ZERROR_DATASET_VAULT_NOT_INITIALIZED -1151

//bad dataset ID
#define ZERROR_BAD_DATASET_ID -1152

//dataset is dead
#define ZERROR_DATASET_IS_DEAD -1153

//dataset is acquired
#define ZERROR_DATASET_IS_ACQUIRED -1154

//some datasets are still acquired!
#define ZERROR_SOME_DATASETS_ARE_STILL_ACQUIRED -1155




//library errors: delay should be >= 0
#define ZERROR_LIB_BAD_DELAY -2001



#endif
