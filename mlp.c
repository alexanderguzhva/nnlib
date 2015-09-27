#include "mlp.h"

#include "matrix_nn.h"
#include "mem.h"
#include "randomgen.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_CheckNetIsntDeadNAcquired(zInt netID)
{
    if (netID < 0 || netID >= NN_netVaultMaxNetsAllocated)
    {
        return ZERROR_BAD_NET_ID;
    }


    if ((*NN_netVault[netID]).netIsDead == NN_NET_IS_DEAD)
    {
        return ZERROR_NET_IS_DEAD;
    }


    if ((*NN_netVault[netID]).netIsAcquired == NN_NET_IS_ACQUIRED)
    {
        //nothing to do
        return ZERROR_NET_IS_ACQUIRED;
    }


    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_CheckNetIsntDead(zInt netID)
{
    if (netID < 0 || netID >= NN_netVaultMaxNetsAllocated)
    {
        return ZERROR_BAD_NET_ID;
    }


    if ((*NN_netVault[netID]).netIsDead == NN_NET_IS_DEAD)
    {
        return ZERROR_NET_IS_DEAD;
    }


    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_CheckNetIsntAcquired(zInt netID)
{
    if (netID < 0 || netID >= NN_netVaultMaxNetsAllocated)
    {
        return ZERROR_BAD_NET_ID;
    }


    if ((*NN_netVault[netID]).netIsAcquired == NN_NET_IS_ACQUIRED)
    {
        //nothing to do
        return ZERROR_NET_IS_ACQUIRED;
    }


    return ZERROR_NO_ERROR;
}





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_InitializeNeuroNetVault(void)
{
    NN_netVaultMaxNetsAllocated = 0;
    NN_netVault = NULL;


#ifdef SIGMOID_CACHE
    Mx_InitializeSigmoidCache();
#endif

#ifdef TANH_CACHE
    Mx_InitializeTanhCache();
#endif

    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_ReleaseNeuroNetVault(void)
{
    zFunctionResult resultID;
    zInt i;

    //first, check all available nets
    for (i = 0; i < NN_netVaultMaxNetsAllocated; i++)
    {
        if ((*NN_netVault[i]).netIsAcquired == NN_NET_IS_ACQUIRED)
        {
            //wow, cannot release vault, some nets are still in use
            return ZERROR_SOME_NETS_ARE_STILL_ACQUIRED;
        }
    }


    while (NN_netVaultMaxNetsAllocated > 0)
    {
        resultID = NNi_DeleteNet(NN_netVaultMaxNetsAllocated - 1);
        if (resultID != ZERROR_NO_ERROR)
        {
            return resultID;
        }
    }

    if (NN_netVault != NULL)
    {
        Mem_Free(NN_netVault);
        NN_netVault = NULL;
    }

    NN_netVaultMaxNetsAllocated = 0;

    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_CreateNet(void)
{
    zFunctionResult resultID;
    zInt i,j;
    zInt flag;

    NeuroNetStruct * newNet;





    j = 0;
    flag = 0;
    if (NN_netVault != NULL)
    {
        //some nets exist, check them

        //are there dead nets among them?
        for (i = 0; i < NN_netVaultMaxNetsAllocated; i++)
        {
            if ((*NN_netVault[i]).netIsDead == NN_NET_IS_DEAD)
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
        //no dead nets
        //so allocating new one

        //step 1. alloc mem for new net
        newNet = (NeuroNetStruct *) Mem_Allocate(sizeof(NeuroNetStruct));
        if (newNet == NULL)
        {
            //oops, realloc failed
            return ZERROR_MEMORY_ALLOCATION_ERROR;
        }

        //step 2. realloc NetVault
        NN_netVault = (NeuroNetStruct **) Mem_Reallocate(NN_netVault, (NN_netVaultMaxNetsAllocated + 1) * sizeof(NeuroNetStruct *));
        if (NN_netVault == NULL)
        {
            //oops, realloc failed
            return ZERROR_MEMORY_ALLOCATION_ERROR;
        }

        NN_netVault[NN_netVaultMaxNetsAllocated] = newNet;

        j = NN_netVaultMaxNetsAllocated;

        NN_netVaultMaxNetsAllocated++;
    }

    //initialize net
    //j points to net
    (*NN_netVault[j]).netIsAcquired = NN_NET_IS_NOT_ACQUIRED;
    (*NN_netVault[j]).netIsDead = NN_NET_IS_NOT_DEAD;
    (*NN_netVault[j]).maxSlabsAllocated = 0;
    (*NN_netVault[j]).slabs = NULL;
    (*NN_netVault[j]).maxLinksAllocated = 0;
    (*NN_netVault[j]).links = NULL;
    (*NN_netVault[j]).totalOutputNeurons = 0;


    return j;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_DeleteSlab(NeuroNetStruct ** net, zInt slabID)
{
    //assuming the net is acquired and slabID is correct
    if ((**net).slabs[slabID].slabIsDead == NN_SLAB_IS_DEAD)
    {
        //nothing to do, slab is dead
        return ZERROR_NO_ERROR;
    }

    Mx_FreeMatrix(&((**net).slabs[slabID].weights));

    Mx_FreeVector(&((**net).slabs[slabID].inputs));
    Mx_FreeIntVector(&((**net).slabs[slabID].inputsWhoPoints));
    Mx_FreeVector(&((**net).slabs[slabID].outputs));
    Mx_FreeVector(&((**net).slabs[slabID].desiredOutputs));
    Mx_FreeVector(&((**net).slabs[slabID].outputDeltas));

    Mx_FreeVector(&((**net).slabs[slabID].supp.neti));
    Mx_FreeVector(&((**net).slabs[slabID].supp.df_dneti));

    Mx_FreeVector(&((**net).slabs[slabID].supp.GVector));
    Mx_FreeMatrix(&((**net).slabs[slabID].supp.GMatrix));

    Mx_FreeMatrix(&((**net).slabs[slabID].supp.dyi_duc));

    Mx_FreeMatrix4(&((**net).slabs[slabID].supp.dyi_dwkl));

    Mx_FreeMatrix(&((**net).slabs[slabID].supp.dweights));
    Mx_FreeMatrix(&((**net).slabs[slabID].supp.old_dweights));


    (**net).slabs[slabID].slabIsDead = NN_SLAB_IS_DEAD;

    //now try to shrink NN_netVault size
    while ((**net).maxSlabsAllocated > 0 &&
           (**net).slabs[(**net).maxSlabsAllocated - 1].slabIsDead == NN_SLAB_IS_DEAD)
    {
        //shrink memory
        (**net).slabs = (SlabStruct *) Mem_Reallocate((**net).slabs, ((**net).maxSlabsAllocated - 1) * sizeof(SlabStruct));
        (**net).maxSlabsAllocated--;
    }

    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_DeleteSlab(zInt netID, zInt slabID)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    if (slabID < 0 ||
        slabID >= net->maxSlabsAllocated)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_SLAB_ID;
    }


    resultID = NNi_DeleteSlab(&net, slabID);
    if (resultID != ZERROR_NO_ERROR)
    {
        NN_ReleaseAcquiredNet(netID);
        return resultID;
    }


    return NN_ReleaseAcquiredNet(netID);
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_DeleteNet(zInt netID)
{
    NeuroNetStruct * netToDie;

    NeuroNetStruct * net;
    zFunctionResult resultID;

    //assuming CS is locked and netID is correct and netID is not acquired

    if ((*NN_netVault[netID]).netIsDead == NN_NET_IS_DEAD)
    {
        //nothing to do
        return ZERROR_NO_ERROR;
    }

    NNi_AcquireNet(netID, &net);

    //kill all slabs
    if (net->maxSlabsAllocated != 0)
    {
        while (net->maxSlabsAllocated != 0)
        {
            resultID = NNi_DeleteSlab(&net, net->maxSlabsAllocated - 1);
            if (resultID != ZERROR_NO_ERROR)
            {
                return resultID;
            }
        }
    }


    //kill all links
    if (net->maxLinksAllocated != 0)
    {
        while (net->maxLinksAllocated != 0)
        {
            resultID = NNi_DeleteLink(&net, net->maxLinksAllocated - 1);
            if (resultID != ZERROR_NO_ERROR)
            {
                return resultID;
            }
        }
    }



    net->netIsDead = NN_NET_IS_DEAD;


    //now try to shrink NN_netVault size
    while (NN_netVaultMaxNetsAllocated > 0 &&
           (*NN_netVault[NN_netVaultMaxNetsAllocated - 1]).netIsDead == NN_NET_IS_DEAD)
    {
        //shrink memory

        //step 1
        //kill old net
        netToDie = NN_netVault[NN_netVaultMaxNetsAllocated - 1];
        Mem_Free(netToDie);

        //step 2
        //realloc netvault
        NN_netVault = (NeuroNetStruct **) Mem_Reallocate(NN_netVault, (NN_netVaultMaxNetsAllocated - 1) * sizeof(NeuroNetStruct*));

//		NN_netVault = (NeuroNetStruct *) Mem_Reallocate(NN_netVault, (NN_netVaultMaxNetsAllocated - 1) * sizeof(NeuroNetStruct));
        NN_netVaultMaxNetsAllocated--;
    }

    return ZERROR_NO_ERROR;
}









//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_TryToAcquireNet(zInt netID, NeuroNetStruct ** net)
{

    zFunctionResult resultID;

    resultID = NN_CheckNetIsntDeadNAcquired(netID);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    resultID = NNi_AcquireNet(netID, net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_AcquireNet(zInt netID, NeuroNetStruct ** net)
{
    //acquire net
    (*NN_netVault[netID]).netIsAcquired = NN_NET_IS_ACQUIRED;
    if (net != NULL)
        *net = &(*NN_netVault[netID]);

    return ZERROR_NO_ERROR;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_ReleaseAcquiredNet(zInt netID)
{
    zFunctionResult resultID;

    resultID = NN_CheckNetIsntDead(netID);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    if ((*NN_netVault[netID]).netIsAcquired != NN_NET_IS_ACQUIRED)
    {
        //nothing to do
        return ZERROR_NO_ERROR;
    }

    //release net
    (*NN_netVault[netID]).netIsAcquired = NN_NET_IS_NOT_ACQUIRED;


    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_DeleteNet(zInt netID)
{
    zFunctionResult resultID;

    resultID = NN_CheckNetIsntAcquired(netID);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    //internal function call
    resultID = NNi_DeleteNet(netID);


    return resultID;
}







//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_CreateSlab(zInt netID, zInt slabType, zInt nNeurons)
{
    zFunctionResult resultID;
    zInt i,j;
    zInt flag;
    NeuroNetStruct * net;

    //number  of neurons check
    if (nNeurons <=0 || nNeurons > NN_SLAB_MAX_NEURONS)
    {
        return ZERROR_BAD_NEURONS_NUMBER;
    }

    //slab type check
    if (slabType != NN_SLAB_INPUT &&
        slabType != NN_SLAB_HIDDEN &&
        slabType != NN_SLAB_OUTPUT)
    {
        return ZERROR_BAD_SLAB_TYPE;
    }


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    if (net->maxSlabsAllocated >= NN_NET_MAX_SLABS)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_NET_TOO_MANY_SLABS;
    }


    //process
    j = 0;
    flag = 0;
    if (net->slabs != NULL)
    {
        //slabs is not NULL, some slabs exist
        //check dead slabs
        flag = 0;
        for (i=0; i < net->maxSlabsAllocated; i++)
        {
            if (net->slabs[i].slabIsDead == NN_SLAB_IS_DEAD)
            {
                //ok, it's dead slab
                //recreate it!
                flag = 1;
                j = i;
                break;
            }
        }
    }

    if (flag == 0)
    {
        //all slabs are alive 8\
        //reallocate slabs
        net->slabs = (SlabStruct *) Mem_Reallocate(net->slabs, (net->maxSlabsAllocated + 1) * sizeof(SlabStruct));
        if (net->slabs == NULL)
        {
            //oops, realloc failed
            return ZERROR_MEMORY_ALLOCATION_ERROR;
        }

        j = net->maxSlabsAllocated;

        net->maxSlabsAllocated++;
    }


    //now j points to slab number
    net->slabs[j].func = NN_SLAB_FUNCTION_IDENTITY;
    net->slabs[j].neuronNWeights = 0;
    net->slabs[j].nNeurons = nNeurons;
    net->slabs[j].slabIsDead = NN_SLAB_IS_NOT_DEAD;

    Mx_CreateMatrix(&(net->slabs[j].weights),0,0);

    Mx_CreateVector(&(net->slabs[j].inputs),0);
    Mx_CreateIntVector(&(net->slabs[j].inputsWhoPoints),0);
    Mx_CreateVector(&(net->slabs[j].outputs),0);
    Mx_CreateVector(&(net->slabs[j].supp.neti),0);
    Mx_CreateVector(&(net->slabs[j].supp.df_dneti),0);
    Mx_CreateVector(&(net->slabs[j].desiredOutputs),0);
    Mx_CreateVector(&(net->slabs[j].outputDeltas),0);

    Mx_CreateVector(&(net->slabs[j].supp.GVector),0);
    Mx_CreateMatrix(&(net->slabs[j].supp.GMatrix),0,0);

    Mx_CreateMatrix(&(net->slabs[j].supp.dyi_duc),0,0);

    Mx_CreateMatrix4(&(net->slabs[j].supp.dyi_dwkl),0,0,0,0);

    Mx_CreateMatrix(&(net->slabs[j].supp.dweights),0,0);
    Mx_CreateMatrix(&(net->slabs[j].supp.old_dweights),0,0);

    net->slabs[j].slabType = slabType;

    net->slabs[j].propagateOrder = NN_ORDER_UNDEFINED;
    net->slabs[j].trainOrder = NN_ORDER_UNDEFINED;


    resultID = NNi_UpdateNumberOfNeurons(&net);
    if (resultID != ZERROR_NO_ERROR)
    {
        NN_ReleaseAcquiredNet(netID);
        return resultID;
    }


    resultID = NN_ReleaseAcquiredNet(netID);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    return j;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_UpdateSlab(NeuroNetStruct ** net, zInt slabID)
{
    zFunctionResult resultID;

    //special case here for inputs
    //special case for IDENTITY-transfer function
    //those slabs have no extra weight per neuron
    if ((**net).slabs[slabID].slabType == NN_SLAB_INPUT ||
        (**net).slabs[slabID].func == NN_SLAB_FUNCTION_IDENTITY)
    {
        //input slab has NULL weights and has as many inputs as neurons.
        //inputWhoPoints is also NULL
        //resize inputs
        resultID = Mx_UpdateVector(&((**net).slabs[slabID].inputs),
                               (**net).slabs[slabID].nNeurons);
        if (resultID != ZERROR_NO_ERROR)
        {
            return resultID;
        }

        //resize outputs
        resultID = Mx_UpdateVector(&((**net).slabs[slabID].outputs),
                           (**net).slabs[slabID].nNeurons);
        if (resultID != ZERROR_NO_ERROR)
        {
            return resultID;
        }

        return ZERROR_NO_ERROR;
    }





    //case for HIDDEN & OUTPUT
    //resize weights
    resultID = Mx_UpdateMatrix(&((**net).slabs[slabID].weights),
                           (**net).slabs[slabID].nNeurons,
                           (**net).slabs[slabID].neuronNWeights+1);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    //resize dweights
    resultID = Mx_UpdateMatrix(&((**net).slabs[slabID].supp.dweights),
                           (**net).slabs[slabID].nNeurons,
                           (**net).slabs[slabID].neuronNWeights+1);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    //resize old_dweights
    resultID = Mx_UpdateMatrix(&((**net).slabs[slabID].supp.old_dweights),
                           (**net).slabs[slabID].nNeurons,
                           (**net).slabs[slabID].neuronNWeights+1);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }



    //resize supp.dyi_duc
    resultID = Mx_UpdateMatrix(&((**net).slabs[slabID].supp.dyi_duc),
                           (**net).slabs[slabID].nNeurons,
                           (**net).slabs[slabID].neuronNWeights+1);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }




    //resize inputs
    resultID = Mx_UpdateVector(&((**net).slabs[slabID].inputs),
                           (**net).slabs[slabID].neuronNWeights+1);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    resultID = Mx_UpdateIntVector(&((**net).slabs[slabID].inputsWhoPoints),
                           (**net).slabs[slabID].neuronNWeights);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    //resize outputs
    resultID = Mx_UpdateVector(&((**net).slabs[slabID].outputs),
                           (**net).slabs[slabID].nNeurons);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    if ((**net).slabs[slabID].slabType == NN_SLAB_OUTPUT)
    {
        //resize desiredOutputs
        resultID = Mx_UpdateVector(&((**net).slabs[slabID].desiredOutputs),
                            (**net).slabs[slabID].nNeurons);
        if (resultID != ZERROR_NO_ERROR)
        {
            return resultID;
        }

        //resize outputDeltas
        resultID = Mx_UpdateVector(&((**net).slabs[slabID].outputDeltas),
                            (**net).slabs[slabID].nNeurons);
        if (resultID != ZERROR_NO_ERROR)
        {
            return resultID;
        }
    }



    //resize supp.neti
    resultID = Mx_UpdateVector(&((**net).slabs[slabID].supp.neti),
                           (**net).slabs[slabID].nNeurons);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    //resize supp.df_dneti
    resultID = Mx_UpdateVector(&((**net).slabs[slabID].supp.df_dneti),
                           (**net).slabs[slabID].nNeurons);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    if ((**net).slabs[slabID].slabType == NN_SLAB_OUTPUT)
    {
        //update GVector
        resultID = Mx_UpdateVector(&((**net).slabs[slabID].supp.GVector),
                           (**net).slabs[slabID].nNeurons);
        if (resultID != ZERROR_NO_ERROR)
        {
            return resultID;
        }
    }



    //
    resultID = Mx_UpdateMatrix4(&((**net).slabs[slabID].supp.dyi_dwkl),
                                (**net).slabs[slabID].numberOfSlabsConnectedWithAsO,
                                (**net).totalOutputNeurons,
                                (**net).slabs[slabID].nNeurons,
                                (**net).slabs[slabID].neuronNWeights+1);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }



    if ((**net).slabs[slabID].slabType == NN_SLAB_HIDDEN)
    {
        //update GMatrix
        resultID = Mx_UpdateMatrix(&((**net).slabs[slabID].supp.GMatrix),
                           (**net).totalOutputNeurons,
                           (**net).slabs[slabID].nNeurons);
        if (resultID != ZERROR_NO_ERROR)
        {
            return resultID;
        }
    }



    return ZERROR_NO_ERROR;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_CreateLink(zInt netID, zInt fromSlab, zInt toSlab)
{
    zFunctionResult resultID;
    zInt i,j;
    zInt flag;
    NeuroNetStruct * net;

    //number  of neurons check
    if (fromSlab <0 || toSlab < 0)
    {
        return ZERROR_BAD_SLAB_ID;
    }


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    if (fromSlab >= net->maxSlabsAllocated ||
        toSlab >= net->maxSlabsAllocated)
    {
        return ZERROR_BAD_SLAB_ID;
    }


    if (net->slabs[fromSlab].slabIsDead == NN_SLAB_IS_DEAD ||
        net->slabs[toSlab].slabIsDead == NN_SLAB_IS_DEAD)
    {
        return ZERROR_BAD_SLAB_ID;
    }



    if (net->maxLinksAllocated >= NN_NET_MAX_LINKS)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_NET_TOO_MANY_LINKS;
    }



    //process
    j = 0;
    flag = 0;
    if (net->links != NULL)
    {
        //slabs is not NULL, some slabs exist
        //check dead slabs
        flag = 0;
        for (i=0; i < net->maxLinksAllocated; i++)
        {
            if (net->links[i].linkIsDead == NN_LINK_IS_DEAD)
            {
                //ok, it's dead slab
                //recreate it!
                flag = 1;
                j = i;
                break;
            }
        }
    }

    if (flag == 0)
    {
        //all slabs are alive 8\
        //reallocate slabs
        net->links = (LinkStruct *) Mem_Reallocate(net->links, (net->maxLinksAllocated + 1) * sizeof(LinkStruct));
        if (net->links == NULL)
        {
            //oops, realloc failed
            return ZERROR_MEMORY_ALLOCATION_ERROR;
        }

        j = net->maxLinksAllocated;

        net->maxLinksAllocated++;
    }

    //now j points to slab number
    net->links[j].linkIsDead = NN_LINK_IS_NOT_DEAD;
    net->links[j].fromSlab = fromSlab;
    net->links[j].toSlab = toSlab;


    //ok, link created
    //recalculate number of weights and neurons in
    resultID = NNi_UpdateNumberOfNeurons(&net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }



    resultID = NN_ReleaseAcquiredNet(netID);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    return j;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_DeleteLink(zInt netID, zInt linkID)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    if (linkID < 0 ||
        linkID >= net->maxLinksAllocated)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_LINK_ID;
    }


    resultID = NNi_DeleteLink(&net, linkID);
    if (resultID != ZERROR_NO_ERROR)
    {
        NN_ReleaseAcquiredNet(netID);
        return resultID;
    }

    return NN_ReleaseAcquiredNet(netID);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_DeleteLink(NeuroNetStruct ** net, zInt linkID)
{
    //assuming the net is acquired and linkID is correct
    if ((**net).links[linkID].linkIsDead == NN_LINK_IS_DEAD)
    {
        //nothing to do, link is dead
        return ZERROR_NO_ERROR;
    }

    (**net).links[linkID].linkIsDead = NN_LINK_IS_DEAD;

    //now try to shrink
    while ((**net).maxLinksAllocated > 0 &&
           (**net).links[(**net).maxLinksAllocated - 1].linkIsDead == NN_LINK_IS_DEAD)
    {
        //shrink memory
        (**net).links = (LinkStruct *) Mem_Reallocate((**net).links, ((**net).maxLinksAllocated - 1) * sizeof(LinkStruct));
        (**net).maxLinksAllocated--;
    }

    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//update nNeurons & numberOfNeuronWeights in the whole net
zFunctionResult NNi_UpdateNumberOfNeurons(NeuroNetStruct ** net)
{
    zInt i,j,k,l;
    zInt n1,n2;
    zInt iFromSlab;
    zFunctionResult resultID;

    ////////////////////////////////////////////////
    //step 1. calculate number of output neurons
    i = 0;
    for (j = 0; j < (**net).maxSlabsAllocated; j++)
        if ((**net).slabs[j].slabIsDead == NN_SLAB_IS_NOT_DEAD && (**net).slabs[j].slabType == NN_SLAB_OUTPUT)
            i += (**net).slabs[j].nNeurons;

    (**net).totalOutputNeurons = i;


    ////////////////////////////////////////////////
    //step 2. reallocate number of weights & neurons
    for (j = 0; j < (**net).maxSlabsAllocated; j++)
        if ((**net).slabs[j].slabIsDead == NN_SLAB_IS_NOT_DEAD)
        {
            n1 = 0;
            n2 = 0;
            for (i = 0; i < (**net).maxLinksAllocated; i++)
            {
                if ((**net).links[i].toSlab == j)
                {
                    iFromSlab = (**net).links[i].fromSlab;
                    n1 += (**net).slabs[iFromSlab].nNeurons;
                }

                if ((**net).links[i].fromSlab == j)
                    n2++;
            }

            (**net).slabs[j].neuronNWeights = n1;
            (**net).slabs[j].numberOfSlabsConnectedWithAsO = n2;

            resultID = NNi_UpdateSlab(net, j);
            if (resultID != ZERROR_NO_ERROR)
            {
                return resultID;
            }
        }



    ////////////////////////////////////////////////
    //step 3. update inputsWhoPoints
    for (j = 0; j < (**net).maxSlabsAllocated; j++)
        if ((**net).slabs[j].slabIsDead == NN_SLAB_IS_NOT_DEAD)
        {
            k = 0;
            for (i = 0; i < (**net).maxLinksAllocated; i++)
            {
                if ((**net).links[i].toSlab == j &&
                    (**net).slabs[j].inputsWhoPoints.data != NULL /* this is actual for recurrent net*/)
                {
                    for (l = 0; l < (**net).slabs[(**net).links[i].fromSlab].nNeurons; l++)
                    {
                        (**net).slabs[j].inputsWhoPoints.data[k] = (**net).links[i].fromSlab;
                        k++;
                    }
                }
            }
        }

    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabPropagate(NeuroNetStruct ** net, zInt slabID)
{
    zFunctionResult resultID;

    resultID = NNi_SlabPropagateWithoutTransfer(net, slabID);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    resultID = NNi_SlabTransferOutput(net, slabID);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabPropagateWithoutTransfer(NeuroNetStruct ** net, zInt slabID)
{
    zFunctionResult resultID;

    switch((**net).slabs[slabID].func)
    {
        case NN_SLAB_FUNCTION_LINEAR:
            resultID = NNi_SlabPropagateLinear(net, slabID);
            break;
        case NN_SLAB_FUNCTION_SIGMOID:
            resultID = NNi_SlabPropagateSigmoid(net, slabID);
            break;
        case NN_SLAB_FUNCTION_TANH:
            resultID = NNi_SlabPropagateTanh(net, slabID);
            break;
        case NN_SLAB_FUNCTION_IDENTITY:
            resultID = NNi_SlabPropagateIdentity(net, slabID);
            break;
    }

    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_SlabPropagate(zInt netID, zInt slabID)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    if (slabID < 0 ||
        slabID >= net->maxSlabsAllocated ||
        net->slabs[slabID].slabIsDead == NN_SLAB_IS_DEAD)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_SLAB_ID;
    }


    resultID = NNi_SlabPropagate(&net, slabID);

    if (resultID != ZERROR_NO_ERROR)
    {
        NN_ReleaseAcquiredNet(netID);
        return resultID;
    }


    return NN_ReleaseAcquiredNet(netID);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabPropagateSigmoid(NeuroNetStruct ** net, zInt slabID)
{
    zFunctionResult resultID;

    (**net).slabs[slabID].inputs.data[0]=1;

    resultID = Mx_MultiplyMatrixVector(&((**net).slabs[slabID].weights),
                                       &((**net).slabs[slabID].inputs),
                                       &((**net).slabs[slabID].supp.neti));

    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    resultID = Mx_VectorProcess_Sigmoid(&((**net).slabs[slabID].supp.neti),
                                        &((**net).slabs[slabID].outputs));
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabPropagateLinear(NeuroNetStruct ** net, zInt slabID)
{
    zFunctionResult resultID;

    (**net).slabs[slabID].inputs.data[0]=1;

    resultID = Mx_MultiplyMatrixVector(&((**net).slabs[slabID].weights),
                                       &((**net).slabs[slabID].inputs),
                                       &((**net).slabs[slabID].supp.neti));

    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    resultID = Mx_CopyVector(&((**net).slabs[slabID].supp.neti),
                             &((**net).slabs[slabID].outputs));
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabPropagateTanh(NeuroNetStruct ** net, zInt slabID)
{
    zFunctionResult resultID;

    (**net).slabs[slabID].inputs.data[0]=1;

    resultID = Mx_MultiplyMatrixVector(&((**net).slabs[slabID].weights),
                                       &((**net).slabs[slabID].inputs),
                                       &((**net).slabs[slabID].supp.neti));

    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    resultID = Mx_VectorProcess_Tanh(&((**net).slabs[slabID].supp.neti),
                                        &((**net).slabs[slabID].outputs));
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }


    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabPropagateIdentity(NeuroNetStruct ** net, zInt slabID)
{
    zFunctionResult resultID;

    //identity slab
    //number of inputs should be the same as outputs
    if ((**net).slabs[slabID].inputs.dim != (**net).slabs[slabID].nNeurons)
        return ZERROR_BAD_IDENTITY_SLAB;

    resultID = Mx_CopyVector(&((**net).slabs[slabID].inputs), &((**net).slabs[slabID].outputs));

    return resultID;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_SetSlabFunc(zInt netID, zInt slabID, zInt func)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    if (slabID < 0 ||
        slabID >= net->maxSlabsAllocated ||
        net->slabs[slabID].slabIsDead == NN_SLAB_IS_DEAD)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_SLAB_ID;
    }

    if (func != NN_SLAB_FUNCTION_LINEAR &&
        func != NN_SLAB_FUNCTION_SIGMOID &&
        func != NN_SLAB_FUNCTION_TANH &&
        func != NN_SLAB_FUNCTION_IDENTITY)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_TRANSFER_FUNCTION;
    }


    net->slabs[slabID].func = func;


    return NN_ReleaseAcquiredNet(netID);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_SetSlabInputs(zInt netID, zInt slabID, zMatrixElementType * inputValues)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    if (slabID < 0 ||
        slabID >= net->maxSlabsAllocated ||
        net->slabs[slabID].slabIsDead == NN_SLAB_IS_DEAD)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_SLAB_ID;
    }


    resultID = NNi_SetSlabInputs(&net, slabID, inputValues);

    return NN_ReleaseAcquiredNet(netID);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SetSlabInputs(NeuroNetStruct ** net, zInt slabID, zMatrixElementType * inputValues)
{
    zInt i;

    //special case for input slab
    //for (i = 0; i < (**net).slabs[slabID].inputs.dim; i++)
    //	(**net).slabs[slabID].inputs.data[i] = inputValues[i];
    memcpy((**net).slabs[slabID].inputs.data,inputValues, (**net).slabs[slabID].inputs.dim * sizeof(zMatrixElementType));

    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SetSlabInputsDoubleDelphi(NeuroNetStruct ** net, zInt slabID, double ** inputValues, int index)
{
    zInt i;

    //special case for input slab
    for (i = 0; i < (**net).slabs[slabID].inputs.dim; i++)
        (**net).slabs[slabID].inputs.data[i] = inputValues[i][index];

    return ZERROR_NO_ERROR;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_SlabGenerateRandomWeights(zInt netID, zInt slabID, zDatatype w, zInt * randseed)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;
    zInt i,j;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    if (slabID < 0 ||
        slabID >= net->maxSlabsAllocated ||
        net->slabs[slabID].slabIsDead == NN_SLAB_IS_DEAD)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_SLAB_ID;
    }


/*	_randseed = randseed;
    for (i = 0; i < net->slabs[slabID].weights.dimX; i++)
        for (j = 0; j < net->slabs[slabID].weights.dimY; j++)
        {
            _randseed = (1367065 * _randseed + 3) % 2147483648;
            net->slabs[slabID].weights.data[i][j] = w * ((_randseed / pow(2.,31)) * 2 - 1);
        };
*/
    for (i = 0; i < net->slabs[slabID].weights.dimX; i++)
        for (j = 0; j < net->slabs[slabID].weights.dimY; j++)
            net->slabs[slabID].weights.data[i][j] = (2 * myrandom(randseed) - 1) * w;

    return NN_ReleaseAcquiredNet(netID);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabTransferOutput(NeuroNetStruct ** net, zInt slabID)
{
    zInt i,j,k;
    zInt toslabID;

    //manual optimize
    SlabStruct * slab;
    SlabStruct * stoSlab;


/*
    //non-optimized

    //transfer outputs of this slab to inputs of other slabs
    for (i = 0; i < (**net).maxLinksAllocated; i++)
    {
        if ((**net).links[i].linkIsDead == NN_LINK_IS_NOT_DEAD)
        {
            if ((**net).links[i].fromSlab == slabID)
            {
                //fill (**net).links[i].toSlab's inputs

                k = 0;

                toslabID = (**net).links[i].toSlab;
                for (j = 0; j < (**net).slabs[toslabID].inputsWhoPoints.dim; j++)
                {
                    if ((**net).slabs[toslabID].inputsWhoPoints.data[j] == slabID)
                    {
                        (**net).slabs[toslabID].inputs.data[j+1] = (**net).slabs[slabID].outputs.data[k];
                        k++;
                    };
                };
            };

        };
    };
*/


    //optimized 1

    slab = &((**net).slabs[slabID]);
    //transfer outputs of this slab to inputs of other slabs
    for (i = 0; i < (**net).maxLinksAllocated; i++)
    {
        if ((**net).links[i].linkIsDead == NN_LINK_IS_NOT_DEAD)
        {
            if ((**net).links[i].fromSlab == slabID)
            {
                //fill (**net).links[i].toSlab's inputs

                k = 0;

                toslabID = (**net).links[i].toSlab;
                stoSlab = &((**net).slabs[toslabID]);
                for (j = 0; j < stoSlab->inputsWhoPoints.dim; j++)
                {
                    if (stoSlab->inputsWhoPoints.data[j] == slabID)
                    {
                        stoSlab->inputs.data[j+1] = slab->outputs.data[k];
                        k++;
                    }
                }
            }

        }
    }



    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabEvaluate(NeuroNetStruct ** net, zInt slabID, zMatrixElementType * correctValues, zMatrixElementType * netOutputValues)
{
    zInt i;

    if (correctValues != NULL)
    {
        //load into desiredOutput
        for (i = 0; i < (**net).slabs[slabID].nNeurons; i++)
            (**net).slabs[slabID].desiredOutputs.data[i] = correctValues[i];
    }

    if (netOutputValues != NULL)
    {
        //load into netOutputValues
        for (i = 0; i < (**net).slabs[slabID].nNeurons; i++)
            netOutputValues[i] = (**net).slabs[slabID].outputs.data[i];
    }


    if (netOutputValues != NULL && correctValues != NULL)
    {
        for (i = 0; i < (**net).slabs[slabID].nNeurons; i++)
            (**net).slabs[slabID].outputDeltas.data[i] = (**net).slabs[slabID].outputs.data[i] - (**net).slabs[slabID].desiredOutputs.data[i];
    }

    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_SlabEvaluate(zInt netID, zInt slabID, zMatrixElementType * correctValues, zMatrixElementType * netOutputValues)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;

    if (correctValues == NULL &&
        netOutputValues == NULL)
    {
        //nothing to do
        return ZERROR_NO_ERROR;
    }

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    if (slabID < 0 ||
        slabID >= net->maxSlabsAllocated ||
        net->slabs[slabID].slabIsDead == NN_SLAB_IS_DEAD)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_SLAB_ID;
    }


    resultID = NNi_SlabEvaluate(&net, slabID, correctValues, netOutputValues);
    if (resultID != ZERROR_NO_ERROR)
    {
        NN_ReleaseAcquiredNet(netID);
        return resultID;
    }

    return NN_ReleaseAcquiredNet(netID);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_SlabBPCalcD(zInt netID, zInt slabID)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    if (slabID < 0 ||
        slabID >= net->maxSlabsAllocated ||
        net->slabs[slabID].slabIsDead == NN_SLAB_IS_DEAD)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_SLAB_ID;
    }

    resultID = NNi_SlabBPCalcD(&net, slabID);
    if (resultID != ZERROR_NO_ERROR)
    {
        NN_ReleaseAcquiredNet(netID);
        return resultID;
    }


    return NN_ReleaseAcquiredNet(netID);

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabBPCalcD(NeuroNetStruct ** net, zInt slabID)
{
    zFunctionResult resultID;

    switch((**net).slabs[slabID].slabType)
    {
        case NN_SLAB_INPUT:
            return ZERROR_BAD_SLAB_TYPE;

        case NN_SLAB_HIDDEN:
            resultID = NNi_SlabBPCalcDHidden(net, slabID);
            break;

        case NN_SLAB_OUTPUT:
            switch((**net).slabs[slabID].func)
            {
                case NN_SLAB_FUNCTION_SIGMOID:
                    resultID = NNi_SlabBPCalcDOutputSigmoid(net, slabID);
                    break;
                case NN_SLAB_FUNCTION_LINEAR:
                    resultID = NNi_SlabBPCalcDOutputLinear(net, slabID);
                    break;
                case NN_SLAB_FUNCTION_TANH:
                    resultID = NNi_SlabBPCalcDOutputTanh(net, slabID);
                    break;
            }
            break;
    }

    return resultID;
}





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabBPCalcDHiddenToHidden(NeuroNetStruct ** net, zInt slabID, zInt linkID, zInt linkOrderID)
{
    //processing slabID, link linkID
    zInt i,k,l,m,n,b;
    zInt iNIndex;
    zInt toSlab;
    zMatrixElementType q;
    zInt outputNeuronsStartingOrder;

    //manual optimize
    SlabStruct * slab;
    SlabStruct * stoSlab;
    SlabStruct * iSlab;
    zMatrixElementType ** buf_gmatrix_index1;
    zMatrixElementType * buf_gmatrix_index2;
    zMatrixElementType * buf_inputs;
    zMatrixElementType * buf_dfdneti;
    zMatrixElementType ** buf_gmatrixb_index1;
    zMatrixElementType * buf_gmatrixb_index2;
    zMatrixElementType ** buf_dyidwkl_index1;
    zMatrixElementType *buf_dyidwkl_index2;


/*
    //non-optimized

    //process all output slabs
    outputNeuronsStartingOrder = 0;
    for (i = 0; i < (**net).maxSlabsAllocated; i++)
    {
        if ((**net).slabs[i].slabType == NN_SLAB_OUTPUT &&
            (**net).slabs[i].slabIsDead == NN_SLAB_IS_NOT_DEAD)
        {
            //ok, we've found an output slab
            //process it

            //recalculate GMatrix
            for (k = 0; k < (**net).slabs[i].nNeurons; k++)
            {
                iNIndex = k + outputNeuronsStartingOrder;

                for (m = 0; m < (**net).slabs[slabID].nNeurons; m++)
                {
                    q = 0;

                    toSlab = (**net).links[linkID].toSlab;
                    for (b = 0; b < (**net).slabs[toSlab].nNeurons; b++)
                    {
                        q += (**net).slabs[toSlab].supp.GMatrix.data[iNIndex][b] *
                             (**net).slabs[toSlab].weights.data[b][m+1] *
                             (**net).slabs[slabID].supp.df_dneti.data[m];
                    };

                    (**net).slabs[slabID].supp.GMatrix.data[iNIndex][m] = q;

                    (**net).slabs[slabID].inputs.data[0] = 1;
                    for (l = 0; l < (**net).slabs[slabID].neuronNWeights + 1; l++)
                        (**net).slabs[slabID].supp.dyi_dwkl.data[linkOrderID][iNIndex][m][l] =
                            (**net).slabs[slabID].supp.GMatrix.data[iNIndex][m] *
                            (**net).slabs[slabID].inputs.data[l];
                };
            };

            outputNeuronsStartingOrder += (**net).slabs[i].nNeurons;
        };
    };
*/



    //non-optimized
/*
    //process all output slabs
    outputNeuronsStartingOrder = 0;
    for (i = 0; i < (**net).maxSlabsAllocated; i++)
    {
        if ((**net).slabs[i].slabType == NN_SLAB_OUTPUT &&
            (**net).slabs[i].slabIsDead == NN_SLAB_IS_NOT_DEAD)
        {

            //ok, we've found an output slab
            //process it

            //recalculate GMatrix
            for (k = 0; k < (**net).slabs[i].nNeurons; k++)
            {
                iNIndex = k + outputNeuronsStartingOrder;

                for (m = 0; m < (**net).slabs[slabID].nNeurons; m++)
                {
                    q = 0;

                    toSlab = (**net).links[linkID].toSlab;
                    for (b = 0; b < (**net).slabs[toSlab].nNeurons; b++)
                    {
                        q += (**net).slabs[toSlab].supp.GMatrix.data[iNIndex][b] *
                             (**net).slabs[toSlab].weights.data[b][m+1] *
                             (**net).slabs[slabID].supp.df_dneti.data[m];
                    };

                    (**net).slabs[slabID].supp.GMatrix.data[iNIndex][m] = q;

                    (**net).slabs[slabID].inputs.data[0] = 1;
                    for (l = 0; l < (**net).slabs[slabID].neuronNWeights + 1; l++)
                        (**net).slabs[slabID].supp.dyi_dwkl.data[linkOrderID][iNIndex][m][l] =
                            (**net).slabs[slabID].supp.GMatrix.data[iNIndex][m] *
                            (**net).slabs[slabID].inputs.data[l];
                };
            };

            outputNeuronsStartingOrder += (**net).slabs[i].nNeurons;
        };
    };
*/




    //optimized 2

    //process all output slabs
    slab = &((**net).slabs[slabID]);
    outputNeuronsStartingOrder = 0;
    for (i = 0; i < (**net).maxSlabsAllocated; i++)
    {
        if ((**net).slabs[i].slabType == NN_SLAB_OUTPUT &&
            (**net).slabs[i].slabIsDead == NN_SLAB_IS_NOT_DEAD)
        {
            //ok, we've found an output slab
            //process it
            iSlab = &((**net).slabs[i]);

            //recalculate GMatrix
            for (k = 0; k < iSlab->nNeurons; k++)
            {
                iNIndex = k + outputNeuronsStartingOrder;

                buf_dfdneti = &(slab->supp.df_dneti.data[0]);

                buf_dyidwkl_index1 = &(slab->supp.dyi_dwkl.data[linkOrderID][iNIndex][0]);

                buf_gmatrix_index2 = &(slab->supp.GMatrix.data[iNIndex][0]);

                for (m = 0; m < slab->nNeurons; m++)
                {
                    q = 0;

                    toSlab = (**net).links[linkID].toSlab;
                    stoSlab = &((**net).slabs[toSlab]);

                    buf_gmatrixb_index2 = &(stoSlab->supp.GMatrix.data[iNIndex][0]);
                    for (b = 0; b < stoSlab->nNeurons; b++)
                    {
                        q += (*buf_gmatrixb_index2++) * //stoSlab->supp.GMatrix.data[iNIndex][b] *
                             stoSlab->weights.data[b][m+1] *
                             (*buf_dfdneti); // slab->supp.df_dneti.data[m];
                    }

                    //debug
//					q = q * 10;

                    //slab->supp.GMatrix.data[iNIndex][m] = q;
                    (*buf_gmatrix_index2++) = q;

                    buf_inputs = &(slab->inputs.data[0]);
                    *buf_inputs = 1;

                    buf_dyidwkl_index2 = *buf_dyidwkl_index1++;

                    switch(slab->neuronNWeights + 1)
                    {
                        case 1:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 2:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 3:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 4:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 5:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 6:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 7:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 8:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 9:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 10:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 11:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 12:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 13:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 14:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 15:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 16:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 17:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 18:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 19:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        case 20:
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                        default:
                            for (l = 0; l < slab->neuronNWeights + 1; l++)
                                (*buf_dyidwkl_index2++) = q * (*buf_inputs++);
                            break;
                    }


                    buf_dfdneti++;
                }
            }

            outputNeuronsStartingOrder += (**net).slabs[i].nNeurons;
        }
    }




    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabBPCalcDHidden(NeuroNetStruct ** net, zInt slabID)
{
    zInt i, linkOrderID, k, l, iNIndex;
    zFunctionResult resultID;
    zInt outputNeuronsStartingOrder;
    zInt destSlab;
    zInt iNeuronIndex;


    //manual optimize
    SlabStruct * slab;
    SlabStruct * stoSlab;
    zMatrixElementType * buf_gvector;
    zMatrixElementType * buf_dfdneti;
    zMatrixElementType * buf_inputs;
    zMatrixElementType ** buf_weights_index1;
    zMatrixElementType * buf_weights_index2;
    zMatrixElementType ** buf_gmatrix_index1;
    zMatrixElementType * buf_gmatrix_index2;
    zMatrixElementType ** buf_dyidwkl_index1;
    zMatrixElementType * buf_dyidwkl_index2;


    slab = &((**net).slabs[slabID]);

    //calculate dWeights into 4d array

    linkOrderID = 0;
    // trace all slabs for which this slab supplies data
    for (i = 0; i < (**net).maxLinksAllocated; i++)
    {
        if ((**net).links[i].linkIsDead == NN_LINK_IS_NOT_DEAD)
        {
            if ((**net).links[i].fromSlab == slabID)
            {
                //ok, we've found a link
                //process it

                outputNeuronsStartingOrder = 0;
                if ((**net).slabs[(**net).links[i].toSlab].slabType == NN_SLAB_OUTPUT)
                {
                    for (k = 0; k < (**net).maxSlabsAllocated; k++)
                        if ((**net).slabs[k].slabIsDead == NN_SLAB_IS_NOT_DEAD &&
                            (**net).slabs[k].slabType == NN_SLAB_OUTPUT)
                        {
                            if (k == (**net).links[i].toSlab)
                            {
                                break;
                            }
                            else
                            {
                                outputNeuronsStartingOrder += (**net).slabs[k].nNeurons;
                            }
                        }
                }



                //now process
                //calculate derivatives
                switch ((**net).slabs[slabID].func)
                {
                    case NN_SLAB_FUNCTION_IDENTITY:
                        break;

                    case NN_SLAB_FUNCTION_LINEAR:
                        resultID = Mx_VectorProcess_LinearD(&((**net).slabs[slabID].outputs),
                                                                &((**net).slabs[slabID].supp.df_dneti));
                        break;

                    case NN_SLAB_FUNCTION_SIGMOID:
                        resultID = Mx_VectorProcess_SigmoidD(&((**net).slabs[slabID].outputs),
                                                                &((**net).slabs[slabID].supp.df_dneti));
                        break;

                    case NN_SLAB_FUNCTION_TANH:
                        resultID = Mx_VectorProcess_TanhD(&((**net).slabs[slabID].outputs),
                                                            &((**net).slabs[slabID].supp.df_dneti));
                        break;
                }


/*
                //non-optimized

                //modify weights
                destSlab = (**net).links[i].toSlab;

                //trace simpliest case: this slab supplies output slab
                if ((**net).slabs[destSlab].slabType == NN_SLAB_OUTPUT)
                {
                    //wow, it's output!
                    //process using GVector, not GMatrix
                    for (iNIndex = 0; iNIndex < (**net).slabs[destSlab].nNeurons; iNIndex++)
                    {
                        iNeuronIndex = iNIndex + outputNeuronsStartingOrder;

                        for (k = 0; k < (**net).slabs[slabID].nNeurons; k++)
                        {
                            //mb unnecessary?
                            (**net).slabs[slabID].inputs.data[0] = 1;


                            //calc GMatrix for this slab
                            (**net).slabs[slabID].supp.GMatrix.data[iNIndex][k] =
                                (**net).slabs[destSlab].supp.GVector.data[iNIndex] *
                                (**net).slabs[destSlab].weights.data[iNIndex][k+1] *
                                (**net).slabs[slabID].supp.df_dneti.data[k];


                            for (l = 0; l < (**net).slabs[slabID].neuronNWeights+1; l++)
                            {
                                (**net).slabs[slabID].supp.dyi_dwkl.data[linkOrderID][iNeuronIndex][k][l] =
                                    (**net).slabs[slabID].supp.GMatrix.data[iNIndex][k] *
                                    (**net).slabs[slabID].inputs.data[l];
                            };
                        };
                    };

                    resultID = ZERROR_NO_ERROR;
                }
                else if ((**net).slabs[destSlab].slabType == NN_SLAB_HIDDEN)
                {
                    //target slab is hidden
                    //use back-prop
                    resultID = NNi_SlabBPCalcDHiddenToHidden(net, slabID, i, linkOrderID);
                };
*/

/*
                //optimized 1
                //modify weights
                destSlab = (**net).links[i].toSlab;
                stoSlab = &((**net).slabs[destSlab]);

                //trace simpliest case: this slab supplies output slab
                if ((**net).slabs[destSlab].slabType == NN_SLAB_OUTPUT)
                {
                    //wow, it's output!
                    //process using GVector, not GMatrix
                    buf_gvector = &((**net).slabs[destSlab].supp.GVector.data[0]);
                    buf_gmatrix_index1 = &(slab->supp.GMatrix.data[0]);
                    for (iNIndex = 0; iNIndex < stoSlab->nNeurons; iNIndex++)
                    {
                        iNeuronIndex = iNIndex + outputNeuronsStartingOrder;

                        buf_dfdneti = &(slab->supp.df_dneti.data[0]);
                        buf_weights_index2 = &(stoSlab->weights.data[iNIndex][1]);
                        buf_gmatrix_index2 = *buf_gmatrix_index1++;
                        buf_dyidwkl_index1 = &(slab->supp.dyi_dwkl.data[linkOrderID][iNeuronIndex][0]);
                        for (k = 0; k < slab->nNeurons; k++)
                        {
                            //mb unnecessary?
//							(**net).slabs[slabID].inputs.data[0] = 1;
                            buf_inputs = &(slab->inputs.data[0]);
                            *buf_inputs = 1;

                            //calc GMatrix for this slab
                            (*buf_gmatrix_index2) = //(**net).slabs[slabID].supp.GMatrix.data[iNIndex][k] =
                                (*buf_gvector) * //(**net).slabs[destSlab].supp.GVector.data[iNIndex] *
                                (*buf_weights_index2++) * //(**net).slabs[destSlab].weights.data[iNIndex][k+1] *
                                (*buf_dfdneti++);//(**net).slabs[slabID].supp.df_dneti.data[k];

                            buf_dyidwkl_index2 = *buf_dyidwkl_index1++;

                            for (l = 0; l < slab->neuronNWeights+1; l++)
                            {
                                (*buf_dyidwkl_index2++) =  //(**net).slabs[slabID].supp.dyi_dwkl.data[linkOrderID][iNeuronIndex][k][l] =
                                    (*buf_gmatrix_index2) * //(**net).slabs[slabID].supp.GMatrix.data[iNIndex][k] *
                                    (*buf_inputs++);//(**net).slabs[slabID].inputs.data[l];
                            };

                            buf_gmatrix_index2++;
                        };

                        buf_gvector++;
                    };

                    resultID = ZERROR_NO_ERROR;
                }
                else if ((**net).slabs[destSlab].slabType == NN_SLAB_HIDDEN)
                {
                    //target slab is hidden
                    //use back-prop
                    resultID = NNi_SlabBPCalcDHiddenToHidden(net, slabID, i, linkOrderID);
                };
*/



                //optimized 2
                //modify weights
                destSlab = (**net).links[i].toSlab;
                stoSlab = &((**net).slabs[destSlab]);

                //trace simpliest case: this slab supplies output slab
                if ((**net).slabs[destSlab].slabType == NN_SLAB_OUTPUT)
                {
                    //wow, it's output!
                    //process using GVector, not GMatrix
                    buf_gvector = &((**net).slabs[destSlab].supp.GVector.data[0]);
                    buf_gmatrix_index1 = &(slab->supp.GMatrix.data[0]);
                    for (iNIndex = 0; iNIndex < stoSlab->nNeurons; iNIndex++)
                    {
                        iNeuronIndex = iNIndex + outputNeuronsStartingOrder;

                        buf_dfdneti = &(slab->supp.df_dneti.data[0]);
                        buf_weights_index2 = &(stoSlab->weights.data[iNIndex][1]);
                        buf_gmatrix_index2 = *buf_gmatrix_index1++;
                        buf_dyidwkl_index1 = &(slab->supp.dyi_dwkl.data[linkOrderID][iNeuronIndex][0]);
                        for (k = 0; k < slab->nNeurons; k++)
                        {
                            //mb unnecessary?
//							(**net).slabs[slabID].inputs.data[0] = 1;
                            buf_inputs = &(slab->inputs.data[0]);
                            *buf_inputs = 1;

                            //calc GMatrix for this slab
                            (*buf_gmatrix_index2) = //(**net).slabs[slabID].supp.GMatrix.data[iNIndex][k] =
                                (*buf_gvector) * //(**net).slabs[destSlab].supp.GVector.data[iNIndex] *
                                (*buf_weights_index2++) * //(**net).slabs[destSlab].weights.data[iNIndex][k+1] *
                                (*buf_dfdneti++);//(**net).slabs[slabID].supp.df_dneti.data[k];

                            buf_dyidwkl_index2 = *buf_dyidwkl_index1++;

                            switch(slab->neuronNWeights+1)
                            {
                                case 1:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 2:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 3:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 4:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 5:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 6:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 7:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 8:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 9:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 10:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 11:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 12:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 13:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 14:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 15:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 16:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 17:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 18:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 19:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                case 20:
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    break;
                                default:
                                    for (l = 0; l < slab->neuronNWeights+1; l++)
                                    {
                                        (*buf_dyidwkl_index2++) = (*buf_gmatrix_index2) * (*buf_inputs++);
                                    }
                                    break;
                            }

                            buf_gmatrix_index2++;
                        }

                        buf_gvector++;
                    }

                    resultID = ZERROR_NO_ERROR;
                }
                else if ((**net).slabs[destSlab].slabType == NN_SLAB_HIDDEN)
                {
                    //target slab is hidden
                    //use back-prop
                    resultID = NNi_SlabBPCalcDHiddenToHidden(net, slabID, i, linkOrderID);
                }




                if (resultID != ZERROR_NO_ERROR)
                {
                    return resultID;
                }

                linkOrderID++;
            }

        }
    }

    return ZERROR_NO_ERROR;

}





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabBPCalcDOutputSigmoid(NeuroNetStruct ** net, zInt slabID)
{
    zInt i,j;
    zFunctionResult resultID;


    //non-optimized

    //calculate df_dneti
    resultID = Mx_VectorProcess_SigmoidD(&((**net).slabs[slabID].outputs), &((**net).slabs[slabID].supp.df_dneti));
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

    //calculate dyi_duc
    for (i = 0; i < (**net).slabs[slabID].nNeurons; i++)
    {
        //->CONVERT INTO MATRIX MULTIPLICATION

        //calculate dyi_duc, c = 0
        (**net).slabs[slabID].supp.dyi_duc.data[i][0] = (**net).slabs[slabID].supp.df_dneti.data[i];

        //calculate dyi_duc, c <> 0
        for (j = 0; j < (**net).slabs[slabID].neuronNWeights; j++)
            (**net).slabs[slabID].supp.dyi_duc.data[i][j+1] = (**net).slabs[slabID].supp.df_dneti.data[i] * (**net).slabs[slabID].inputs.data[j+1];
    }

    //save GVector
    Mx_CopyVector(&((**net).slabs[slabID].supp.df_dneti), &((**net).slabs[slabID].supp.GVector));



    return ZERROR_NO_ERROR;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabBPCalcDOutputLinear(NeuroNetStruct ** net, zInt slabID)
{
    zInt i,j;
    zFunctionResult resultID;

    //calculate df_dneti
    resultID = Mx_VectorProcess_LinearD(&((**net).slabs[slabID].outputs), &((**net).slabs[slabID].supp.df_dneti));
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

//	for (i = 0; i < (**net).slabs[slabID].nNeurons; i++)
//	{
//		(**net).slabs[slabID].supp.df_dneti.data[i] = 1;
//	};

    //calculate dyi_duc
    for (i = 0; i < (**net).slabs[slabID].nNeurons; i++)
    {
        //->CONVERT INTO MATRIX MULTIPLICATION

        //calculate dyi_duc, c = 0
        (**net).slabs[slabID].supp.dyi_duc.data[i][0] = (**net).slabs[slabID].supp.df_dneti.data[i];

        //calculate dyi_duc, c <> 0
        for (j = 0; j < (**net).slabs[slabID].neuronNWeights; j++)
            (**net).slabs[slabID].supp.dyi_duc.data[i][j+1] = (**net).slabs[slabID].supp.df_dneti.data[i] * (**net).slabs[slabID].inputs.data[j+1];
    }

    //save GVector
    Mx_CopyVector(&((**net).slabs[slabID].supp.df_dneti), &((**net).slabs[slabID].supp.GVector));

    return ZERROR_NO_ERROR;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabBPCalcDOutputTanh(NeuroNetStruct ** net, zInt slabID)
{
    zInt i,j;
    zFunctionResult resultID;

    //calculate df_dneti
    resultID = Mx_VectorProcess_TanhD(&((**net).slabs[slabID].outputs), &((**net).slabs[slabID].supp.df_dneti));
    if (resultID != ZERROR_NO_ERROR)
        return resultID;

//	for (i = 0; i < (**net).slabs[slabID].nNeurons; i++)
//	{
//		(**net).slabs[slabID].supp.df_dneti.data[i] = 1. - (**net).slabs[slabID].outputs.data[i] * (**net).slabs[slabID].outputs.data[i];
//	};


    //calculate dyi_duc
    for (i = 0; i < (**net).slabs[slabID].nNeurons; i++)
    {
        //->CONVERT INTO MATRIX MULTIPLICATION

        //calculate dyi_duc, c = 0
        (**net).slabs[slabID].supp.dyi_duc.data[i][0] = (**net).slabs[slabID].supp.df_dneti.data[i];

        //calculate dyi_duc, c <> 0
        for (j = 0; j < (**net).slabs[slabID].neuronNWeights; j++)
            (**net).slabs[slabID].supp.dyi_duc.data[i][j+1] = (**net).slabs[slabID].supp.df_dneti.data[i] * (**net).slabs[slabID].inputs.data[j+1];
    }

    //save GVector
    Mx_CopyVector(&((**net).slabs[slabID].supp.df_dneti), &((**net).slabs[slabID].supp.GVector));

    return ZERROR_NO_ERROR;
}





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_SlabAdjustWeightsErr(zInt netID, zInt slabID)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    if (slabID < 0 ||
        slabID >= net->maxSlabsAllocated ||
        net->slabs[slabID].slabIsDead == NN_SLAB_IS_DEAD)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_SLAB_ID;
    }

    resultID = NNi_SlabAdjustWeightsErr(&net, slabID);

    if (resultID != ZERROR_NO_ERROR)
    {
        NN_ReleaseAcquiredNet(netID);
        return resultID;
    }


    return NN_ReleaseAcquiredNet(netID);

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabAdjustWeightsErr(NeuroNetStruct ** net, zInt slabID)
{
    zFunctionResult resultID;

    switch((**net).slabs[slabID].slabType)
    {
        case NN_SLAB_INPUT:
            //cannot work with input slab
            return ZERROR_BAD_SLAB_TYPE;

        case NN_SLAB_HIDDEN:
            resultID = NNi_SlabAdjustWeightsErrHidden(net, slabID, &((**net).slabs[slabID]));
            break;

        case NN_SLAB_OUTPUT:
            resultID = NNi_SlabAdjustWeightsErrOutput(net, slabID, &((**net).slabs[slabID]));
            break;
    }

    return resultID;
}





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabAdjustWeightsErrOutput(NeuroNetStruct ** net, zInt slabID, SlabStruct * slab)
{
    zInt i,j;
    zMatrixElementType delta;

    //manual optimization
    zMatrixElementType * buf_outputdeltas;
    zMatrixElementType ** buf_dyiduc_index1;
    zMatrixElementType ** buf_dweights_index1;
    zMatrixElementType * buf_dyiduc_index2;
    zMatrixElementType * buf_dweights_index2;

/*
    //non-optimized
    for (i = 0; i < (**net).slabs[slabID].nNeurons; i++)
    {
        delta = (**net).slabs[slabID].outputDeltas.data[i];

        for (j = 0; j < (**net).slabs[slabID].neuronNWeights+1; j++)
        {
            (**net).slabs[slabID].supp.dweights.data[i][j] +=
                   (**net).slabs[slabID].supp.dyi_duc.data[i][j] * delta;
        };

    };
*/


    //optimize 1
    for (i = 0; i < slab->nNeurons; i++)
    {
        delta = slab->outputDeltas.data[i];

        for (j = 0; j < slab->neuronNWeights + 1; j++)
        {
            slab->supp.dweights.data[i][j] +=
                   slab->supp.dyi_duc.data[i][j] * delta;
        }

    }



/*
    //optimize 2
    buf_dyiduc_index1 = &(slab->supp.dyi_duc.data[0]);
    buf_dweights_index1 = &(slab->supp.dweights.data[0]);
    buf_outputdeltas = &(slab->outputDeltas.data[0]);
    for (i = 0; i < slab->nNeurons; i++)
    {
//		buf_dyiduc_index2 = *buf_dyiduc_index1++;
//		buf_dweights_index2 = *buf_dweights_index1++;
        buf_dyiduc_index2 = &(slab->supp.dyi_duc.data[i][0]);
        buf_dweights_index2 = &(slab->supp.dweights.data[i][0]);
        delta = slab->outputDeltas.data[i];

        for (j = 0; j < slab->neuronNWeights + 1; j++)
        {
            (*buf_dweights_index2++) += (*buf_dyiduc_index2) * delta;//(*buf_outputdeltas);
        };

        buf_outputdeltas++;
    };
*/

    return ZERROR_NO_ERROR;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabAdjustWeightsErrHiddenToHidden(NeuroNetStruct ** net, zInt slabID, SlabStruct * slab, zInt linkID, zInt linkOrderID)
{
    //processing slabID, link linkID
    zInt i,j,k,l,m,n,b;
    zInt iNIndex;
    zInt toSlab;
    zMatrixElementType q0;
    zInt outputNeuronsStartingOrder;
    zMatrixElementType delta;


    //manual optimization
    zMatrixElementType * buf_yiwkl_index2;
    zMatrixElementType * buf_dweights_index2;
    zMatrixElementType ** buf_yiwkl_index1;
    zMatrixElementType ** buf_dweights_index1;
    SlabStruct * stoSlab;


    //process all output slabs
    outputNeuronsStartingOrder = 0;
    for (i = 0; i < (**net).maxSlabsAllocated; i++)
    {
        if ((**net).slabs[i].slabType == NN_SLAB_OUTPUT &&
            (**net).slabs[i].slabIsDead == NN_SLAB_IS_NOT_DEAD)
        {
            //ok, we've found an output slab
            //process it

            //adjust weights

/*
            //non-optimized
            for (l = 0; l < (**net).slabs[i].nNeurons; l++)
            {
                iNIndex = outputNeuronsStartingOrder + l;
                delta = (**net).slabs[i].outputDeltas.data[l];

                for (k = 0; k < (**net).slabs[slabID].nNeurons; k++)
                {
                    toSlab = (**net).links[linkID].toSlab;

                    for (j = 0; j < (**net).slabs[slabID].neuronNWeights+1; j++)
                    {
                        q0 = (**net).slabs[slabID].supp.dyi_dwkl.data[linkOrderID][iNIndex][k][j] * delta;

                        (**net).slabs[slabID].supp.dweights.data[k][j] += q0;
                    }
                };
            };
*/

            //optimized
            stoSlab = &((**net).slabs[i]);
            for (l = 0; l < stoSlab->nNeurons; l++)
            {
                iNIndex = outputNeuronsStartingOrder + l;
                delta = stoSlab->outputDeltas.data[l];

                buf_yiwkl_index1 = &(slab->supp.dyi_dwkl.data[linkOrderID][iNIndex][0]);
                buf_dweights_index1 = &(slab->supp.dweights.data[0]);
                for (k = 0; k < slab->nNeurons; k++)
                {
                    buf_yiwkl_index2 = *buf_yiwkl_index1++;
                    buf_dweights_index2 = *buf_dweights_index1++;

                    /*for (j = 0; j < slab->neuronNWeights + 1; j++)
                    {
                        (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                    };
*/
                    switch(slab->neuronNWeights + 1)
                    {
                        case 1:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 2:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 3:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 4:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 5:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 6:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 7:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 8:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 9:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 10:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 11:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 12:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 13:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 14:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 15:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 16:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 17:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 18:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 19:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        case 20:
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            break;
                        default:
                            for (j = 0; j < slab->neuronNWeights+1; j++)
                            {
                                (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            }
                            break;
                    }

                }
            }


            outputNeuronsStartingOrder += (**net).slabs[i].nNeurons;
        }
    }

    return ZERROR_NO_ERROR;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabAdjustWeightsErrHidden(NeuroNetStruct ** net, zInt slabID, SlabStruct * slab)
{
    zInt i,j,k,l;
    zInt iLinkID;
    zMatrixElementType q0;
    zInt toSlab;
    zInt outputNeuronsStartingOrder;
    zInt iNIndex;
    zMatrixElementType delta;

    //manual optimization
    zMatrixElementType * buf_yiwkl_index2;
    zMatrixElementType * buf_dweights_index2;
    zMatrixElementType ** buf_yiwkl_index1;
    zMatrixElementType ** buf_dweights_index1;
    SlabStruct * stoSlab;
    LinkStruct * slink;

    iLinkID = 0;
    for (i = 0; i < (**net).maxLinksAllocated; i++)
    {
/*
        //non-optimized
        if ((**net).links[i].linkIsDead == NN_LINK_IS_NOT_DEAD)
        {
            if ((**net).links[i].fromSlab == slabID)
            {
                //ok, we've found a link
                //process it

                toSlab = (**net).links[i].toSlab;

                outputNeuronsStartingOrder = 0;
                if ((**net).slabs[toSlab].slabType == NN_SLAB_OUTPUT)
                {
                    for (k = 0; k < (**net).maxSlabsAllocated; k++)
                        if ((**net).slabs[k].slabIsDead == NN_SLAB_IS_NOT_DEAD &&
                            (**net).slabs[k].slabType == NN_SLAB_OUTPUT)
                        {
                            if (k == (**net).links[i].toSlab)
                            {
                                break;
                            }
                            else
                            {
                                outputNeuronsStartingOrder += (**net).slabs[k].nNeurons;
                            }
                        }
                }
*/


        //optimized
        slink = &((**net).links[i]);
        if (slink->linkIsDead == NN_LINK_IS_NOT_DEAD)
        {
            if (slink->fromSlab == slabID)
            {
                //ok, we've found a link
                //process it

                toSlab = slink->toSlab;

                outputNeuronsStartingOrder = 0;
                if ((**net).slabs[toSlab].slabType == NN_SLAB_OUTPUT)
                {
                    for (k = 0; k < (**net).maxSlabsAllocated; k++)
                        if ((**net).slabs[k].slabIsDead == NN_SLAB_IS_NOT_DEAD &&
                            (**net).slabs[k].slabType == NN_SLAB_OUTPUT)
                        {
                            if (k == slink->toSlab)
                            {
                                break;
                            }
                            else
                            {
                                outputNeuronsStartingOrder += (**net).slabs[k].nNeurons;
                            }
                        }
                }




/*
                //non-optimized
                if ((**net).slabs[toSlab].slabType == NN_SLAB_OUTPUT)
                {

                    for (l = 0; l < (**net).slabs[toSlab].nNeurons; l++)
                    {
                        iNIndex = outputNeuronsStartingOrder + l;
                        delta = (**net).slabs[toSlab].outputDeltas.data[l];
                        for (k = 0; k < (**net).slabs[slabID].nNeurons; k++)
                        {
                            //normalize?
                            for (j = 0; j < (**net).slabs[slabID].neuronNWeights+1; j++)
                            {
                                q0 = (**net).slabs[slabID].supp.dyi_dwkl.data[iLinkID][iNIndex][k][j] * delta;

                                (**net).slabs[slabID].supp.dweights.data[k][j] += q0;
                            }
                        };
                    };
                }
                else if ((**net).slabs[toSlab].slabType == NN_SLAB_HIDDEN)
                {
                    NNi_SlabAdjustWeightsErrHiddenToHidden(net, slabID, i, iLinkID);
                };
*/

                //optimized
                stoSlab = &((**net).slabs[toSlab]);
                if (stoSlab->slabType == NN_SLAB_OUTPUT)
                {
                    for (l = 0; l < stoSlab->nNeurons; l++)
                    {
                        iNIndex = outputNeuronsStartingOrder + l;
                        delta = stoSlab->outputDeltas.data[l];

                        buf_yiwkl_index1 = &(slab->supp.dyi_dwkl.data[iLinkID][iNIndex][0]);
                        buf_dweights_index1 = &(slab->supp.dweights.data[0]);

                        for (k = 0; k < slab->nNeurons; k++)
                        {
                            buf_yiwkl_index2 = *buf_yiwkl_index1++;
                            buf_dweights_index2 = *buf_dweights_index1++;
/*							for (j = 0; j < slab->neuronNWeights+1; j++)
                            {
                                (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                            }
*/
                            switch(slab->neuronNWeights + 1)
                            {
                                case 1:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 2:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 3:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 4:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 5:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 6:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 7:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 8:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 9:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 10:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 11:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 12:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 13:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 14:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 15:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 16:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 17:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 18:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 19:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                case 20:
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    break;
                                default:
                                    for (j = 0; j < slab->neuronNWeights+1; j++)
                                    {
                                        (*buf_dweights_index2++) += (*buf_yiwkl_index2++) * delta;
                                    }
                                    break;
                            }
                        }

                    }
                }
                else if (stoSlab->slabType == NN_SLAB_HIDDEN)
                {
                    NNi_SlabAdjustWeightsErrHiddenToHidden(net, slabID, slab, i, iLinkID);
                };



                iLinkID++;

            }
        }
    }



    return ZERROR_NO_ERROR;
}





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_SlabSetPropagateOrder(zInt netID, zInt slabID, zInt order)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    if (slabID < 0 ||
        slabID >= net->maxSlabsAllocated ||
        net->slabs[slabID].slabIsDead == NN_SLAB_IS_DEAD)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_SLAB_ID;
    }

    net->slabs[slabID].propagateOrder = order;


    return NN_ReleaseAcquiredNet(netID);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_SlabSetTrainOrder(zInt netID, zInt slabID, zInt order)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;

    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    if (slabID < 0 ||
        slabID >= net->maxSlabsAllocated ||
        net->slabs[slabID].slabIsDead == NN_SLAB_IS_DEAD)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_SLAB_ID;
    }

    net->slabs[slabID].trainOrder = order;

    return NN_ReleaseAcquiredNet(netID);
}


#define SlabApplyChanges_macro1	-r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2)


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NNi_SlabApplyChanges(NeuroNetStruct ** net, zInt slabID, SlabStruct * slab, zDatatype r, zDatatype m)
{
    zInt i,j;
    zDatatype q;

    //manual optimize
    zMatrixElementType ** buf_weights_index1;
    zMatrixElementType ** buf_dweights_index1;
    zMatrixElementType ** buf_olddweights_index1;
    zMatrixElementType * buf_weights_index2;
    zMatrixElementType * buf_dweights_index2;
    zMatrixElementType * buf_olddweights_index2;


/*
    //thrown away as useless

    if (r < NN_MINIMAL_R ||
        r > NN_MAXIMAL_R)
    {
        return ZERROR_NN_BAD_R;
    };

    if (m < NN_MINIMAL_M ||
        m > NN_MAXIMAL_M)
    {
        return ZERROR_NN_BAD_M;
    };
*/

/*
    //non-optimized
    for (i = 0; i < (**net).slabs[slabID].nNeurons; i++)
        for (j = 0; j < (**net).slabs[slabID].neuronNWeights + 1; j++)
        {
            q = (-r * (**net).slabs[slabID].supp.dweights.data[i][j] + m * (**net).slabs[slabID].supp.old_dweights.data[i][j]);
            (**net).slabs[slabID].weights.data[i][j] += q;
            (**net).slabs[slabID].supp.old_dweights.data[i][j] = q;
        };
*/

/*
    //optimized 1
    buf_olddweights_index1 = &(slab->supp.old_dweights.data[0]);
    buf_weights_index1 = &(slab->weights.data[0]);
    buf_dweights_index1 = &(slab->supp.dweights.data[0]);
    for (i = 0; i < slab->nNeurons; i++)
    {
        buf_olddweights_index2 = *buf_olddweights_index1++;
        buf_weights_index2 = *buf_weights_index1++;
        buf_dweights_index2 = *buf_dweights_index1++;
        for (j = 0; j < slab->neuronNWeights + 1; j++)
        {
            q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
            (*buf_weights_index2++) += q;
            (*buf_olddweights_index2++) = q;
        };
    };
*/



    //optimized 2
    buf_olddweights_index1 = &(slab->supp.old_dweights.data[0]);
    buf_weights_index1 = &(slab->weights.data[0]);
    buf_dweights_index1 = &(slab->supp.dweights.data[0]);
    for (i = 0; i < slab->nNeurons; i++)
    {
        buf_olddweights_index2 = *buf_olddweights_index1++;
        buf_weights_index2 = *buf_weights_index1++;
        buf_dweights_index2 = *buf_dweights_index1++;


        switch(slab->neuronNWeights + 1)
        {
            case 1:
//				q = SlabApplyChanges_macro1;
//				(*buf_weights_index2++) += q;
//				(*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                break;
            case 2:
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                break;
            case 3:
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                break;
            case 4:
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                break;
            case 5:
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                break;
            case 6:
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                break;
            case 7:
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                break;
            case 8:
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                break;
            case 9:
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                break;
            case 10:
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                break;
            case 11:
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                break;
            case 12:
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                break;
            case 13:
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                break;
            case 14:
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                break;
            case 15:
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                break;
            case 16:
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                break;
            case 17:
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                break;
            case 18:
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                break;
            case 19:
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                q = SlabApplyChanges_macro1; (*buf_weights_index2++) += q; (*buf_olddweights_index2++) = q;
                break;
            case 20:
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                (*buf_weights_index2++) += q;
                (*buf_olddweights_index2++) = q;
                break;
            default:
                for (j = 0; j < slab->neuronNWeights + 1; j++)
                {
                    q = -r * (*buf_dweights_index2++) + m * (*buf_olddweights_index2);
                    (*buf_weights_index2++) += q;
                    (*buf_olddweights_index2++) = q;
                }
                break;
        }


    }




    return ZERROR_NO_ERROR;
}





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
zFunctionResult NN_SlabApplyChanges(zInt netID, zInt slabID, zDatatype r, zDatatype m)
{
    zFunctionResult resultID;
    NeuroNetStruct * net;

    if (r < NN_MINIMAL_R ||
        r > NN_MAXIMAL_R)
    {
        return ZERROR_NN_BAD_R;
    }

    if (m < NN_MINIMAL_M ||
        m > NN_MAXIMAL_M)
    {
        return ZERROR_NN_BAD_M;
    }


    resultID = NN_TryToAcquireNet(netID, &net);
    if (resultID != ZERROR_NO_ERROR)
    {
        return resultID;
    }

    if (slabID < 0 ||
        slabID >= net->maxSlabsAllocated ||
        net->slabs[slabID].slabIsDead == NN_SLAB_IS_DEAD)
    {
        NN_ReleaseAcquiredNet(netID);
        return ZERROR_BAD_SLAB_ID;
    }


    resultID = NNi_SlabApplyChanges(&net, slabID, &((*net).slabs[slabID]), r, m);
    if (resultID != ZERROR_NO_ERROR)
    {
        NN_ReleaseAcquiredNet(netID);
        return resultID;
    }


    return NN_ReleaseAcquiredNet(netID);
}



