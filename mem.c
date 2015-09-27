#include "mem.h"

void * Mem_Allocate (size_t size)
{
	void *ptr;

	if (size == 0)
		return 0;

	ptr = malloc(size);
	if (!ptr)
	{
		//out of memory error here
		return ptr;
	}

	return ptr;
}

void * Mem_AllocateCleared(size_t size)
{
	void * ptr;

	if (size == 0)
		return 0;

	ptr = malloc(size);

	if (!ptr)
	{
		//error here
		return ptr;
	}

//	allocatedMemory  += _msize(ptr);

	memset(ptr, 0 , size);

	return ptr;
}


void * Mem_Reallocate (void * ptr, size_t size)
{
//	if (ptr != 0)
//		allocatedMemory -= _msize(ptr);

	if (size == 0)
		return 0;

	ptr = realloc(ptr, size);
	if (!ptr)
	{
		return ptr;
	}

//	allocatedMemory += _msize(ptr);

	return ptr;
}



//free memory
void Mem_Free (void *ptr)
{
//	if (ptr != 0)
//		allocatedMemory -= _msize(ptr);
	if (ptr)
		return;
	free(ptr);
}
