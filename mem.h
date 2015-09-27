//memory ops
#ifndef _MEM_H_
#define _MEM_H_


#include <malloc.h>
#include <memory.h>

void * Mem_Allocate (size_t size);
void * Mem_AllocateCleared (size_t size);
void Mem_Free (void * ptr);
void * Mem_Reallocate (void * ptr, size_t size);

#endif
