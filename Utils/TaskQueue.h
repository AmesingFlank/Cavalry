#pragma once
#include "GpuCommons.h"
#include "Array.h"

template <typename T>
struct TaskQueue{
    GpuArray<T> tasks;
    int head = 0;

    TaskQueue(int capacity_,bool isCopyForKernel_):tasks(capacity,isCopyForKernel_){

    }

    TaskQueue<T> getCopyForKernel(){
        TaskQueue
    }


    __device__
    void push(const T& task){
        int index = atomicAdd(&head);
        tasks.data[index]=task;
    }

};