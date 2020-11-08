#pragma once
#include "GpuCommons.h"
#include "Array.h"

template <typename T>
struct TaskQueue{
    GpuArray<T> tasks;
    GpuArray<int> head;

    TaskQueue(int capacity_,bool isCopyForKernel_):tasks(capacity,isCopyForKernel_),head(1,isCopyForKernel_){
       
       
    }

    TaskQueue<T> getCopyForKernel(){
        TaskQueue<T> copy(tasks.N,true);
        copy.tasks.data = tasks.data;
        copy.head.data = head.data;
        return copy;
    }


    __device__
    void push(const T& task){
        int index = atomicAdd(head);
        tasks.data[index]=task;
    }

    int count() {
        int result;
        HANDLE_ERROR(cudaMemcpy(&result,head.data, sizeof(int),cudaMemcpyDeviceToHost));
        return result;
    }

};