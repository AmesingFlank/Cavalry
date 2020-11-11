#pragma once
#include "GpuCommons.h"
#include "Array.h"
#include <iostream>

template<typename Func,typename T>
__global__
inline void runAllTasks(T* tasks, int count, Func f) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    f(tasks[index]);
    //printf("here!\n");

}

template <typename T>
struct TaskQueue{
    GpuArray<T> tasks;
    GpuArray<int> head;

    TaskQueue(int capacity_,bool isCopyForKernel_ = false):tasks(capacity_,isCopyForKernel_),head(1,isCopyForKernel_){
       
       
    }

    TaskQueue<T> getCopyForKernel(){
        TaskQueue<T> copy(tasks.N,true);
        copy.tasks = tasks.getCopyForKernel();
        copy.head = head.getCopyForKernel();
        return copy;
    }


    __device__
    void push(const T& task){
        int index = atomicAdd(head.data,1);
        tasks.data[index]=task;
    }

    __host__ __device__
    int count() {
#ifdef __CUDA_ARCH__
        return *head.data;
#else
        int result;
        HANDLE_ERROR(cudaMemcpy(&result,head.data, sizeof(int),cudaMemcpyDeviceToHost));
        return result;
#endif
    }

    void clear() {
        HANDLE_ERROR(cudaMemset(head.data, 0, sizeof(int)));
    }

    template<typename Func>
    void forAll(Func f) {
        int tasksCount = count();
        if (tasksCount == 0) {
            std::cout << "no tasks to be done!" << std::endl;
            return;
        }
        int numThreads = min(tasksCount, MAX_THREADS_PER_BLOCK);
        int numBlocks = divUp(tasksCount, numThreads);
        std::cout << "tasksCount " << tasksCount << std::endl;
        runAllTasks << <numBlocks, numThreads >> > (tasks.data, tasksCount, f);
        CHECK_CUDA_ERROR("run for all");

    }

};