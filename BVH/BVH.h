#pragma once
#include "../Core/BoundingBox.h"
#include "../Utils/Array.h"
#include "../Core/Triangle.h"

struct BVHNode
{
    AABB box;
    int primitiveIndex;
    int leftChild;
    int rightChild;
    bool isLeaf;
};

struct BVH{
    int primitivesCount;
    ArrayPair<BVHNode> nodes;

    BVH();
    BVH(int primitivesCount_,bool isCopyForKernel_ = false);

    BVH getCopyForKernel();

    static BVH build(Triangle* trianglesDevice, int trianglesCount, const AABB& sceneBounds);

    __host__ __device__
    bool intersect(IntersectionResult& result, const Ray& ray, Triangle* primitives) const {
#ifdef __CUDA_ARCH__
        BVHNode* nodesData = nodes.gpu.data;
#else
        BVHNode* nodesData = nodes.cpu.data;
#endif

        int stack[64];
        stack[0] = 0;
        int top = 0;

        result.intersected = false;
         

        while(top > 0){
            int curr = top;
            --top;

            BVHNode& node = nodesData[curr];
            if(node.box.intersect(ray)){
                if(node.isLeaf){
                    IntersectionResult thisResult;
                    if(primitives[node.primitiveIndex].intersect(thisResult,ray)){
                        if(result.intersected == false || thisResult.distance < result.distance){
                            result = thisResult;
                        }
                    }
                }
                else{
                    ++top;
                    stack[top] = node.leftChild;
                    ++top;
                    stack[top] = node.rightChild;
                }
            }
        }

        return result.intersected;
    }
};

