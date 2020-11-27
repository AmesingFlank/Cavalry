#pragma once
#include "../Core/BoundingBox.h"
#include "../Utils/Array.h"
#include "../Core/Triangle.h"
#include "../Core/VisibilityTest.h"

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

    // BVH Traversal Optimisation: go to nearest child first, and don't expand a node if minDist < result.distance

    __host__ __device__
    bool intersect(IntersectionResult& result, const Ray& ray, Triangle* primitives) const {
#ifdef __CUDA_ARCH__
        BVHNode* nodesData = nodes.gpu.data;
#else
        BVHNode* nodesData = nodes.cpu.data;
#endif
        result.intersected = false;
        result.distance = FLT_MAX;

        BVHNode& root = nodesData[0];
        float minDistanceToRoot;
        if (!root.box.intersect(ray,minDistanceToRoot)) {
            return false;
        }


        int stack[64];
        stack[0] = 0;
        int top = 0;

#define PUSH(x) ++top; stack[top]=x;


        while(top >= 0){
            int curr = stack[top];
            --top;

            BVHNode& node = nodesData[curr];

            if(node.isLeaf){
                IntersectionResult thisResult;

                if(primitives[node.primitiveIndex].intersect(thisResult,ray)){
                    if(result.intersected == false || thisResult.distance < result.distance){
                        result = thisResult;
                    }
                }
            }
            else{
                float minDistLeft;
                float minDistRight;

                nodesData[node.leftChild].box.intersect(ray,minDistLeft);
                nodesData[node.rightChild].box.intersect(ray, minDistRight);

                if (minDistLeft >= 0 &&  minDistLeft < result.distance && minDistRight >= 0 && minDistRight < result.distance) {
                    if (minDistLeft > minDistRight) {
                        PUSH(node.leftChild);
                        PUSH(node.rightChild);
                    }
                    else {
                        PUSH(node.rightChild);
                        PUSH(node.leftChild);
                    }
                }

                else if (minDistLeft >= 0 && minDistLeft < result.distance){
                    PUSH(node.leftChild);
                }
                else if (minDistRight >= 0 && minDistRight < result.distance) {
                    PUSH(node.rightChild);
                }

            }
            
        }

        return result.intersected;

#undef PUSH
    }

    __host__ __device__
    bool testVisibility(const VisibilityTest& test, Triangle* primitives) const {
#ifdef __CUDA_ARCH__
        BVHNode* nodesData = nodes.gpu.data;
#else
        BVHNode* nodesData = nodes.cpu.data;
#endif

        Ray ray = test.ray;

        BVHNode& root = nodesData[0];
        float minDistanceToRoot;
        if (!root.box.intersect(ray, minDistanceToRoot)) {
            SIGNAL_ERROR("shadow ray doesn't intersect root node");
        }

        int stack[64];
        stack[0] = 0;
        int top = 0;

#define PUSH(x) ++top; stack[top]=x;


        while (top >= 0) {
            int curr = stack[top];
            --top;

            BVHNode& node = nodesData[curr];
            

            if (node.isLeaf) {
                Triangle* prim = &primitives[node.primitiveIndex];
                if (prim->mesh->getID() == test.sourceGeometry || prim->mesh->getID() == test.targetGeometry) {
                    continue;
                }
                IntersectionResult thisResult;
                if (prim->intersect(thisResult, ray)) {
                    if (test.useDistanceLimit) {
                        if (thisResult.distance < test.distanceLimit) {
                            return false;
                        }
                    }
                    else {
                        return false;
                    }
                }
            }
            else {
                
                float minDistLeft;
                float minDistRight;

                nodesData[node.leftChild].box.intersect(ray, minDistLeft);
                nodesData[node.rightChild].box.intersect(ray, minDistRight);

                if (minDistLeft >= 0 && minDistRight >= 0 ) {
                    if (minDistLeft > minDistRight) {
                        PUSH(node.leftChild);
                        PUSH(node.rightChild);
                    }
                    else {
                        PUSH(node.rightChild);
                        PUSH(node.leftChild);
                    }
                }

                else if (minDistLeft >= 0 ) {
                    PUSH(node.leftChild);
                }
                else if (minDistRight >= 0 ) {
                    PUSH(node.rightChild);
                }
            }
        }
#undef PUSH

        return true;
    }
};

