#pragma once
#include "../Core/BoundingBox.h"
#include "../Utils/Array.h"
#include "../Core/Triangle.h"
#include "../Core/VisibilityTest.h"
#include "BVHNode.h"

// In this file, a "Primitive" means a Triangle.

struct BVH{
    int primitivesCount;
    ArrayPair<BVHNode> nodes;

    BVH();
    BVH(int primitivesCount_,bool isCopyForKernel_ = false);

    BVH getCopyForKernel();

    static BVH build(Triangle* trianglesDevice, int trianglesCount, const AABB& sceneBounds);

    // BVH Traversal Optimisation: go to nearest child first, and don't expand a node if minDist < result.distance

    __device__
    bool intersect(IntersectionResult& result, const Ray& ray, Triangle* primitives) const {

        BVHNode* nodesData = nodes.gpu.data;

        float3 invDir = make_float3(1, 1, 1) / ray.direction;

        result.intersected = false;
        result.distance = FLT_MAX;
        int resultTriangle = -1;

        BVHNode& root = nodesData[0];
        float minDistanceToRoot;
        if (!root.box.intersect(ray,minDistanceToRoot,invDir)) {
            return false;
        }


        int stack[64];
        stack[0] = 0;
        int top = 0;

        float2 resultUV;

#define PUSH(x) ++top; stack[top]=x;


        while(top >= 0){
            int curr = stack[top];
            --top;

            BVHNode& node = nodesData[curr];

            if(node.isLeaf){
                IntersectionResult thisResult;

                for(int index = node.primitiveIndexBegin; index <= node.primitiveIndexEnd;++index){
                    float u, v;
                    if(primitives[index].intersect(thisResult,ray,u,v)){
                        if(result.intersected == false || thisResult.distance < result.distance){
                            result = thisResult;
                            resultTriangle = index;
                            resultUV.x = u;
                            resultUV.y = v;
                        }
                    }
                }
            }
            else{
                float minDistLeft;
                float minDistRight;

                nodesData[node.leftChild].box.intersect(ray,minDistLeft,invDir);
                nodesData[node.rightChild].box.intersect(ray, minDistRight,invDir);

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

        if (result.intersected) {
            primitives[resultTriangle].fillIntersectionInformation(result, ray, resultUV.x, resultUV.y);
        }

        return result.intersected;

#undef PUSH
    }

    __device__
    bool testVisibility(const VisibilityTest& test, Triangle* primitives) const {
        BVHNode* nodesData = nodes.gpu.data;

        Ray ray = test.ray;
        float3 invDir = make_float3(1,1,1) / ray.direction;

        BVHNode& root = nodesData[0];
        float minDistanceToRoot;
        if (!root.box.intersect(ray, minDistanceToRoot,invDir)) {
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
                for (int index = node.primitiveIndexBegin; index <= node.primitiveIndexEnd; ++index) {
                    Triangle* prim = &primitives[index];
                    if (prim->mesh == test.sourceMesh || prim->mesh == test.targetMesh) {
                        continue;
                    }
                    IntersectionResult thisResult;
                    float u, v;
                    if (prim->intersect(thisResult, ray,u,v)) {
                        if (test.useDistanceLimit) {
                            if (thisResult.distance < test.distanceLimit) {
                                //printf("intersected!  source: %d, target: %d, intersection:%d\n",  test.sourceTriangleIndex, test.targetTriangleIndex, index);
                                return false;
                            }
                        }
                        else {
                            //printf("intersected!  source: %d, target: %d, intersection:%d\n", test.sourceTriangleIndex, test.targetTriangleIndex, index);
                            return false;
                        }
                    }
                }
                
            }
            else {
                
                float minDistLeft;
                float minDistRight;

                nodesData[node.leftChild].box.intersect(ray, minDistLeft,invDir);
                nodesData[node.rightChild].box.intersect(ray, minDistRight,invDir);

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

