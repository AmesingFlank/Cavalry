#pragma once
#include "../Core/BoundingBox.h"
#include "../Utils/Array.h"
#include "../Core/Triangle.h"
#include "../Core/VisibilityTest.h"


struct BVHNode
{
    AABB box;
    int primitiveIndexBegin;
    int primitiveIndexEnd;
    int leftChild;
    int rightChild;
    bool isLeaf;

    __device__
    int primitivesCount(){
        return primitiveIndexEnd-primitiveIndexBegin+1;
    }
};



// Nodes for Building and Optmization
struct BVHLeafNode{
    AABB box;
    int primitiveIndexBegin;
    int primitiveIndexEnd;
    int parent;
    float surfaceArea;
};

struct BVHInternalNode{
    AABB box;
    int leftChild;
    int rightChild;
    int parent;
    unsigned int visited;
    bool leftChildIsLeaf;
    bool rightChildIsLeaf;
    float surfaceArea;
};

struct BVHRestructureNode {
    AABB box;
    int primitiveIndexBegin;
    int primitiveIndexEnd;
    int leftChild;
    int rightChild;
    int parent;
    bool isLeaf;
    float surfaceArea;

    __device__
        int primitivesCount() {
        return primitiveIndexEnd - primitiveIndexBegin + 1;
    }
};