#include "../Utils/GpuCommons.h"
#include "../Utils/MathsCommons.h"


#include "BVH.h"
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// do not change
#define MORTEN_BITS_PER_DIMENSION 10



BVH::BVH():primitivesCount(0),nodes(0,true){

}

BVH::BVH(int primitivesCount_,bool isCopyForKernel_):primitivesCount(primitivesCount_),nodes(primitivesCount_ * 2 - 1,isCopyForKernel_){

}

BVH BVH::getCopyForKernel(){
    BVH copy(primitivesCount,true);
    copy.nodes.gpu.data = nodes.gpu.data;
    return copy;
}

struct BVHLeafNode{
    AABB box;
    int primitiveIndex;
    int parent;
};

struct BVHInternalNode{
    AABB box;
    int leftChild;
    int rightChild;
    int parent;
    unsigned int visited;
    bool leftChildIsLeaf;
    bool rightChildIsLeaf;
};


__global__
void fillLeafBoundingBoxes(Triangle* primitivesDevice, int primitivesCount,BVHLeafNode* nodes ){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= primitivesCount) return;

    nodes[index].box = primitivesDevice[index].getBoundingBox();
    nodes[index].primitiveIndex = index;
}


__host__ __device__
unsigned int shiftMorton(unsigned int x){
    if (x >= (1 << MORTEN_BITS_PER_DIMENSION)) 
        x= 1 << MORTEN_BITS_PER_DIMENSION-1;
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    return x;
};


__global__
void fillLeafMortonCodes(Triangle* primitivesDevice, int primitivesCount,BVHLeafNode* leaves,unsigned int* leafMortonCodes,AABB sceneBounds ){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= primitivesCount) return;

    float3 centroid = (leaves[index].box.centroid() - sceneBounds.centroid()) / sceneBounds.extent();
    int factor = 1<<MORTEN_BITS_PER_DIMENSION;
    unsigned int x = centroid.x * factor;
    unsigned int y = centroid.y * factor;
    unsigned int z = centroid.z * factor;

    unsigned int code = (shiftMorton(x) << 2) | (shiftMorton(y) << 1) | shiftMorton(z);
    leafMortonCodes[index] = code;
}



__host__ __device__
int sign(int val) {
    return (val >= 0)?1:-1;
}

__host__ __device__
int commonPrefixLength(unsigned int* mortonCodes, unsigned int i,unsigned int j){
    unsigned int mortonI = mortonCodes[i];
    unsigned int mortonJ = mortonCodes[j];
    if(mortonI != mortonJ){
        for(int bit = 3*MORTEN_BITS_PER_DIMENSION-1; bit>=0;--bit){
            if(((mortonI >> bit) & 1) != ((mortonJ >> bit) & 1)){
                return 3*MORTEN_BITS_PER_DIMENSION-1 - bit;
            }
        }
    }

    for(int bit = 31; bit>=0;--bit){
        if(((i >> bit) & 1) != ((j >> bit) & 1)){
            return (31 - bit) + 3*MORTEN_BITS_PER_DIMENSION;
        }
    }
    
    return 32+3*MORTEN_BITS_PER_DIMENSION; // shuoldn't happen
}

__global__ 
void buildRadixTree(int leavesCount,unsigned int* leafMortonCodes, BVHLeafNode* leaves, BVHInternalNode* internals){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= leavesCount - 1) return;

    int d = 1;
    if(i > 0){
        d = sign(commonPrefixLength(leafMortonCodes,i,i+1) - commonPrefixLength(leafMortonCodes,i,i-1));
    }

    int minCommonPrefixLength = 0;
    if(i > 0){
        minCommonPrefixLength = commonPrefixLength(leafMortonCodes,i,i-d);
    }

    unsigned int maxLength = 2;

    while( i+ maxLength*d >=0 && i+maxLength*d < leavesCount && commonPrefixLength(leafMortonCodes,i,i+maxLength*d) > minCommonPrefixLength){
        maxLength *= 2;
    }
    

    unsigned int length = 0;
    for(int power = 1;; ++power){
        unsigned int t = maxLength >> power;

        if(t==0){
            break;
        }
        int j = i+(length+t)*d;
        if( j >= 0 && j <leavesCount && commonPrefixLength(leafMortonCodes,i,j) > minCommonPrefixLength){
            length += t;
        }
    }

    int j = i + length *  d; // other end;
    int prefixLength = commonPrefixLength(leafMortonCodes,i,j);

    if (i == 0) {
        j = leavesCount - 1;
        prefixLength = 0;
    }


    unsigned int distanceToSplit = 0;
    for(int power = 1;; ++power){
        unsigned int t = ceil( (float)length / ((float) (1<<power)) );
        
        int j = i + (distanceToSplit + t)*d;
        if(j >= 0 && j <leavesCount && commonPrefixLength(leafMortonCodes,i,j) > prefixLength){
            distanceToSplit += t;
        }

        if (t <= 1) {
            break;
        }
    }

    

    int splitPos = i + distanceToSplit*d + min(0,d);

    int leftChild = splitPos;
    int rightChild = splitPos + 1;

    internals[i].leftChild = leftChild;
    internals[i].rightChild = rightChild;

    internals[i].leftChildIsLeaf = min(i,j)==leftChild;
    internals[i].rightChildIsLeaf = (max(i,j) == rightChild) || (rightChild == leavesCount - 1);

    if(internals[i].leftChildIsLeaf){
        leaves[leftChild].parent = i;
    }
    else{
        internals[leftChild].parent = i;
    }

    if(internals[i].rightChildIsLeaf){
        leaves[rightChild].parent = i;
    }
    else{
        internals[rightChild].parent = i;
    }
    /*
    if (i >= 0) {
        printf("i:%d,\t j:%d,\t d:%d,\t splitPos: %d,\t minDelta:%d,\t delta:%d,\t maxLength:%d,\t   length: %d,\t   leavesCount:%d ,\n", 
            i, j, d,splitPos, minCommonPrefixLength, prefixLength, maxLength,length,leavesCount);
    }
    */

    internals[i].visited = 0;
}

__global__ 
void computeBounds(int leavesCount, BVHLeafNode* leaves, BVHInternalNode* internals){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= leavesCount) return;

    int curr = leaves[i].parent;
    //printf("this is leaf %d, next is %d\n", i,curr);
    while(atomicInc(&(internals[curr].visited),2) > 0){
        //printf("doing bounds for node %d\n", curr);
        AABB boundsLeft;
        AABB boundsRight;
        BVHInternalNode& node = internals[curr];
        if(node.leftChildIsLeaf){
            boundsLeft = leaves[node.leftChild].box;
        }
        else{
            boundsLeft = internals[node.leftChild].box;
        }

        if(node.rightChildIsLeaf){
            boundsRight = leaves[node.rightChild].box;
        }
        else{
            boundsRight = internals[node.rightChild].box;
        }

        node.box = unionBoxes(boundsLeft,boundsRight);
        if (curr == 0) {
            break;
        } 

        curr = node.parent;
    }
}

__device__
void copyLeafNode(BVHLeafNode& leaf, BVHNode& node){
    node.box = leaf.box;
    node.isLeaf = true;
    node.primitiveIndex = leaf.primitiveIndex;
}

__global__ 
void mergeNodesArray(int leavesCount, BVHLeafNode* leaves, BVHInternalNode* internals, BVHNode* nodes){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= leavesCount-1) return;

    nodes[i].isLeaf = false;

    nodes[i].box = internals[i].box;

    if(internals[i].leftChildIsLeaf){
        int leftChildLeaf = leavesCount - 1 + internals[i].leftChild;
        nodes[i].leftChild = leftChildLeaf;
        copyLeafNode(leaves[internals[i].leftChild],nodes[leftChildLeaf]);
    }
    else{
        nodes[i].leftChild = internals[i].leftChild;
    }

    if(internals[i].rightChildIsLeaf){
        int rightChildLeaf = leavesCount - 1 + internals[i].rightChild;
        nodes[i].rightChild = rightChildLeaf;
        copyLeafNode(leaves[internals[i].rightChild],nodes[rightChildLeaf]);
    }
    else{
        nodes[i].rightChild = internals[i].rightChild;
    }
}

BVH BVH::build(Triangle* primitivesDevice, int primitivesCount,const AABB& sceneBounds){
    std::cout << "started building bvh" << std::endl;
    BVH bvh(primitivesCount);

    GpuArray<BVHLeafNode> leaves(primitivesCount);

    
    int numThreadsPrimitives = min(primitivesCount,MAX_THREADS_PER_BLOCK);
    int numBlocksPrimitives = divUp(primitivesCount,numThreadsPrimitives);
    fillLeafBoundingBoxes <<< numBlocksPrimitives, numThreadsPrimitives >>> (primitivesDevice,primitivesCount, leaves.data);
    CHECK_IF_CUDA_ERROR("fill leaf bounding boxes");

    GpuArray<unsigned int> leafMortonCodes(primitivesCount);
    fillLeafMortonCodes<<< numBlocksPrimitives, numThreadsPrimitives >>> (primitivesDevice,primitivesCount,leaves.data, leafMortonCodes.data,sceneBounds);
    CHECK_IF_CUDA_ERROR("fill leaf morton");

    thrust::stable_sort_by_key(thrust::device, leafMortonCodes.data,leafMortonCodes.data+primitivesCount,leaves.data,thrust::less<unsigned int>());

    GpuArray<BVHInternalNode> internals(primitivesCount-1);

    int numThreadsInternals = min(primitivesCount-1,MAX_THREADS_PER_BLOCK);
    int numBlocksInternals = divUp(primitivesCount-1,numThreadsPrimitives);
    buildRadixTree <<< numBlocksInternals, numThreadsInternals >>> (primitivesCount,leafMortonCodes.data, leaves.data,internals.data);
    CHECK_IF_CUDA_ERROR("build radix tree");

    computeBounds <<< numBlocksPrimitives, numThreadsPrimitives >>> (primitivesCount,leaves.data,internals.data);
    CHECK_IF_CUDA_ERROR("compute bounds");


    mergeNodesArray <<< numBlocksInternals, numThreadsInternals >>> (primitivesCount,leaves.data,internals.data, bvh.nodes.gpu.data);
    CHECK_IF_CUDA_ERROR("merge nodes array");    

    return bvh;
}