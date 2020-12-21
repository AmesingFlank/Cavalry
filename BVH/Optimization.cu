#include "../Utils/GpuCommons.h"
#include "Optimization.h"
#include "../Utils/Array.h"
#include "../Utils/Utils.h"
#include <vector>

std::vector<unsigned int> scheduleCPU = {

/*round 1*/
0b00011,0b00110,0b00101,0b01001,0b01010,0b01100,0b10001,0b10010,0b10100,0b11000, /*2bits; greatest at 5 or less*/ 
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, /*round 1 vacancies*/

/*round 2*/
0b00111,0b01011,0b01101,0b01110,0b10011,0b10101,0b10110,0b11001,0b11010,0b11100,/*3bits; greatest at 5 or less*/     
0b100001,0b100010,0b100100,0b101000,0b110000,/*2 bits; 6th bit set*/  
0b1000001,0b1000010,0b1000100,0b1001000,0b1010000,0b1100000/*2 bits; 7th bit set*/, 
0,0,0,0,0,0,0,0,0,0,0, /*round 2 vacancies*/


/*round 3*/
0b100011,0b100101,0b100110,0b101001,0b101010,0b101100,0b110001,0b110010,0b110100,0b111000, /*3bits; 6th set*/
0b1000011,0b1000101,0b1000110,0b1001001,0b1001010,0b1001100,0b1010001,0b1010010,0b1010100,0b1011000,0b1100001,0b1100010,0b1100100,0b1101000,0b1110000,/*3bits; 7th set*/
0b0001111,0b0010111,0b0011110, /*some of 4bits*/
0,0,0,0, /*round 3 vacancies*/


/*round 4; all the remaining 4 bits*/
0b1000111,0b0011011,0b0101011,0b1001011,0b0110011,0b1010011,0b1100011,0b0011101,0b0101101,0b1001101,0b0110101,0b1010101,0b1100101,0b0111001,0b1011001,0b1101001,0b1110001,0b0100111,0b0101110,0b1001110,0b0110110,0b1010110,0b1100110,0b0111010,0b1011010,0b1101010,0b1110010,0b0111100,0b1011100,0b1101100,0b1110100,0b1111000,

/*round 5; all the 5 bits*/
0b0011111,0b0101111,0b1001111,0b0110111,0b1010111,0b1100111,0b0111011,0b1011011,0b1101011,0b1110011,0b0111101,0b1011101,0b1101101,0b1110101,0b1111001,0b0111110,0b1011110,0b1101110,0b1110110,0b1111010,0b1111100,
0,0,0,0,0,0,0,0,0,0,0

};

__global__
void tryCollapseNodes(int nodesCount, BVHNode* nodes, unsigned int* visited){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nodesCount) return;

    if(!nodes[i].isLeaf){
        return;
    }
    int curr = nodes[i].parent;

    while(atomicInc(&(visited[curr]),2) > 0){
        BVHNode& node = nodes[curr];
      
        BVHNode& leftChild = nodes[node.leftChild];
        BVHNode& rightChild = nodes[node.rightChild];

        if(leftChild.isLeaf && rightChild.isLeaf){
            constexpr float primCost = 1;
            constexpr float boxCost = 0.0006;
            
            float originalCost =
                (leftChild.primitivesCount() * primCost * leftChild.surfaceArea/node.surfaceArea) + boxCost+
                (rightChild.primitivesCount() * primCost * rightChild.surfaceArea / node.surfaceArea) + boxCost;

            float temp = leftChild.surfaceArea / node.surfaceArea + rightChild.surfaceArea / node.surfaceArea;
            
            float collapsedCost = (leftChild.primitivesCount() + rightChild.primitivesCount()) * primCost;

            if(collapsedCost <  originalCost ){
                node.isLeaf = true;
                node.primitiveIndexBegin = leftChild.primitiveIndexBegin;
                node.primitiveIndexEnd = rightChild.primitiveIndexEnd;
                continue;
            }
        }
        return;
    }
}


void collapseNodes(int primitivesCount,GpuArray<BVHNode>& nodes){
    GpuArray<unsigned int> visited(nodes.N);
    int numBlocks,numThreads;
    setNumBlocksThreads(nodes.N, numBlocks, numThreads);
    tryCollapseNodes<<<numBlocks,numThreads>>> (nodes.N,nodes.data,visited.data);
    CHECK_IF_CUDA_ERROR("collapseNodes");
}


void optimizeBVH(int primitivesCount,GpuArray<BVHNode>& nodes){

}