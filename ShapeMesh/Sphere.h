#pragma once

#include "../Core/TriangleMesh.h"
#include <vector>
#include "../Utils/MathsCommons.h"

inline TriangleMesh createSphereMesh(float radius, const glm::mat4& transform){
    std::cout << "creating sphere mesh" << std::endl;
    int thetaSegments = 360;
    int phiSegments = 180;

    int trianglesCount = thetaSegments * phiSegments * 2;
    int verticesCount = thetaSegments * (phiSegments+1);

    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<int3> indices;

    glm::mat3 normalMat = getTransformForNormal(transform);

    for(int theta = 0;theta < thetaSegments;++theta){
        for(int phi = 0 ;phi <= phiSegments;++phi){
            float3 pos = make_float3(
                sin(theta) * sin(phi),
                cos(phi),
                cos(theta) * sin(phi)
            );
            float3 normal = pos;

            positions.push_back(apply(transform,pos*radius));
            normals.push_back(normalMat * normal);

            if(phi < phiSegments){
                int thisIndex = positions.size()-1;
                int downIndex = thisIndex + 1;
                int rightIndex = (thisIndex + phiSegments) % verticesCount;
                int downRightIndex = rightIndex + 1;
                int3 triangle0 = make_int3(thisIndex,downIndex,downRightIndex);
                int3 triangle1 = make_int3(thisIndex,downRightIndex,rightIndex);
                indices.push_back(triangle0);
                indices.push_back(triangle1);
            }
        }
    }

    if(positions.size()!=verticesCount){
        SIGNAL_ERROR("positions size wrong");
    }
    if(normals.size()!=verticesCount){
        SIGNAL_ERROR("normals size wrong");
    }
    if(indices.size()!=trianglesCount){
        SIGNAL_ERROR("indices size wrong");
    }
    TriangleMesh mesh(trianglesCount,verticesCount,true,false);
    mesh.positions.cpu = positions;
    mesh.normals.cpu = normals;
    mesh.indices.cpu = indices;

    std::cout << "created sphere mesh" << std::endl;
    return mesh;
}