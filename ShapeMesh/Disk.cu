#include "Disk.h"


TriangleMesh createDiskMesh(float radius,float height, const glm::mat4& transform){
    std::cout << "creating sphere mesh" << std::endl;
    int segments = 360;

    int trianglesCount = segments ;
    int verticesCount = segments + 1;

    std::vector<float3> positions;
    std::vector<int3> indices;


    for(int segmentID = 0 ; segmentID < segments; ++segmentID){
        float theta = 2.f * M_PI * (float)segmentID / (float)segments;
        float3 p = make_float3(
            sin(theta) * radius,
            cos(theta) * radius,
            height
        );
        positions.push_back(apply(transform,p));
    }

    float3 center = make_float3(0,0,height);
    positions.push_back(apply(transform, center));

    
    if(positions.size()!=verticesCount){
        SIGNAL_ERROR("positions size wrong in Disk shape mesh %d %d \n",(int)positions.size(),verticesCount);
    }

    for(int segmentID = 0 ; segmentID < segments; ++segmentID){
        int3 triangle = make_int3( (segmentID+1) % segments, segmentID,  verticesCount-1);

        indices.push_back(triangle);
    }
    

    if(indices.size()!=trianglesCount){
        SIGNAL_ERROR("indices size wrong %d %d\n",indices.size(),trianglesCount);
    }

    TriangleMesh mesh(trianglesCount,verticesCount,false,false);
    mesh.positions.cpu = positions;
    mesh.indices.cpu = indices;
    mesh.shapeType = MeshShapeType::Disk;

    mesh.computeArea(); std::cout << "disk area " << mesh.area() <<"   "<<radius<<"   "<<height<< std::endl;
    std::cout << "created disk mesh " << std::endl;
    return mesh;
}