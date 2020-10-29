#include "TriangleMesh.h"
#include <iostream>
#include <happly.h>

__host__ 
TriangleMesh::TriangleMesh():
trianglesCount(0),
verticesCount(0),
hasVertexNormals(0),
positions(0,true),
normals(0,true),
texCoords(0,true),
indices(0,true)
{

}

__host__ 
TriangleMesh::TriangleMesh(int trianglesCount_, int verticesCount_,bool hasVertexNormals_, bool isCopyForKernel_):
trianglesCount(trianglesCount_),
verticesCount(verticesCount_),
hasVertexNormals(hasVertexNormals_),
positions(verticesCount_,isCopyForKernel_),
normals(verticesCount_,isCopyForKernel_),
texCoords(verticesCount,isCopyForKernel_),
indices(trianglesCount,isCopyForKernel_)
{

}


TriangleMesh TriangleMesh::getCopyForKernel(){
    TriangleMesh copy(trianglesCount,verticesCount,hasVertexNormals,true);
    copy.positions.gpu.data = positions.gpu.data;
    copy.normals.gpu.data = normals.gpu.data;
    copy.texCoords.gpu.data = texCoords.gpu.data;
    copy.indices.gpu.data = indices.gpu.data;
    return copy;
}

void TriangleMesh::copyToDevice() {
    positions.copyToDevice();
    normals.copyToDevice();
    texCoords.copyToDevice();
    indices.copyToDevice();
}


TriangleMesh TriangleMesh::createFromPLY(const std::string& filename,const glm::mat4& transform){
    happly::PLYData plyIn(filename);

    std::vector<float> positionsX = plyIn.getElement("vertex").getProperty<float>("x");
    std::vector<float> positionsY= plyIn.getElement("vertex").getProperty<float>("y");
    std::vector<float> positionsZ= plyIn.getElement("vertex").getProperty<float>("z");

    std::vector<std::vector<int>> indices = 
        plyIn.getElement("face").getListProperty<int>("vertex_indices");

    int trianglesCount = indices.size();
    int verticesCount = positionsX.size();
    TriangleMesh mesh(trianglesCount,verticesCount,false,false);

    for(int i = 0;i<verticesCount;++i){
        float3 pos = make_float3(positionsX[i],positionsY[i],positionsZ[i]);
        pos = to_float3(transform * glm::vec4(to_vec3(pos), 1.f));
        mesh.positions.cpu.data[i] = pos;
    }

    for(int i = 0;i<trianglesCount;++i){
        int3 thisIndices = make_int3(indices[i][0],indices[i][1],indices[i][2]);
        mesh.indices.cpu.data[i] = thisIndices;
    }

    return mesh;

}


TriangleMesh TriangleMesh::createFromParams(const Parameters& params,const glm::mat4& transform,const std::filesystem::path& basePath){
    
    if (params.hasString("filename")) {
        std::string plyPathString = params.getString("filename");
        std::filesystem::path plyRelativePath(plyPathString);
        std::string filename = (basePath / plyRelativePath).generic_string();

        return TriangleMesh::createFromPLY(filename, transform);
    }
    else if(params.hasNumList("P") && params.hasNumList("indices")){
        std::vector<float> positionsFloats = params.getNumList("P");
        std::vector<float> indicesFloats = params.getNumList("indices");

        int trianglesCount = indicesFloats.size() / 3;
        int verticesCount = positionsFloats.size() / 3;


        std::vector<int3> indices(trianglesCount);
        for (int i = 0; i < trianglesCount; ++i) {
            indices[i] = make_int3(indicesFloats[i * 3], indicesFloats[i * 3+1], indicesFloats[i * 3+2]);
        }

        std::vector<float3> positions(verticesCount);
        for (int i = 0; i < verticesCount; ++i) {
            positions[i] = make_float3(positionsFloats[i * 3], positionsFloats[i * 3 + 1], positionsFloats[i * 3 + 2]);
        }

        bool hasNormal = params.hasNumList("N");

        TriangleMesh mesh(trianglesCount, verticesCount, hasNormal, false);
        mesh.positions.cpu = positions;
        mesh.indices.cpu = indices;

        if (hasNormal) {
            std::vector<float> normalsFloats = params.getNumList("N");
            std::vector<float3> normals(verticesCount);
            for (int i = 0; i < verticesCount; ++i) {
                normals[i] = make_float3(normalsFloats[i * 3], normalsFloats[i * 3 + 1], normalsFloats[i * 3 + 2]);
            }
            mesh.normals.cpu = normals;
        }

        return mesh;
    }
    else {
        SIGNAL_ERROR("No valid inputs for triangle mesh");
    }
    
}

void TriangleMesh::computeArea(){
    surfaceArea = 0;
    for(int i = 0;i<trianglesCount;++i){
        int3 vertices = indices.cpu.data[i];
        float3 p0 = positions.cpu.data[vertices.x];
        float3 p1 = positions.cpu.data[vertices.y];
        float3 p2 = positions.cpu.data[vertices.z];
        surfaceArea += length(cross(p1-p0,p2-p0)) * 0.5f;
    }
}