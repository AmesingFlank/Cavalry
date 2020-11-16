#include "TriangleMesh.h"
#include <iostream>
#include <happly.h>
#include "Scene.h"

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

/*
TriangleMesh TriangleMesh::getCopyForKernel(){
    TriangleMesh copy(trianglesCount,verticesCount,hasVertexNormals,true);
    copy.positions.gpu = positions.gpu.getCopyForKernel();
    copy.normals.gpu = normals.gpu.getCopyForKernel();
    copy.texCoords.gpu = texCoords.gpu.getCopyForKernel();
    copy.indices.gpu = indices.gpu.getCopyForKernel();
    return copy;
}
*/

void TriangleMesh::copyToDevice() {
    positions.copyToDevice();
    normals.copyToDevice();
    texCoords.copyToDevice();
    indices.copyToDevice();
}


TriangleMesh TriangleMesh::createFromPLY(const std::string& filename,const glm::mat4& transform){
    happly::PLYData plyIn(filename);

    auto& vertices = plyIn.getElement("vertex");

    std::vector<float> positionsX = vertices.getProperty<float>("x");
    std::vector<float> positionsY= vertices.getProperty<float>("y");
    std::vector<float> positionsZ= vertices.getProperty<float>("z");

    std::vector<std::vector<int>> indices = 
        plyIn.getElement("face").getListProperty<int>("vertex_indices");

    bool hasVertexNormal = vertices.hasProperty("nx") && vertices.hasProperty("ny") && vertices.hasProperty("nz");

    int trianglesCount = indices.size();
    int verticesCount = positionsX.size();
    TriangleMesh mesh(trianglesCount,verticesCount,hasVertexNormal,false);

    for(int i = 0;i<verticesCount;++i){
        float3 pos = make_float3(positionsX[i],positionsY[i],positionsZ[i]);
        pos = to_float3(transform * glm::vec4(to_vec3(pos), 1.f));
        mesh.positions.cpu.data[i] = pos;
    }

    for(int i = 0;i<trianglesCount;++i){
        int3 thisIndices = make_int3(indices[i][0],indices[i][1],indices[i][2]);
        mesh.indices.cpu.data[i] = thisIndices;
    }

    if (hasVertexNormal) {
        std::vector<float> normalX = vertices.getProperty<float>("nx");
        std::vector<float> normalY = vertices.getProperty<float>("ny");
        std::vector<float> normalZ = vertices.getProperty<float>("nz");
        glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(transform)));
        for (int i = 0; i < verticesCount; ++i) {
            float3 normal = make_float3(normalX[i], normalY[i], normalZ[i]);
            normal = normalize(to_float3(normalMat * to_vec3(normal)));
            mesh.normals.cpu.data[i] = normal;
        }
    }

    bool hasUV = vertices.hasProperty("u") && vertices.hasProperty("v");
    if (hasUV) {
        std::vector<float> u = vertices.getProperty<float>("u");
        std::vector<float> v = vertices.getProperty<float>("v");

        for (int i = 0; i < verticesCount; ++i) {
            float2 uv = make_float2(u[i], v[i]);
            mesh.texCoords.cpu.data[i] = uv;
        }
    }



    return mesh;

}


TriangleMesh TriangleMesh::createFromObjectDefinition(const ObjectDefinition& def, const glm::mat4& transform, const std::filesystem::path& basePath) {
    if (def.objectName != "plymesh" && def.objectName != "trianglemesh") {
        SIGNAL_ERROR((std::string("unsupported shape type :")+def.objectName).c_str());
    }
    return TriangleMesh::createFromParams(def.params, transform, basePath);
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

        if (params.hasNumList("uv")) {
            std::vector<float> uvFloats = params.getNumList("uv");
            std::vector<float2> texCoords(verticesCount);
            for (int i = 0; i < verticesCount; ++i) {
                texCoords[i] = make_float2(uvFloats[i * 2], uvFloats[i * 2 + 1]);
            }
            mesh.texCoords.cpu = texCoords;
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

void TriangleMesh::copyTrianglesToScene(Scene& scene,int meshID){
    for(int i = 0;i<trianglesCount;++i){
        Triangle triangle(meshID,i);
        scene.trianglesHost.push_back(triangle);
    }
}