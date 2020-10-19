#include "TriangleMesh.h"

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