#pragma once


#include "../Core/TriangleMesh.h"
#include <vector>
#include "../Utils/MathsCommons.h"

TriangleMesh createDiskMesh(float radius,float height, const glm::mat4& transform);