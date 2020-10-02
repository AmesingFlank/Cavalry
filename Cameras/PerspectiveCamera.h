#pragma once

#include "../Core/Camera.h"
class PerspectiveCamera: public Camera{
public:
 Ray genRay(const CameraSample& cameraSample) override;
};