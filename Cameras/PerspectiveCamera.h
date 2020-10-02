#pragma once

#include "../Core/Camera.h"
class PerspectiveCamera: public Camera{
public:
	virtual Ray genRay(const CameraSample& cameraSample) const override;
};