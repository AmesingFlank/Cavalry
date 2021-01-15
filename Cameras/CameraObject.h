#pragma once

#include "PerspectiveCamera.h"


#include "Color.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/Variant.h"
#include "../Core/Parameters.h"



class CameraObject : public Variant<PerspectiveCamera> {
public:

	__host__ __device__
	CameraObject() {}

	template<typename V>
	__host__ __device__
	CameraObject(const V& v) : Variant<PerspectiveCamera>(v) {}

	__host__ __device__
	CameraObject(const CameraObject& other) : Variant<PerspectiveCamera>(other.value) {}



	__host__ __device__
	CameraObject& operator=(const CameraObject& other) {
		value = other.value;
		return *this;
	}

	__host__ __device__
	Ray genRay(const CameraSample& cameraSample) const {
		auto visitor = [&](auto& arg) -> Ray {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Camera,typename T>::value) {
				return arg.T::genRay(cameraSample);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	
    __host__ __device__
    void pdf(const Ray &ray, float* outputPositionProbability, float* outputDirectionProbability) const {
		auto visitor = [&](auto& arg){
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Camera,typename T>::value) {
				arg.T::pdf(ray,outputPositionProbability,outputDirectionProbability);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		visit(visitor);
	}

	static CameraObject createFromObjectDefinition(const ObjectDefinition& def,const glm::mat4& cameraToWorld, int filmWidth_, int filmHeight_){
		return CameraObject(PerspectiveCamera::createFromParams(def.params,cameraToWorld,filmWidth_,filmHeight_));
	}
};
