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

	static CameraObject createFromObjectDefinition(const ObjectDefinition& def,const float3& eye_, const float3& center_, const float3& up_, int filmWidth_, int filmHeight_){
		return CameraObject(PerspectiveCamera::createFromParams(def.params,eye_,center_,up_,filmWidth_,filmHeight_));
	}
};
