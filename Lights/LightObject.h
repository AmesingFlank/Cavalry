#pragma once

#include "PointLight.h"

#include "../Utils/Variant.h"

using LightVariant = Variant<PointLight,EnvironmentMap>;

class LightObject : public LightVariant {
public:

	__host__ __device__
	LightObject() {}

	template<typename V>
	__host__ __device__
	LightObject(const V& v) :LightVariant(v) {}

	__host__ __device__
	LightObject(const LightObject& other) : LightVariant(other.value) {}


	__host__ __device__
	LightObject& operator=(const LightObject& other) {
		value = other.value;
		return *this;
	}

	__host__ __device__
	Spectrum sampleRayToPoint(const float3& position,const float2& randomSource, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const {
		auto visitor = [&](auto& arg) -> Spectrum {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Light,typename T>::value) {
				return arg.T::sampleRayToPoint(position,randomSource,outputProbability,outputRay,outputVisibilityTest);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	static LightObject createFromObjectDefinition(const ObjectDefinition& def,const glm::mat4 transform){
		return LightObject(EnvironmentMap());
	}
};
