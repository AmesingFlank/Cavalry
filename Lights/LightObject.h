#pragma once

#include "PointLight.h"
#include "DiffuseAreaLight.h"

#include "../Utils/Variant.h"

using LightVariant = Variant<PointLight,EnvironmentMap,DiffuseAreaLight>;

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
	Spectrum sampleRayToPoint(const float3& position,const float4& randomSource, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const {
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

	void buildCpuReferences(const SceneHandle& scene){
		auto visitor = [&](auto& arg) {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Light,typename T>::value) {
				return arg.T::buildCpuReferences(scene);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	};


	__device__
	void buildGpuReferences(const SceneHandle& scene){
		auto visitor = [&](auto& arg) {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Light,typename T>::value) {
				return arg.T::buildGpuReferences(scene);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	};

	static LightObject createFromObjectDefinition(const ObjectDefinition& def,const glm::mat4 transform){
		if (def.objectName == "diffuse") {
			Spectrum color = make_float3(1, 1, 1);
			
			if (def.params.hasNumList("L")) {
				std::vector<float> colorVec = def.params.getNumList("L");
				if(colorVec.size()==3){
					color.x = colorVec[0];
					color.y = colorVec[1];
					color.z = colorVec[2];
				}
				else{
					float kelvin = colorVec[0];
					float scale = colorVec[1];
					color = colorTemperatureToRGB(kelvin)*scale;
				}
			}
			
			return DiffuseAreaLight(color);
		}
		return LightObject(EnvironmentMap());
	}

	void prepareForRender() {
		auto visitor = [&](auto& arg) {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Light, typename T>::value) {
				return arg.T::prepareForRender();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		visit(visitor);
	}
};
