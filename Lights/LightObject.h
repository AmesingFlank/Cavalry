#pragma once

#include "PointLight.h"
#include "DiffuseAreaLight.h"

#include "../Utils/Variant.h"
#include "../Utils/Utils.h"
#include <filesystem>

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
	Spectrum sampleRayToPoint(const float3& position, SamplerObject& sampler, float& outputProbability, Ray& outputRay,VisibilityTest& outputVisibilityTest) const {
		auto visitor = [&](auto& arg) -> Spectrum {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Light,typename T>::value) {
				return arg.T::sampleRayToPoint(position,sampler,outputProbability,outputRay,outputVisibilityTest);
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

	static Spectrum vectorToSpectrum(const std::vector<float>& vec) {
		Spectrum color;
		if (vec.size() == 3) {
			color.x = vec[0];
			color.y = vec[1];
			color.z = vec[2];
		}
		else {
			float kelvin = vec[0];
			float scale = vec[1];
			color = colorTemperatureToRGB(kelvin) * scale;
		}
		return color;
	}

	static LightObject createFromObjectDefinition(const ObjectDefinition& def,const glm::mat4 transform,const std::filesystem::path& basePath, const std::unordered_map<std::string, Texture2D>& textures){
		Spectrum color = make_float3(1, 1, 1);
		// The "L"/"l" param is the same for all the light source supported.
		if (def.params.hasNumList("L")) {
			std::vector<float> colorVec = def.params.getNumList("L");
			color = vectorToSpectrum(colorVec);
		}
		else if (def.params.hasNumList("l")) {
			std::vector<float> colorVec = def.params.getNumList("l");
			color = vectorToSpectrum(colorVec);
		}
		
		if (def.objectName == "diffuse") {
			return DiffuseAreaLight(color);
		}
		if (def.objectName == "infinite") {
			// for infinite area light, "scale" is an alternative for "L"
			if (def.params.hasNumList("scale")) {
				std::vector<float> colorVec = def.params.getNumList("scale");
				color = vectorToSpectrum(colorVec);
			}
			if (def.params.hasString("mapname")) {
				std::string relativePathStr = def.params.getString("mapname");
				std::filesystem::path relativePath(relativePathStr);
				std::string fileName = (basePath / relativePath).generic_string();

				std::string postfix = getFileNamePostfix(fileName);
				bool shouldInvertGamma = postfix == "tga" || postfix == "png";

				Texture2D texture = Texture2D::createTextureFromFile(fileName, shouldInvertGamma);
				return EnvironmentMap(transform,color,texture);
			}
			else {
				return EnvironmentMap(transform, color);
			}
		}
		std::cout << "unrecognied light source " << def.objectName << std::endl;
		return EnvironmentMap(transform,make_float3(0,0,0));
		
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

	__device__
    Spectrum sampleRay(SamplerObject& sampler, Ray& outputRay, float3& outputLightNormal, float& outputPositionProbability, float& outputDirectionProbability) const {
        auto visitor = [&](auto& arg) -> Spectrum {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Light,typename T>::value) {
				return arg.T::sampleRay(sampler,outputRay,outputLightNormal,outputPositionProbability,outputDirectionProbability);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	__device__
	void sampleRayPdf(const Ray& sampledRay, const float3& sampledLightNormal, float& outputPositionProbability, float& outputDirectionProbability) const {
		auto visitor = [&](auto& arg) {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Light, typename T>::value) {
				return arg.T::sampleRayPdf(sampledRay,sampledLightNormal,outputPositionProbability,outputDirectionProbability);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		visit(visitor);
	}
};
