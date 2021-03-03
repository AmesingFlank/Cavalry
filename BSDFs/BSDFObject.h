#pragma once

#include "Lambertian.h"
#include "Fresnel.h"
#include "Microfacet.h"
#include "MirrorBSDF.h"
#include "Specular.h"
#include "PlasticBSDF.h"

#include "../Utils/GpuCommons.h"
#include "../Utils/Variant.h"


using BSDFVariant = Variant<LambertianBSDF,FresnelBlendBSDF,MirrorBSDF,MicrofacetBSDF,SpecularBSDF,PlasticBSDF>;


class BSDFObject : public BSDFVariant {
public:

	__host__ __device__
		BSDFObject() {}

	template<typename V>
	__host__ __device__
	BSDFObject(const V& v) :  BSDFVariant(v) {}

	__host__ __device__
	BSDFObject(const BSDFObject& other) :  BSDFVariant(other.value) {}


	__host__ __device__
	BSDFObject& operator=(const BSDFObject& other) {
		value = other.value;
		return *this;
	}

	__device__
	Spectrum eval(float3 incident, float3 exitant) const {
		auto visitor = [&](auto& arg) -> Spectrum {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<BSDF,typename T>::value) {
				return arg.T::eval(incident, exitant);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	__device__
	bool isDelta() const {
		auto visitor = [&](auto& arg) -> bool {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<BSDF, typename T>::value) {
				return arg.T::isDelta();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	__device__
	bool isAlmostDelta() const {
		auto visitor = [&](auto& arg) -> bool {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<BSDF, typename T>::value) {
				return arg.T::isAlmostDelta();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	__device__
	Spectrum sample(float2 randomSource, float3& incidentOutput, const float3& exitant, float* probabilityOutput) const {
		auto visitor = [&](auto& arg) -> Spectrum {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<BSDF, typename T>::value) {
				return arg.T::sample(randomSource,incidentOutput,exitant,probabilityOutput);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	__device__
    float pdf(const float3& incident, const float3& exitant) const {
		auto visitor = [&](auto& arg) -> float {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<BSDF, typename T>::value) {
				return arg.T::pdf(incident,exitant);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}
};
