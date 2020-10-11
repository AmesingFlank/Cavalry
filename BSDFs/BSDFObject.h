#pragma once

#include "Lambertian.h"

#include "Color.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/Variant.h"


using BSDFVariant = Variant<LambertianBSDF>;


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

	__host__ __device__
	Spectrum eval(float3 incident, float3 exitant) const {
		auto visitor = [&](auto& arg) -> Spectrum {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<BSDF,typename T>::value) {
				return arg.eval(incident, exitant);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}
};
