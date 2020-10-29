#pragma once

#include "MatteMaterial.h"

#include "Color.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/Variant.h"
#include "../Core/Parameters.h"


using MaterialVariant = Variant<MatteMaterial>;


class MaterialObject : public MaterialVariant {
public:

	__host__ __device__
		MaterialObject() {}

	template<typename V>
	__host__ __device__
	MaterialObject(const V& v) :  MaterialVariant(v) {}

	__host__ __device__
	MaterialObject(const MaterialObject& other) :  MaterialVariant(other.value) {}


	__host__ __device__
	MaterialObject& operator=(const MaterialObject& other) {
		value = other.value;
		return *this;
	}

	__host__ __device__
	Spectrum eval(const Ray& incidentRay, const Spectrum& incidentSpectrum, const Ray& exitantRay, const IntersectionResult& intersection) const {
		auto visitor = [&](auto& arg) -> Spectrum {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Material,typename T>::value) {
				return arg.T::eval(incidentRay,incidentSpectrum,exitantRay,intersection);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	static MaterialObject createFromObjectDefinition(const ObjectDefinition& def) {
		if (def.objectName == "matte") {
			std::vector<float> color = def.params.getNumList("Kd");
			return MaterialObject(MatteMaterial(make_float3(color[0], color[1], color[2])));
		}
		return MaterialObject(MatteMaterial(make_float3(1,1,1)));
	}
};
