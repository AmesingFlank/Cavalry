#pragma once

#include "MatteMaterial.h"
#include "Mirror.h"
#include "Substrate.h"
#include "Metal.h"
#include "Glass.h"


#include "Color.h"
#include "../Utils/GpuCommons.h"
#include "../Utils/Variant.h"
#include "../Core/Parameters.h"


using MaterialVariant = Variant<MatteMaterial,MirrorMaterial,SubstrateMaterial,MetalMaterial,GlassMaterial>;


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

	__device__
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

	__device__
	BSDFObject getBSDF(const IntersectionResult& intersection) const {
		auto visitor = [&](auto& arg) -> BSDFObject {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Material, typename T>::value) {
				return arg.T::getBSDF(intersection);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	__device__
	MaterialType getType() const  {
		auto visitor = [&](auto& arg) -> MaterialType {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Material, typename T>::value) {
				return arg.T::getType();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	};

	static MaterialObject createFromObjectDefinition(const ObjectDefinition& def,const std::unordered_map<std::string,Texture2D>& textures) {
		std::string materialType = def.params.getString("type");
		if (materialType == "matte") {
			return MaterialObject(MatteMaterial::createFromParams(def.params,textures));
		}
		if (materialType == "substrate") {
			return MaterialObject(SubstrateMaterial::createFromParams(def.params,textures));
		}
		if (materialType == "mirror") {
			return MaterialObject(MirrorMaterial::createFromParams(def.params,textures));
		}
		if (materialType == "metal") {
			return MaterialObject(MetalMaterial::createFromParams(def.params,textures));
		}
		if (materialType == "glass") {
			return MaterialObject(GlassMaterial::createFromParams(def.params, textures));
		}
		return MaterialObject(MatteMaterial::createFromParams(def.params,textures));
	}

	void prepareForRender() {
		auto visitor = [&](auto& arg) {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Material, typename T>::value) {
				return arg.T::prepareForRender();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		visit(visitor);
	}
};
