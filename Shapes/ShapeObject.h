#pragma once
#include "Sphere.h"
#include "TriangleMesh.h"

#include "../Utils/Variant.h"

using ShapeVariant = Variant<Sphere,TriangleMesh>;

class ShapeObject : public ShapeVariant {
public:

	__host__ __device__
	ShapeObject() {}

	template<typename V>
	__host__ __device__
	ShapeObject(const V& v) :ShapeVariant(v) {}

	__host__ __device__
	ShapeObject(const ShapeObject& other) : ShapeVariant(other.value) {}



	__host__ __device__
	ShapeObject& operator=(const ShapeObject& other) {
		value = other.value;
		return *this;
	}

	__host__ __device__
	bool intersect(IntersectionResult& result, const Ray& ray) const {
		auto visitor = [&](auto& arg) -> bool {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Shape,typename T>::value) {
				return arg.T::intersect(result,ray);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

    __host__ __device__
    ShapeID getID() const {
        auto visitor = [&](auto& arg) -> ShapeID {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Shape,typename T>::value) {
				return arg.T::getID();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	__host__
	ShapeObject getCopyForKernel(){
		auto visitor = [&](auto& arg) -> ShapeObject{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Shape,typename T>::value) {
				return ShapeObject(arg.T::getCopyForKernel());
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	__host__ __device__
	IntersectionResult sample(const float4& randomSource,float* outputProbability) const{
		auto visitor = [&](auto& arg) -> IntersectionResult{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Shape,typename T>::value) {
				return arg.T::sample(randomSource,outputProbability);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}


	static ShapeObject createFromObjectDefinition(const ObjectDefinition& def,const glm::mat4 transform, const std::filesystem::path& basePath){
		return ShapeObject(TriangleMesh::createFromParams(def.params,transform,basePath));
	}

	void prepareForRender() {
		auto visitor = [&](auto& arg){
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Shape, typename T>::value) {
				return arg.T::prepareForRender();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		visit(visitor);
	}
};
