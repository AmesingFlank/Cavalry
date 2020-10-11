#pragma once
#include "Sphere.h"

#include "../Utils/Variant.h"

using ShapeVariant = Variant<Sphere>;

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
				return arg.intersect(result,ray);
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
				return arg.getID();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }
};