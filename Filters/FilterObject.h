#pragma once
#include "../Core/Filter.h"
#include "BoxFilter.h"
#include "TriangleFilter.h"

using FilterVariant = Variant<BoxFilter,TriangleFilter>;


class FilterObject : public FilterVariant {
public:

	__host__ __device__
    FilterObject() {}

	template<typename V>
	__host__ __device__
	FilterObject(const V& v) :  FilterVariant(v) {}

	__host__ __device__
	FilterObject(const FilterObject& other) :  FilterVariant(other.value) {}


	__host__ __device__
	FilterObject& operator=(const FilterObject& other) {
		value = other.value;
		return *this;
	}

	__device__
	float xwidth() const {
		auto visitor = [&](auto& arg) -> float {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Filter, typename T>::value) {
				return arg.T::xwidth;
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	__device__
	float ywidth() const {
		auto visitor = [&](auto& arg) -> float {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Filter, typename T>::value) {
				return arg.T::ywidth;
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	__device__
	float contribution(int x, int y, const CameraSample& cameraSample) const{
		auto visitor = [&](auto& arg) -> float {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Filter,typename T>::value) {
				return arg.T::contribution(x,y,cameraSample);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

	static FilterObject createFromObjectDefinition(const ObjectDefinition& def) {
		float xwidth = def.params.getNum("xwidth");
		float ywidth = def.params.getNum("ywidth");
		if (def.objectName == "box") {
			return FilterObject(BoxFilter(xwidth,ywidth));
		}
		if (def.objectName == "triangle") {
			return FilterObject(BoxFilter(xwidth, ywidth));
		}
		std::cout << "unsupported filter type: " << def.objectName << std::endl;
		return FilterObject(BoxFilter(xwidth, ywidth));
	}

};
