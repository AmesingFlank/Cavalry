#pragma once
#include "SimpleSamplerCPU.h"
#include "SimpleSamplerGPU.h"

#include "../Utils/Variant.h"

using SamplerVariant = Variant<SimpleSamplerCPU,SimpleSamplerGPU>;

class SamplerObject : public SamplerVariant {
public:

	__host__ __device__
	SamplerObject() {}

	template<typename V>
	__host__ __device__
	SamplerObject(const V& v) :SamplerVariant(v) {}

	__host__ __device__
	SamplerObject(const SamplerObject& other) : SamplerVariant(other.value) {}



	__host__ __device__
	SamplerObject& operator=(const SamplerObject& other) {
		value = other.value;
		return *this;
	}

    __host__ __device__
    float rand1(){
        auto visitor = [&](auto& arg) -> float{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler,typename T>::value) {
				return arg.T::rand1();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	__host__ __device__
    float2 rand2(){
        auto visitor = [&](auto& arg) -> float2{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler,typename T>::value) {
				return arg.T::rand2();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	__host__ __device__
    float4 rand4(){
        auto visitor = [&](auto& arg) -> float4{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler,typename T>::value) {
				return arg.T::rand4();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	__host__
	SamplerObject getCopyForKernel(){
		auto visitor = [&](auto& arg) -> SamplerObject{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler,typename T>::value) {
				return SamplerObject(arg.T::getCopyForKernel());
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

   
};
