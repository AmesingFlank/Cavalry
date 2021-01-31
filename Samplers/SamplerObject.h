#pragma once

#include "SimpleSampler.h"
#include "HaltonSampler.h"

#include "../Utils/Variant.h"

using SamplerVariant = Variant<SimpleSampler,HaltonSampler>;

class SamplerObject : public SamplerVariant {
public:

	__host__ __device__
	SamplerObject() {}

	template<typename V>
	__host__ __device__
	SamplerObject(const V& v) :SamplerVariant(v) {}

	__host__ __device__
	SamplerObject(const SamplerObject& other) : SamplerVariant(other.value) {}



	__device__
	SamplerObject& operator=(const SamplerObject& other) {
		value = other.value;
		return *this;
	}

	__device__
	int randInt(int N) {
		auto visitor = [&](auto& arg) -> int {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler, typename T>::value) {
				return arg.T::randInt(N);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
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

	__device__
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

	__device__
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

	GpuArray<CameraSample> genAllCameraSamples(const CameraObject& camera, FilmObject& film, int bytesNeededPerSample) {
		auto visitor = [&](auto& arg) -> GpuArray<CameraSample> {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler, typename T>::value) {
				return arg.T::genAllCameraSamples(camera, film,bytesNeededPerSample);
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

	__device__ __host__
	int getSamplesPerPixel(){
		auto visitor = [&](auto& arg) -> int{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler,typename T>::value) {
				return arg.samplesPerPixel;
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}


	static SamplerObject createFromObjectDefinition(const ObjectDefinition& def) {
		int samplesPerPixel = def.params.getNum("pixelsamples");
		std::cout << "pixelsamples in file : " << samplesPerPixel << std::endl;
		if (def.objectName == "random") {
			return SamplerObject(SimpleSampler(samplesPerPixel));
		}
		else if (def.objectName == "halton") {
			return SamplerObject(HaltonSampler(samplesPerPixel));
		}
		std::cout << "unsupported sampler type: " << def.objectName << std::endl;
		return SamplerObject(HaltonSampler(samplesPerPixel));
	}


	__device__
	void startPixel(){
		auto visitor = [&](auto& arg){
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler,typename T>::value) {
				arg.T::startPixel();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		visit(visitor);
	}

	void prepare(int threadsCount) {
		auto visitor = [&](auto& arg) {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler, typename T>::value) {
				arg.T::prepare(threadsCount);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		visit(visitor);
	}

	void reorderStates(GpuArray<int>& taskIndices) {
		auto visitor = [&](auto& arg) {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler, typename T>::value) {
				arg.T::reorderStates(taskIndices);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		visit(visitor);
	};

	void syncDimension(){
		auto visitor = [&](auto& arg) {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler, typename T>::value) {
				arg.T::syncDimension();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		visit(visitor);
	};

	int bytesNeededPerThread() {
		auto visitor = [&](auto& arg) -> int{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler,typename T>::value) {
				return arg.bytesNeededPerThread();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}
};
