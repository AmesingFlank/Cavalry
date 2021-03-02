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
	int randInt(int N, SamplingState& samplingState) {
		auto visitor = [&](auto& arg) -> int {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler, typename T>::value) {
				return arg.T::randInt(N,samplingState);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

    __host__ __device__
    float rand1(SamplingState& samplingState){
        auto visitor = [&](auto& arg) -> float{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler,typename T>::value) {
				return arg.T::rand1(samplingState);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	__device__
    float2 rand2(SamplingState& samplingState){
        auto visitor = [&](auto& arg) -> float2{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler,typename T>::value) {
				return arg.T::rand2(samplingState);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	__device__
    float4 rand4(SamplingState& samplingState){
        auto visitor = [&](auto& arg) -> float4{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler,typename T>::value) {
				return arg.T::rand4(samplingState);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	GpuArray<CameraSample> genAllCameraSamples(const CameraObject& camera, Film& film, int bytesNeededPerSample,int maxSamplesPerRound = -1) {
		auto visitor = [&](auto& arg) -> GpuArray<CameraSample> {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler, typename T>::value) {
				return arg.T::genAllCameraSamples(camera, film,bytesNeededPerSample,maxSamplesPerRound);
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
		std::cout << "spp: " << samplesPerPixel << std::endl;
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
	void startPixel(SamplingState& samplingState, unsigned long long lastIndex){
		auto visitor = [&](auto& arg){
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Sampler,typename T>::value) {
				arg.T::startPixel(samplingState,lastIndex);
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
