#pragma once

#include "../Core/Sampler.h"
#include <random>


class SimpleSampler: public Sampler{
public:
    std::uniform_real_distribution<> dis = std::uniform_real_distribution<>(0.0, 1.0);
    std::mt19937 gen = std::mt19937(std::random_device()());

	virtual float rand1() {
        return dis(gen);
    };
	virtual float2 rand2() {
        return make_float2(dis(gen),dis(gen));
    };
};