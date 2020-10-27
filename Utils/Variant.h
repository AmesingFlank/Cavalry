#pragma once

#include <type_traits>

// https://github.com/cselab/cuda_variant
#include <variant/variant.h>
#include "GpuCommons.h"


#define SIGNAL_VARIANT_ERROR SIGNAL_ERROR("variant error\n")


#define SIGNAL_VARIANT_GET_ERROR SIGNAL_ERROR("variant get error\n")



template<typename... Ts>
class Variant {

public:
    variant :: variant<typename Ts...> value;

    __host__ __device__
    Variant(){}

    template<typename V>
    __host__ __device__
    Variant(const V& v): value(v){}

	template<>
	__host__ __device__
	Variant(const Variant& other) : value(other.value) {};


    template<typename V>
    __host__ __device__
    Variant& operator=(const V& v) {
        return value = v;
    }


    __host__ __device__
	Variant& operator=(const Variant& other) {
		value = other.value;
        return *this;
	}

	template<typename Fn>
	__host__ __device__
	auto visit(Fn fn){
		return variant::apply_visitor(fn, value);
	}

    template<typename Fn>
	__host__ __device__
    auto visit(Fn fn) const{
        return variant::apply_visitor(fn, value);
    }

    

    
    template<typename T>
    __host__ __device__
    const T& get() const{
        return visit([&](const auto& arg)-> const T& {
            using ConstX = typename std::remove_reference<decltype(arg)>::type;
            using X = typename std::remove_cv<ConstX>::type;
            if constexpr (std::is_same<X, T>::value) {
                return arg;
            }
            else {
                SIGNAL_VARIANT_GET_ERROR;
            }
        });
    }

    template<typename T>
    __host__ __device__
    T& get() {
        return visit([&](auto& arg)-> T& {
            using X = typename std::remove_reference<decltype(arg)>::type;
            if constexpr (std::is_same<X, T>::value) {
                return arg;
            }
            else {
                SIGNAL_VARIANT_GET_ERROR;
            }
        });
    }



};


