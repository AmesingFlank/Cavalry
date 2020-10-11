#pragma once

#include <type_traits>

// https://github.com/cselab/cuda_variant
#include <variant/variant.h>

#ifndef SIGNAL_VARIANT_ERROR
#define SIGNAL_VARIANT_ERROR printf("variant error!\n");
#endif


#ifndef SIGNAL_VARIANT_GET_ERROR
#define SIGNAL_VARIANT_GET_ERROR printf("variant get error!\n");
#endif



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
                return *static_cast<T*>(nullptr);
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
                return *static_cast<T*>(nullptr);
            }
        });
    }



};


