#pragma once

#include "SimpleFilm.h"
#include "BoxFilterFilm.h"
#include "../Utils/Variant.h"
#include "../Core/Parameters.h"

using FilmVariant = Variant<SimpleFilm,BoxFilterFilm>;

class FilmObject : public FilmVariant {
public:

	__host__ __device__
	FilmObject() {}

	template<typename V>
	__host__ __device__
	FilmObject(const V& v) :FilmVariant(v) {}

	__host__ __device__
	FilmObject(const FilmObject& other) : FilmVariant(other.value) {}


	__host__ __device__
	FilmObject& operator=(const FilmObject& other) {
		value = other.value;
		return *this;
	}

	__device__
	void addSample(const CameraSample& sample, const Spectrum& spectrum) {
		auto visitor = [&](auto& arg) {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Film,typename T>::value) {
				arg.T::addSample(sample,spectrum);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}

    __host__ __device__
    RenderResult readCurrentResult(){
        auto visitor = [&](auto& arg) -> RenderResult{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Film,typename T>::value) {
				return arg.T::readCurrentResult();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

    __host__ __device__
    int getWidth(){
        auto visitor = [&](auto& arg) -> int{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Film,typename T>::value) {
				return arg.T::getWidth();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

    __host__ __device__
    int getHeight(){
        auto visitor = [&](auto& arg) -> int{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Film,typename T>::value) {
				return arg.T::getHeight();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	 __host__ __device__
    int getCompletedSamplesPerPixel(){
        auto visitor = [&](auto& arg) -> int{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Film,typename T>::value) {
				return arg.T::getCompletedSamplesPerPixel();
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	 __host__ __device__
    void setCompletedSamplesPerPixel(int spp){
        auto visitor = [&](auto& arg) {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Film,typename T>::value) {
				return arg.T::setCompletedSamplesPerPixel(spp);
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	__host__
	FilmObject getCopyForKernel(){
        auto visitor = [&](auto& arg) -> FilmObject{
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<Film,typename T>::value) {
				return FilmObject(arg.T::getCopyForKernel());
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
    }

	static FilmObject createFromObjectDefinition(const ObjectDefinition& def){
		return FilmObject(BoxFilterFilm::createFromParams(def.params));
	}
};
