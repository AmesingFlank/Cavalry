#pragma once
#include "../Utils/GpuCommons.h"
#include "../Utils/MathsCommons.h"
#include <filesystem>
#include "Parameters.h"

enum class TextureType {
	Byte, Float
};

struct Texture2D{
    
	TextureType textureType;
	cudaArray* array = nullptr;
	cudaTextureObject_t texture;
    bool isCopyForKernel;
	int width;
	int height;

	Texture2D(int width_, int height_, bool isCopyForKernel_ = false, TextureType textureType_ = TextureType::Byte):
		textureType(textureType_),width(width_), height(height_), isCopyForKernel(isCopyForKernel_)
	{
		if (!isCopyForKernel) {
			allocate();
		}
	}
    
	Texture2D (const Texture2D& other) {
		textureType = other.textureType;
		isCopyForKernel = other.isCopyForKernel;
		width = other.width;
		height = other.height;

		if (!isCopyForKernel) {
			allocate();
			copyFrom(other);
		}
		else {
			moveFrom(other);
		}
	}

	Texture2D(uchar4* data, int width_, int height_) {
		textureType = TextureType::Byte;
		isCopyForKernel = false;
		width = width_;
		height = height_;
		allocate();
		copyFrom(data);
	}

	Texture2D(float4* data, int width_, int height_) {
		textureType = TextureType::Float;
		isCopyForKernel = false;
		width = width_;
		height = height_;
		allocate();
		copyFrom(data);
	}

	Texture2D operator=(const Texture2D& other) {
		textureType = other.textureType;
		isCopyForKernel = other.isCopyForKernel;
		width = other.width;
		height = other.height;
		if (!isCopyForKernel) {
			if (array != nullptr) {
				release();
			}
			allocate();
			copyFrom(other);
		}
		else {
			moveFrom(other);
		}
		return *this;
	}

	~Texture2D() {
		if (isCopyForKernel) return;
		release();
	}

	Texture2D getCopyForKernel() {
		Texture2D copy(width,height, true, textureType);
		copy.array = array;
		copy.texture = texture;
		return copy;
	}


	__device__
	float4 readTexture(float2 coords) const{
		return tex2D<float4>(texture, coords.x , coords.y);
	}

	void allocate() {
		cudaChannelFormatDesc channelDesc;
		if (textureType == TextureType::Byte) {
			channelDesc = cudaCreateChannelDesc<uchar4>();
		}
		else {
			channelDesc = cudaCreateChannelDesc<float4>();
		}
		HANDLE_ERROR(cudaMallocArray(&array, &channelDesc, width, height));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = array;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));

		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		if (textureType == TextureType::Byte) {
			texDesc.readMode = cudaReadModeNormalizedFloat;
		}
		else {
			texDesc.readMode = cudaReadModeElementType;
		}
		texDesc.normalizedCoords = 1;

		HANDLE_ERROR(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));
	}

	void copyFrom(uchar4* data) {
		if (textureType != TextureType::Byte) {
			SIGNAL_ERROR("error! attempting to copy bytes into a float texture\n");
		}
		HANDLE_ERROR(cudaMemcpyToArray(array,0,0,data,width*height*sizeof(uchar4),cudaMemcpyHostToDevice));
	}
	void copyFrom(float4* data) {
		if (textureType != TextureType::Float) {
			SIGNAL_ERROR("error! attempting to copy floats into a byte texture\n");
		}
		HANDLE_ERROR(cudaMemcpyToArray(array, 0, 0, data, width * height * sizeof(float4), cudaMemcpyHostToDevice));
	}

	void copyFrom(const Texture2D& that) {
		if (textureType != that.textureType) {
			SIGNAL_ERROR("error! attempting to from a texture with a different type. %d %d\n",(int)textureType,(int)that.textureType);
		}
		size_t size;
		if (textureType == TextureType::Byte) {
			size = width * height * sizeof(uchar4);
		}
		else {
			size = width * height * sizeof(float4);
		}
		HANDLE_ERROR(cudaMemcpyArrayToArray(array, 0, 0, that.array, 0, 0, size, cudaMemcpyDeviceToDevice));
	}

	void moveFrom(const Texture2D& that) {
		array = that.array;
		texture = that.texture;
	}

	void release() {
		HANDLE_ERROR(cudaFreeArray(array));
		HANDLE_ERROR(cudaDestroyTextureObject(texture));
	}
	
	static Texture2D createFromObjectDefinition(const ObjectDefinition& def, const glm::mat4& transform, const std::filesystem::path& basePath) ;

	static Texture2D createTextureFromFile(const std::string& filename, bool shouldInvertGamma);

};


