#pragma once
#include "../Utils/GpuCommons.h"
#include "../Dependencies/include/stb_image.h"


struct Texture2D{
    
	cudaArray* array = nullptr;
	cudaTextureObject_t texture;
    bool isCopyForKernel;
	int width;
	int height;

	Texture2D(int width_, int height_, bool isCopyForKernel_ = false) :width(width_), height(height_), isCopyForKernel(isCopyForKernel_) {
		if (!isCopyForKernel) {
			allocate();
		}
	}
    
	Texture2D (const Texture2D& other) {
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
		isCopyForKernel = false;
		width = width_;
		height = height_;
		allocate();
		copyFrom(data);
	}

	Texture2D operator=(const Texture2D& other) {
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
		Texture2D copy(width,height, true);
		copy.array = array;
		copy.texture = texture;
		return copy;
	}


	__device__
	float4 readTexture(float x, float y) {
		return tex2D<float4>(texture, x , y);
	}

	void allocate() {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
		HANDLE_ERROR(cudaMallocArray(&array, &channelDesc, width, height));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = array;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));

		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 1;

		HANDLE_ERROR(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));
	}

	void copyFrom(uchar4* data) {
		HANDLE_ERROR(cudaMemcpyToArray(array,0,0,data,width*height*sizeof(uchar4),cudaMemcpyHostToDevice));
	}

	void copyFrom(const Texture2D& that) {
		HANDLE_ERROR(cudaMemcpyArrayToArray(array, 0, 0, that.array, 0, 0, width * height * sizeof(uchar4), cudaMemcpyDeviceToDevice));
	}

	void moveFrom(const Texture2D& that) {
		array = that.array;
		texture = that.texture;
	}

	void release() {
		HANDLE_ERROR(cudaFreeArray(array));
		HANDLE_ERROR(cudaDestroyTextureObject(texture));
	}


};


inline Texture2D createTextureFromFile(const std::string& filename) {
	int width;
	int height;
	uchar4* data = (uchar4*)stbi_load(filename.c_str(), &width, &height, 0, STBI_rgb_alpha);
	return Texture2D(width,height,data);
}