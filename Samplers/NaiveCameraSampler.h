#pragma

#include "../Core/Sampler.h"

class NaiveCameraSampler :public CameraSampler {
public:
	virtual std::vector<CameraSample> genAllSamples(const CameraObject& camera, const Film& film) {
		int width = film.width;
		int height = film.height;
		std::vector<CameraSample> result;
		for (float x = 0; x < width; x += 1) {
			for (float y = 0; y < height; y += 1) {
				CameraSample sample{ x / (float)(width - 1),y / (float)(height - 1) };
				result.push_back(sample);
			}
		}
		return result;
	}
};