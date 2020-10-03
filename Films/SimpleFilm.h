#pragma once

#include "../Core/Film.h"

class SimpleFilm : public Film {
public:
	SimpleFilm(int width_, int height_);
	virtual void addSample(const CameraSample& sample, const Spectrum& spectrum) override;
	virtual RenderResult readCurrentResult()  override;

	RenderResult result;
};