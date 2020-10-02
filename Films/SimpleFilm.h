#pragma once

#include "../Core/Film.h"

class SimpleFilm : public Film{
public:
    SimpleFilm(int width_, int height_);
    void addSample(float2 position, Color color) override;
    RenderResult readCurrentResult() override;

    RenderResult result;

}