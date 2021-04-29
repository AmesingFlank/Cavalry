#pragma once

#include "../Core/Film.h"



unsigned long long decideSppCount(int resolution, int totalSPP, int completedSPPs, int bytesNeededPerSample);

unsigned long long decideSampleCount(int resolution, int totalSPP, int completedPixels, int bytesNeededPerSample);
