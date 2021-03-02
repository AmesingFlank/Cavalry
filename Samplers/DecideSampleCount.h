#pragma once

#include "../Core/Film.h"


int decideSamplesPerPixel(Film& film, int totalSPP,int bytesNeededPerSample, int maxSamplesPerRound = -1);