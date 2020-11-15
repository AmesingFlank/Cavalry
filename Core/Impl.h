#pragma once

// this file is needed so that implementations of these functions needs to be out-of-class,
// which is because the class definitions includes incomplete defitions.
// These function implementations are put inside an *Impl.h file instead of a .cpp/.cu file.
// in order to avoid requiring relocatable GPU code (i.e., -rdc=true).


#include "IntersectionResultImpl.h"
#include "../Lights/DiffuseAreaLightImpl.h"
#include "PrimitiveImpl.h"
#include "TriangleImpl.h"