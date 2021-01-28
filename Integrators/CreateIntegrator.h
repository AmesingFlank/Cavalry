#pragma once

#include <memory>
#include "../Core/Parameters.h"
#include "../Core/Integrator.h"
#include <iostream>

namespace CreateIntegrator{
    std::unique_ptr<Integrator> createFromObjectDefinition (const ObjectDefinition& def);
}

