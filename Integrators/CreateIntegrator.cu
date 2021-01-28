#include "CreateIntegrator.h"
#include "DirectLightingIntegrator.h"
#include "PathTracingIntegrator.h"
#include "ReinforcementLearningPathTracing.h"
#include <memory>
#include "../Core/Parameters.h"
#include <iostream>

namespace CreateIntegrator{
    std::unique_ptr<Integrator> createFromObjectDefinition (const ObjectDefinition& def){

        if(def.objectName == "directlighting"){
            return std::make_unique<DirectLighting::DirectLightingIntegrator>();
        }
        else if(def.objectName == "path"){
            int depth = def.params.getNum("maxdepth");
            return std::make_unique<PathTracing::PathTracingIntegrator>(depth);
        }
        else if(def.objectName == "rlpath"){
            int depth = def.params.getNum("maxdepth");
            return std::make_unique<ReinforcementLearningPathTracing::RLPTIntegrator>(depth);
        }
        std::cout<<"unrecognized integrator name. using pathtracer as default."<<std::endl;
        return std::make_unique<PathTracing::PathTracingIntegrator>(8);
    }
}

