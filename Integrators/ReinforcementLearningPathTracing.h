#pragma once

#include "../Core/Integrator.h"
#include "../Core/Sampler.h"
#include <memory>

namespace ReinforcementLearningPathTracing {


    struct QEntry{
        static const int NUM_X = 16;
        static const int NUM_Y = 8;
        static const int NUM_XY = NUM_X * NUM_Y;

        float Q[NUM_XY];
        float cdf[NUM_XY];
        float maxQ;
        
        __device__
        QEntry(){
            maxQ = 0;
            for (int y = 0; y < NUM_Y; y++){
                for (int x = 0; x < NUM_X; x++){
                    const float initDensity = y / float(NUM_Y);
                    Q[x + y * NUM_X] = initDensity;
                    maxQ = max(maxQ,initDensity);
                }
            }
        }

        __device__
        float alpha() const{
            return 0.85f;
        }

        __device__
        float updateCDF(){
            float sumQ = 0;
            for (int y = 0; y < NUM_Y; y++){
                for (int x = 0; x < NUM_X; x++){
                    sumQ += Q[x + y * NUM_X];
                }
            }

            float accumulatedDensity = 0;
            for (int y = 0; y < NUM_Y; y++){
                for (int x = 0; x < NUM_X; x++){
                    accumulatedDensity +=  Q[x + y * NUM_X] / sumQ;
                    cdf[x+y*NUM_X] = accumulatedDensity;
                }
            }
        }

        __device__
        const float getValueFunc() const{
            return maxQ;
        }

        // should only be called by one thread for each cellIndex
        __device__
        void prepareForUpdateQ(int cellIndex){
            float* temp = &(cdf[0]); // to save memory, use the cdf array to help update Q;
            temp[cellIndex] = 0;
            Q[cellIndex] *= (1.f-alpha());
        }

        __device__
        void proposeNextQ(const float QVal, int cellIndex){
            float* temp = &(cdf[0]); // to save memory, use the cdf array to help update Q;
            atomicAdd(&(temp[cellIndex]), QVal);
        }

        // should only be called by one thread for each cellIndex
        __device__
        void finishUpdateQ(int cellIndex){
            float* temp = &(cdf[0]); // to save memory, use the cdf array to help update Q;
            Q[cellIndex] += alpha() * temp[cellIndex];
            updateCDF();
        }

        __device__
        int getCorrespondingIndex(float f) const{
            int l = 0;
            int r = NUM_XY -1;
            while(l<r){
                int m = (l+r)/2;
                if(cdf[m] >= f){
                    if(m==0){
                        return 0;
                    }
                    if(cdf[m-1] < f){
                        return m;
                    }
                    r = m-1;
                }
                else{ // cdf[m] < f
                    l = m+1;
                }
            }
            return l;
        }

        __device__
        float3 sampleDirectionProportionalToQ(SamplerObject& sampler, float& outputProbability, int& outputCellIndex) const
        {
            float f = sampler.rand1();
            outputCellIndex = getCorrespondingIndex(f);
            outputProbability = cdf[outputCellIndex];
            if(outputCellIndex > 0){
                outputProbability -= cdf[outputCellIndex-1];
            }
            outputProbability = (NUM_XY * outputProbability / (2*M_PI)); // Solid angle probability
            
            const int thetaIdx = outputCellIndex / NUM_X;
            const int phiIdx = outputCellIndex % NUM_X;
            const float u = ((float)thetaIdx + sampler.rand1()) / NUM_Y;
            const float v = ((float)phiIdx + sampler.rand1()) / NUM_X;

            return make_float3(
                sqrt(1.0f - u * u) * cos(2 * M_PI * v),
                sqrt(1.0f - u * u) * sin(2 * M_PI * v),
                u);
        }
    };

    class RLPTIntegrator : public Integrator {
    public:

        int maxDepth;
        RLPTIntegrator(int maxDepth_);

        virtual void render(const Scene& scene, const CameraObject& camera, FilmObject& film) override;

        GpuArray<QEntry> QTable;

    };
}

