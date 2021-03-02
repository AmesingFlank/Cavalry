#pragma once

#include "../Core/Integrator.h"
#include "../Core/Sampler.h"
#include <memory>

namespace ReinforcementLearningPathTracing {

    struct QEntry{
        static constexpr int NUM_X = 16;
        static constexpr int NUM_Y = 8;
        static constexpr int NUM_XY = NUM_X * NUM_Y;

        __host__ __device__
        static constexpr float INV_NUM_X() {
            return 1.f / (float)NUM_X;
        }

        __host__ __device__
        static constexpr float INV_NUM_Y() {
            return 1.f / (float)NUM_Y;
        }

        float Q[NUM_XY];

        float newQ[NUM_XY];
        float proposalCount[NUM_XY];
        float totalProposalCount[NUM_XY];

        __device__
        float defaultQ(int cellIndex)const {
            return 1;
        }
        
        __device__
        QEntry(){
            for (int i = 0; i < NUM_XY; ++i) {
                Q[i] = defaultQ(i);
                totalProposalCount[i] = 0;
            }
        }

        __device__
        float alpha(int cellIndex) const{
            //printf("computing alpha[%d]:  %f %f\n", cellIndex,proposalCount[cellIndex],totalProposalCount[cellIndex]);
            if (proposalCount[cellIndex] == 0) {
                return 0;
            }
            return (proposalCount[cellIndex]) / (totalProposalCount[cellIndex] + proposalCount[cellIndex]);
        }

        __device__
        float averageQ() {
            float sumQ = 0;
            int count = 0;
            for (int i = 0; i < NUM_XY; ++i) {
                if (totalProposalCount[i] > 0) {
                    sumQ += Q[i];
                    count += 1;
                }
            }
            if (count == 0) {
                return defaultQ(0);
            }
            float avg = sumQ / (float)count;
            for (int i = 0; i < NUM_XY; ++i) {
                if (totalProposalCount[i] == 0) {
                    Q[i] = avg;
                    //printf("updaing avg%f\n", avg);
                }
            }
            return avg;
        }

        __device__
        float sumQ() {
            float sumQ = 0;
            for (int i = 0; i < NUM_XY; ++i) {
                sumQ += Q[i];    
            }
            return sumQ;
        }


        // should only be called by one thread for each cellIndex
        __device__
        void prepareForUpdateQ(int cellIndex){
            newQ[cellIndex] = 0;
            proposalCount[cellIndex]=0;
        }

        __device__
        void proposeNextQ(float QVal, int cellIndex){
            atomicAdd(&(newQ[cellIndex]), QVal);
            atomicAdd(&(proposalCount[cellIndex]), 1);
            //printf("proposing %d %f\n", cellIndex, QVal);
        }

        // should only be called by one thread for each cellIndex
        __device__
        void finishUpdateQ(int cellIndex){
            float a = alpha(cellIndex);
            if (a == 0) {
                return;
            }
            float updatedQ = Q[cellIndex] * (1.f - a) + a * newQ[cellIndex] / proposalCount[cellIndex];
            //printf("Q[%d]:   old:%f  new:%f  alpha:%f \n", cellIndex, Q[cellIndex], updatedQ, a);
            Q[cellIndex] = updatedQ;
            totalProposalCount[cellIndex] += proposalCount[cellIndex];
        }


        static __host__ __device__  int dirToCellIndex(float3 dir) {
            float u = dir.z;
            float y = (u + 1.f) / 2.f;
            int thetaIndex = clampF((int)(y * QEntry::NUM_Y), 0, QEntry::NUM_Y - 1);

            dir.x /= sqrt(1.f - u * u);
            dir.y /= sqrt(1.f - u * u);

            float v = acos(dir.x);
            if (dir.y < 0) {
                v = (2.f * M_PI)-v;
            }
            v = v / (2.f * M_PI);
            int phiIndex = clampF((int)(v * QEntry::NUM_X), 0, QEntry::NUM_X - 1);
            return thetaIndex * NUM_X + phiIndex;
        }

        __device__
        float3 sampleDirectionInCell(float2 randomSource,int cellIndex) const
        {
            int thetaIdx = cellIndex / NUM_X;
            int phiIdx = cellIndex % NUM_X;
            float u = ((float)thetaIdx + randomSource.x) * INV_NUM_Y();
            u = u * 2 - 1.f;
            float v = ((float)phiIdx + randomSource.y) * INV_NUM_X();

            float xyScale = sqrt(1.0f - u * u);
            float phi = 2 * M_PI * v;

            float3 dir = make_float3(
                xyScale * cos(phi),
                xyScale * sin(phi),
                u);
            return dir;
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

