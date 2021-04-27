#include "HaltonSampler.h"
#include "DecideSampleCount.h"
#include "../Utils/Utils.h"
#include <cstdlib>      // std::rand, std::srand

void shuffle(std::vector<unsigned short>& vec)
{
    for (int i = vec.size() - 1; i > 0; --i) {
        std::swap(vec[i], vec[std::rand()%(i+1)]);
    }
}

std::vector<unsigned int> computePrimes(int N) {
    std::vector<unsigned int> primes;
    primes.push_back(2);
    for (int i = 3;; ++i) {
        if (primes.size() >= N) {
            break;
        }
        bool isPrime = true;
        for (int p : primes) {
            if (p * p > i) {
                break;
            }
            if (i % p == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime) {
            primes.push_back(i);
        }
    }
    return primes;
}

void buildPermutations(GpuArray<unsigned short>& permutations, GpuArray<unsigned int>& permutationsStart, const std::vector<unsigned int>& primesHost){

    std::vector<unsigned int> permsStart(primesHost.size());

    int size = 0;
    for(int i = 0;i<primesHost.size();++i){
        permsStart[i] = size;
        int p = primesHost[i];
        size += p;
    }

    std::vector<unsigned short> perms(size);

    std::srand ( 533799 ); //just a number i liked.

    for(int i = 0;i<primesHost.size();++i){
        int p = primesHost[i];
        std::vector<unsigned short> thisPerm(p);
        for(int j = 0;j<p;++j){
            thisPerm[j]=j;
        }
        shuffle ( thisPerm);
        for(int j=0;j<p;++j){
            perms[permsStart[i]+j] = thisPerm[j];
        }
    }
    permutations = perms;
    permutationsStart = permsStart;

}

HaltonSampler::HaltonSampler(int samplesPerPixel_, bool isCopyForKernel_):
threadsCount(0),primes(0,true),permutations(0,true),permutationsStart(0,true){
    samplesPerPixel = samplesPerPixel_;
    if (!isCopyForKernel_) {
        auto primesHost = computePrimes(1024);
        primes = primesHost;
        buildPermutations(permutations,permutationsStart,primesHost);
    }
}

HaltonSampler::HaltonSampler():
threadsCount(0),samplesPerPixel(0),primes(0,true),permutations(0,true),permutationsStart(0,true) {
    
}



void HaltonSampler::prepare(int threadsCount_) {
    if (threadsCount < threadsCount_) {
        threadsCount = threadsCount_;
    }
}

HaltonSampler HaltonSampler::getCopyForKernel(){
    HaltonSampler copy(samplesPerPixel,true);
    copy.threadsCount = threadsCount;
    copy.primes = primes.getCopyForKernel();
    copy.permutations = permutations.getCopyForKernel();
    copy.permutationsStart = permutationsStart.getCopyForKernel();
    return copy;
}


__global__
void genHaltonCameraSample(CameraSample* resultPointer, int samplesCount, int width, int height,int samplesPerPixel,HaltonSampler sampler,int pixelIndexStart){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= samplesCount){
        return;
    }

    SamplingState state;
    state.dimension = 0;
    state.index = index + pixelIndexStart * samplesPerPixel;

    int pixelIndex = pixelIndexStart + index / samplesPerPixel;

    int x = pixelIndex % width;
    int y = pixelIndex / width;

    CameraSample sample{ x , y, state };
    sample.x += sampler.HaltonSampler::rand1(state);
    sample.y += sampler.HaltonSampler::rand1(state);

    resultPointer[index] = sample;
}


GpuArray<CameraSample> HaltonSampler::genAllCameraSamples(const CameraObject& camera, Film& film, int bytesNeededPerSample,int maxSamplesPerRound) {
    int width = film.width;
    int height = film.height;

    unsigned long long sampleCount = decideSampleCount(film, samplesPerPixel, bytesNeededPerSample);

    unsigned long long pixelsCount = sampleCount / samplesPerPixel;

    prepare(sampleCount);

    GpuArray<CameraSample> result(sampleCount);

    int numBlocks, numThreads;
    setNumBlocksThreads(sampleCount, numBlocks, numThreads);

    genHaltonCameraSample <<<numBlocks,numThreads>>> (result.data,sampleCount,width,height, samplesPerPixel,getCopyForKernel(),film.completedPixels);
    CHECK_IF_CUDA_ERROR("gen halton camera samples");

    film.completedPixels += pixelsCount;

    return result;
}
