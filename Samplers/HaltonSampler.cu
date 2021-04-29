#include "HaltonSampler.h"
#include "DecideSampleCount.h"
#include "../Utils/Utils.h"
#include <cstdlib>      // std::rand, std::srand
#include "SamplerObject.h"

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
threadsCount(0),primes(0,true),permutations(0,true),permutationsStart(0,true) {
    
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


SamplerObject HaltonSampler::getObjectFromThis() {
    return getCopyForKernel();
}