// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h

// Includes
#define NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "particleSystem.h"


const uint width = 640, height = 480;

const uint num_old = 300;
const uint num_new = 600;
const float radius = 0.1;

int main() {
    std::vector<float> pos_old(3*num_old);
    std::vector<float> val_old(4*num_old);
    std::vector<float> pos_new(3*num_new);
    std::vector<float> val_new(3*num_new);
    
    assert(pos_old.size() * 4 == val_old.size() * 3);// "position: x,y,z; value: x,y,z,weight."
    assert(pos_new.size() == val_new.size());

    auto no = pos_old.size() / 3;
    ParticleSystem particle_system(no, radius);
    particle_system.inputData(pos_old.data(), val_old.data());
    float* dddd = (float*)malloc(sizeof(float) * 3 * num_new);
    particle_system.interpolate(pos_new.size() / 3, pos_new.data(), dddd);

    // here, val_new have been rewritten.
    return 0;
}
