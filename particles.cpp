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
#include <time.h>

#include "particleSystem.h"

const uint num_old = 16*16*16;
const uint num_new = 600;
const float radius = 0.125;

void data_generate(std::vector<float> pos_old,
                   std::vector<float> pos_new,
                   std::vector<float> val_old) {

    // lambda function for random data generation.
    srand(static_cast<uint>(time(0)));
    auto rrr = [] {return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);};
    
    for (size_t i = 0; i < 16; i++)
    {
        for (size_t j = 0; j < 16; j++)
        {
            for (size_t k = 0; k < 16; k++)
            {
                auto n = i * 256 + j * 16 + k;
                pos_old[n * 3    ] = i * 0.125;
                pos_old[n * 3 + 1] = j * 0.125;
                pos_old[n * 3 + 2] = k * 0.125;
            }
        }
    }
    for (size_t i = 0; i < val_old.size()/4; i++)
    {
        val_old[i * 4    ] = 4.0;
        val_old[i * 4 + 1] = 3.0;
        val_old[i * 4 + 2] = 2.0;
        val_old[i * 4 + 3] = 0.001953125;
    }
    for (size_t i = 0; i < pos_old.size()/3; i++)
    {
        printf("pos[%f, %f, %f]\n", pos_old[3 * i], pos_old[3 * i + 1], pos_old[3 * i + 2]);
        printf("val[%f, %f, %f, %f]\n", val_old[4 * i], val_old[4 * i + 1], val_old[4 * i + 2], val_old[4 * i + 3]);

    }
    for (size_t i = 0; i < pos_new.size(); i++)
    {
        pos_new[i] = rrr();
    }
}


int main() {
    std::vector<float> pos_old(3*num_old);
    std::vector<float> val_old(4*num_old);
    std::vector<float> pos_new(3*num_new);
    std::vector<float> val_new(3*num_new);

    data_generate(pos_old, pos_new, val_old);
    
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
