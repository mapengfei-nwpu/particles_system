/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

// simulation parameters in constant memory
__constant__ SimParams params;
__constant__ float fpi = 3.14159265f;

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float3 *pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // get address in grid
    int3 gridPos = calcGridPos(pos[index]);
    uint hash = calcGridHash(gridPos);
    if (hash == 56204) {
        printf("position: %f, %f, %f\n grid position: %d, %d, %d\n", pos[index].x, pos[index].y, pos[index].z, gridPos.x, gridPos.y, gridPos.z);
    }

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float3 *sortedPos,        // output: sorted positions
                                  float4 *sortedVel,        // output: sorted velocities
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  float3 *oldPos,           // input: sorted position array
                                  float4 *oldVel,           // input: sorted velocity array
                                  uint    numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    cg::sync(cta);

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        float3 pos = oldPos[sortedIndex];
        float4 vel = oldVel[sortedIndex];

        sortedPos[index] = pos;
        sortedVel[index] = vel;
    }
}

__device__ float phi(const float h) {
    float hh = abs(h);
    if (hh > 2.0) return 0.0;
    else return 0.25 * (1 + cos(fpi * h * 0.5));
}

__device__ float delta(const float3 pos1, const float3 pos2) {

    auto r = params.cellRadius/2.0;
    auto x = phi((pos1.x - pos2.x) / r);
    auto y = phi((pos1.y - pos2.y) / r);
    auto z = phi((pos1.z - pos2.z) / r);
    //printf("position1: %f, %f, %f\nposition2: % f, % f, % f\nphi: %f, %f, %f\n",
    //    pos1.x, pos1.y, pos1.z, pos2.x, pos2.y, pos2.z,x,y,z);
    return x * y * z;
}

__device__ float3 collideOne(const float3 pos1, const float3 pos2, const float4 val) {
    float d = delta(pos1, pos2);
    //printf("delta: %f\n", d);
    auto force = make_float3(val.w * d * val.x, val.w * d * val.y, val.w * d * val.z);
    //printf("position1: %f, %f, %f\n", pos1.x, pos1.y, pos1.z);
    //printf("position2: %f, %f, %f\n", pos2.x, pos2.y, pos2.z);
    //printf("x: %f, y: %f, z: %f, w: %f, d: %f \n", val.x, val.y, val.z,val.w,d);
    
    return force;
}

// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
                   uint    index,
                   float3  pos,
                   float3 *oldPos,
                   float4 *oldVel,
                   uint   *cellStart,
                   uint   *cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];
        if (index == 0) printf("cell size:%d\n", endIndex - startIndex);
        for (uint j=startIndex; j<endIndex; j++)
        {
            force += collideOne(oldPos[j], pos, oldVel[j]);
            if (index == 0) {
                printf("neighbour particle position: %f, %f, %f\n", oldPos[j].x, oldPos[j].y, oldPos[j].z);
            }
        }
    }


    return force;
}


__global__
void collideD(float3 *newVel,               // output: new velocity
              float3 *newPos,               // input: new positions
              float4 *oldVel,               // input: sorted velocities
              float3 *oldPos,               // input: sorted positions
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles_new)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles_new) return;

    // read particle data from sorted arrays
    float3 pos = newPos[index];

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);
    if (index == 0) {
        printf("position: %f, %f, %f\n", pos.x, pos.y, pos.z);
    }
    if (index == 0) {
        printf("grid position: %d, %d, %d\n", gridPos.x, gridPos.y, gridPos.z);
    }
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                if (index == 0) {
                    printf("neighbour grid position: %d, %d, %d\n", neighbourPos.x, neighbourPos.y, neighbourPos.z);
                }
                force += collideCell(neighbourPos, index, pos, oldPos, oldVel, cellStart, cellEnd);

            }
        }
    }

    // printf("x: %f, y: %f, z: %f \n", force.x, force.y, force.z);
    // write new velocity back to original unsorted location
    newVel[index] = force;
}

#endif
