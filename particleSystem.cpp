
#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <algorithm>

ParticleSystem::ParticleSystem(uint numParticles, float cellRadius) :
    m_bInitialized(false),
    m_numParticles(numParticles),
    m_dPos(nullptr),
    m_dVal(nullptr)
{
    // set the radius of a cell in the grid. 
    m_params.cellSize = make_float3(cellRadius, cellRadius, cellRadius);

    // set the simulation world as [-1£¬-1£¬-1]X[1£¬1£¬1].
    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    m_params.worldSize = make_float3(2.0f, 2.0f, 2.0f);

    // compute the grid size.
    auto num_cellx = static_cast<uint>(ceil(m_params.worldSize.x / cellRadius));
    auto num_celly = static_cast<uint>(ceil(m_params.worldSize.y / cellRadius));
    auto num_cellz = static_cast<uint>(ceil(m_params.worldSize.z / cellRadius));
    m_gridSize = make_uint3(num_cellx, num_celly, num_cellz);
    m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;

    // set simulation parameters.
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numParticles = m_numParticles;

    _initialize(numParticles);
}


ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

void
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 3 * m_numParticles;

    allocateArray((void **)&m_dVal, memSize);

    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVal, memSize);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

    setParameters(&m_params);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    freeArray(m_dVal);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVal);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);
}

void ParticleSystem::inputData(float* pos, float* val) {
    std::vector<float> pos1;
    auto data = pos1.data();
    copyArrayToDevice(pos, m_dPos, sizeof(float)*m_numParticles*3);


}

void ParticleSystem::interpolate(uint numParticleNew, float* pos, float* val) {

}
