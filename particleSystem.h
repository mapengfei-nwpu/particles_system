

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, float cellRadius);
        ~ParticleSystem();

    protected: // methods
        ParticleSystem() {}

        void _initialize(int numParticles);
        void _finalize();

    protected: // data
        bool m_bInitialized;
        uint m_numParticles;

        // CPU data
        float *m_hPos;              // particle positions
        float *m_hVal;              // particle velocities

        uint  *m_hParticleHash;
        uint  *m_hCellStart;
        uint  *m_hCellEnd;

        // GPU data
        float *m_dPos;
        float *m_dVal;

        float *m_dSortedPos;
        float *m_dSortedVal;

        // grid data for sorting method
        uint  *m_dGridParticleHash; // grid hash value for each particle
        uint  *m_dGridParticleIndex;// particle index for each particle
        uint  *m_dCellStart;        // index of start of each cell in sorted list
        uint  *m_dCellEnd;          // index of end of cell

        uint   m_gridSortBits;


        // params
        SimParams m_params;
        uint3 m_gridSize;
        uint m_numGridCells;

};

#endif // __PARTICLESYSTEM_H__
