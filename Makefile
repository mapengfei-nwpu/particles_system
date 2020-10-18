# Location of the CUDA Toolkit
# CUDA_PATH ?= /usr/local/cuda
CUDA_PATH ?= /usr

# architecture
TARGET_SIZE := 64

# host compiler
HOST_COMPILER := g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     := -fPIC
LDFLAGS     :=

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I./include
LIBRARIES :=

################################################################################
# Gencode arguments
# SMS ?= 30 35 37 50 52 60 61 70 75
SMS ?= 30

ifeq ($(SMS),)
    $(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
endif

ifeq ($(GENCODE_FLAGS),)
    # Generate SASS code for each SM architecture listed in $(SMS)
    $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

    # Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
    HIGHEST_SM := $(lastword $(sort $(SMS)))
    ifneq ($(HIGHEST_SM),)
        GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
    endif
endif

################################################################################

# Target rules
all: build

build: particles

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

particleSystem.o:particleSystem.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

particleSystem_cuda.o:particleSystem_cuda.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

libparticles.so: particleSystem.o particleSystem_cuda.o
	$(EXEC) $(NVCC) -shared -o $@ $+ $(LIBRARIES)
	rm -f particleSystem.o particleSystem_cuda.o

particles.o:particles.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

particles: particleSystem.o particleSystem_cuda.o particles.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) rm -f particleSystem.o particleSystem_cuda.o particles.o

run: build
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f particles particleSystem.o particleSystem_cuda.o particles.o libparticles.so

clobber: clean
