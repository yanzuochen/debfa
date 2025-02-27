/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"

uint8_t *constantWeightVarsAddr{nullptr};
uint8_t *mutableWeightVarsAddr{nullptr};
uint8_t *activationsAddr{nullptr};

/// \returns the index of the element at x,y,z,w.
ssize_t getXYZW(const ssize_t *dims, ssize_t x, ssize_t y, ssize_t z, ssize_t w) {
  return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
         (z * dims[3]) + w;
}

/// Find in the bundle's symbol table a weight variable whose name starts with
/// \p name.
const SymbolTableEntry *getWeightVar(const BundleConfig &config,
                                     const char *name) {
  for (unsigned i = 0, e = config.numSymbols; i < e; ++i) {
    if (!strncmp(config.symbolTable[i].name, name, strlen(name))) {
      return &config.symbolTable[i];
    }
  }
  return nullptr;
}

/// Find in the bundle's symbol table a mutable weight variable whose name
/// starts with \p name.
const SymbolTableEntry &getMutableWeightVar(const BundleConfig &config,
                                            const char *name) {
  const SymbolTableEntry *mutableWeightVar = getWeightVar(config, name);
  assert(mutableWeightVar && "Expected to find a mutable weight variable");
  assert(mutableWeightVar->kind != 0 &&
         "Weight variable is expected to be mutable");
  return *mutableWeightVar;
}

/// Allocate an aligned block of memory.
void *alignedAlloc(const BundleConfig &config, size_t size) {
  void *ptr;
  // Properly align the memory region.
  int res = posix_memalign(&ptr, config.alignment, size);
  assert(res == 0 && "posix_memalign failed");
  assert((size_t)ptr % config.alignment == 0 && "Wrong alignment");
  memset(ptr, 0, size);
  (void)res;
  return ptr;
}

/// Initialize the constant weights memory block by loading the weights from the
/// weights file.
static uint8_t *initConstantWeights(const char *weightsFileName,
                                    const BundleConfig &config) {
  // Load weights.
  FILE *weightsFile = fopen(weightsFileName, "rb");
  assert(weightsFile && "Could not open the weights file");
  fseek(weightsFile, 0, SEEK_END);
  size_t fileSize = ftell(weightsFile);
  fseek(weightsFile, 0, SEEK_SET);
  uint8_t *baseConstantWeightVarsAddr =
      static_cast<uint8_t *>(alignedAlloc(config, fileSize));
  // printf("Allocated weights of size: %lu\n", fileSize);
  // printf("Expected weights of size: %" PRIu64 "\n", config.constantWeightVarsMemSize);
  assert(fileSize == config.constantWeightVarsMemSize && "Wrong weights file size");
  int result = fread(baseConstantWeightVarsAddr, fileSize, 1, weightsFile);
  assert(result == 1 && "Could not read the weights file");
  (void)result;
  // printf("Loaded weights of size: %lu from the file %s\n", fileSize, weightsFileName);
  fclose(weightsFile);
  return baseConstantWeightVarsAddr;
}

static uint8_t *allocateMutableWeightVars(const BundleConfig &config) {
  auto *weights = static_cast<uint8_t *>(
      alignedAlloc(config, config.mutableWeightVarsMemSize));
  // printf("Allocated mutable weight variables of size: %" PRIu64 "\n", config.mutableWeightVarsMemSize);
  return weights;
}

static uint8_t *initActivations(const BundleConfig &config) {
  return static_cast<uint8_t *>(alignedAlloc(config, config.activationsMemSize));
}

static void dumpInferenceResults(uint8_t *mutableWeightVars, float *outarr) {
  auto &outputWeights = getMutableWeightVar(model_config, "output0");
  float *results = (float *)(mutableWeightVars + outputWeights.offset);
  memcpy(outarr, results, outputWeights.size * sizeof(float));
}

// Loads the NCHW (RGB) numpy array, converts it to NCHW (BGR) and stores it in
// the input data area.
static void loadNumpyArray(float *arr, ssize_t *strides, ssize_t *dims) {
  const SymbolTableEntry &inputVar = getMutableWeightVar(model_config, "input0");
  float *inputData = (float *)(mutableWeightVarsAddr + inputVar.offset);
  ssize_t float_size = sizeof(float);
  ssize_t skips[4] = {strides[0] / float_size, strides[1] / float_size,
                      strides[2] / float_size, strides[3] / float_size};

  assert(inputVar.size == dims[0] * dims[1] * dims[2] * dims[3]);

  for (ssize_t n = 0; n < dims[0]; n++) {
    for (ssize_t c = 0; c < dims[1]; c++) {
      for (ssize_t h = 0; h < dims[2]; h++) {
        for (ssize_t w = 0; w < dims[3]; w++) {
          inputData[getXYZW(dims, n, c, h, w)] =
              arr[n * skips[0] + c * skips[1] + h * skips[2] + w * skips[3]];
        }
      }
    }
  }
}

extern "C" {
  extern void init(char *weights_fpath);
  extern void deinit();
  extern int run_model(float *input, ssize_t *strides, ssize_t *dims, float *outarr);
}

void init(char *weights_fpath) {
  // Allocate and initialize constant and mutable weights.
  constantWeightVarsAddr = initConstantWeights(weights_fpath, model_config);
  mutableWeightVarsAddr = allocateMutableWeightVars(model_config);
  activationsAddr = initActivations(model_config);
}

void deinit() {
  free(activationsAddr);
  free(constantWeightVarsAddr);
  free(mutableWeightVarsAddr);
}

int run_model(float *inarr, ssize_t *strides, ssize_t *dims, float *outarr) {
  // Set input data.
  loadNumpyArray(inarr, strides, dims);

  // Perform the computation.
  int errCode = forward(constantWeightVarsAddr, mutableWeightVarsAddr, activationsAddr);
  if (errCode != GLOW_SUCCESS)
    return errCode;

  // Report the results.
  dumpInferenceResults(mutableWeightVarsAddr, outarr);
  return 0;
}
