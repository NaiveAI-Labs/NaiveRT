/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#pragma once

#include <cstdint>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cfloat>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

static inline __device__ void rtcGridDependencySynchronize()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;\n");
#endif
}

static inline __device__ void rtcTriggerProgrammaticLaunchCompletion()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;\n");
#endif
}

template <int step>
static inline __device__ uint32_t extractBinIdx(float x)
{
    if constexpr (step == 0)
    {
        __half hx = __float2half(x);
        uint16_t bits = __half_as_ushort(hx);
        bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
        return bits >> 5;
    }
    else
    {
        uint32_t bits = __float_as_uint(x);
        bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

        if constexpr (step == 1)
        {
            return bits >> 21;
        }
        else if constexpr (step == 2)
        {
            return (bits >> 10) & 0x7ff;
        }
        else if constexpr (step == 3)
        {
            return bits & 0x3ff;
        }
    }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern)
{
    if constexpr (shift == 0)
    {
        return true;
    }
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
    return (bits ^ pattern) >> shift == 0;
}

static inline __device__ float topKScoreToFloat(float x)
{
    return x;
}

static inline __device__ float topKScoreToFloat(__half x)
{
    return __half2float(x);
}

static inline __device__ float topKScoreToFloat(__nv_bfloat16 x)
{
    return __bfloat162float(x);
}

template <typename T, typename IdxT, typename Func>
__device__ void vectorizedProcess(size_t threadRank, size_t numThreads, T const* in, IdxT len, Func f)
{
    constexpr int WARP_SIZE = 32;
    using WideT = float4;
    if constexpr (sizeof(T) >= sizeof(WideT))
    {
        for (IdxT i = threadRank; i < len; i += numThreads)
        {
            f(in[i], i);
        }
    }
    else
    {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        constexpr int itemsPerScalar = sizeof(WideT) / sizeof(T);

        union
        {
            WideT scalar;
            T array[itemsPerScalar];
        } wide;

        int skipCount = (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
            : 0;
        if (skipCount > len)
        {
            skipCount = len;
        }
        WideT const* inCast = reinterpret_cast<WideT const*>(in + skipCount);
        IdxT const lenCast = (len - skipCount) / itemsPerScalar;

        for (IdxT i = threadRank; i < lenCast; i += numThreads)
        {
            wide.scalar = inCast[i];
            IdxT const realIdx = skipCount + i * itemsPerScalar;
#pragma unroll
            for (int j = 0; j < itemsPerScalar; ++j)
            {
                f(wide.array[j], realIdx + j);
            }
        }

        static_assert(WARP_SIZE >= itemsPerScalar);
        if (threadRank < skipCount)
        {
            f(in[threadRank], threadRank);
        }

        IdxT const remainIdx = skipCount + lenCast * itemsPerScalar + threadRank;
        if (remainIdx < len)
        {
            f(in[remainIdx], remainIdx);
        }
    }
}

struct EmptyTopKScoreAccessor
{
    using InputType = float;
    static constexpr bool kSupportsVectorizedProcess = false;

    __device__ __forceinline__ float score(int) const
    {
        return 0.0f;
    }
};

template <typename ScoreAccessor, typename Func>
__device__ void vectorizedAccessorProcess(
    size_t threadRank, size_t numThreads, int rowStart, int rowEnd, ScoreAccessor const& scoreAccessor, Func f)
{
    using InputT = typename ScoreAccessor::InputType;
    int segmentStart = rowStart;
    while (segmentStart < rowEnd)
    {
        int const segmentLen = scoreAccessor.contiguousLength(segmentStart, rowEnd);
        InputT const* segmentPtr = scoreAccessor.contiguousPtr(segmentStart);
        auto processSegment = [&](InputT logit, int segmentOffset)
        {
            f(logit, segmentStart + segmentOffset);
        };
        vectorizedProcess(threadRank, numThreads, segmentPtr, segmentLen, processSegment);
        segmentStart += segmentLen;
    }
}

template <int step, int kNumThreadsPerBlock, int kNumBins, int kNumFinalItems, bool multipleBlocksPerRow,
    bool mergeBlocks, bool useAccessor, typename SmemFinalType, typename SmemOutputType, typename InputT = float,
    typename ScoreAccessor = EmptyTopKScoreAccessor>
__device__ bool processHistogramStep(int const* indices, InputT const* logits, int rowEnd, uint32_t& logitPattern,
    int& thresholdBinIdx, SmemOutputType& smemOutput, int* smemThresholdBinIdx, int* smemFinalDstIdx,
    int* smemFinalBinSize, int* smemFoundTopKValues, SmemFinalType& smemFinal, int stride1, int rowStart, int topK,
    ScoreAccessor const& scoreAccessor)
{
#pragma unroll
    for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock)
    {
        smemFinal.histo.data[idx] = 0;
    }
    __syncthreads();

    constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
    if constexpr (step == 2)
    {
        logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    }
    else if constexpr (step == 3)
    {
        logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    }

    auto distributeToBins = [&](InputT logitIn, int)
    {
        float const logit = topKScoreToFloat(logitIn);
        if (isPartialMatch<patternShift>(logit, logitPattern))
        {
            uint32_t binIdx = extractBinIdx<step>(logit);
            atomicAdd(&smemFinal.histo.data[binIdx], 1);
        }
    };

    if constexpr (useAccessor)
    {
        if constexpr (ScoreAccessor::kSupportsVectorizedProcess)
        {
            vectorizedAccessorProcess(threadIdx.x, kNumThreadsPerBlock, rowStart, rowEnd, scoreAccessor, distributeToBins);
        }
        else
        {
            for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock)
            {
                distributeToBins(scoreAccessor.score(idx), idx);
            }
        }
    }
    else if (stride1 == 1)
    {
        vectorizedProcess(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, distributeToBins);
    }
    else
    {
        for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock)
        {
            distributeToBins(logits[idx * stride1], idx);
        }
    }
    __syncthreads();

    int lastValue = smemFoundTopKValues[0];
    for (int round = 0; round < kNumBins / kNumThreadsPerBlock; ++round)
    {
        int idx = threadIdx.x + kNumThreadsPerBlock * round;
        int binCount = smemFinal.histo.data[idx];
        __syncthreads();

        int prefixSum = 0;
        int totalSum = 0;
        using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
        Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

        prefixSum += lastValue;
        totalSum += lastValue;
        smemFinal.histo.data[idx] = prefixSum;
        __syncthreads();

        bool foundThreshold = false;
        if (prefixSum < topK)
        {
            int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1 ? totalSum : smemFinal.histo.data[idx + 1];
            if (nextPrefixSum >= topK)
            {
                smemThresholdBinIdx[0] = idx;
                smemFinalBinSize[0] = nextPrefixSum - prefixSum;
                foundThreshold = true;
            }
        }

        if (__syncthreads_or(foundThreshold))
        {
            break;
        }
        lastValue = totalSum;
    }

    __syncthreads();
    thresholdBinIdx = smemThresholdBinIdx[0];

    // Half bins from step 0 do not align with the full-precision filters used
    // in later steps, so defer auto-promotes when step 0 needs refinement.
    bool const step0WillContinue = (step == 0) && (smemFinalBinSize[0] > kNumFinalItems);

    auto processBins = [&](InputT logitIn, int idx)
    {
        float const logit = topKScoreToFloat(logitIn);
        if (isPartialMatch<patternShift>(logit, logitPattern))
        {
            uint32_t binIdx = extractBinIdx<step>(logit);
            if (binIdx < thresholdBinIdx && !step0WillContinue)
            {
                int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);
                if constexpr (mergeBlocks)
                {
                    smemOutput[dstIdx] = indices[idx];
                }
                else if constexpr (multipleBlocksPerRow)
                {
                    smemOutput[dstIdx] = idx + rowStart;
                    reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                }
                else
                {
                    smemOutput[dstIdx] = idx;
                }
            }

            if constexpr (step < 3)
            {
                if (binIdx == thresholdBinIdx && smemFinalBinSize[0] <= kNumFinalItems)
                {
                    int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
                    smemFinal.items.logits[dstIdx] = logit;
                    if constexpr (mergeBlocks)
                    {
                        smemFinal.items.indices[dstIdx] = indices[idx];
                    }
                    else if constexpr (multipleBlocksPerRow)
                    {
                        smemFinal.items.indices[dstIdx] = idx + rowStart;
                    }
                    else
                    {
                        smemFinal.items.indices[dstIdx] = idx;
                    }
                }
            }
            else
            {
                if (binIdx == thresholdBinIdx)
                {
                    int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
                    if (dstIdx < topK)
                    {
                        if constexpr (mergeBlocks)
                        {
                            smemOutput[dstIdx] = indices[idx];
                        }
                        else if constexpr (multipleBlocksPerRow)
                        {
                            smemOutput[dstIdx] = idx + rowStart;
                            reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                        }
                        else
                        {
                            smemOutput[dstIdx] = idx;
                        }
                    }
                }
            }
        }
    };

    if constexpr (useAccessor)
    {
        if constexpr (ScoreAccessor::kSupportsVectorizedProcess)
        {
            vectorizedAccessorProcess(threadIdx.x, kNumThreadsPerBlock, rowStart, rowEnd, scoreAccessor, processBins);
        }
        else
        {
            for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock)
            {
                processBins(scoreAccessor.score(idx), idx);
            }
        }
    }
    else if (stride1 == 1)
    {
        vectorizedProcess(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, processBins);
    }
    else
    {
        for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock)
        {
            processBins(logits[idx * stride1], idx);
        }
    }

    __syncthreads();
    return smemFinalBinSize[0] > kNumFinalItems;
}

static inline constexpr __device__ float negativeMaxFloat()
{
    return -cuda::std::numeric_limits<float>::max();
}

template <int kNumThreadsPerBlock>
__device__ void writeDenseDecodeRow(int* outIndices, int rowEnd, int topK)
{
    int validCount = rowEnd;
    if (validCount < 0)
    {
        validCount = 0;
    }
    else if (validCount > topK)
    {
        validCount = topK;
    }

    for (int rowIt = threadIdx.x; rowIt < validCount; rowIt += kNumThreadsPerBlock)
    {
        outIndices[rowIt] = rowIt;
    }
    for (int rowIt = validCount + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock)
    {
        outIndices[rowIt] = -1;
    }
}

template <int kNumThreadsPerBlock>
__device__ void writeDenseDecodeRowWithLogits(int* outIndices, float* outLogits, int rowEnd, int topK)
{
    constexpr float _negativeMaxFloat = negativeMaxFloat();
    int validCount = rowEnd;
    if (validCount < 0)
    {
        validCount = 0;
    }
    else if (validCount > topK)
    {
        validCount = topK;
    }

    for (int rowIt = threadIdx.x; rowIt < validCount; rowIt += kNumThreadsPerBlock)
    {
        outIndices[rowIt] = rowIt;
        outLogits[rowIt] = static_cast<float>(rowIt);
    }
    for (int rowIt = validCount + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock)
    {
        outIndices[rowIt] = -1;
        outLogits[rowIt] = _negativeMaxFloat;
    }
}

template <int kNumThreadsPerBlock>
__device__ void writeInvalidDecodeSplitBlock(int* outIndices, float* outLogits, int topK)
{
    constexpr float _negativeMaxFloat = negativeMaxFloat();
    for (int rowIt = threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock)
    {
        outIndices[rowIt] = -1;
        outLogits[rowIt] = _negativeMaxFloat;
    }
}

template <int kNumThreadsPerBlock>
__device__ void topKPerRowDecodeDenseImpl(int const* seqLens, int* outIndices, int topK, int nextN)
{
    int rowIdx = blockIdx.x;
    int seqLen = seqLens[rowIdx / nextN];
    int rowEnd = seqLen - nextN + (rowIdx % nextN) + 1;

    outIndices += static_cast<long long>(rowIdx) * topK;
    writeDenseDecodeRow<kNumThreadsPerBlock>(outIndices, rowEnd, topK);
}

template <int kNumThreadsPerBlock, int kNumBins>
struct TopKSmem
{
    static constexpr int kNumFinalItems = 2048;
    static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
    using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;
    using FinalSortTempStorage = typename FinalSort::TempStorage;
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

    struct FinalItems
    {
        int indices[kNumFinalItems];
        float logits[kNumFinalItems];
    };

    struct Histogram
    {
        typename Scan::TempStorage scan;
        int data[kNumBins];
    };

    union Final
    {
        FinalItems items;
        FinalSortTempStorage finalSort;
        Histogram histo;
    };

    Final smemFinal;
    int smemThresholdBinIdx[1];
    int smemFinalDstIdx[1];
    int smemFinalBinSize[1];
    int smemFoundTopKValues[1];
};

template <int kNumThreadsPerBlock, int kNumBins, bool multipleBlocksPerRow = false, bool mergeBlocks = false,
    bool useAccessor = false, typename InputT = float, typename ScoreAccessor = EmptyTopKScoreAccessor>
static __device__ void topKPerRowJob(int const* indices, InputT const* logits, int rowStart, int rowEnd,
    int* outIndices, float* outLogits, int stride1, int topK, TopKSmem<kNumThreadsPerBlock, kNumBins>& smem,
    ScoreAccessor const& scoreAccessor = EmptyTopKScoreAccessor{})
{
    static constexpr int kNumFinalItems = TopKSmem<kNumThreadsPerBlock, kNumBins>::kNumFinalItems;
    static constexpr int kNumFinalItemsPerThread = TopKSmem<kNumThreadsPerBlock, kNumBins>::kNumFinalItemsPerThread;
    using FinalSort = typename TopKSmem<kNumThreadsPerBlock, kNumBins>::FinalSort;

    auto& smemFinal = smem.smemFinal;
    int* smemThresholdBinIdx = smem.smemThresholdBinIdx;
    int* smemFinalDstIdx = smem.smemFinalDstIdx;
    int* smemFinalBinSize = smem.smemFinalBinSize;
    int* smemFoundTopKValues = smem.smemFoundTopKValues;

    extern __shared__ int32_t smemOutput[];

    int rowLen = rowEnd - rowStart;
    if (rowLen <= topK)
    {
        for (int rowIt = threadIdx.x; rowIt < rowLen; rowIt += kNumThreadsPerBlock)
        {
            if constexpr (multipleBlocksPerRow)
            {
                outIndices[rowIt] = rowIt + rowStart;
                outLogits[rowIt] = static_cast<float>(logits[rowIt + rowStart]);
            }
            else
            {
                outIndices[rowIt] = rowIt;
            }
        }
        for (int rowIt = rowLen + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock)
        {
            outIndices[rowIt] = -1;
            if constexpr (multipleBlocksPerRow)
            {
                constexpr float _negativeMaxFloat = negativeMaxFloat();

                outLogits[rowIt] = _negativeMaxFloat;
            }
        }
        return;
    }

    if (threadIdx.x == 0)
    {
        smemFinalDstIdx[0] = 0;
        smemFoundTopKValues[0] = 0;
    }
    __syncthreads();

    int thresholdBinIdx = -1;
    uint32_t logitPattern = 0;
    bool continueToNextStep = processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
        multipleBlocksPerRow, mergeBlocks, useAccessor>(indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
        smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart,
        topK, scoreAccessor);

    if (continueToNextStep)
    {
        continueToNextStep = processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
            multipleBlocksPerRow, mergeBlocks, useAccessor>(indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1,
            rowStart, topK, scoreAccessor);
    }

    if (continueToNextStep)
    {
        continueToNextStep = processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
            multipleBlocksPerRow, mergeBlocks, useAccessor>(indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1,
            rowStart, topK, scoreAccessor);
    }

    if (continueToNextStep)
    {
        processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks,
            useAccessor>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
            smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK,
            scoreAccessor);
    }

    if (!continueToNextStep)
    {
        constexpr int kInsertionSortBranchThreshold = 512;
        int const finalCount = smemFinalDstIdx[0];
        if (finalCount > kInsertionSortBranchThreshold)
        {
            float finalLogits[kNumFinalItemsPerThread];
            int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
            {
                // CUB radix-sort transforms 0xffffffff to its lowest key (0).
                // This negative NaN sentinel sorts below valid -inf; -FLT_MAX
                // does not, and could promote padding with uninitialized IDs.
                finalLogits[ii] = __int_as_float(-1);
            }

#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
            {
                int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
                if (srcIdx < finalCount)
                {
                    finalLogits[ii] = smemFinal.items.logits[srcIdx];
                    finalIndices[ii] = smemFinal.items.indices[srcIdx];
                }
            }
            __syncthreads();

            FinalSort(smemFinal.finalSort).SortDescendingBlockedToStriped(finalLogits, finalIndices);
            int baseIdx = smemFoundTopKValues[0];

#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
            {
                int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
                int dstIdx = baseIdx + srcIdx;
                if (dstIdx < topK)
                {
                    smemOutput[dstIdx] = finalIndices[ii];
                    if constexpr (multipleBlocksPerRow)
                    {
                        reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = finalLogits[ii];
                    }
                }
            }
        }
        else
        {
            int baseIdx = smemFoundTopKValues[0];
            for (int i = threadIdx.x; i < finalCount; i += kNumThreadsPerBlock)
            {
                int outIndex = 0;
                float logit = smemFinal.items.logits[i];
                for (int j = 0; j < finalCount; ++j)
                {
                    float otherLogit = smemFinal.items.logits[j];
                    if (logit < otherLogit || (logit == otherLogit && i < j))
                    {
                        outIndex++;
                    }
                }
                if (outIndex + baseIdx < topK)
                {
                    smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
                    if constexpr (multipleBlocksPerRow)
                    {
                        reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] = smemFinal.items.logits[i];
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < topK; i += kNumThreadsPerBlock)
    {
        if constexpr (multipleBlocksPerRow)
        {
            outIndices[i] = smemOutput[i];
            outLogits[i] = reinterpret_cast<float*>(smemOutput + topK)[i];
        }
        else if constexpr (mergeBlocks)
        {
            outIndices[i] = smemOutput[i];
        }
        else
        {
            if (stride1 == 1)
            {
                outIndices[i] = smemOutput[i];
            }
            else
            {
                outIndices[i] = smemOutput[i] - rowStart;
            }
        }
    }
}

template <int kNumThreadsPerBlock, int kNumBins, typename InputT = float>
__device__ void topKPerRowPrefillImpl(InputT const* logits, int const* rowStarts, int const* rowEnds, int* outIndices,
    int stride0, int stride1, int topK, int offsetIndex)
{
    int rowIdx = blockIdx.x + offsetIndex;
    int rowStart = rowStarts[rowIdx];
    int rowEnd = rowEnds[rowIdx];

    outIndices += static_cast<long long>(rowIdx) * topK;
    logits += static_cast<long long>(rowIdx) * stride0;
    __shared__ TopKSmem<kNumThreadsPerBlock, kNumBins> smem;
    topKPerRowJob<kNumThreadsPerBlock, kNumBins>(
        nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1, topK, smem);
}

template <int kNumThreadsPerBlock, int kNumBins, bool multipleBlocksPerRow = false, bool mergeBlocks = false,
    typename InputT = float>
__device__ void topKPerRowDecodeImpl(InputT const* logits, int const* seqLens, int* outIndices, int stride0,
    int stride1, int topK, int nextN, float* outLogits, int numBlocksToMerge, int const* indices)
{
    int rowIdx = blockIdx.x;
    int rowStart = 0;
    int seqLen = seqLens[rowIdx / nextN];
    int rowEnd = seqLen - nextN + (rowIdx % nextN) + 1;
    int effectiveRowEnd = rowEnd;

    if constexpr (!multipleBlocksPerRow && !mergeBlocks)
    {
        outIndices += static_cast<long long>(rowIdx) * topK;
    }
    else if constexpr (multipleBlocksPerRow)
    {
        outIndices += static_cast<long long>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
        outLogits += static_cast<long long>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;

        if (effectiveRowEnd <= topK)
        {
            if (blockIdx.y == 0)
            {
                writeDenseDecodeRowWithLogits<kNumThreadsPerBlock>(outIndices, outLogits, effectiveRowEnd, topK);
            }
            else
            {
                writeInvalidDecodeSplitBlock<kNumThreadsPerBlock>(outIndices, outLogits, topK);
            }
            return;
        }

        auto const blockSize = rowEnd / gridDim.y;
        rowStart = blockSize * blockIdx.y;
        rowEnd = gridDim.y == blockIdx.y + 1 ? rowEnd : rowStart + blockSize;
    }
    else if constexpr (mergeBlocks)
    {
        outIndices += static_cast<long long>(rowIdx) * topK;

        if (effectiveRowEnd <= topK)
        {
            writeDenseDecodeRow<kNumThreadsPerBlock>(outIndices, effectiveRowEnd, topK);
            return;
        }

        rowEnd = numBlocksToMerge * topK;
        indices += static_cast<long long>(rowIdx) * numBlocksToMerge * topK;
    }

    logits += static_cast<long long>(rowIdx) * stride0;
    __shared__ TopKSmem<kNumThreadsPerBlock, kNumBins> smem;
    topKPerRowJob<kNumThreadsPerBlock, kNumBins, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowStart, rowEnd, outIndices, outLogits, stride1, topK, smem);
}

template <typename InputT>
struct CpMajorSeparateTopKAccessor
{
    using InputType = InputT;
    static constexpr bool kSupportsVectorizedProcess = true;

    InputT const* scores;
    int row;
    int numRows;
    int localTopK;

    __device__ __forceinline__ InputT score(int idx) const
    {
        int part = idx / localTopK;
        int pos = idx - part * localTopK;
        return scores[(static_cast<long long>(part) * numRows + row) * localTopK + pos];
    }

    __device__ __forceinline__ InputT const* contiguousPtr(int idx) const
    {
        int part = idx / localTopK;
        int pos = idx - part * localTopK;
        return scores + (static_cast<long long>(part) * numRows + row) * localTopK + pos;
    }

    __device__ __forceinline__ int contiguousLength(int idx, int rowEnd) const
    {
        int pos = idx & (localTopK - 1);
        int len = localTopK - pos;
        int remain = rowEnd - idx;
        return len < remain ? len : remain;
    }
};

template <typename InputT>
struct CpMajorPackedTopKAccessor
{
    using InputType = InputT;
    static constexpr bool kSupportsVectorizedProcess = true;

    InputT const* packed;
    int row;
    int numRows;
    int localTopK;
    int packedStride;

    __device__ __forceinline__ InputT score(int idx) const
    {
        int part = idx / localTopK;
        int pos = idx - part * localTopK;
        return packed[(static_cast<long long>(part) * numRows + row) * packedStride + pos];
    }

    __device__ __forceinline__ InputT const* contiguousPtr(int idx) const
    {
        int part = idx / localTopK;
        int pos = idx - part * localTopK;
        return packed + (static_cast<long long>(part) * numRows + row) * packedStride + pos;
    }

    __device__ __forceinline__ int contiguousLength(int idx, int rowEnd) const
    {
        int pos = idx & (localTopK - 1);
        int len = localTopK - pos;
        int remain = rowEnd - idx;
        return len < remain ? len : remain;
    }
};

template <int kNumThreadsPerBlock, int kNumBins, typename InputT = float>
__device__ void topKPerRowCpMajorSeparateImpl(InputT const* scores, int* outIndices, int rowEnd, int numRows,
    int localTopK, int topK)
{
    int rowIdx = blockIdx.x;
    outIndices += static_cast<long long>(rowIdx) * topK;
    CpMajorSeparateTopKAccessor<InputT> accessor{scores, rowIdx, numRows, localTopK};
    __shared__ TopKSmem<kNumThreadsPerBlock, kNumBins> smem;
    topKPerRowJob<kNumThreadsPerBlock, kNumBins, false, false, true, InputT, CpMajorSeparateTopKAccessor<InputT>>(
        nullptr, nullptr, 0, rowEnd, outIndices, nullptr, 1, topK, smem, accessor);
}

template <int kNumThreadsPerBlock, int kNumBins, typename InputT = float>
__device__ void topKPerRowCpMajorPackedImpl(InputT const* packed, int* outIndices, int rowEnd, int numRows,
    int localTopK, int packedStride, int topK)
{
    int rowIdx = blockIdx.x;
    outIndices += static_cast<long long>(rowIdx) * topK;
    CpMajorPackedTopKAccessor<InputT> accessor{packed, rowIdx, numRows, localTopK, packedStride};
    __shared__ TopKSmem<kNumThreadsPerBlock, kNumBins> smem;
    topKPerRowJob<kNumThreadsPerBlock, kNumBins, false, false, true, InputT, CpMajorPackedTopKAccessor<InputT>>(
        nullptr, nullptr, 0, rowEnd, outIndices, nullptr, 1, topK, smem, accessor);
}

static constexpr int kRadixBins = 2048;

struct alignas(64) RadixState
{
    int candCount;
    int outIdx;
    int kRemaining;
    int filterCnt;
    int thresholdBin;
    int finishedBlocks;
    int thresholdLess;
    int padding[1];
};

template <int kThreads, int step>
__device__ __forceinline__ void radixLastBlockTrailer(int* gHist, RadixState& st, int topK, int rowFullLen)
{
    using Scan = cub::BlockScan<int, kThreads>;
    __shared__ typename Scan::TempStorage scanStorage;
    __shared__ int sThresholdBin;
    __shared__ int sRunningBefore;
    __shared__ int sThresholdCount;

    if (threadIdx.x == 0)
    {
        sThresholdBin = -1;
        sRunningBefore = 0;
        sThresholdCount = 0;
    }
    __syncthreads();

    int const kRem = (step == 1) ? topK : (topK - st.outIdx);
    if (kRem <= 0)
    {
        if (threadIdx.x == 0)
        {
            st.thresholdBin = -1;
            st.candCount = 0;
            st.kRemaining = 0;
            st.finishedBlocks = 0;
            st.filterCnt = 0;
            (void) rowFullLen;
        }
        if constexpr (step < 3)
        {
            __syncthreads();
            for (int i = threadIdx.x; i < kRadixBins; i += kThreads)
            {
                gHist[i] = 0;
            }
        }
        return;
    }

    constexpr int kRoundsPerScan = kRadixBins / kThreads;
    int running = 0;
    for (int r = 0; r < kRoundsPerScan; ++r)
    {
        int bin = r * kThreads + threadIdx.x;
        int c = gHist[bin];
        int prefix = 0;
        int total = 0;
        Scan(scanStorage).ExclusiveSum(c, prefix, total);
        prefix += running;
        int next = prefix + c;
        if (prefix < kRem && next >= kRem && sThresholdBin == -1)
        {
            atomicCAS(&sThresholdBin, -1, bin);
            if (sThresholdBin == bin)
            {
                sRunningBefore = prefix;
                sThresholdCount = c;
            }
        }
        running += total;
        __syncthreads();
        if (sThresholdBin != -1)
        {
            break;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        st.thresholdBin = (sThresholdBin == -1) ? kRadixBins : sThresholdBin;
        st.thresholdLess = sRunningBefore;
        st.finishedBlocks = 0;
        if constexpr (step == 1)
        {
            st.candCount = rowFullLen;
            (void) sThresholdCount;
            (void) kRem;
        }
        else
        {
            st.candCount = st.filterCnt;
            int newKRem = topK - st.outIdx;
            st.kRemaining = newKRem < 0 ? 0 : newKRem;
            st.filterCnt = 0;
        }
    }
    if constexpr (step < 3)
    {
        __syncthreads();
        for (int i = threadIdx.x; i < kRadixBins; i += kThreads)
        {
            gHist[i] = 0;
        }
    }
}

template <int kThreads, int step, bool useAccessor = false, typename InputT = float,
    typename ScoreAccessor = EmptyTopKScoreAccessor>
__device__ void radixPassKernelImpl(InputT const* logits, int const* seqLens, int* outIndices, int const* candBufIn,
    int* candBufOut, int* histograms, RadixState* state, int stride0, int nextN, int topK,
    ScoreAccessor const& scoreAccessor = EmptyTopKScoreAccessor{}, int accessorRowEnd = -1)
{
    int rowIdx = blockIdx.y;
    int blockInRow = blockIdx.x;
    int blocksPerRow = gridDim.x;

    RadixState& st = state[rowIdx];
    int* gHist = histograms + static_cast<long long>(rowIdx) * kRadixBins;

    __shared__ int sHist[kRadixBins];
    for (int i = threadIdx.x; i < kRadixBins; i += kThreads)
    {
        sHist[i] = 0;
    }
    __syncthreads();

    if constexpr (step == 1)
    {
        int const rowEnd = useAccessor ? accessorRowEnd : (seqLens[rowIdx / nextN] - nextN + (rowIdx % nextN) + 1);
        // NaiveOps keeps dense decode rows canonical even in the multi-pass
        // radix tier: [0, rowEnd) followed by -1 padding.
        if (rowEnd <= topK)
        {
            int* outIdxArr = outIndices + static_cast<long long>(rowIdx) * topK;
            if (blockInRow == 0)
            {
                writeDenseDecodeRow<kThreads>(outIdxArr, rowEnd, topK);
                if (threadIdx.x == 0)
                {
                    st.candCount = 0;
                    st.outIdx = 0;
                    st.kRemaining = 0;
                    st.filterCnt = 0;
                    st.thresholdBin = -2;
                    st.finishedBlocks = 0;
                    st.thresholdLess = 0;
                }
            }
            return;
        }
        size_t threadRank = static_cast<size_t>(blockInRow) * kThreads + threadIdx.x;
        size_t numThreads = static_cast<size_t>(blocksPerRow) * kThreads;
        auto f = [&](InputT vIn, int)
        {
            float const v = topKScoreToFloat(vIn);
            uint32_t bin = extractBinIdx<step>(v);
            atomicAdd(&sHist[bin], 1);
        };
        if constexpr (useAccessor)
        {
            vectorizedAccessorProcess(threadRank, numThreads, 0, rowEnd, scoreAccessor, f);
        }
        else
        {
            InputT const* in = logits + static_cast<long long>(rowIdx) * stride0;
            vectorizedProcess(threadRank, numThreads, in, rowEnd, f);
        }
    }
    else if constexpr (step == 2)
    {
        if (st.thresholdBin == -2)
        {
            return;
        }
        int const rowEnd = useAccessor ? accessorRowEnd : st.candCount;
        int* outIdxArr = outIndices + static_cast<long long>(rowIdx) * topK;
        int* candArr = candBufOut + static_cast<long long>(rowIdx) * stride0;
        int const prevThresh = st.thresholdBin;
        size_t threadRank = static_cast<size_t>(blockInRow) * kThreads + threadIdx.x;
        size_t numThreads = static_cast<size_t>(blocksPerRow) * kThreads;
        auto f = [&](InputT vIn, int i)
        {
            float const v = topKScoreToFloat(vIn);
            int bin1 = static_cast<int>(extractBinIdx<1>(v));
            if (bin1 < prevThresh)
            {
                int pos = atomicAdd(&st.outIdx, 1);
                if (pos < topK)
                {
                    outIdxArr[pos] = static_cast<int>(i);
                }
            }
            else if (bin1 == prevThresh)
            {
                int pos = atomicAdd(&st.filterCnt, 1);
                candArr[pos] = static_cast<int>(i);
                uint32_t bin2 = extractBinIdx<step>(v);
                atomicAdd(&sHist[bin2], 1);
            }
        };
        if constexpr (useAccessor)
        {
            vectorizedAccessorProcess(threadRank, numThreads, 0, rowEnd, scoreAccessor, f);
        }
        else
        {
            InputT const* in = logits + static_cast<long long>(rowIdx) * stride0;
            vectorizedProcess(threadRank, numThreads, in, rowEnd, f);
        }
    }
    else
    {
        if (st.thresholdBin == -2)
        {
            return;
        }
        int const candCnt = st.candCount;
        int const* candArrIn = candBufIn + static_cast<long long>(rowIdx) * stride0;
        int* outIdxArr = outIndices + static_cast<long long>(rowIdx) * topK;
        int* candArrOut = candBufOut + static_cast<long long>(rowIdx) * stride0;
        int const prevThresh = st.thresholdBin;
        for (int i = blockInRow * kThreads + threadIdx.x; i < candCnt; i += blocksPerRow * kThreads)
        {
            int srcIdx = candArrIn[i];
            float v;
            if constexpr (useAccessor)
            {
                v = topKScoreToFloat(scoreAccessor.score(srcIdx));
            }
            else
            {
                InputT const* in = logits + static_cast<long long>(rowIdx) * stride0;
                v = topKScoreToFloat(in[srcIdx]);
            }
            int bin2 = static_cast<int>(extractBinIdx<2>(v));
            if (bin2 < prevThresh)
            {
                int pos = atomicAdd(&st.outIdx, 1);
                if (pos < topK)
                {
                    outIdxArr[pos] = srcIdx;
                }
            }
            else if (bin2 == prevThresh)
            {
                int pos = atomicAdd(&st.filterCnt, 1);
                candArrOut[pos] = srcIdx;
                uint32_t bin3 = extractBinIdx<step>(v);
                atomicAdd(&sHist[bin3], 1);
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < kRadixBins; i += kThreads)
    {
        int c = sHist[i];
        if (c)
        {
            atomicAdd(&gHist[i], c);
        }
    }

    __threadfence();
    __shared__ int isLast;
    if (threadIdx.x == 0)
    {
        int prev = atomicAdd(&st.finishedBlocks, 1);
        isLast = (prev == blocksPerRow - 1) ? 1 : 0;
    }
    __syncthreads();
    if (!isLast)
    {
        return;
    }

    int const rowFullLen = (step == 1)
        ? (useAccessor ? accessorRowEnd : (seqLens[rowIdx / nextN] - nextN + (rowIdx % nextN) + 1))
        : 0;
    radixLastBlockTrailer<kThreads, step>(gHist, st, topK, rowFullLen);

    if constexpr (step == 3)
    {
        __syncthreads();
        int const filterCnt = st.candCount;
        int const thresh3 = st.thresholdBin;
        int const prefix3 = st.thresholdLess;
        int const* candArr = candBufOut + static_cast<long long>(rowIdx) * stride0;
        int* outIdxArr = outIndices + static_cast<long long>(rowIdx) * topK;
        int const ltBase = st.outIdx;
        int const eqBase = ltBase + prefix3;
        int const eqCap = topK - eqBase;
        __shared__ int sEqEmitted;
        if (threadIdx.x == 0)
        {
            sEqEmitted = 0;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < filterCnt; i += kThreads)
        {
            int srcIdx = candArr[i];
            float v;
            if constexpr (useAccessor)
            {
                v = topKScoreToFloat(scoreAccessor.score(srcIdx));
            }
            else
            {
                InputT const* in = logits + static_cast<long long>(rowIdx) * stride0;
                v = topKScoreToFloat(in[srcIdx]);
            }
            int bin3 = static_cast<int>(extractBinIdx<3>(v));
            if (bin3 < thresh3)
            {
                int pos = atomicAdd(&st.outIdx, 1);
                outIdxArr[pos] = srcIdx;
            }
            else if (bin3 == thresh3)
            {
                int pos = atomicAdd(&sEqEmitted, 1);
                if (pos < eqCap)
                {
                    outIdxArr[eqBase + pos] = srcIdx;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int eq = sEqEmitted < eqCap ? sEqEmitted : eqCap;
            int filled = eqBase + eq;
            if (filled > topK)
            {
                filled = topK;
            }
            for (int i = filled; i < topK; ++i)
            {
                outIdxArr[i] = -1;
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < kRadixBins; i += kThreads)
        {
            gHist[i] = 0;
        }
        if (threadIdx.x == 0)
        {
            st.outIdx = 0;
        }
    }
}

template <int kThreads, int step, typename InputT>
__device__ void radixPassCpMajorSeparateImpl(InputT const* scores, int* outIndices, int const* candBufIn,
    int* candBufOut, int* histograms, RadixState* state, int numCandidates, int numRows, int localTopK, int topK)
{
    CpMajorSeparateTopKAccessor<InputT> accessor{scores, static_cast<int>(blockIdx.y), numRows, localTopK};
    radixPassKernelImpl<kThreads, step, true, InputT, CpMajorSeparateTopKAccessor<InputT>>(
        nullptr, nullptr, outIndices, candBufIn, candBufOut, histograms, state, numCandidates, 1, topK, accessor,
        numCandidates);
}

template <int kThreads, int step, typename InputT>
__device__ void radixPassCpMajorPackedImpl(InputT const* packed, int* outIndices, int const* candBufIn,
    int* candBufOut, int* histograms, RadixState* state, int numCandidates, int numRows, int localTopK, int packedStride,
    int topK)
{
    CpMajorPackedTopKAccessor<InputT> accessor{
        packed, static_cast<int>(blockIdx.y), numRows, localTopK, packedStride};
    radixPassKernelImpl<kThreads, step, true, InputT, CpMajorPackedTopKAccessor<InputT>>(
        nullptr, nullptr, outIndices, candBufIn, candBufOut, histograms, state, numCandidates, 1, topK, accessor,
        numCandidates);
}
