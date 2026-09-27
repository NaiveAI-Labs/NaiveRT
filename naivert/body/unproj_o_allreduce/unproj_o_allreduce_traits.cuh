#pragma once

#include "naivert/common/executor_traits.hpp"
#include "naivert/common/model_geometry.cuh"
#include "naivert/common/tp8_mailbox.cuh"
#include "naivert/common/role_layout.hpp"

namespace naivert_replica_unproj_o_allreduce {

using naivert::ExecutorTraits;
using naivert::StandardRoleLayout;

#ifndef NAIVERT_UNPROJ_O_ALLREDUCE_OUTPUTS_PER_TILE
#error "NAIVERT_UNPROJ_O_ALLREDUCE_OUTPUTS_PER_TILE must come from the model's kernel geometry"
#endif
#ifndef NAIVERT_UNPROJ_O_ALLREDUCE_PROJECTION_K
#error "NAIVERT_UNPROJ_O_ALLREDUCE_PROJECTION_K must come from the model's kernel geometry"
#endif
constexpr int kHidden = naivert::geometry::kHidden;
constexpr int kProjectionK = NAIVERT_UNPROJ_O_ALLREDUCE_PROJECTION_K;
constexpr int kInputSlots = 4;
constexpr int kInter = kProjectionK / kInputSlots;
// Each slot's K is split across the 8 consumer warps in 32-wide MMA steps; the
// weight pane holds 32-wide K chunks per 16-row block; one scale per 128-block.
constexpr int kInterChunks = kInter / 32;
constexpr int kWarpK = kInter / 8;
constexpr int kScaleColumns = kProjectionK / 128;
constexpr int kBlocksPerSlot = kInter / 128;
static_assert(kWarpK % 32 == 0 && kInter % 128 == 0,
              "unproj needs whole 32-wide MMA steps per warp and whole 128-wide scale blocks per slot");
constexpr int kOutputsPerTile = NAIVERT_UNPROJ_O_ALLREDUCE_OUTPUTS_PER_TILE;
constexpr int kTiles = kHidden / kOutputsPerTile;
constexpr int kSlotDepth = 5;
constexpr int kBlockThreads = StandardRoleLayout::kThreadCount;
constexpr int kConsumerThreads = StandardRoleLayout::kConsumerActiveThreads;
constexpr int kHelperThreads = StandardRoleLayout::kHelperThreads;

constexpr int kInputOffset = 0x0000;
constexpr int kPartialOffset = 0x1000;
constexpr int kStagedPayloadOffset = 0x1800;
constexpr int kStagedResidualOffset = 0x1880;
constexpr int kBarrierGroup0Offset = 0x3c00;
constexpr int kBarrierGroup1Offset = 0x3c80;
constexpr int kPayloadBase = 0x4000;
constexpr int kWeightPaneBytes = kOutputsPerTile * kInter;
constexpr int kPayloadStride = kWeightPaneBytes;
constexpr int kSharedBytes = kPayloadBase + kSlotDepth * kPayloadStride;

static_assert(kBlockThreads == 384);
static_assert(kConsumerThreads == 256);
static_assert(kHelperThreads == 128);
static_assert(kInputSlots * kInter == kProjectionK);
static_assert(kHidden % kOutputsPerTile == 0, "tile width must divide hidden");
static_assert(std::size_t(kTiles) * naivert::kMailboxWarpStride <= naivert::kMailboxPhaseStride,
              "unproj tiles must fit one mailbox phase lane");
static_assert(kInputOffset +
                      kProjectionK * int(sizeof(__nv_bfloat16)) <=
                  kPartialOffset);
static_assert(kPartialOffset +
                      8 * kOutputsPerTile * int(sizeof(float)) <=
                  kStagedPayloadOffset);
static_assert(kStagedPayloadOffset +
                      (kOutputsPerTile / 2) *
                          int(sizeof(std::uint32_t)) <=
                  kStagedResidualOffset);
static_assert(kStagedResidualOffset +
                      kOutputsPerTile * int(sizeof(__nv_bfloat16)) <=
                  kBarrierGroup0Offset);

using UnprojOAllreduceTraits =
    ExecutorTraits<kSlotDepth>;

}  // namespace naivert_replica_unproj_o_allreduce
