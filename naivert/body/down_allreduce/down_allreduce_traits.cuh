#pragma once

#include <cuda_bf16.h>

#include "naivert/common/executor_traits.hpp"
#include "naivert/common/model_geometry.cuh"
#include "naivert/common/role_layout.hpp"

#ifndef NAIVERT_DOWN_ALLREDUCE_INTER
#define NAIVERT_DOWN_ALLREDUCE_INTER 256
#endif

#ifndef NAIVERT_DOWN_ALLREDUCE_OUTPUTS_PER_TILE
#error "NAIVERT_DOWN_ALLREDUCE_OUTPUTS_PER_TILE must come from the model's kernel geometry"
#endif

namespace naivert_replica_down_allreduce {

using naivert::ExecutorTraits;
using naivert::StandardRoleLayout;

constexpr int kHidden = naivert::geometry::kHidden;
constexpr int kInter = NAIVERT_DOWN_ALLREDUCE_INTER;
constexpr int kInputSlots = naivert::geometry::kDenseInter / kInter;
constexpr int kLocalInter = kInputSlots * kInter;
constexpr int kOutputsPerTile = NAIVERT_DOWN_ALLREDUCE_OUTPUTS_PER_TILE;
constexpr int kTiles = kHidden / kOutputsPerTile;
// pack_down's tile: kFullMTiles 16-row MMA tiles plus an optional 8-row tail
// (hidden 7168: 56 = 3 x 16 + 8, Naive N 64 = 4 x 16).
constexpr int kFullMTiles = kOutputsPerTile / 16;
constexpr int kTailRows = kOutputsPerTile - kFullMTiles * 16;
static_assert(kFullMTiles >= 1 && kFullMTiles <= 4 && (kTailRows == 0 || kTailRows == 8),
              "a down tile is up to four 16-row MMA tiles plus at most one 8-row tail");
constexpr int kSlotDepth = 5;
constexpr int kBlockThreads = StandardRoleLayout::kThreadCount;
constexpr int kConsumerThreads = StandardRoleLayout::kConsumerActiveThreads;
constexpr int kHelperThreads = StandardRoleLayout::kHelperThreads;

constexpr int kPartialRows = 64;
constexpr int kPartialOffset = 0x3000;
constexpr int kStagedPayloadOffset = 0x3800;
constexpr int kStagedResidualOffset = kStagedPayloadOffset + 0x200;
constexpr int kBarrierGroup0Offset = 0x3c00;
constexpr int kBarrierGroup1Offset = 0x3c80;
constexpr int kPayloadBase = 0x4000;
constexpr int kWeightPaneBytes = kOutputsPerTile * kInter;
constexpr int kScalePaneOffset = kWeightPaneBytes;
constexpr int kScalePaneBytes = 32;
// Native stages a weight pane followed by its scale payload.  Keep every
// ring slot 128-byte aligned so UBLKCP destinations remain naturally aligned.
constexpr int kPayloadStride =
    ((kWeightPaneBytes + kScalePaneBytes + 127) / 128) * 128;
constexpr int kSharedBytes =
    kPayloadBase + kSlotDepth * kPayloadStride;

constexpr int kRowEpilogueStride = 0x80;

static_assert(kOutputsPerTile > 0,
              "down_allreduce tile width must be positive");
static_assert(kBlockThreads == 384,
              "down_allreduce uses the SASS-observed 384-thread CTA");
static_assert(kConsumerThreads == 256,
              "down_allreduce follows the 256-consumer role layout");
static_assert(kHelperThreads == 128,
              "down_allreduce follows the 128-helper role layout");
static_assert(kLocalInter == naivert::geometry::kDenseInter,
              "the dense down consumes the per-rank intermediate as whole native K=256 panes");
static_assert(kHidden % kOutputsPerTile == 0,
              "tile width must divide hidden");
static_assert(kInputSlots * kInter * int(sizeof(__nv_bfloat16)) <=
                  kPartialOffset,
              "shared up-gate staging overlaps partial scratch");
static_assert(kPartialOffset +
                      8 * kPartialRows * int(sizeof(float)) <=
                  kStagedPayloadOffset,
              "down-allreduce partial scratch overlaps barriers");
static_assert(kStagedPayloadOffset +
                      (kOutputsPerTile / 2) *
                          int(sizeof(std::uint32_t)) <=
                  kStagedResidualOffset,
              "down-allreduce tagged payload overlaps staged residual");
static_assert(kStagedResidualOffset +
                      kOutputsPerTile * int(sizeof(__nv_bfloat16)) <=
                  kBarrierGroup0Offset,
              "down-allreduce staged residual overlaps barriers");

using DownAllreduceTraits =
    ExecutorTraits<kSlotDepth>;

// rows8 (verify-8): eight staged K vectors (32 KiB), eight rows of warp partials, eight
// row epilogue strips; the five-pane weight ring follows, loaded once for all eight rows.
constexpr int kRows8Rows = 8;
constexpr int kRows8UpGateBytes =
    kRows8Rows * kLocalInter * int(sizeof(__nv_bfloat16));
// 0x8000 for Naive N (2048 per rank); a 2304-per-rank build rounds up to 0x9000.
constexpr int kRows8PartialOffset = ((kRows8UpGateBytes + 0x3ff) / 0x400) * 0x400;
constexpr int kRows8PartialBytes =
    kRows8Rows * 8 * kPartialRows * int(sizeof(float));
constexpr int kRows8StagedPayloadOffset =
    kRows8PartialOffset + kRows8PartialBytes;
constexpr int kRows8StagedResidualOffset =
    kRows8StagedPayloadOffset + kRows8Rows * kRowEpilogueStride;
constexpr int kRows8BarrierGroup0Offset =
    kRows8StagedResidualOffset + kRows8Rows * kRowEpilogueStride;
constexpr int kRows8BarrierGroup1Offset = kRows8BarrierGroup0Offset + 0x80;
constexpr int kRows8PayloadBase =
    ((kRows8BarrierGroup1Offset + 0x80 + 0x3ff) / 0x400) * 0x400;
constexpr int kRows8SharedBytes =
    kRows8PayloadBase + kSlotDepth * kPayloadStride;

static_assert(kRows8UpGateBytes <= kRows8PartialOffset,
              "eight dense-down rows overlap rows8 partial scratch");
static_assert(kRows8PayloadBase % 128 == 0,
              "dense-down rows8 weight panes must stay 128-byte aligned");
static_assert(kRows8SharedBytes <= 0x37000,
              "dense-down rows8 shared layout must fit the SM100 CTA limit");

using DownAllreduceRows8Traits =
    ExecutorTraits<kSlotDepth>;

}  // namespace naivert_replica_down_allreduce
