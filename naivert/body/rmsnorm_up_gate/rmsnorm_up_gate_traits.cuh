#pragma once

#include <cuda_bf16.h>

#include "naivert/common/executor_traits.hpp"
#include "naivert/common/model_geometry.cuh"
#include "naivert/common/role_layout.hpp"

#ifndef NAIVERT_RMSNORM_UP_GATE_OUTPUTS_PER_TILE
#define NAIVERT_RMSNORM_UP_GATE_OUTPUTS_PER_TILE 16
#endif

namespace naivert_replica_rmsnorm_up_gate {

using naivert::ExecutorTraits;
using naivert::StandardRoleLayout;

constexpr int kHidden = naivert::geometry::kHidden;
constexpr int kInter = naivert::geometry::kDenseInter;
constexpr int kOutputsPerTile = NAIVERT_RMSNORM_UP_GATE_OUTPUTS_PER_TILE;
constexpr int kPageK = 1024;
constexpr int kPages = kHidden / kPageK;
constexpr int kSlotDepth = 4;
constexpr int kBlockThreads = StandardRoleLayout::kThreadCount;
constexpr int kConsumerThreads = StandardRoleLayout::kConsumerActiveThreads;
constexpr int kHelperThreads = StandardRoleLayout::kHelperThreads;

// Each ring pane carries one 16x1024 gate tile followed by the matching
// 16x1024 up tile.  Keeping both branches in one 32-KiB transaction preserves
// the native producer/consumer overlap.  The New ABI keeps the original
// weight pointers as capture-time packing sources; timed replay transports
// weights only through the fixed packed-workspace address.
constexpr int kMetadataOffset = 0x400;
constexpr int kBarrierGroup0Offset = 0x3900;
constexpr int kBarrierGroup1Offset = 0x3980;
constexpr int kPaneBaseOffset = 0x4000;
constexpr int kPaneStride = 0x9000;
constexpr int kBranchBytes = kOutputsPerTile * kPageK;
constexpr int kScaleValuesPerPage = kPageK / 128;
constexpr int kScaleBranchBytes =
    kScaleValuesPerPage * int(sizeof(__nv_bfloat16));
constexpr int kScaleSidebandOffset = 2 * kBranchBytes;
constexpr int kPaneVisibleBytes =
    kScaleSidebandOffset + 2 * kScaleBranchBytes;
constexpr int kScratchOffset = kPaneBaseOffset + kSlotDepth * kPaneStride;
constexpr int kScratchBytes = 0xD400;
constexpr int kStoreMetaOffset = kScratchOffset + kScratchBytes;
constexpr int kStoreMetaBytes = 0x400;
constexpr int kSharedBytes = kStoreMetaOffset + kStoreMetaBytes;

constexpr int kRowActivationBytes = kHidden * int(sizeof(__half));

static_assert(kOutputsPerTile > 0,
              "rmsnorm_up_gate tile width must be positive");
static_assert(kBlockThreads == 384,
              "rmsnorm_up_gate uses the SASS-observed 384-thread CTA");
static_assert(kConsumerThreads == 256,
              "rmsnorm_up_gate follows the 256-consumer role layout");
static_assert(kHelperThreads == 128,
              "rmsnorm_up_gate follows the 128-helper role layout");
static_assert(kHidden % kPageK == 0,
              "rmsnorm_up_gate stages hidden in 1024-wide pages");
static_assert(kInter % kOutputsPerTile == 0,
              "rmsnorm_up_gate output tiles must divide TP-local inter");
static_assert(kPaneVisibleBytes == 0x8020,
              "gate/up payload and native scale sideband must fit");
static_assert(kScratchOffset + kScratchBytes <= kSharedBytes,
              "rmsnorm_up_gate shared layout must fit");

using RMSNormUpGateTraits =
    ExecutorTraits<kSlotDepth>;

// rows8 (verify-8): eight fp16 activation rows (64 KiB) after a tighter pane ring.  The
// pane contents keep the rows1 fragment layout; only the stride between the four ring
// panes shrinks from 0x9000 to 0x8100 (0x8020 visible, 256-byte aligned) so the activations,
// the sum state and the per-warp accumulators fit below the 227 KiB opt-in limit.
constexpr int kRows8Rows = 8;
constexpr int kRows8PaneStride = 0x8100;
constexpr int kRows8ScratchOffset = kPaneBaseOffset + kSlotDepth * kRows8PaneStride;
constexpr int kRows8ActivationBytes = kRows8Rows * kRowActivationBytes;
constexpr int kRows8SumStateOffset = kRows8ScratchOffset + kRows8ActivationBytes;
constexpr int kRows8SumStateBytes = 0x200;
constexpr int kRows8OutputAccumOffset = kRows8SumStateOffset + kRows8SumStateBytes;
constexpr int kRows8OutputAccumBytes =
    kRows8Rows * (256 / 32) * 2 * kOutputsPerTile * int(sizeof(float));
constexpr int kRows8SharedBytes =
    kRows8OutputAccumOffset + kRows8OutputAccumBytes + 0x400;

static_assert(kRows8PaneStride >= kPaneVisibleBytes && kRows8PaneStride % 256 == 0,
              "rows8 pane stride must hold the visible pane, 256-byte aligned");
static_assert(kRows8SumStateBytes >= (kRows8Rows * (256 / 32) + kRows8Rows) * int(sizeof(float)),
              "rows8 sum state holds the per-warp partials and the inverse RMS per row");
// The eight-row layout fits the SM100 CTA limit for hidden 4096 (Naive N); a hidden-7168 build
// compiles the variant but its CTAs exit at dispatch and the launch keeps the
// rows1 shared size.
constexpr bool kRows8Fits = kRows8SharedBytes <= 0x37000;
constexpr int kRows8LaunchSharedBytes = kRows8Fits ? kRows8SharedBytes : kSharedBytes;

using RMSNormUpGateRows8Traits =
    ExecutorTraits<kSlotDepth>;

}  // namespace naivert_replica_rmsnorm_up_gate
