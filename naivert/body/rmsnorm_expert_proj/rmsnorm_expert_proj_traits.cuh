#pragma once

#include <cuda_bf16.h>
#include "naivert/common/expert_activation.hpp"

#include "naivert/common/executor_traits.hpp"
#include "naivert/common/model_geometry.cuh"
#include "naivert/common/role_layout.hpp"

namespace naivert_replica_rmsnorm_expert_proj {

using naivert::ExecutorTraits;
using naivert::StandardRoleLayout;

constexpr int kRmsHiddenDim = naivert::geometry::kHidden;
constexpr int kRmsExperts = naivert::geometry::kRoutedExperts;
constexpr int kRmsSlotDepth = 4;
constexpr int kRmsThreads = StandardRoleLayout::kThreadCount;
constexpr int kRmsConsumerThreads = StandardRoleLayout::kConsumerActiveThreads;
constexpr int kRmsHelperThreads = StandardRoleLayout::kHelperThreads;
constexpr int kRmsPersistentCtas = naivert::geometry::kSmCount;
constexpr int kRmsExpertsPerTask = 2;
constexpr int kRmsTaskCtas = kRmsExperts / kRmsExpertsPerTask;
constexpr int kRmsConsumerGroupThreads = kRmsConsumerThreads / 2;
constexpr int kRmsConsumerGroupWarps = kRmsConsumerGroupThreads / 32;

constexpr int kRmsBarrierGroup0Offset = 0x3d00;
constexpr int kRmsBarrierGroup1Offset = 0x3d80;
constexpr int kRmsDescriptorWindowBytes = 0x3800;
constexpr int kRmsScratchBase = 0x20000;
constexpr int kRmsScratchStride = 0x4000;
constexpr int kRmsSharedBytes = 0x30000;
constexpr int kRmsRouterPaneBytes =
    kRmsHiddenDim * int(sizeof(naivert::RouterWeight));

constexpr int kRmsHiddenPaneOffset = 0x4000;
constexpr int kRmsXNormPaneOffset = 0x8000;
constexpr int kRmsStateOffset = 0xc000;
constexpr int kRmsScoreTaskOffset = kRmsStateOffset + 0x10;
constexpr int kRmsScoreValueOffset = kRmsStateOffset + 0x20;
constexpr int kRmsRmsPartialOffset = kRmsStateOffset + 0x40;
constexpr int kRmsRouterPartialOffset = kRmsStateOffset + 0x80;
constexpr int kRmsConsumerWarps = kRmsConsumerThreads / 32;
constexpr int kRmsPartialBytes = kRmsConsumerWarps * int(sizeof(float));
constexpr int kRmsHiddenPaneBytes =
    kRmsHiddenDim * int(sizeof(__nv_bfloat16));
constexpr int kRmsMaxTasksPerCta =
    (kRmsTaskCtas + kRmsPersistentCtas - 1) / kRmsPersistentCtas;

// Shared row staging strides for the multi-row router.
constexpr int kRmsRowsXNormOffset = 0;
constexpr int kRmsRowsXNormStride = 0x3800;
constexpr int kRmsRowsRouterStride = 0x4000;

static_assert(kRmsThreads == 384,
              "RMSNormExpertProj uses the SASS-observed 384-thread CTA");
static_assert(kRmsConsumerThreads == 256,
              "RMSNormExpertProj follows the 256-consumer role layout");
static_assert(kRmsHelperThreads == 128,
              "RMSNormExpertProj follows the 128-helper role layout");
static_assert(kRmsRouterPaneBytes <= kRmsScratchStride,
              "router row must fit in one helper copy-ring pane");
static_assert(kRmsHiddenPaneOffset + kRmsHiddenPaneBytes <= kRmsXNormPaneOffset,
              "hidden staging pane must not overlap the x_norm pane");
static_assert(kRmsXNormPaneOffset + kRmsHiddenPaneBytes <= kRmsStateOffset,
              "x_norm staging pane must not overlap the RMS cache header");
static_assert(kRmsScoreValueOffset +
                      kRmsMaxTasksPerCta * kRmsExpertsPerTask *
                          int(sizeof(float)) <=
                  kRmsRmsPartialOffset,
              "score epilogue scratch must not overlap RMS partials");
static_assert(kRmsRmsPartialOffset + kRmsPartialBytes <=
                  kRmsRouterPartialOffset,
              "RMS partials must not overlap router partials");
static_assert(kRmsRouterPartialOffset +
                      kRmsSlotDepth * kRmsPartialBytes <=
                  kRmsScratchBase,
              "slot-isolated router partials must stay below copy panes");
static_assert(kRmsStateOffset < kRmsScratchBase,
              "native staging panes must stay below the helper router panes");
static_assert(kRmsBarrierGroup1Offset + kRmsSlotDepth * 8 <= kRmsHiddenPaneOffset,
              "Region A/B mbarriers must not overlap the hidden staging pane");
static_assert(kRmsHiddenDim % 8 == 0,
              "hidden dim must be a multiple of 8 for 128-bit access");
static_assert(kRmsExperts % kRmsExpertsPerTask == 0,
              "paired task mapping requires an even expert count");
static_assert(kRmsMaxTasksPerCta == 1,
              "fixed decode shape requires one paired task per active CTA");

static_assert(kRmsRowsXNormStride >= kRmsHiddenPaneBytes, "normalized row must fit its pane");
static_assert(kRmsRowsRouterStride >= kRmsRouterPaneBytes, "router row must fit its pane");

using RMSNormExpertProjTraits =
    ExecutorTraits<kRmsSlotDepth>;

}  // namespace naivert_replica_rmsnorm_expert_proj
