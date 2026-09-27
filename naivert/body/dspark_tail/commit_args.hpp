#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include "naivert/entry/persistent_sequencer_launch.hpp"
#include "naivert/entry/speculative_decode_abi.hpp"
namespace naivert_seq1_commit {
// Preserve the native output/cancel protocol and shared 80-byte control ABI.
// This branch calls mode value 1 kMtp3; the draft supplies depth 7 / rows 8.
static_assert(static_cast<unsigned>(naivert_runtime_speculative::DecodeMode::kMtp3) == 1);
inline constexpr std::uint32_t kEosDisabled = 0xffffffffu;
inline constexpr std::uint32_t kSeq1BoundShift = 16u;
inline constexpr std::uint32_t kSeq1BoundNone = 0u;
inline constexpr std::uint32_t kSeq1BoundOutputLimit = 1u;
inline constexpr std::uint32_t kSeq1BoundPositionLimit = 2u;
inline constexpr std::uint32_t kSeq1BoundCanceled = 3u;
inline constexpr std::uint32_t kSeq1ChainLaunchFailed = 0x40000000u;
inline constexpr std::uint32_t kSeq1Canceled = 0x20000000u;
inline constexpr std::uint32_t kSeq1CancelRequestedWord = 0u;
inline constexpr std::uint32_t kSeq1CancelAtGenerationWord = 1u;
inline constexpr std::uint32_t kSeq1CancelWords = 2u;
inline constexpr std::uint32_t kSeq1CancelMargin = 2u;
struct NaiveN0p5Seq1CommitArgs {
  naivert_runtime_speculative::DecodeControlState* control;
  const std::int32_t* token_out;
  std::uint64_t* previous_complete_ns;
  naivert_runtime_persistent_sequencer::SequencerOutputRecord* host_output_ring;
  std::uint32_t* host_flags_ring;
  std::uint32_t output_ring_capacity;
  std::uint32_t publish;
  std::uint32_t chain;
  std::uint32_t* cancel;
  const std::uint32_t* accepted_count;
  std::uint32_t token_row_stride;
  std::uint32_t token_out_rows;
  // Repeat-penalty token counts (full vocabulary); every committed token increments its entry.
  std::int32_t* rep_count;
};
static_assert(sizeof(NaiveN0p5Seq1CommitArgs) == 88);
}  // namespace naivert_seq1_commit
