#pragma once
#include "naivert/body/dspark_tail/commit_args.hpp"
#include <cuda/atomic>
#include <cuda_runtime.h>
namespace naivert_seq1_commit {
using naivert_runtime_persistent_sequencer::SequencerOutputRecord;
using naivert_runtime_speculative::DecodeControlState;
using naivert_runtime_speculative::DecodeMode;
using naivert_runtime_speculative::kDecodeControlBridgeError;
using naivert_runtime_speculative::kDecodeControlEos;
using naivert_runtime_speculative::kDecodeControlOutputLimit;
using naivert_runtime_speculative::kDecodeControlStop;
template <typename T>
__device__ __forceinline__ void system_store(T* pointer, T desired) {
  cuda::atomic_ref<T, cuda::thread_scope_system> value(*pointer);
  value.store(desired, cuda::memory_order_release);
}
template <typename T>
__device__ __forceinline__ T system_load(const T* pointer) {
  cuda::atomic_ref<T, cuda::thread_scope_system> value(
      *const_cast<T*>(pointer));
  return value.load(cuda::memory_order_acquire);
}
template <typename T>
__device__ __forceinline__ T system_load_relaxed(const T* pointer) {
  cuda::atomic_ref<T, cuda::thread_scope_system> value(
      *const_cast<T*>(pointer));
  return value.load(cuda::memory_order_relaxed);
}
// The two cancel words arrive preloaded (relaxed loads issued before the tail's rendezvous,
// ordered by its system fence), so the commit no longer pays one or two PCIe round trips.
__device__ __forceinline__ std::uint32_t cancel_at_generation(
    const NaiveN0p5Seq1CommitArgs& args, std::uint32_t generation,
    std::uint32_t rows_per_commit, std::uint32_t cancel_at_word,
    std::uint32_t cancel_requested_word) {
  if (args.cancel == nullptr) {
    return 0u;
  }
  std::uint32_t at = cancel_at_word;
  if (at == 0u && args.publish != 0u && cancel_requested_word != 0u) {
    at = generation + kSeq1CancelMargin * rows_per_commit;
    system_store(&args.cancel[kSeq1CancelAtGenerationWord], at);
  }
  return at;
}
__device__ __forceinline__ std::uint64_t global_timer_ns() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
__device__ __forceinline__ void fail_closed(
    DecodeControlState* control, std::uint32_t flags) {
  control->flags = flags | kDecodeControlStop | kDecodeControlBridgeError;
}
__device__ __forceinline__ std::uint32_t token_at(
    const NaiveN0p5Seq1CommitArgs& args,
    std::uint32_t row) {
  if (row == 0u) {
    return static_cast<std::uint32_t>(args.token_out[0]);
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(args.token_out);
  return *reinterpret_cast<const std::uint32_t*>(
      bytes + static_cast<std::uint64_t>(row) * args.token_row_stride);
}
__device__ __forceinline__ bool commit_body(const NaiveN0p5Seq1CommitArgs& args,
                                            std::uint32_t cancel_at_word,
                                            std::uint32_t cancel_requested_word) {
  const DecodeControlState control = *args.control;
  if (control.active_rows == 0u) {
    return false;
  }
  if ((control.flags & kDecodeControlStop) != 0u) {
    fail_closed(args.control, control.flags);
    return false;
  }
  const std::uint32_t index = control.output_count;
  const bool mtp =
      control.decode_mode ==
      static_cast<std::uint32_t>(DecodeMode::kMtp3);
  const std::uint32_t proposed =
      mtp
      ? (args.accepted_count == nullptr ? 0u : *args.accepted_count)
      : 1u;
  const bool valid =
      (mtp
           ? (args.accepted_count != nullptr &&
              control.active_rows ==
                  control.speculative_depth + 1u &&
              control.active_rows <= control.row_capacity &&
              args.token_row_stride >= sizeof(std::uint32_t) &&
              args.token_out_rows >= control.active_rows &&
              proposed >= 1u &&
              proposed <= control.active_rows)
           : (control.decode_mode ==
                  static_cast<std::uint32_t>(DecodeMode::kSeq1) &&
              control.active_rows == 1u)) &&
      control.max_output_tokens != 0u &&
      args.output_ring_capacity != 0u &&
      index < control.max_output_tokens &&
      index < args.output_ring_capacity &&
      (args.publish == 0u ||
       (args.host_output_ring != nullptr &&
        args.host_flags_ring != nullptr));
  if (!valid) {
    fail_closed(args.control, control.flags);
    return false;
  }
  const std::uint32_t eos_token_id = control.reserved0;
  const std::uint32_t max_position = control.reserved1;
  std::uint32_t committed = proposed;
  const std::uint32_t remaining_out =
      control.max_output_tokens - index;
  if (committed > remaining_out) {
    committed = remaining_out;
  }
  const std::uint32_t remaining_ring =
      args.output_ring_capacity - index;
  if (committed > remaining_ring) {
    committed = remaining_ring;
  }
  if (max_position != 0u) {
    const std::uint32_t room =
        max_position > control.committed_pos
        ? max_position - control.committed_pos
        : 1u;
    if (committed > room) {
      committed = room;
    }
  }
  std::uint32_t flags = 0u;
  std::uint32_t bound = kSeq1BoundNone;
  bool stopped_on_eos = false;
  if (eos_token_id != kEosDisabled) {
    for (std::uint32_t i = 0u; i < committed; ++i) {
      if (token_at(args, i) == eos_token_id) {
        committed = i + 1u;
        stopped_on_eos = true;
        break;
      }
    }
  }
  const std::uint32_t cancel_at =
      cancel_at_generation(
          args, index + 1u, mtp ? control.active_rows : 1u, cancel_at_word,
          cancel_requested_word);
  const bool canceled =
      cancel_at != 0u && index + committed >= cancel_at;
  if (canceled) {
    const std::uint32_t room =
        cancel_at > index ? cancel_at - index : 1u;
    if (committed > room) {
      committed = room;
    }
  }
  if (stopped_on_eos) {
    flags |= kDecodeControlStop | kDecodeControlEos;
  }
  if (index + committed >= control.max_output_tokens) {
    flags |= kDecodeControlStop | kDecodeControlOutputLimit;
    bound = kSeq1BoundOutputLimit;
  }
  if (max_position != 0u &&
      control.committed_pos + committed >= max_position) {
    flags |= kDecodeControlStop | kDecodeControlOutputLimit;
    bound = kSeq1BoundPositionLimit;
  }
  if (canceled) {
    flags |= kDecodeControlStop | kSeq1Canceled;
    bound = kSeq1BoundCanceled;
  }
  if (args.rep_count != nullptr) {
    for (std::uint32_t i = 0u; i < committed; ++i) {
      ++args.rep_count[token_at(args, i)];
    }
  }
  const std::uint32_t next_position =
      control.committed_pos + committed;
  const std::uint64_t now = global_timer_ns();
  const std::uint64_t previous =
      args.previous_complete_ns == nullptr ? 0ull
                                           : *args.previous_complete_ns;
  if (args.publish != 0u) {
    for (std::uint32_t i = 0u; i < committed; ++i) {
      const std::uint32_t slot = index + i;
      SequencerOutputRecord* record = &args.host_output_ring[slot];
      record->token = token_at(args, i);
      record->parent_start_ns = previous;
      record->parent_complete_ns = now;
      record->launch_marker_ns = previous;
      const std::uint32_t position = control.committed_pos + i + 1u;
      record->cur_pos = position;
      record->sampling_position = position;
      record->rope_position = position;
      record->output_count = slot + 1u;
      const std::uint32_t published_flags =
          (flags & ((1u << kSeq1BoundShift) - 1u)) | (bound << kSeq1BoundShift);
      const bool last = (i + 1u == committed);
      args.host_flags_ring[slot] = last ? published_flags : 0u;
    }
  }
  if (args.previous_complete_ns != nullptr) {
    *args.previous_complete_ns = now;
  }
  DecodeControlState next = control;
  next.iteration = control.iteration + 1ull;
  next.committed_pos = next_position;
  next.speculative_base = next_position;
  next.output_count = index + committed;
  next.accepted_count = 1u;
  next.main_cache_committed = next_position;
  next.main_cache_speculative_end =
      mtp ? next_position + control.active_rows : next_position;
  if (mtp) {
    next.mtp_cache_committed = next_position;
    next.mtp_cache_speculative_end = next_position + control.active_rows;
  }
  next.flags = flags;
  *args.control = next;
  if (args.publish != 0u) {
    // One system fence publishes every record and flag written above; the per-slot
    // generation words are then relaxed stores. The host reader waits on one slot's
    // generation at a time (production_request.py), so it needs each generation to be
    // ordered after its record, never after the other slots' generations. The former
    // per-slot release store paid a PCIe drain per committed token.
    __threadfence_system();
    for (std::uint32_t i = 0u; i < committed; ++i) {
      const std::uint32_t slot = index + i;
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> generation(
          args.host_output_ring[slot].generation);
      generation.store(slot + 1u, cuda::memory_order_relaxed);
    }
  }
  return (flags & kDecodeControlStop) == 0u;
}
}
