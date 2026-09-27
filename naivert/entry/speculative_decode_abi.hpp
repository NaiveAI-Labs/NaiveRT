#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace naivert_runtime_speculative {

enum class DecodeMode : std::uint32_t {
  kSeq1 = 0,
  kMtp3 = 1,
};

enum DecodeControlFlag : std::uint32_t {
  kDecodeControlStop = 1u << 0,
  kDecodeControlEos = 1u << 1,
  kDecodeControlOutputLimit = 1u << 2,
  kDecodeControlBridgeError = 1u << 31,
};

// Common state captured by both the seq1 and MTP3 graph plans.  row_stride is
// measured in bytes.  The mode changes values, never this layout.
struct alignas(16) DecodeControlState {
  std::uint32_t decode_mode;
  std::uint32_t speculative_depth;
  std::uint32_t active_rows;
  std::uint32_t row_capacity;
  std::uint64_t row_stride;
  std::uint64_t iteration;
  std::uint32_t committed_pos;
  std::uint32_t speculative_base;
  std::uint32_t output_count;
  std::uint32_t max_output_tokens;
  std::uint32_t accepted_count;
  std::uint32_t main_cache_committed;
  std::uint32_t main_cache_speculative_end;
  std::uint32_t mtp_cache_committed;
  std::uint32_t mtp_cache_speculative_end;
  std::uint32_t flags;
  std::uint32_t reserved0;
  std::uint32_t reserved1;
};

static_assert(std::is_standard_layout_v<DecodeControlState>);
static_assert(std::is_trivially_copyable_v<DecodeControlState>);
static_assert(sizeof(DecodeControlState) == 80);
static_assert(alignof(DecodeControlState) == 16);
static_assert(offsetof(DecodeControlState, decode_mode) == 0);
static_assert(offsetof(DecodeControlState, speculative_depth) == 4);
static_assert(offsetof(DecodeControlState, active_rows) == 8);
static_assert(offsetof(DecodeControlState, row_capacity) == 12);
static_assert(offsetof(DecodeControlState, row_stride) == 16);
static_assert(offsetof(DecodeControlState, iteration) == 24);
static_assert(offsetof(DecodeControlState, committed_pos) == 32);
static_assert(offsetof(DecodeControlState, speculative_base) == 36);
static_assert(offsetof(DecodeControlState, output_count) == 40);
static_assert(offsetof(DecodeControlState, max_output_tokens) == 44);
static_assert(offsetof(DecodeControlState, accepted_count) == 48);
static_assert(offsetof(DecodeControlState, main_cache_committed) == 52);
static_assert(
    offsetof(DecodeControlState, main_cache_speculative_end) == 56);
static_assert(offsetof(DecodeControlState, mtp_cache_committed) == 60);
static_assert(
    offsetof(DecodeControlState, mtp_cache_speculative_end) == 64);
static_assert(offsetof(DecodeControlState, flags) == 68);
static_assert(offsetof(DecodeControlState, reserved0) == 72);
static_assert(offsetof(DecodeControlState, reserved1) == 76);

// uintptr_t keeps the pointer manifest inspectable without dereferencing it.
// The final three addresses are MTP-only and must be zero in seq1 mode.
struct alignas(8) DecodeTensorPointers {
  std::uintptr_t control_state;
  std::uintptr_t input_tokens;
  std::uintptr_t input_hidden;
  std::uintptr_t target_logits;
  std::uintptr_t target_tokens;
  std::uintptr_t target_last_hidden;
  std::uintptr_t accepted_tokens;
  std::uintptr_t accepted_count;
  std::uintptr_t accepted_last_hidden;
  std::uintptr_t output_tokens;
  std::uintptr_t main_cache;
  std::uintptr_t draft_tokens;
  std::uintptr_t draft_hidden;
  std::uintptr_t mtp_cache;
};

static_assert(std::is_standard_layout_v<DecodeTensorPointers>);
static_assert(std::is_trivially_copyable_v<DecodeTensorPointers>);
static_assert(sizeof(std::uintptr_t) == 8);
static_assert(sizeof(DecodeTensorPointers) == 112);
static_assert(alignof(DecodeTensorPointers) == 8);
static_assert(offsetof(DecodeTensorPointers, control_state) == 0);
static_assert(offsetof(DecodeTensorPointers, input_tokens) == 8);
static_assert(offsetof(DecodeTensorPointers, input_hidden) == 16);
static_assert(offsetof(DecodeTensorPointers, target_logits) == 24);
static_assert(offsetof(DecodeTensorPointers, target_tokens) == 32);
static_assert(offsetof(DecodeTensorPointers, target_last_hidden) == 40);
static_assert(offsetof(DecodeTensorPointers, accepted_tokens) == 48);
static_assert(offsetof(DecodeTensorPointers, accepted_count) == 56);
static_assert(offsetof(DecodeTensorPointers, accepted_last_hidden) == 64);
static_assert(offsetof(DecodeTensorPointers, output_tokens) == 72);
static_assert(offsetof(DecodeTensorPointers, main_cache) == 80);
static_assert(offsetof(DecodeTensorPointers, draft_tokens) == 88);
static_assert(offsetof(DecodeTensorPointers, draft_hidden) == 96);
static_assert(offsetof(DecodeTensorPointers, mtp_cache) == 104);

}  // namespace naivert_runtime_speculative
