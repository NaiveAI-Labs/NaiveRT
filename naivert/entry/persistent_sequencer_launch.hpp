#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "naivert/entry/speculative_decode_abi.hpp"

namespace naivert_runtime_persistent_sequencer {

// Replay-local resource sets.  One epoch owns one slot (slot = epoch %
// kReplaySlotCount); the count is how many epochs may be in flight before
// an earlier epoch's completion has to be visible (see the generation
// handshake in persistent_sequencer_entry.cu).
inline constexpr std::size_t kReplaySlotCount = 2;
inline constexpr std::size_t kMaxPipelineStageCount = 4;

// One fixed block is owned by every replay-slot/rank row.  Slot zero also
// carries the rank-local request cursor used by the device tail-launch
// sequencer while the public pointer contract remains TP8 x 3.
struct alignas(64) SequencerState {
  std::uint64_t request_id;
  std::uint64_t epoch;
  std::uint64_t completed_epoch_plus_one;
  std::uint64_t launch_count;
  std::uint32_t replay_slot;
  std::uint32_t slot_generation;
  std::uint32_t active_slot;
  std::uint32_t active_generation;
  std::uint32_t stop;
  std::uint32_t error;
  std::uint32_t started;
  std::uint32_t reserved;
  std::uint32_t cur_pos;
  std::uint32_t sampling_position;
  std::uint64_t sampling_seed;
  std::uint32_t rope_position;
  std::uint32_t token_id;
  std::uint32_t source_slot;
  std::uint32_t source_generation;
  std::uint32_t output_generation;
  // Python publishes this request bound once. The completion kernel derives
  // subsequent epochs after the preceding parent completes.
  std::uint32_t max_epochs;
  std::uint32_t initial_cur_pos;
  std::uint32_t initial_sampling_position;
  std::uint32_t initial_rope_position;
  std::uint32_t output_ring_capacity;
};

static_assert(sizeof(SequencerState) == 128,
              "persistent sequencer state ABI changed");

struct alignas(8) SequencerOutputRecord {
  std::uint32_t token;
  std::uint32_t generation;
  // GPU-global nanosecond timestamps.  parent_start_ns is written by the
  // existing runtime prelude immediately before its device-control work;
  // parent_complete_ns is written by the dependent sequencer tail before
  // generation is release-published to the host.
  std::uint64_t parent_start_ns;
  std::uint64_t parent_complete_ns;
  // Written by the existing runtime prelude.  It isolates the TP8/C++ launch
  // boundary without adding a production-only timing node to the graph.
  std::uint64_t launch_marker_ns;
  // Device-published post-commit progression.  These fields make the
  // autonomous request state auditable without a per-epoch D2H copy.
  std::uint32_t cur_pos;
  std::uint32_t sampling_position;
  std::uint32_t rope_position;
  std::uint32_t output_count;
};

static_assert(sizeof(SequencerOutputRecord) == 48,
              "persistent sequencer output record ABI changed");

// Debug-only progress codes written into the token word of the *next* mapped
// host output record. They are overwritten by the actual sampled token before
// generation is release-published, so they never change the production output
// ABI or the normal mapped-reader result.
enum SequencerDebugStage : std::uint32_t {
  kSequencerDebugStageNone = 0,
  kSequencerDebugStageNextMetadataReady = 0xD0010001u,
  kSequencerDebugStageTailLaunchAttempted = 0xD0010002u,
  kSequencerDebugStageTailLaunchAccepted = 0xD0010003u,
  kSequencerDebugStagePreludeEntered = 0xD0010004u,
  kSequencerDebugStagePreludeWorkComplete = 0xD0010005u,
  kSequencerDebugStagePreludeTp8RendezvousComplete = 0xD0010006u,
  kSequencerDebugStageBodyComplete = 0xD0010007u,
  kSequencerDebugStageCompletionEntered = 0xD0010008u,
  kSequencerDebugStagePipeline0Complete = 0xD0010010u,
  kSequencerDebugStagePipeline1Complete = 0xD0010011u,
  kSequencerDebugStagePipeline2Complete = 0xD0010012u,
  kSequencerDebugStagePipeline3Complete = 0xD0010013u,
};

// Every replay-slot/rank pair owns one peer-visible rendezvous row.  Source
// body graphs capture distinct allreduce-mailbox banks per replay slot, so
// their gate rows must be distinct as well.
struct alignas(8) SequencerTp8Barrier {
  std::uint64_t body_arrivals[8];
  std::uint64_t body_phase;
  std::uint64_t stop;
  std::uint64_t abort;
};

static_assert(sizeof(SequencerTp8Barrier) == 88,
              "persistent sequencer TP8 barrier ABI changed");

struct SequencerSlotPointers {
  SequencerState* state;
  std::uint64_t* wakeup;
  std::uint64_t* completion;
  SequencerOutputRecord* output_ring;
  SequencerOutputRecord* host_output_ring;
  const std::uint32_t* token_out;
  // Null for the unchanged Seq1 fast path. A four-stage MTP plan binds one
  // slot-local state row that its commit stage updates before this common
  // completion kernel derives the next epoch.
  naivert_runtime_speculative::DecodeControlState* decode_control;
};

struct PersistentSequencerOwner;

std::size_t persistent_sequencer_state_bytes();

PersistentSequencerOwner* create_persistent_sequencer(
    int device,
    cudaStream_t sequencer_stream,
    const std::array<
        std::array<cudaGraph_t, kMaxPipelineStageCount>,
        kReplaySlotCount>&
        pipeline_graphs,
    std::uint32_t pipeline_stage_count,
    std::uint32_t max_epochs,
    const std::array<
        std::array<SequencerTp8Barrier*, 8>,
        kReplaySlotCount>&
        tp8_barriers,
    const std::array<SequencerSlotPointers, kReplaySlotCount>& slots);

void start_persistent_sequencer(PersistentSequencerOwner* owner);

// Python publishes one initial token. The completion kernel derives every
// later epoch and the GPU prelude consumes the preceding TOKEN_OUT.
void arm_persistent_sequencer_initial(
    PersistentSequencerOwner* owner,
    std::uint64_t request_id,
    std::uint32_t cur_pos,
    std::uint32_t sampling_position,
    std::uint64_t sampling_seed,
    std::uint32_t rope_position,
    std::uint32_t token_id,
    std::uint32_t max_epochs,
    cudaStream_t arm_stream);

// Fail-closed escape hatch for a partially armed TP8 request.  This launch
// must use a stream independent of the sequencer stream so an already-running
// prelude cannot prevent abort publication.
void abort_persistent_sequencer(
    PersistentSequencerOwner* owner,
    cudaStream_t abort_stream);

// Request close is the only synchronizing operation.
void stop_persistent_sequencer(
    PersistentSequencerOwner* owner,
    cudaStream_t arm_stream);

std::uint64_t completed_flat_graph_launches(
    PersistentSequencerOwner* owner);

void destroy_persistent_sequencer(PersistentSequencerOwner* owner);

}  // namespace naivert_runtime_persistent_sequencer
