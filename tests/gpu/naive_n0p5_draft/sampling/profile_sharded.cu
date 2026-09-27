// Kernel-boundary event instrumentation. Timings include event overhead and
// rank-launch skew; use bench_sharded.cu for production latency comparisons.
#define main sharded_benchmark_main
#include "bench_sharded.cu"
#undef main

const char* stage_names[] = {
    "ready", "partial_maximum", "local_maximum", "sync_maximum", "global_maximum",
    "exponentiate", "local_sum", "sync_sum", "global_inverse", "normalize",
    "histogram24", "merge24", "sync24", "select24",
    "histogram16", "merge16", "sync16", "select16",
    "histogram8", "merge8", "sync8", "select8",
    "histogram0", "merge0", "sync0", "select0",
    "normalize_nucleus", "local_cdf", "sync_cdf",
    "choose_token", "sync_token", "broadcast_tokens", "consumed"};
constexpr int stages = sizeof(stage_names) / sizeof(*stage_names);

void instrument(cudaGraph_t graph, cudaEvent_t* events) {
  size_t count = 0;
  CK(cudaGraphGetNodes(graph, nullptr, &count));
  REQUIRE(count == stages);
  cudaGraphNode_t node;
  count = 1;
  CK(cudaGraphGetRootNodes(graph, &node, &count));
  REQUIRE(count == 1);
  cudaGraphNode_t first;
  CK(cudaGraphAddEventRecordNode(&first, graph, nullptr, 0, events[0]));
  CK(cudaGraphAddDependencies(graph, &first, &node, nullptr, 1));
  for (int i = 0; i < stages; ++i) {
    cudaGraphNodeType type;
    CK(cudaGraphNodeGetType(node, &type));
    REQUIRE(type == cudaGraphNodeTypeKernel);
    cudaGraphNode_t next;
    count = 1;
    CK(cudaGraphNodeGetDependentNodes(node, &next, nullptr, &count));
    REQUIRE(count == (i + 1 < stages ? 1 : 0));
    cudaGraphNode_t after;
    CK(cudaGraphAddEventRecordNode(&after, graph, &node, 1, events[i + 1]));
    if (count) {
      CK(cudaGraphRemoveDependencies(graph, &node, &next, nullptr, 1));
      CK(cudaGraphAddDependencies(graph, &after, &next, nullptr, 1));
      node = next;
    }
  }
}

int main(int argc, char** argv) {
  REQUIRE(argc == 2);
  std::string dir = argv[1];
  auto p = read<float>(dir + "/target.bin", 8 * V);
  auto q = read<float>(dir + "/draft.bin", 7 * V);
  auto ids = read<int64_t>(dir + "/proposals.bin", 7);
  Rank* ranks[8];
  cudaEvent_t events[8][stages + 1];
  for (int r = 0; r < 8; ++r) {
    CK(cudaSetDevice(r));
    for (int peer = 0; peer < 8; ++peer) if (peer != r) {
      auto status = cudaDeviceEnablePeerAccess(peer, 0);
      REQUIRE(status == cudaSuccess || status == cudaErrorPeerAccessAlreadyEnabled);
      cudaGetLastError();
    }
    auto& x = *(ranks[r] = new Rank);
    x.a.rank = r;
    std::vector<float> sp(8 * SHARD), sq(7 * SHARD);
    for (int row = 0; row < 8; ++row)
      std::copy_n(p.data() + row * V + r * SHARD, SHARD, sp.data() + row * SHARD);
    for (int row = 0; row < 7; ++row)
      std::copy_n(q.data() + row * V + r * SHARD, SHARD, sq.data() + row * SHARD);
    int64_t state[] = {1234, 0};
    CK(cudaMemcpy(x.input_p, sp.data(), sp.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x.input_q, sq.data(), sq.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x.ids, ids.data(), ids.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x.rng, state, sizeof(state), cudaMemcpyHostToDevice));
    for (auto& event : events[r]) CK(cudaEventCreate(&event));
  }
  for (int r = 0; r < 8; ++r) {
    auto& x = *ranks[r];
    for (int peer = 0; peer < 8; ++peer) {
      x.a.peers[peer] = ranks[peer]->mailbox;
      x.a.flags[peer] = ranks[peer]->flags;
      x.reference.a.logits[peer] = ranks[peer]->input_p;
      x.reference.a.draft_logits[peer] = ranks[peer]->input_q;
      x.reference.a.flags[peer] = ranks[peer]->flags;
    }
    x.reference.a.rank = r;
    x.reference.a.rng = x.rng;
    x.reference.a.proposals = x.ids;
  }
  printf("{\"instrumented\":true,\"repeats\":20,\"cases\":[");
  for (bool anchor : {false, true}) {
    for (int r = 0; r < 8; ++r) { CK(cudaSetDevice(r)); ranks[r]->launch(0, anchor); }
    for (int r = 0; r < 8; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(ranks[r]->stream)); }
    for (int r = 0; r < 8; ++r) {
      CK(cudaSetDevice(r));
      auto& x = *ranks[r];
      CK(cudaStreamBeginCapture(x.stream, cudaStreamCaptureModeGlobal));
      x.launch(2, anchor);
      CK(cudaStreamEndCapture(x.stream, &x.graph[2]));
      instrument(x.graph[2], events[r]);
      CK(cudaGraphInstantiate(&x.exec[2], x.graph[2], 0));
    }
    double sums[8][stages] = {};
    for (int rep = 0; rep < 25; ++rep) {
      for (int r = 0; r < 8; ++r) {
        CK(cudaSetDevice(r));
        CK(cudaGraphLaunch(ranks[r]->exec[2], ranks[r]->stream));
      }
      for (int r = 0; r < 8; ++r) {
        CK(cudaSetDevice(r));
        auto& x = *ranks[r];
        CK(cudaStreamSynchronize(x.stream));
        REQUIRE(download(x.a.watchdog, 1)[0] == 0);
        REQUIRE(download(x.a.tokens, anchor ? 1 : 8) == download(x.reference.a.tokens, anchor ? 1 : 8));
        for (int i = 0; i < stages; ++i) {
          float ms;
          CK(cudaEventElapsedTime(&ms, events[r][i], events[r][i + 1]));
          if (rep >= 5) sums[r][i] += ms * 1000;
        }
      }
    }
    printf("%s{\"anchor\":%s,\"stage_us_by_rank\":{", anchor ? "," : "", anchor ? "true" : "false");
    for (int i = 0; i < stages; ++i) {
      printf("%s\"%s\":[", i ? "," : "", stage_names[i]);
      for (int r = 0; r < 8; ++r) printf("%s%.3f", r ? "," : "", sums[r][i] / 20);
      printf("]");
    }
    printf("}}");
    for (int r = 0; r < 8; ++r) {
      CK(cudaSetDevice(r));
      CK(cudaGraphExecDestroy(ranks[r]->exec[2]));
      CK(cudaGraphDestroy(ranks[r]->graph[2]));
    }
  }
  printf("]}\n");
}
