// Bounded whole-sharded-sampler memory check. Capture first, then launch each
// rank from its own host thread so instrumentation cannot strand peer launches
// behind an eager rank's rendezvous. This harness does not measure performance.
#include <thread>
#define main sharded_benchmark_main
#include "bench_sharded.cu"
#undef main

int main(int argc, char** argv) {
  REQUIRE(argc == 2);
  std::string dir = argv[1];
  auto p = read<float>(dir + "/target.bin", 8 * V);
  auto q = read<float>(dir + "/draft.bin", 7 * V);
  auto ids = read<int64_t>(dir + "/proposals.bin", 7);
  Rank* ranks[8];
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
  fprintf(stderr, "setup complete\n");
  for (bool anchor : {false, true}) {
    for (int r = 0; r < 8; ++r) {
      CK(cudaSetDevice(r));
      auto& x = *ranks[r];
      CK(cudaStreamBeginCapture(x.stream, cudaStreamCaptureModeGlobal));
      x.launch(2, anchor);
      CK(cudaStreamEndCapture(x.stream, &x.graph[2]));
      CK(cudaGraphInstantiate(&x.exec[2], x.graph[2], 0));
    }
    fprintf(stderr, "captured anchor=%d\n", anchor);
    std::thread workers[8];
    for (int r = 0; r < 8; ++r) workers[r] = std::thread([&, r]() {
      CK(cudaSetDevice(r));
      auto& x = *ranks[r];
      for (int repeat = 0; repeat < 2; ++repeat) {
        CK(cudaGraphLaunch(x.exec[2], x.stream));
        CK(cudaStreamSynchronize(x.stream));
      }
    });
    for (auto& worker : workers) worker.join();
    std::vector<int> expected_tokens;
    for (int r = 0; r < 8; ++r) {
      CK(cudaSetDevice(r));
      auto& x = *ranks[r];
      REQUIRE(download(x.a.watchdog, 1)[0] == 0);
      auto tokens = download(x.a.tokens, anchor ? 1 : 8);
      if (expected_tokens.empty()) expected_tokens = tokens;
      REQUIRE(tokens == expected_tokens);
      auto actual_p = download(x.a.p, (anchor ? 1 : 8) * SHARD);
      auto actual_q = download(x.a.q, anchor ? 0 : 7 * SHARD);
      for (int row = 0; row < (anchor ? 1 : 8); ++row) {
        auto expected = cpu_probability(p.data() + row * V, true);
        for (int i = 0; i < SHARD; ++i)
          REQUIRE(fabsf(actual_p[row * SHARD + i] - expected[r * SHARD + i]) < 2e-7);
      }
      if (!anchor) for (int row = 0; row < 7; ++row) {
        auto expected = cpu_probability(q.data() + row * V, false);
        for (int i = 0; i < SHARD; ++i)
          REQUIRE(fabsf(actual_q[row * SHARD + i] - expected[r * SHARD + i]) < 2e-7);
      }
      CK(cudaGraphExecDestroy(x.exec[2]));
      CK(cudaGraphDestroy(x.graph[2]));
    }
    fprintf(stderr, "validated anchor=%d\n", anchor);
  }
  printf("{\"pass\":true,\"ranks\":8,\"shapes\":2,\"replays_per_shape\":2,\"cpu_probability_oracle\":true}\n");
}
