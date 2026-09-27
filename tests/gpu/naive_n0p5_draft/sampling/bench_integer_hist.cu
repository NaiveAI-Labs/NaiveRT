// Paired whole-TP8 sampler timing against a frozen baseline header.
// Generate sharded_fp64_baseline.cuh from the named baseline commit outside the repo and add its directory to -I.
#define main sharded_benchmark_main
#include "bench_sharded.cu"
#undef main
#define naivert_dspark_sharded fp64_sharded
#include "sharded_fp64_baseline.cuh"
#undef naivert_dspark_sharded
static_assert(sizeof(shard::Args) == sizeof(fp64_sharded::Args));
static_assert(sizeof(shard::Scratch) == sizeof(fp64_sharded::Scratch));
static_assert(sizeof(shard::Mailbox) == sizeof(fp64_sharded::Mailbox));

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
  constexpr int batch = 16, repeats = 40;
  printf("{\"batch\":16,\"repeats\":40,\"timing\":[");
  bool first = true;
  for (bool anchor : {false, true}) {
    for (int r = 0; r < 8; ++r) {
      CK(cudaSetDevice(r));
      auto& x = *ranks[r];
      x.a.anchor = anchor;
      for (int impl = 0; impl < 2; ++impl) {
        CK(cudaStreamBeginCapture(x.stream, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < batch; ++i) {
          if (impl == 0) {
            fp64_sharded::Args args;
            std::memcpy(&args, &x.a, sizeof(args));
            fp64_sharded::run(args, reinterpret_cast<fp64_sharded::Scratch*>(x.scratch), false, x.stream);
          } else shard::run(x.a, x.scratch, false, x.stream);
        }
        CK(cudaStreamEndCapture(x.stream, &x.graph[impl]));
        CK(cudaGraphInstantiateWithFlags(&x.exec[impl], x.graph[impl], cudaGraphInstantiateFlagDeviceLaunch));
      }
    }
    std::vector<float> times[2];
    std::vector<int32_t> expected;
    for (int rep = 0; rep < repeats + 4; ++rep) {
      for (int order = 0; order < 2; ++order) {
        int impl = rep % 2 ? 1 - order : order;
        for (int r = 0; r < 8; ++r) {
          CK(cudaSetDevice(r));
          auto& x = *ranks[r];
          CK(cudaEventRecord(x.start, x.stream));
          CK(cudaGraphLaunch(x.exec[impl], x.stream));
          CK(cudaEventRecord(x.end, x.stream));
        }
        float maximum = 0;
        for (int r = 0; r < 8; ++r) {
          CK(cudaSetDevice(r));
          auto& x = *ranks[r];
          CK(cudaStreamSynchronize(x.stream));
          float ms;
          CK(cudaEventElapsedTime(&ms, x.start, x.end));
          maximum = std::max(maximum, ms * 1000 / batch);
          REQUIRE(download(x.a.watchdog, 1)[0] == 0);
          auto actual = download(x.a.tokens, anchor ? 1 : 8);
          if (expected.empty()) expected = actual;
          REQUIRE(actual == expected);
        }
        if (rep >= 4) times[impl].push_back(maximum);
      }
    }
    for (int impl = 0; impl < 2; ++impl) {
      auto& t = times[impl];
      std::sort(t.begin(), t.end());
      printf("%s{\"anchor\":%s,\"implementation\":\"%s\",\"mean_us\":%.3f,\"median_us\":%.3f,\"min_us\":%.3f,\"max_us\":%.3f}",
             first ? "" : ",", anchor ? "true" : "false", impl ? "candidate" : "baseline",
             std::accumulate(t.begin(), t.end(), 0.0) / t.size(), t[t.size()/2], t.front(), t.back());
      first = false;
    }
    for (int r = 0; r < 8; ++r) {
      CK(cudaSetDevice(r));
      for (int impl = 0; impl < 2; ++impl) {
        CK(cudaGraphExecDestroy(ranks[r]->exec[impl]));
        CK(cudaGraphDestroy(ranks[r]->graph[impl]));
      }
    }
  }
  printf("]}\n");
}
