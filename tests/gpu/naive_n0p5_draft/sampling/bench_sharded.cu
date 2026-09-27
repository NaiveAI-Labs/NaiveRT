#include "harness.cuh"
#include "naivert/body/dspark_sampling/sharded.cuh"
namespace shard = naivert_dspark_sharded;
__global__ void reference_ready(Args a, bool consumed) {
  naivert_dspark_ops::sampling_rendezvous(a.flags, a.rank, a.watchdog,
                                          consumed);
}
__global__ void reference_gather(Args a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < a.rows * V) {
    int row = i / V, token = i % V;
    a.full[i] = a.logits[token / SHARD][row * SHARD + token % SHARD];
    if (!a.anchor && row < 7)
      a.draft_full[i] =
          a.draft_logits[token / SHARD][row * SHARD + token % SHARD];
  }
}
struct Rank {
  shard::Args a{};
  shard::Scratch* scratch;
  shard::Mailbox* mailbox;
  float *input_p, *input_q;
  int64_t *rng, *ids;
  unsigned long long* flags;
  Buffers reference;
  cudaStream_t stream;
  cudaGraph_t graph[3];
  cudaGraphExec_t exec[3];
  cudaEvent_t start, end;
  Rank() {
    alloc(input_p, 8 * SHARD);
    alloc(input_q, 7 * SHARD);
    alloc(rng, 2);
    alloc(ids, 7);
    alloc(flags, 3);
    alloc(a.p, 8 * SHARD);
    alloc(a.q, 7 * SHARD);
    alloc(a.tokens, 8);
    alloc(a.watchdog, 1);
    alloc(scratch, 1);
    alloc(mailbox, 1);
    a.logits = input_p;
    a.draft_logits = input_q;
    a.proposals = ids;
    a.rng = rng;
    CK(cudaMemset(flags, 0, 3 * sizeof(*flags)));
    CK(cudaMemset(mailbox, 0, sizeof(*mailbox)));
    CK(cudaMemset(a.watchdog, 0, 4));
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&end));
  }
  void launch(int impl, bool anchor) {
    a.anchor = anchor;
    if (impl == 2)
      shard::run(a, scratch, false, stream);
    else {
      auto ref = reference.a;
      ref.anchor = anchor;
      ref.rows = anchor ? 1 : 8;
      reference_ready<<<1, 1, 0, stream>>>(ref, false);
      reference_gather<<<(ref.rows * V + 255) / 256, 256, 0, stream>>>(ref);
      reference_ready<<<1, 1, 0, stream>>>(ref, true);
      if (impl == 0)
        sample_target<<<ref.rows, 256, 0, stream>>>(ref);
      else
        fast::run({ref.full, ref.draft_full, ref.tokens, ref.proposals, ref.rng,
                   ref.watchdog, ref.anchor},
                  reference.scratch, stream);
    }
    CK(cudaGetLastError());
  }
};
int main(int argc, char** argv) {
  REQUIRE(argc >= 2);
  std::string dir = argv[1];
  int repeats = argc > 2 ? atoi(argv[2]) : 20;
  int batch = argc > 3 ? atoi(argv[3]) : 16;
  REQUIRE(repeats > 0 && batch > 0);
  auto real_p = read<float>(dir + "/target.bin", 8 * V),
       real_q = read<float>(dir + "/draft.bin", 7 * V);
  auto real_ids = read<int64_t>(dir + "/proposals.bin", 7);
  Rank* ranks[8];
  for (int r = 0; r < 8; ++r) {
    CK(cudaSetDevice(r));
    for (int peer = 0; peer < 8; ++peer)
      if (peer != r) {
        auto e = cudaDeviceEnablePeerAccess(peer, 0);
        REQUIRE(e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled);
        cudaGetLastError();
      }
    ranks[r] = new Rank;
    ranks[r]->a.rank = r;
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
  auto sync = [&]() {
    for (int r = 0; r < 8; ++r) {
      CK(cudaSetDevice(r));
      CK(cudaStreamSynchronize(ranks[r]->stream));
    }
  };
  auto input = [&](const std::vector<float>& p, const std::vector<float>& q,
                   const std::vector<int64_t>& ids, int seed, int counter) {
    for (int r = 0; r < 8; ++r) {
      CK(cudaSetDevice(r));
      auto& x = *ranks[r];
      std::vector<float> sp(8 * SHARD), sq(7 * SHARD);
      for (int row = 0; row < 8; ++row)
        std::copy_n(p.data() + row * V + r * SHARD, SHARD,
                    sp.data() + row * SHARD);
      for (int row = 0; row < 7; ++row)
        std::copy_n(q.data() + row * V + r * SHARD, SHARD,
                    sq.data() + row * SHARD);
      int64_t state[2] = {seed, counter};
      CK(cudaMemcpy(x.input_p, sp.data(), sp.size() * 4,
                    cudaMemcpyHostToDevice));
      CK(cudaMemcpy(x.input_q, sq.data(), sq.size() * 4,
                    cudaMemcpyHostToDevice));
      CK(cudaMemcpy(x.rng, state, sizeof(state), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(x.ids, ids.data(), ids.size() * 8, cudaMemcpyHostToDevice));
      CK(cudaMemset(x.a.watchdog, 0, 4));
      CK(cudaMemset(x.reference.a.watchdog, 0, 4));
    }
  };
  int cases = 0;
  double max_abs = 0;
  size_t diff = 0, compared = 0;
  std::mt19937 gen(42);
  std::normal_distribution<float> normal(0, 4);
  for (int kind = 0; kind < (argc > 4 ? 1 : 11); ++kind) {
    auto p = real_p, q = real_q;
    auto ids = real_ids;
    if (kind == 1) {
      std::fill(p.begin(), p.end(), 0);
      std::fill(q.begin(), q.end(), 0);
    }
    if (kind == 2) {
      for (float& v : p) v = normal(gen);
      for (float& v : q) v = normal(gen);
      for (auto& id : ids) id = gen() % V;
    }
    if (kind >= 3) {
      int accept = kind - 3;
      std::fill(p.begin(), p.end(), -10000);
      std::fill(q.begin(), q.end(), -10000);
      for (int r = 0; r < 7; ++r) {
        ids[r] = r * SHARD;
        q[r * V + ids[r]] = 0;
        p[r * V + ids[r] + (r < accept ? 0 : 1)] = 0;
      }
      p[7 * V + V - 1] = 0;
    }
    for (bool anchor : {false, true})
      for (int trial = 0; trial < (argc > 4 ? 1 : 4); ++trial) {
        input(p, q, ids, 1234 + trial, trial * 7);
        int rows = anchor ? 1 : 8;
        for (int r = 0; r < 8; ++r) {
          CK(cudaSetDevice(r));
          auto& x = *ranks[r];
          CK(cudaMemset(x.scratch, trial % 2 ? 0xa5 : 0x5a,
                        sizeof(shard::Scratch)));
          x.launch(0, anchor);
        }
        sync();
        for (int r = 0; r < 8; ++r) {
          CK(cudaSetDevice(r));
          ranks[r]->launch(2, anchor);
        }
        sync();
        for (int r = 0; r < 8; ++r) {
          CK(cudaSetDevice(r));
          auto& x = *ranks[r];
          auto old = download(x.reference.a.tokens, rows),
               now = download(x.a.tokens, rows);
          if (old != now) {
            fprintf(stderr,
                    "token mismatch kind=%d anchor=%d trial=%d rank=%d\n", kind,
                    anchor, trial, r);
            for (int row = 0; row < rows; ++row)
              fprintf(stderr, " %d:%d/%d", row, old[row], now[row]);
            fprintf(stderr, "\n");
            return 3;
          }
          REQUIRE(download(x.a.watchdog, 1)[0] == 0);
          REQUIRE(download(x.reference.a.watchdog, 1)[0] == 0);
          if (trial == 0) {
            auto op = download(x.reference.a.full, rows * V),
                 oq = download(x.reference.a.draft_full, anchor ? 0 : 7 * V);
            auto np = download(x.a.p, rows * SHARD),
                 nq = download(x.a.q, anchor ? 0 : 7 * SHARD);
            for (size_t i = 0; i < np.size(); ++i) {
              float ref = op[(i / SHARD) * V + r * SHARD + i % SHARD];
              max_abs = std::max(max_abs, double(fabsf(np[i] - ref)));
              diff += np[i] != ref;
              ++compared;
              REQUIRE((np[i] == 0) == (ref == 0));
            }
            for (size_t i = 0; i < nq.size(); ++i) {
              float ref = oq[(i / SHARD) * V + r * SHARD + i % SHARD];
              max_abs = std::max(max_abs, double(fabsf(nq[i] - ref)));
              diff += nq[i] != ref;
              ++compared;
            }
          }
        }
        ++cases;
      }
  }
  REQUIRE(max_abs < 1e-7);
  printf(
      "{\"cases\":%d,\"probability_values\":%zu,\"different_values\":%zu,\"max_"
      "abs\":%.9g,\"token_mismatches\":0,\"scratch_bytes_per_rank\":%zu,"
      "\"mailbox_bytes_per_rank\":%zu,\"local_vocabulary\":%d,\"timing\":[",
      cases, compared, diff, max_abs, sizeof(shard::Scratch),
      sizeof(shard::Mailbox), SHARD);
  fflush(stdout);
  bool first = true;
  for (bool anchor : {false, true}) {
    input(real_p, real_q, real_ids, 1234, 0);
    for (int r = 0; r < 8; ++r) {
      CK(cudaSetDevice(r));
      auto& x = *ranks[r];
      for (int impl = 0; impl < 3; ++impl) {
        CK(cudaStreamBeginCapture(x.stream, cudaStreamCaptureModeGlobal));
        for (int b = 0; b < batch; ++b) x.launch(impl, anchor);
        CK(cudaStreamEndCapture(x.stream, &x.graph[impl]));
        CK(cudaGraphInstantiateWithFlags(&x.exec[impl], x.graph[impl],
                                         cudaGraphInstantiateFlagDeviceLaunch));
      }
    }
    std::vector<float> times[3];
    std::vector<int32_t> expected_tokens;
    for (int rep = 0; rep < repeats + 2; ++rep)
      for (int order = 0; order < 3; ++order) {
        int impl = rep % 2 ? 2 - order : order;
        for (int r = 0; r < 8; ++r) {
          CK(cudaSetDevice(r));
          auto& x = *ranks[r];
          CK(cudaEventRecord(x.start, x.stream));
          CK(cudaGraphLaunch(x.exec[impl], x.stream));
          CK(cudaEventRecord(x.end, x.stream));
        }
        sync();
        float maximum = 0;
        for (int r = 0; r < 8; ++r) {
          CK(cudaSetDevice(r));
          auto& x = *ranks[r];
          float ms;
          CK(cudaEventElapsedTime(&ms, x.start, x.end));
          maximum = std::max(maximum, ms * 1000 / batch);
          REQUIRE(download(impl == 2 ? x.a.watchdog : x.reference.a.watchdog,
                           1)[0] == 0);
          auto actual = download(impl == 2 ? x.a.tokens : x.reference.a.tokens,
                                 anchor ? 1 : 8);
          if (expected_tokens.empty()) expected_tokens = actual;
          REQUIRE(actual == expected_tokens);
        }
        if (rep >= 2) times[impl].push_back(maximum);
      }
    for (int impl = 0; impl < 3; ++impl) {
      auto& t = times[impl];
      std::sort(t.begin(), t.end());
      printf(
          "%s{\"anchor\":%s,\"implementation\":\"%s\",\"batch\":%d,\"repeats\":"
          "%d,\"mean_us\":%.3f,\"median_us\":%.3f,\"min_us\":%.3f,\"max_us\":%."
          "3f}",
          first ? "" : ",", anchor ? "true" : "false",
          impl == 0   ? "legacy_tp8"
          : impl == 1 ? "replicated_radix_tp8"
                      : "sharded_tp8",
          batch, repeats, std::accumulate(t.begin(), t.end(), 0.) / t.size(),
          t[t.size() / 2], t.front(), t.back());
      first = false;
    }
    for (int r = 0; r < 8; ++r) {
      CK(cudaSetDevice(r));
      for (int impl = 0; impl < 3; ++impl) {
        CK(cudaGraphExecDestroy(ranks[r]->exec[impl]));
        CK(cudaGraphDestroy(ranks[r]->graph[impl]));
      }
    }
  }
  printf("]}\n");
  return 0;
}
