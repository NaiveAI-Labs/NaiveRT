#include "harness.cuh"
int main(int argc, char** argv) {
  REQUIRE(argc >= 2);
  std::string dir = argv[1];
  int repeats = argc > 2 ? atoi(argv[2]) : 50;
  REQUIRE(repeats > 0);
  auto real_p = read<float>(dir + "/target.bin", 8 * V),
       real_q = read<float>(dir + "/draft.bin", 7 * V);
  auto real_ids = read<int64_t>(dir + "/proposals.bin", 7);
  Buffers old, newer;
  float* device_uniform;
  alloc(device_uniform, 16);
  std::mt19937 gen(42);
  std::normal_distribution<float> normal(0, 4);
  double max_abs = 0, max_cpu_abs = 0;
  size_t different = 0, total_prob = 0;
  int cases = 0;
  for (int kind = 0; kind < (argc > 3 ? 0 : 7); ++kind) {
    auto p = real_p, q = real_q;
    auto ids = real_ids;
    if (kind == 1) {
      std::fill(p.begin(), p.end(), 0);
      std::fill(q.begin(), q.end(), 0);
    }
    if (kind == 2) {
      for (size_t i = 0; i < p.size(); ++i) p[i] = float(i % 19) - 9;
      for (size_t i = 0; i < q.size(); ++i) q[i] = float(i % 23) - 11;
    }
    if (kind == 3) {
      std::fill(p.begin(), p.end(), -INFINITY);
      std::fill(q.begin(), q.end(), -INFINITY);
      for (int r = 0; r < 8; ++r) p[r * V + 100 + r] = 0;
      for (int r = 0; r < 7; ++r) {
        q[r * V + 100 + r] = 0;
        ids[r] = 100 + r;
      }
    }
    if (kind == 4) {
      for (float& v : p) v = normal(gen);
      for (float& v : q) v = normal(gen);
      for (auto& id : ids) id = gen() % V;
    }
    if (kind == 5) {
      std::fill(p.begin(), p.end(), -INFINITY);
      std::fill(q.begin(), q.end(), -INFINITY);
      for (int r = 0; r < 8; ++r) {
        p[r * V + V - 1] = 0;
        p[r * V + V - 2] = -1;
      }
      for (int r = 0; r < 7; ++r) {
        q[r * V] = 0;
        ids[r] = 0;
      }
    }
    if (kind == 6) {
      for (float& v : p) v = normal(gen) * 10;
      for (float& v : q) v = normal(gen) * 10;
      for (auto& id : ids) id = gen() % V;
    }
    for (bool anchor : {false, true}) {
      int rows = anchor ? 1 : 8;
      std::vector<std::vector<float>> cpu_p, cpu_q;
      for (int r = 0; r < rows; ++r) {
        cpu_p.push_back(cpu_probability(p.data() + r * V, true));
        if (!anchor && r < 7)
          cpu_q.push_back(cpu_probability(q.data() + r * V, false));
      }
      for (int trial = 0; trial < 16; ++trial) {
        old.input(p, q, ids, anchor, 1234 + trial, trial * 7);
        newer.input(p, q, ids, anchor, 1234 + trial, trial * 7);
        // Poison scratch on alternating rounds to detect reliance on previous
        // data.
        CK(cudaMemset(newer.scratch, trial % 2 ? 0xa5 : 0x5a,
                      sizeof(fast::Scratch)));
        old.launch(false);
        newer.launch(true);
        CK(cudaDeviceSynchronize());
        auto ot = download(old.a.tokens, rows),
             nt = download(newer.a.tokens, rows);
        if (ot != nt) {
          fprintf(stderr, "token mismatch kind=%d anchor=%d trial=%d\n", kind,
                  anchor, trial);
          for (int r = 0; r < rows; ++r)
            fprintf(stderr, " %d:%d/%d", r, ot[r], nt[r]);
          fprintf(stderr, "\n");
          return 3;
        }
        REQUIRE(download(newer.a.watchdog, 1)[0] == 0);
        REQUIRE(download(old.a.watchdog, 1)[0] == 0);
        auto np = download(newer.a.full, rows * V);
        auto nq = download(newer.a.draft_full, anchor ? 0 : 7 * V);
        if (trial == 0) {
          auto op = download(old.a.full, rows * V),
               oq = download(old.a.draft_full, anchor ? 0 : 7 * V);
          for (size_t i = 0; i < np.size(); ++i) {
            max_abs = std::max(max_abs, double(fabsf(np[i] - op[i])));
            different += np[i] != op[i];
            ++total_prob;
            max_cpu_abs = std::max(max_cpu_abs,
                                   double(fabsf(np[i] - cpu_p[i / V][i % V])));
            REQUIRE(fabsf(np[i] - cpu_p[i / V][i % V]) <=
                    2e-7f + 2e-5f * fabsf(cpu_p[i / V][i % V]));
            REQUIRE((np[i] == 0) == (op[i] == 0));
          }
          for (size_t i = 0; i < nq.size(); ++i) {
            max_abs = std::max(max_abs, double(fabsf(nq[i] - oq[i])));
            different += nq[i] != oq[i];
            ++total_prob;
            max_cpu_abs = std::max(max_cpu_abs,
                                   double(fabsf(nq[i] - cpu_q[i / V][i % V])));
            REQUIRE(fabsf(nq[i] - cpu_q[i / V][i % V]) <=
                    2e-7f + 2e-5f * fabsf(cpu_q[i / V][i % V]));
          }
        }
        uniforms<<<1, 32>>>(newer.rng, device_uniform, anchor);
        auto u = download(device_uniform, 16);
        for (int r = 0; r < rows; ++r) {
          const auto& cp = cpu_p[r];
          bool reject =
              !anchor && r < 7 && !(u[r] * cpu_q[r][ids[r]] < cp[ids[r]]);
          int token = -1;
          if (!anchor && r < 7 && !reject)
            token = ids[r];
          else {
            double mass = 0;
            for (int i = 0; i < V; ++i)
              mass += reject ? fmaxf(cp[i] - cpu_q[r][i], 0) : cp[i];
            double cut = u[8 + r] * mass, acc = 0;
            for (int i = 0; i < V; ++i) {
              float v = reject ? fmaxf(cp[i] - cpu_q[r][i], 0) : cp[i];
              acc += v;
              if (v > 0 && acc > cut) {
                token = i;
                break;
              }
            }
          }
          if (token != nt[r]) {
            fprintf(stderr,
                    "CPU token mismatch kind=%d anchor=%d trial=%d row=%d "
                    "cpu=%d gpu=%d\n",
                    kind, anchor, trial, r, token, nt[r]);
            return 4;
          }
        }
        ++cases;
      }
    }
  }
  REQUIRE(max_abs < 1e-7);
  printf(
      "{\"validation_cases\":%d,\"probability_values\":%zu,\"different_"
      "probability_values\":%zu,\"max_abs_old_new\":%.9g,\"max_abs_cpu_new\":%."
      "9g,\"token_mismatches\":0,\"watchdog\":0,\"scratch_bytes\":%zu,",
      cases, total_prob, different, max_abs, max_cpu_abs,
      sizeof(fast::Scratch));
  fflush(stdout);
  printf("\"timing\":[");
  for (bool anchor : {false, true}) {
    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graphs[2];
    cudaGraphExec_t execs[2];
    for (int impl = 0; impl < 2; ++impl) {
      newer.input(real_p, real_q, real_ids, anchor, 1234, 0);
      CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      newer.launch(impl, stream);
      CK(cudaStreamEndCapture(stream, &graphs[impl]));
      CK(cudaGraphInstantiate(&execs[impl], graphs[impl], 0));
    }
    cudaEvent_t start, end;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&end));
    std::vector<float> times[2];
    if (!anchor) CK(cudaProfilerStart());
    for (int i = 0; i < repeats + 5; ++i)
      for (int order = 0; order < 2; ++order) {
        int impl = (i + order) % 2;
        newer.input(real_p, real_q, real_ids, anchor, 1234, 0);
        CK(cudaEventRecord(start, stream));
        CK(cudaGraphLaunch(execs[impl], stream));
        CK(cudaEventRecord(end, stream));
        CK(cudaEventSynchronize(end));
        float ms;
        CK(cudaEventElapsedTime(&ms, start, end));
        if (i >= 5) times[impl].push_back(ms * 1000);
        REQUIRE(download(newer.a.watchdog, 1)[0] == 0);
      }
    if (!anchor) CK(cudaProfilerStop());
    for (int impl = 0; impl < 2; ++impl) {
      auto& t = times[impl];
      std::sort(t.begin(), t.end());
      printf(
          "%s{\"anchor\":%s,\"implementation\":\"%s\",\"repeats\":%d,\"mean_"
          "us\":%.3f,\"median_us\":%.3f,\"min_us\":%.3f,\"max_us\":%.3f}",
          anchor || impl ? "," : "", anchor ? "true" : "false",
          impl ? "radix" : "baseline", repeats,
          std::accumulate(t.begin(), t.end(), 0.) / t.size(), t[t.size() / 2],
          t.front(), t.back());
      CK(cudaGraphExecDestroy(execs[impl]));
      CK(cudaGraphDestroy(graphs[impl]));
    }
    CK(cudaStreamDestroy(stream));
    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(end));
  }
  printf("]}\n");
}
