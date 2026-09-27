// Instrumented stage timings include event-node overhead; use bench.cu for
// latency.
#include "harness.cuh"
int main(int argc, char** argv) {
  REQUIRE(argc == 2);
  std::string dir = argv[1];
  auto real_p = read<float>(dir + "/target.bin", 8 * V),
       real_q = read<float>(dir + "/draft.bin", 7 * V);
  auto real_ids = read<int64_t>(dir + "/proposals.bin", 7);
  Buffers newer;
  cudaStream_t stream;
  CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaEvent_t events[17];
  for (auto& e : events) CK(cudaEventCreate(&e));
  newer.input(real_p, real_q, real_ids, false, 1234, 0);
  fast::Args a{
      newer.a.full, newer.a.draft_full, newer.a.tokens, newer.a.proposals,
      newer.a.rng,  newer.a.watchdog,   false};
  auto* s = newer.scratch;
  int rows = 8, distributions = 15;
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CK(cudaEventRecordWithFlags(events[0], stream, cudaEventRecordExternal));
  fast::partial_maximum<<<dim3(fast::tiles, distributions), fast::threads, 0,
                          stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[1], stream, cudaEventRecordExternal));
  fast::maximum<<<distributions, fast::threads, 0, stream>>>(s);
  CK(cudaEventRecordWithFlags(events[2], stream, cudaEventRecordExternal));
  fast::exponentiate<<<dim3(fast::tiles, distributions), fast::threads, 0,
                       stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[3], stream, cudaEventRecordExternal));
  fast::inverse_sum<<<distributions, fast::threads, 0, stream>>>(s);
  CK(cudaEventRecordWithFlags(events[4], stream, cudaEventRecordExternal));
  fast::
      normalize<<<dim3(fast::tiles, distributions), fast::threads, 0, stream>>>(
          a, s);
  CK(cudaEventRecordWithFlags(events[5], stream, cudaEventRecordExternal));
  fast::radix_histogram<24>
      <<<dim3(fast::tiles, rows), fast::threads, 0, stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[6], stream, cudaEventRecordExternal));
  fast::select_digit<24><<<rows, fast::threads, 0, stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[7], stream, cudaEventRecordExternal));
  fast::radix_histogram<16>
      <<<dim3(fast::tiles, rows), fast::threads, 0, stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[8], stream, cudaEventRecordExternal));
  fast::select_digit<16><<<rows, fast::threads, 0, stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[9], stream, cudaEventRecordExternal));
  fast::radix_histogram<8>
      <<<dim3(fast::tiles, rows), fast::threads, 0, stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[10], stream, cudaEventRecordExternal));
  fast::select_digit<8><<<rows, fast::threads, 0, stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[11], stream, cudaEventRecordExternal));
  fast::radix_histogram<0>
      <<<dim3(fast::tiles, rows), fast::threads, 0, stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[12], stream, cudaEventRecordExternal));
  fast::select_digit<0><<<rows, fast::threads, 0, stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[13], stream, cudaEventRecordExternal));
  fast::
      normalize_nucleus<<<dim3(fast::tiles, rows), fast::threads, 0, stream>>>(
          a, s);
  CK(cudaEventRecordWithFlags(events[14], stream, cudaEventRecordExternal));
  fast::choose_tile<<<rows, fast::threads, 0, stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[15], stream, cudaEventRecordExternal));
  fast::choose_token<<<rows, fast::threads, 0, stream>>>(a, s);
  CK(cudaEventRecordWithFlags(events[16], stream, cudaEventRecordExternal));
  CK(cudaStreamEndCapture(stream, &graph));
  CK(cudaGraphInstantiate(&exec, graph, 0));
  double sums[16] = {};
  for (int rep = 0; rep < 25; ++rep) {
    newer.input(real_p, real_q, real_ids, false, 1234, 0);
    CK(cudaGraphLaunch(exec, stream));
    CK(cudaStreamSynchronize(stream));
    for (int i = 0; i < 16; ++i) {
      float ms;
      CK(cudaEventElapsedTime(&ms, events[i], events[i + 1]));
      if (rep >= 5) sums[i] += ms * 1000;
    }
  }
  printf("partial_maximum %.3f us\n", sums[0] / 20);
  printf("maximum %.3f us\n", sums[1] / 20);
  printf("exponentiate %.3f us\n", sums[2] / 20);
  printf("inverse_sum %.3f us\n", sums[3] / 20);
  printf("normalize %.3f us\n", sums[4] / 20);
  printf("radix_histogram<24> %.3f us\n", sums[5] / 20);
  printf("select_digit<24> %.3f us\n", sums[6] / 20);
  printf("radix_histogram<16> %.3f us\n", sums[7] / 20);
  printf("select_digit<16> %.3f us\n", sums[8] / 20);
  printf("radix_histogram<8> %.3f us\n", sums[9] / 20);
  printf("select_digit<8> %.3f us\n", sums[10] / 20);
  printf("radix_histogram<0> %.3f us\n", sums[11] / 20);
  printf("select_digit<0> %.3f us\n", sums[12] / 20);
  printf("normalize_nucleus %.3f us\n", sums[13] / 20);
  printf("choose_tile %.3f us\n", sums[14] / 20);
  printf("choose_token %.3f us\n", sums[15] / 20);
}
