// Check the integer weighted histograms against direct FP64 summation. There
// are no peer waits in this kernel, so this also permits a bounded memcheck run.
#include "harness.cuh"
#include "naivert/body/dspark_sampling/sharded.cuh"
namespace shard = naivert_dspark_sharded;

int main() {
  shard::Args a{};
  shard::Scratch* scratch;
  alloc(a.p, 8 * shard::vocabulary);
  alloc(scratch, 1);
  std::vector<float> values(8 * shard::vocabulary);
  std::mt19937 gen(8193);
  size_t compared = 0;
  for (int kind = 0; kind < 6; ++kind) {
    for (size_t i = 0; i < values.size(); ++i) {
      unsigned bits;
      if (kind == 0) bits = 0;                      // empty bins, +0
      else if (kind == 1) bits = 0x3f800000;       // p=1; maximum warp mass
      else if (kind == 2) bits = gen() % 0x3f800001u; // all probability exponents
      else if (kind == 3) bits = gen() & 0xffffff; // subnormal/first normal bin
      else if (kind == 4) bits = (unsigned(i % 64) << 24) | (i & 1 ? 0xffffff : 0);
      else bits = 0x3dffffff - (gen() & 31);       // concentrated high-byte bucket
      std::memcpy(&values[i], &bits, sizeof(bits));
    }
    CK(cudaMemcpy(a.p, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    for (int shift : {24, 16, 8, 0}) {
      CK(cudaMemset(scratch, 0xa5, sizeof(*scratch)));
      unsigned mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
      unsigned prefixes[8];
      for (int row = 0; row < 8; ++row) {
        unsigned bits;
        std::memcpy(&bits, &values[row * shard::vocabulary + 19], sizeof(bits));
        prefixes[row] = bits & mask;
      }
      CK(cudaMemcpy(scratch->prefix, prefixes, sizeof(prefixes), cudaMemcpyHostToDevice));
      if (shift == 24) shard::radix_histogram<24><<<dim3(shard::tiles, 8), shard::threads>>>(a, scratch);
      if (shift == 16) shard::radix_histogram<16><<<dim3(shard::tiles, 8), shard::threads>>>(a, scratch);
      if (shift == 8) shard::radix_histogram<8><<<dim3(shard::tiles, 8), shard::threads>>>(a, scratch);
      if (shift == 0) shard::radix_histogram<0><<<dim3(shard::tiles, 8), shard::threads>>>(a, scratch);
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
      auto actual = download(scratch->histogram, 8 * shard::tiles * shard::bins);
      std::vector<double> expected(actual.size(), 0.0);
      for (size_t i = 0; i < values.size(); ++i) {
        unsigned bits;
        std::memcpy(&bits, &values[i], sizeof(bits));
        size_t row = i / shard::vocabulary, part = i % shard::vocabulary / shard::tile;
        if ((bits & mask) == prefixes[row])
          expected[(row * shard::tiles + part) * shard::bins + ((bits >> shift) & 255)] += double(values[i]);
      }
      REQUIRE(actual == expected);
      compared += actual.size();
    }
  }
  printf("{\"pass\":true,\"cases\":24,\"histogram_bins_compared\":%zu,\"different_bins\":0}\n", compared);
  CK(cudaFree(a.p));
  CK(cudaFree(scratch));
}
