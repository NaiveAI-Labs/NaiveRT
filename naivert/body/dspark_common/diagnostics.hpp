#pragma once

// Intermediate exports are correctness probes, never serving/benchmark work.
// Build the explicit diagnostics extension to inspect them.
#ifndef NAIVERT_DSPARK_DIAGNOSTICS
#define NAIVERT_DSPARK_DIAGNOSTICS 0
#endif
static_assert(NAIVERT_DSPARK_DIAGNOSTICS == 0 || NAIVERT_DSPARK_DIAGNOSTICS == 1);
namespace naivert_dspark_ops {
inline constexpr bool kExportDiagnostics = NAIVERT_DSPARK_DIAGNOSTICS != 0;
}
