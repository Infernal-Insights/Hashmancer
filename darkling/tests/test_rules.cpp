#include <assert.h>
#include <cuda_runtime.h>
#include "darkling_rules.h"

int main() {
  dl_rules_load_json("configs/ruleset_baked256.json");
  DlRuleMC host;
  cudaMemcpyFromSymbol(&host, g_rules, sizeof(DlRuleMC));
  assert(host.shape == PREFIX_1);
  return 0;
}
