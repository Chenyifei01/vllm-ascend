# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner for msprof op collection of the fused dequant_situ_quant op.

msprof op needs a plain command line; this script just creates the inputs and
calls the op repeatedly so the profiler can capture steady-state launches.

Usage (from the repo root):
  msprof op --kernel-name=dequant_situ_quant --output=./profiling \
      --aic-metrics=PipeUtilization \
      python benchmarks/dequant_situ_quant_msprof.py 1040 shared

Positional args: rows, mode. Mode selects the production caller variant:
  shared : int32 [rows, 2H], weight_scale [2H], activation_scale [rows]
  routed : int32 [rows, 2H], weight_scale/bias [14, 2H], group_index [14]
  bf16   : bf16 [rows, 2H], no scales (predequantized GMM1 output)

2H defaults to the Kimi K3 routed width 6144; pass a third arg to override.
"""

import sys

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 1040
MODE = sys.argv[2] if len(sys.argv) > 2 else "shared"
INPUT_WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 6144
BETA = 4.0
LINEAR_BETA = 25.0
K3_LOCAL_EXPERTS = 14
ITERS = 30

torch.manual_seed(7)
x = torch.randint(-20000, 20000, (ROWS, INPUT_WIDTH), dtype=torch.int32,
                  device="npu")
weight_scale = torch.randn(INPUT_WIDTH, dtype=torch.float32,
                           device="npu") * 0.001
bias = None
group_index = None
if MODE == "routed":
    weight_scale = torch.randn(K3_LOCAL_EXPERTS, INPUT_WIDTH,
                               dtype=torch.float32, device="npu") * 0.001
    bias = torch.randn(K3_LOCAL_EXPERTS, INPUT_WIDTH, dtype=torch.float32,
                       device="npu") * 0.1
    base = ROWS // K3_LOCAL_EXPERTS
    sizes = [base] * K3_LOCAL_EXPERTS
    for i in range(ROWS - base * K3_LOCAL_EXPERTS):
        sizes[i] += 1
    group_index = torch.tensor(sizes, dtype=torch.int64, device="npu")
elif MODE == "bf16":
    x = (torch.randn(ROWS, INPUT_WIDTH, dtype=torch.float32, device="npu")
         * 8.0).to(torch.bfloat16)
    weight_scale = None
activation_scale = None
if MODE != "bf16":
    activation_scale = torch.rand(ROWS, 1, dtype=torch.float32,
                                  device="npu") * 4 + 0.5


def call():
    return torch.ops._C_ascend.dequant_situ_quant(
        x=x, weight_scale=weight_scale, activation_scale=activation_scale,
        bias=bias, quant_scale=None, quant_offset=None,
        group_index=group_index, beta=BETA, linear_beta=LINEAR_BETA,
        activate_left=True, quant_mode="dynamic")


for _ in range(10):
    call()
torch.npu.synchronize()
for _ in range(ITERS):
    call()
torch.npu.synchronize()
print("done")
