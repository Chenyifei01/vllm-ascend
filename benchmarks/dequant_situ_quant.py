# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused dequant_situ_quant custom op.

Run: python benchmarks/dequant_situ_quant.py

Production callers (Kimi K3 W4A8 SituAndMul, every decoder layer):
  * shared experts : x=int32 [rows, 2H], weight_scale [2H],
                     activation_scale [rows] (AscendSharedExperts, 2H=12288/TP)
  * routed experts : x=int32 [tokens*topk, 2H], weight_scale/bias [E, 2H],
                     group_index [E] (apply_gmm1_act_quant, 2H=6144)
  * predequantized : x=bf16 [rows, 2H], no scales (w4a8 GMM1 with bf16
                     output, 2H=6144)

Times only torch.ops._C_ascend.dequant_situ_quant (the op under
optimization) against an unfused reference composition. Times exclude
compilation, graph capture and host tensor allocation. Repetition counts
adapt to a warmup measurement so slow shapes stay bounded.
"""

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

BETA = 4.0
LINEAR_BETA = 25.0
K3_LOCAL_EXPERTS = 14
K3_ROUTED_INPUT_WIDTH = 6144


def graph_latency_us(fn, *args):
    for _ in range(3):
        fn(*args)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    fn(*args)
    end.record()
    end.synchronize()
    estimate_ms = max(start.elapsed_time(end), 0.001)
    # Capture at most about 10 ms of work and measure about 50 ms per sample.
    batch = min(32, max(1, int(10 / estimate_ms)))
    repeats = min(20, max(1, int(50 / (batch * estimate_ms))))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local",
                         auto_dispatch_capture=True):
        for _ in range(batch):
            output = fn(*args)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (batch * repeats))
    # Keep the captured outputs alive until timing completes.
    del output
    return statistics.median(samples)


def situ_reference(gate, up):
    gate = BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate)
    up = LINEAR_BETA * torch.tanh(up / LINEAR_BETA)
    situ = gate * up
    scale = situ.abs().amax(dim=-1) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    y = torch.round(situ / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return y, scale


def run_op(x, weight_scale, activation_scale, bias, group_index):
    return torch.ops._C_ascend.dequant_situ_quant(
        x=x, weight_scale=weight_scale, activation_scale=activation_scale,
        bias=bias, quant_scale=None, quant_offset=None,
        group_index=group_index, beta=BETA, linear_beta=LINEAR_BETA,
        activate_left=True, quant_mode="dynamic")


def fused_shared(x, weight_scale, activation_scale):
    return run_op(x, weight_scale, activation_scale, None, None)


def ref_shared(x, weight_scale, activation_scale):
    dequant = x.float() * weight_scale * activation_scale.reshape(-1, 1)
    gate, up = dequant.chunk(2, dim=-1)
    return situ_reference(gate, up)


def fused_routed(x, weight_scale, activation_scale, bias, group_index,
                 row_expert):
    return run_op(x, weight_scale, activation_scale, bias, group_index)


def ref_routed(x, weight_scale, activation_scale, bias, group_index,
               row_expert):
    dequant = x.float() * weight_scale[row_expert] \
        * activation_scale.reshape(-1, 1)
    dequant = dequant + bias[row_expert]
    gate, up = dequant.chunk(2, dim=-1)
    return situ_reference(gate, up)


def fused_bf16(x):
    return run_op(x, None, None, None, None)


def ref_bf16(x):
    gate, up = x.float().chunk(2, dim=-1)
    return situ_reference(gate, up)


def check(name, actual, expected, y_atol=1.0, scale_rtol=5e-3):
    actual_y, actual_scale = actual
    expected_y, expected_scale = expected
    torch.testing.assert_close(actual_y.float(), expected_y.float(),
                               rtol=0, atol=y_atol)
    torch.testing.assert_close(actual_scale, expected_scale,
                               rtol=scale_rtol, atol=1e-5)
    print(f"  check {name}: OK")


def make_shared(rows, input_width):
    torch.manual_seed(7)
    x = torch.randint(-20000, 20000, (rows, input_width), dtype=torch.int32,
                      device="npu")
    weight_scale = torch.randn(input_width, dtype=torch.float32,
                               device="npu") * 0.001
    activation_scale = torch.rand(rows, 1, dtype=torch.float32,
                                  device="npu") * 4 + 0.5
    return (x, weight_scale, activation_scale)


def make_routed(rows, input_width):
    torch.manual_seed(7)
    x = torch.randint(-20000, 20000, (rows, input_width), dtype=torch.int32,
                      device="npu")
    weight_scale = torch.randn(K3_LOCAL_EXPERTS, input_width,
                               dtype=torch.float32, device="npu") * 0.001
    bias = torch.randn(K3_LOCAL_EXPERTS, input_width, dtype=torch.float32,
                       device="npu") * 0.1
    activation_scale = torch.rand(rows, 1, dtype=torch.float32,
                                  device="npu") * 4 + 0.5
    base = rows // K3_LOCAL_EXPERTS
    sizes = [base] * K3_LOCAL_EXPERTS
    for i in range(rows - base * K3_LOCAL_EXPERTS):
        sizes[i] += 1
    group_index = torch.tensor(sizes, dtype=torch.int64, device="npu")
    # repeat_interleave is a dynamic-shape op: build row_expert outside any
    # NPUGraph capture (it stream-syncs and cannot be captured).
    row_expert = torch.repeat_interleave(
        torch.arange(K3_LOCAL_EXPERTS, device="npu"), group_index)
    return (x, weight_scale, activation_scale, bias, group_index, row_expert)


def make_bf16(rows, input_width):
    torch.manual_seed(7)
    x = (torch.randn(rows, input_width, dtype=torch.float32, device="npu")
         * 8.0).to(torch.bfloat16)
    return (x,)


SHAPES = {
    # (mode, rows, input_width) — input_width is 2H = 12288 / TP degree.
    "shared": [
        (1, 12288), (4, 12288), (16, 12288), (64, 12288),
        (128, 12288), (512, 12288), (1024, 12288), (2048, 12288),
        (1, 6144), (16, 6144), (128, 6144), (1024, 6144), (4096, 6144),
    ],
    "routed": [
        (16, K3_ROUTED_INPUT_WIDTH), (1040, K3_ROUTED_INPUT_WIDTH),
        (65536, K3_ROUTED_INPUT_WIDTH),
    ],
    "bf16": [
        (1, K3_ROUTED_INPUT_WIDTH), (16, K3_ROUTED_INPUT_WIDTH),
        (1040, K3_ROUTED_INPUT_WIDTH), (4096, K3_ROUTED_INPUT_WIDTH),
    ],
}

CASES = {
    "shared": (make_shared, fused_shared, ref_shared),
    "routed": (make_routed, fused_routed, ref_routed),
    "bf16": (make_bf16, fused_bf16, ref_bf16),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="*", default=["shared", "routed", "bf16"],
                        choices=list(SHAPES), help="which mode families to run")
    parser.add_argument("--json", help="optional output path for the results")
    args = parser.parse_args()

    results = {}
    for mode in args.modes:
        maker, fused, ref = CASES[mode]
        for rows, input_width in SHAPES[mode]:
            name = f"{mode} rows={rows} 2H={input_width}"
            inputs = maker(rows, input_width)
            check(name, fused(*inputs), ref(*inputs))
            fused_us = graph_latency_us(fused, *inputs)
            ref_us = graph_latency_us(ref, *inputs)
            results[name] = {"fused_us": round(fused_us, 2),
                             "ref_us": round(ref_us, 2),
                             "speedup": round(ref_us / fused_us, 2)}
            print(f"{name}: fused {fused_us:.2f}us  ref {ref_us:.2f}us  "
                  f"speedup {ref_us / fused_us:.2f}x", flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"results written to {args.json}")


if __name__ == "__main__":
    main()
