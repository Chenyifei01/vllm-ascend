# dequant_situ_quant 优化记录（dequant-situ-quant-perf）

> 910B3 (dav_c220, 40 AIV, UB 192KB/核), CANN 9.1.0。
> 方法论沿用前三役（rms_norm_cast 910B_perf / add_rms_norm_bias
> add-rms-norm-bias-perf / dequant_swiglu_quant dequant-swiglu-quant-perf）：
> 事件四件套、退出标志清零契约、Gather+stride-0 广播、未动路径作噪声金丝雀。
> 分支自 2cb4aeab4（上游 main 同步点）切出，仅携带 skill 文档提交。

## 第 0 步：基线采集（2026-10-08，零代码改动）

### 0.1 算子契约与生产画像

反量化 + Situ 激活 + 动态量化三合一（Kimi K3 W4A8 的 GMM1 后处理）：

```
int32/bf16 路径（K3Kernel，本役主战场）:
  f      = fp32(x) [× weight_scale[expert,2H] × activation_scale[row] (+ bias)]
  gate   = f 左半 (activate_left=1), up = f 右半
  situ_a = β·tanh(gate/β) · σ(gate)
  up'    = linearβ·tanh(up/linearβ)     # linear_beta>0 时
  situ   = situ_a · up'
  scale  = max|situ行| / 127 (fp32, 逐行, 0 → 1)
  y      = int8( situ / scale )
```

数值链（CastFloatToInt8）：fp32 → Cast RINT → int32 → SetDeqScale(1.0)
Cast ROUND → half → Cast TRUNC → int8。与 dequant_swiglu_quant 相同，
half 中转对 |v|≤127 整数值无损。

生产调用点（vllm_ascend，全部 Kimi K3 W4A8，每层每次前向）：

| 调用方 | x | 附属输入 | 2H | 备注 |
|---|---|---|---|---|
| shared_experts.py:382 | int32 [rows, 2H] | weight_scale [2H], activation_scale [rows] | 12288/TP | quant GMM 后逐行反量化 |
| w4a8.py:476 (apply_gmm1_act_quant) | bf16 [rows, 2H] | 无（GMM 已反量化，output_dtype=bf16） | 6144 | β=1.0, linear_beta=0.0 |
| w4a8.py:476（int32 变体） | int32 [rows, 2H] | weight_scale/bias [E,2H], group_index [E] | 6144 | routed, E=14, topk=16 |

注意 bf16 生产调用 β=1.0/linear_beta=0：tanh(β⁻¹x)·β 退化为 tanh(x)，
linear_beta=0 跳过 up 的 clamp 链——pass 数比 shared 少 4 个。

### 0.2 kernel 分派（静态走读）

op_kernel/dequant_situ_quant.cpp 按 ORIG_DTYPE_X 编译期分派：

| x dtype | kernel | 列切分 | 备注 |
|---|---|---|---|
| int8 | DequantSituQuantKernel\<hasDequantBias\> | 有（列切分时动态量化走两遍重算） | **无生产调用方**，只保回归 |
| int32 | DequantSituQuantK3Kernel\<int32_t\> | 无（整行进 UB） | shared + routed 生产路径 |
| bf16 | DequantSituQuantK3Kernel\<bfloat16_t\> | 无（整行进 UB） | w4a8 生产路径 |

tiling（dequant_situ_quant_tiling.cpp CalcTiling）：int32/bf16 直接
baseRowLen=1, baseColLen=outDimy, usedCoreNum=min(rows, totalCore)；
int8 才有 UB 预算推导的列切分。**整行进 UB → H=3072(2H=6144) 时每核
UB 占用 ≈ 2H×4(dequant) + 2H×4(tmp/cast) + 2H(x) + out ≈ 76KB < 192KB，
无 UB 压力。**

### 0.3 K3 kernel 数据流走读（pass 清单，bf16 路径 H=3072）

ProcessGroup 对每行严格串行：CopyInRow → ComputeRow → CopyOutRow，
xQueue_/outQueue_ 深度均为 1 → **MTE2/V/MTE3 三流水零重叠**（与
add_rms_norm_bias 彼役的行间串行病灶同型，且更彻底）。

ComputeRow 的向量 pass 清单（H=3072）：

| # | pass | 说明 |
|---|---|---|
| 1 | Cast bf16→fp32 (2H) | 必须 |
| 2 | Mul weightScale (2H) | 仅 int32 路径 |
| 3 | Muls activationScale (2H) | **activationScaleGm_.GetValue(rowIdx) 每行标量读 GM** |
| 4 | Add bias (2H) | 仅 routed |
| 5 | Adds(sigmoid←gate, 0.0f) (H) | **纯拷贝 pass，冗余**（下一步 Muls(-1) 可直写） |
| 6-8 | Muls(gate,1/β) → Tanh → Muls(gate,β) (H) | |
| 9-11 | Muls(sigmoid,-1) → Exp → Adds(+1) (H) | |
| 12 | Duplicate(ones) (H) | **每行重做，可 init 一次** |
| 13 | Div(sigmoid, ones, sigmoid) (H) | stride-0 广播可去 ones 缓冲 |
| 14 | Mul(gate, sigmoid) (H) | |
| 15-17 | Muls/Tanh/Muls (H×3) | 仅 linear_beta>0（bf16 生产路径无） |
| 18 | Mul(gate·up) (H) | |
| 19 | Abs (H) | |
| 20 | Max 链 (~H/64 rep) + WholeReduceMax ×2 | **DynamicQuant 里 WholeReduceMax 连做两次，冗余** |
| 21 | V_S 标量往返: GetValue(0) → 1/scale | **每行一次，issue 停等** |
| 22 | Muls(situ, 1/scale) (H) | |
| 23-25 | Cast fp32→int32(RINT)→half(ROUND)→int8(TRUNC) | 必须 |

CopyOutRow：y (H 字节) + scale (4 字节) 两次 DataCopyPad 小包。

### 0.4 候选方向（待 pipe 基线定位后排序，负结果也记录）

- R1 消冗余 pass：#5 拷贝、#12 ones、#20 双重 WholeReduceMax（位级等价）
- R2 行间双缓冲：xQueue_/outQueue_ 加深 + 核内行号奇偶乒乓，MTE2/V/MTE3 重叠
- R3 消 V_S 往返：scale 链路留向量域（Gather/stride-0），Muls 的标量操作数处理
- R4 activation_scale 每行标量读 → 批量 MTE2 + Gather（int32 路径）

### 0.5 精度基线 — ✅ 全绿

`pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_dequant_situ_quant.py`
→ **16 passed**（未改动代码）。覆盖：routed 多专家 int32（含零组）、
bf16 预反量化 rows=1/16/1040、shared int32 全 TP 档、零行空输出。

### 0.6 性能基线（NPUGraph replay，5 样本取中位）

脚本：benchmarks/dequant_situ_quant.py（fused vs unfused 参照，
shared/routed/bf16 三模式）；数据见 profiling_dsq_situ_baseline/bench_baseline.json。
同口径两次运行偏差 <2%。

| shape | fused µs | ref µs | 加速比 |
|---|---|---|---|
| shared rows=1 2H=12288 | 5.83 | 36.04 | 6.19x |
| shared rows=4 2H=12288 | 6.15 | 102.20 | 16.63x |
| shared rows=16 2H=12288 | 7.97 | 134.57 | 16.88x |
| shared rows=64 2H=12288 | 15.85 | 175.13 | 11.05x |
| shared rows=128 2H=12288 | 23.91 | 218.10 | 9.12x |
| shared rows=512 2H=12288 | 59.53 | 410.71 | 6.90x |
| shared rows=1024 2H=12288 | 110.89 | 683.37 | 6.16x |
| shared rows=2048 2H=12288 | 213.63 | 1512.72 | 7.08x |
| shared rows=1 2H=6144 | 4.32 | 30.60 | 7.09x |
| shared rows=16 2H=6144 | 6.13 | 130.09 | 21.22x |
| shared rows=128 2H=6144 | 17.38 | 182.32 | 10.49x |
| shared rows=1024 2H=6144 | 69.40 | 412.64 | 5.95x |
| shared rows=4096 2H=6144 | 251.48 | 1498.16 | 5.96x |
| **routed rows=16 2H=6144** | **40.22** | 170.86 | 4.25x |
| routed rows=1040 2H=6144 | 80.70 | 494.69 | 6.13x |
| routed rows=65536 2H=6144 | 4400.47 | 46345.12 | 10.53x |
| bf16 rows=1 2H=6144 | 3.73 | 26.25 | 7.04x |
| bf16 rows=16 2H=6144 | 5.67 | 115.93 | 20.43x |
| **bf16 rows=1040 2H=6144** | **63.59** | 356.20 | 5.60x |
| bf16 rows=4096 2H=6144 | 231.74 | 1247.04 | 5.38x |

首跑信号（2026-10-08）：

- **routed rows=16 拖 40.22µs**——同行数 shared/bf16 只要 6.13/5.67µs，7 倍差。
  走读对应 0.3 的 ProcessGroup 设计：14 个 expert 串行，每组
  CopyInExpertParams 重搬 weight_scale+bias（2H×8B=48KB），每组 1 至 2 行
  使 40 核里最多 2 核干活，参数搬运时其余 38 核空转。
- rows=1 各模式 3.7 至 5.8µs：单行整链 ~25 个 pass 串行 + launch/收尾开销，
  记分板地板由发射线程决定，优化空间在 pass 数。
- 大 rows 段（bf16 4096 → 231.74µs）：x 读 50.3MB + y 写 12.6MB ≈ 63MB /
  231.74µs ≈ 272GB/s 有效搬运率，偏低，疑似主因是行间零流水（MTE2 等上一行
  V 算完才能搬下一行，三流水无法重叠）——待 0.7 pipe 基线证实。

### 0.7 pipe 基线（msprof op PipeUtilization，2026-10-08）

`--kernel-name=DequantSituQuant*`（驼峰通配；小写 extern C 名匹配不到，
见 skill/记忆）。CSV 在 profiling_dsq_situ_baseline/pipe_*/OPPROF_*/。

| shape | aiv_time/核 | vec | mte2 | mte3 | scalar | mte2 有效带宽 |
|---|---|---|---|---|---|---|
| bf16 rows=1040 | 65.1µs | 75.0% | 15.0% | 5.9% | 15.4% | 30.6GB/s |
| shared rows=1040 | 116.4µs | 78.0% | 17.1% | 3.4% | 8.5% | 62.3GB/s |
| routed rows=16 | 均值 5.4µs | 8.2% | 18.8% | 0.5% | 81.0% | 10.9GB/s |

**routed rows=16 每核分布**（40 核只起了 16 个）：

- block0：**44.16µs**（vec 67% / mte2 27% / scalar 16%）——14 个 expert
  串行压在核 0：每组 CopyInExpertParams 搬 48KB（weight_scale+bias）+ 单行
  全链计算，14 组循环完成才算完。
- block1 至 15：1.6 至 8.4µs，vec 1%、scalar 75 至 99%——只做 14 次
  group_index 标量读然后空转。

**归因（实测）**：
1. bf16/shared 大 rows：vec-bound（75 至 78%），标量占 8.5 至 15.4%（每行
   V_S 往返 + int32 路径每行 activation_scale GetValue），MTE2 仅 15 至 17%
   且被行间串行压着——R1 消 pass、R2 行间流水、R3 消标量往返各有一份收益，
   叠加预计 1.4 倍上下。
2. routed rows=16（decode 主路径）：瓶颈不是计算而是**负载分配**——
   一核独扛 44µs、39 核空转。行均摊到全核（每核至多 1 行，只搬该行
   expert 的参数）可把墙钟压到单行成本 ≈5µs，预期 8 倍量级。

### 轮次计划

- **R1 消冗余 pass**（位级等价，先做）：Adds 拷贝 pass、每行 Duplicate(ones)、
  DynamicQuant 双重 WholeReduceMax、β=1/linearβ=0 时的无效 Muls 对。
- **R2 行间双缓冲**：xQueue/outQueue 加深 + 核内行号奇偶乒乓；前后各采
  trace.json（`--aic-metrics=TimelineDetail`）验证三流水重叠。
- **R3 标量域消除**：视 R2 后 scalar 占比决定（V_S 往返可被流水掩盖则降级）。
- **R4 routed 负载再分配**：行级均摊 + 按 expert 参数按需搬运；事件/标志
  契约按 skill 910B 要点走，重点验证混合组行数边界（0 行组、组行数 < 核数）。
