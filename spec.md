# adaln_gate_residual：WS1 实现与验证规格

状态：设计稿；本次只提交规格，尚未实现或运行 GPU 验收。
分支：`feat/adaln-gate-residual`。
基线：2026-09-05 拉取的 `upstream/main`，`01b4ae410ae27aa0ff93bb48300c05158ef2968e`。

## 1. 需求与范围

已阅读 [issue #386](https://github.com/RL-Align/RL-Kernel/issues/386) 全文、两张架构图和唯一一条认领评论。
该行由 @BruceLoveDecimal 认领，要求 forward、backward，并为后续 WS2 保留一致性基础。
交付参考 [PR #204](https://github.com/RL-Align/RL-Kernel/pull/204) 的后端、binding、registry、fallback、测试、benchmark 和环境记录方式；数值标准以 #386 为准。

算子消费已计算好的子层输出 `s = sublayer(h)`，计算 `y = x + gate * s`。
覆盖一个 MMDiT block 中图像和文本各自的 attention、MLP 后四处残差连接。
不计算 LayerNorm、shift/scale、gate 投影、attention、MLP，也不把 gate 经过 sigmoid。
首版为 out-of-place，避免破坏 autograd 保存的张量；“write-back”不表示原地覆盖输入。
WS2 通信、完整 block 组装、60 层误差曲线和 SDE/logp 是后续整体工作，不作为本算子已经完成的成果。

调用位置依据：[Diffusers QwenImageTransformerBlock](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage.py)，2026-09-05 阅读。
普通 gate 为 `[B,1,D]`，条件选择路径也可能提供 `[B,S,D]`；实现验证时须固定该源码 revision，并记录在报告中。
本规格数值约定优先于原始低精度 PyTorch 表达式可能产生的中间舍入。

## 2. 接口与输入契约

建议公共入口位于 `rl_engine/kernels/adaln_gate_residual.py`：

```python
adaln_gate_residual(x, gate, sublayer_out, *, backend="auto", trace=None) -> Tensor
```

`KernelRegistry.get_op("adaln_gate_residual")` 提供默认后端；三个后端采用相同参数命名和 `forward` 接口。
显式 `backend="cuda" | "triton" | "pytorch"` 用于验证和严格选择，不能悄悄改用别的实现。

| 项目 | v1 契约 |
| --- | --- |
| x、sublayer_out | 相同 `[B,S,D]`，相同设备、相同 dtype |
| gate | `[B,D]` 或 `[B,1,D]`：仅沿 S 广播；`[B,S,D]`：逐元素 |
| dtype | x/s 为 FP32 或 BF16；gate 可与 x 相同或为 FP32 |
| 输出 | `[B,S,D]`，x.dtype，连续存储，独立分配 |
| 梯度 | dx、ds、dgate 分别恢复对应输入的 shape/dtype |
| 尺寸 | B、D 必须正数；S=0 返回空输出及零 dgate |
| 布局 | 接受合法 strided view；包装层按需 contiguous，记录拷贝 |
| 不支持 | FP16/FP64 生产调用、跨设备、任意广播、标量 gate、out/in-place 参数 |

特别测试 `chunk` 得到的非连续 gate、切片 storage offset、transpose、expand 的只读输入和非连续 dy。
规范化不能切断 autograd；底层 CUDA binding 仍独立校验 shape、dtype、device、layout。
`[B,1,D]` 在 S=1 时统一按广播模式解释，两种算法在此处也必须一致。
输入相互 alias 时，本算子返回各输入槽位的局部 VJP；共享叶子的额外累加见第 4 节。
二阶梯度不属于 v1；用明确的 once-differentiable 行为避免给出错误结果。

## 3. 冻结数值契约

契约 ID：`adaln_gate_residual.fp32_separate_mul_add.chunk256_tree.v1`。
以下是设计选择，需要三后端共同实现，不能靠调大 tolerance 代替。

### 3.1 前向及逐元素反向

`rn32` 表示 IEEE FP32 round-to-nearest-even。所有输入先转 FP32：

```text
p = rn32(gate * s)
y32 = rn32(x + p)             # 禁止将这两步融合成 FMA
y = cast_to_x_dtype(y32)

dx = dy                      # 直接传递；不经过数值求和
ds32 = rn32(dy32 * gate32)
q = rn32(dy32 * s32)
```

逐元素 gate 的 dgate 为 q；广播 gate 的 dgate 使用下一节归约。
ds/dgate 只在最终写回各自输出时转换一次到输入 dtype。
中间乘积和归约 partial 始终 FP32，不落 BF16；不使用 fast-math、TF32、Split-K、Stream-K 或 atomic accumulation。
不允许编译器重关联。CUDA 用显式 `__fmul_rn`、`__fadd_rn`，Triton 关闭 FP fusion，并检查生成代码。
不复用全局 `--use_fast_math` 构建结果作为严格后端：若现有构建开关开启，则拒绝构建该严格算子并给出原因，不能只在 trace 写 false。
保持 FP32 subnormal，不开启 FTZ；对对应编译参数和 subnormal 用例进行验收。

FP32 输入/输出与 CPU FP32 oracle 比较原始位；BF16 输出与 CPU FP32 oracle 最后转换一次后的 BF16 位比较。
混合 dtype 同理，不能拿 BF16 输出和未转换的 FP32 输出要求 byte-equal。
有限值（含 signed zero/subnormal）为 strict corpus；NaN/Inf 单独测试传播和安全性，NaN payload 不作跨后端逐位承诺。
有限输入溢出导致 Inf 的用例也单独记录，不混入“所有有限结果通过”的统计。

### 3.2 广播 dgate 的固定归约

数学意义为 `dgate[b,d] = sum_s q[b,s,d]`，但不能直接用框架 `sum` 或无约束 `tl.sum` 定义逐位 oracle。

固定 token chunk 大小 C=256，chunk 边界由原始样本内 token 索引决定，与 B、SM 数、grid、warp 数无关：

1. chunk j 读取 `q[b,256*j : 256*(j+1),d]`；尾部缺项填 FP32 `+0`。
2. 在 256 个叶子上按相邻配对归约：`(0+1),(2+3),...`，下一层继续相邻配对，共 8 层，每次 rn32。
3. 得到 FP32 `partial[b,j,d]`，每个位置只有一个 writer。
4. `acc=+0`，按 j 从小到大执行 `acc=rn32(acc+partial[b,j,d])`，最后才 cast 到 gate.dtype。
5. S=0 时不启动空 grid，返回正零 dgate。

CPU oracle 显式切片、逐层相加和按 j 循环；可并行处理 b/d，不能并行重排归约轴。
CUDA 与 Triton 按相同 DAG 实现；测试用 trace 中的 schedule ID 和源码/编译代码核对实际算法。
两阶段 partial 是确定性归约缓冲区，不是 GEMM Split-K；不使用原子合并。
优化可以改变一次处理多少 d、CTA 分配和执行次序，不能改变上述算术 DAG。
若以后修改 C 或跨 chunk 合并规则，必须升级契约 ID 并重新验收，不能运行时 autotune 归约树。

## 4. Autograd 与实现路线

三个后端均需可训练；广播 dgate 必须使用自定义 backward，不能让 PyTorch 自动广播反向选择 sum 算法。
CPU 固定顺序参考同时提供显式 `reference_forward_fp32` 和 `reference_backward_fp32`，供独立测试直接调用。
PyTorch 生产 fallback 可在输入所在设备执行显式 FP32 运算序列；CPU oracle 不允许由 candidate 的 backward 反过来生成。

CUDA：通用 NVIDIA kernel，无需 Hopper/TMA 专属指令。前向融合一次读取和输出；反向先做 ds 与 dgate partial，再启动最终 dgate 合并。
逐元素 gate 不需要归约；根据 `needs_input_grad` 跳过不需要的输出/launch。
保存 s 与 gate（按实际所需），不保存完整 FP32 x、y 或 q；q 在生成 partial 时计算。
FP32 partial 内存为 `4 * B * ceil(S/256) * D` 字节，不物化整块 FP32 q。
CUDA 使用当前设备 guard 和当前 stream，检查 launch error，不额外全局同步。
Triton 实现相同 forward/backward 和固定树，不把同一 warp 的默认 reduction 视为已证明相同。

本算子返回的 dx 是残差直连贡献，ds 交给上游 sublayer backward。
当 x 也是 h 的祖先时，完整 dx 还包含子层路径贡献。外部 autograd 图的任意多路累加不在局部 kernel 控制范围。
集成 fixture 明确按“直连贡献 + 子层贡献”一次 FP32 相加后 cast；多于两路按命名顺序左折叠。
共享 gate、x/s alias 和重复调用的外部梯度合并必须区分局部 VJP 的逐位保证与整图累加保证，不能宣称此算子自动解决任意图的确定性。

## 5. Registry、fallback 和 trace

基线已包含 #204 的 Native/Triton/CUDA 路径和 semantic descriptor；复用现有注册机制，不重构无关算子。

- 新增 `PYTORCH_ADALN_GATE_RESIDUAL`、`TRITON_ADALN_GATE_RESIDUAL`、`CUDA_ADALN_GATE_RESIDUAL`。
- CUDA auto 优先 CUDA，再 Triton，再 PyTorch；必须探测扩展 forward/backward 符号、设备能力、构建契约。
- CPU 默认 PyTorch；ROCm/NPU 首版仅提供可运行且被验证的 PyTorch 路径，不能标注未测设备为 strict-certified。
- 无扩展或无 Triton 时包仍可导入；auto fallback 输出原因日志并写入 trace。
- 显式后端不可用时抛错；非法输入直接抛错。CUDA 执行错误不得捕获后重算来伪装成功。
- 实际设备检查基于输入，不能只依赖 registry 的进程级缓存。缓存后换设备、缺失 backward 符号也要覆盖。

trace 为每次调用/验证报告的可序列化记录，不用不安全的全局 last_trace。
至少包含：op、contract ID、实际 fwd/bwd backend、请求 backend、fallback 原因、shape、stride、dtype、gate 模式、layout copy、设备/SM、launch 配置、归约树和 C、FP32 accumulator、FMA/FTZ/fast-math/TF32 状态、Split-K/Stream-K/atomics 状态、源码和构建 fingerprint、工具链版本、tolerance profile（严格模式为 none）。
fingerprint 应来自源码及有效构建/JIT 参数的 hash，不能只写类名或固定版本字符串。
同一数据集默认要求 CPU↔CUDA、CUDA↔Triton/PyTorch 逐位一致；硬件差异若确实无法满足，只能显式登记限定硬件、dtype、fwd/bwd 的 tolerance profile，并单独报告，不能算 strict pass。

## 6. 文件落点

以下新增路径是实现计划，现阶段并不存在：

| 文件/目录 | 内容 |
| --- | --- |
| `rl_engine/kernels/adaln_gate_residual.py` | 公共入口、输入契约、trace |
| `rl_engine/kernels/ops/{pytorch,triton,cuda}/modulation/adaln_gate_residual.py` | 三后端及 autograd；增加包初始化文件 |
| `csrc/cuda/adaln_gate_residual.cu` | CUDA forward、pointwise backward、固定归约 |
| `csrc/ops.cpp`、`rl_engine/_C.pyi`、`setup.py` | forward/backward binding、stub、构建与平台 gating |
| `rl_engine/kernels/registry.py` | 默认选择和后端注册；按现有 schema 补充描述 |
| `rl_engine/testing/adaln_gate_residual_reference.py` | CPU 显式 FP32 oracle |
| `tests/test_adaln_gate_residual.py` | 正确性、边界、VJP、autograd |
| `tests/test_adaln_gate_residual_invariance.py` | batch/launch 不变性、原始位比较 |
| `tests/test_adaln_gate_residual_dispatch.py` | 缺依赖、显式后端、trace、错误路径 |
| `scripts/validate_adaln_gate_residual.py` | strict matrix、环境信息、JSON 报告 |
| `benchmarks/benchmark_adaln_gate_residual.py` | 三后端性能与内存 |
| `rl_engine/kernels/gtest/operator_{specs,inputs}.py` | 通用验证工具接入；不能替代专门 bit harness |
| `docs/operators/adaln-gate-residual.md`、`docs/.nav.yml`、`docs/operators/README.md` | 使用、数值契约、复现结果、导航 |

## 7. 验证方案：合并的主要门槛

### 7.1 独立 oracle 与 bit harness

固定 CPU RNG seed 生成输入，先量化到测试 dtype，再把同一份字节复制到各设备；上游 dy 也由 CPU 固定生成。
分别比较 y、dx、ds、dgate，不用 `output.sum().backward()` 作为唯一反向测试。
直接调用 VJP 传相同 dy，覆盖随机 dy、ones、zeros；逐项测试三种输入的 requires_grad 组合。
比较 shape/dtype 后，将 contiguous CPU 张量 view 为整数/uint8 比较原始位；`torch.equal` 对 signed zero 的数值相等不足以证明 byte-equal。
失败报告包含首个坐标、原始 hex bits、mismatch 数、最大绝对误差、seed、输入 hash、后端、契约和环境；非零退出。
CPU oracle 使用独立小尺寸手算向量验证。另用纯 FP64 数学表达式做小规模导数/有限差分 sanity check，区别于 FP32 bit oracle，不能向生产入口强塞 FP64 gradcheck。

### 7.2 覆盖矩阵

| 维度 | 必须覆盖 |
| --- | --- |
| 后端 | CPU oracle、PyTorch、CUDA、Triton；显式选择并断言实际执行路径 |
| dtype | 全 FP32、全 BF16、BF16 x/s + FP32 gate |
| B | 1、2、4、7；大 S 完整矩阵至少 B=1、2、4，其余可小尺寸 |
| S | 0、1、2、31、255、256、257、511、512、513；真实图像/文本长度 |
| D | 1、31、32、33、127、128、129、3071、3072、3073 |
| gate | 三种 shape；0、1、负值、随机、非常小/大有限值 |
| 数值 | 抵消、BF16 halfway 舍入、FMA 敏感三元组、signed zero、subnormal、归约顺序敏感 q |
| 布局 | contiguous、chunk gate、非零 offset、transpose、expand view、非连续 dy |
| 错误 | 不匹配 shape/device、非法广播/dtype、缺符号、无依赖、非法显式后端 |

小尺寸覆盖上述组合及 pairwise；真实尺寸固定 D=3072，不能仅用缩小形状代替。
由 issue 图的 VAE 8 倍下采样与 2×2 patch 推导 `S_img=(H/16)*(W/16)`：

| 图像分辨率 | token 网格 | S_img |
| --- | --- | --- |
| 1024×1024 | 64×64 | 4096 |
| 1328×1328 | 83×83 | 6889 |
| 1664×928 | 104×58 | 6032 |

文本 S 使用 1、77、256、512 和一组真实捕获长度，记录 tokenizer/model revision。
6889 和 6032 特别覆盖非整 chunk 尾部。验证逐后端、逐 case 执行以控制内存，OOM 应记为未完成而非通过。

### 7.3 Batch、token 与 launch invariance

1. 保存目标样本的 x/s/gate/dy，单独运行；放到 B=2/4/7 中不同位置，旁边填不同样本；目标 y/dx/ds/dgate 全部 byte-equal。
2. 重排 batch、拆成 microbatch 再拼回，保持目标输入和 dy 原始位不变；禁止引入 batch mean loss 缩放来改变 dy。
3. 对逐元素输出 y/dx/ds，改变 S、目标 token 位置或 token 分块，保持目标 token 和 gate 相同，结果位相同。
4. 广播 dgate 依赖该样本的所有 token：修改/删除有效 token 后不要求梯度相同。不能把它误当成逐行输出。
5. 验证跨 launch 的 token 分块时，必须保留原始全局 token offset、256 边界和 partial 合并顺序。独立 micro-sequence 求 dgate 再框架 sum 不提供相同保证。
6. 仅在 256 chunk 边界追加完整的零贡献 chunk，验证非零有限 dgate 不变；signed-zero 和非整 chunk padding 另按固定 oracle 检查。
7. 测试专用 launch 参数改变 CTA 调度/grid cap/feature tile（例如 CUDA 128/256 threads、Triton 4/8 warps）；算术 DAG 不变。逐后端至少两种配置，不能只重复同一 launch。
8. 单进程重复 100 次、独立进程/JIT 冷启动至少 3 次；非默认 stream。第二种 NVIDIA SM 上重复同一 corpus，记录指纹和结果。

### 7.4 反向、集成和安全

- CPU/CUDA/Triton 的所有局部梯度对同一个固定 FP32 VJP oracle，分别逐位比较。
- gate=0 时 ds 为零但 dgate 通常不为零；冻结 gate 时仍要得到正确 dx/ds。
- 输入数据与 version counter 未被原地修改；梯度 shape/dtype、保存张量生命周期、重复 forward/backward 正确。
- 两级 gated residual fixture 覆盖 attention/MLP 两处调用；再分别模拟 text/image 两流。
- 小型冻结 base + 可训练 LoRA A/B fixture 检查 `base.grad is None`、dA/dB finite/nonzero；采用非零初始化避免 B=0 时 dA 合理为零的误判。
- 局部 VJP 做 bit 检查；完整小图先做数学正确性检查，固定多路合并的 fixture 单独做 bit 检查。不得把普通外部 GEMM 差异归咎于本算子。
- CUDA 尾部/offset/非默认 stream 用例跑 compute-sanitizer memcheck；若使用共享内存归约，再跑 racecheck/synccheck。

### 7.5 验证环境与执行命令

需要 Linux + NVIDIA CUDA GPU；建议先 SM80/SM86 通用设备，再用 SM90 作跨 SM 验证。具体 Python、PyTorch、Triton、CUDA toolkit、driver、host compiler 版本以实际跑通记录锁定，当前不虚构版本或通过结果。
报告保存 git SHA、dirty 状态、GPU/SM/显存、OS、版本、完整构建参数、FTZ/FMA 状态、seed、数据 hash、pass/fail/skip、trace、耗时和峰值内存。
CPU 单测可以先跑；没有 GPU 的机器只证明 CPU/dispatch，不证明 CUDA/Triton 已验收。

下列为实现后的计划 CLI，新增参数/脚本须随实现交付：

```bash
python -m pytest tests/test_adaln_gate_residual.py tests/test_adaln_gate_residual_dispatch.py -q -rs
python -m pytest tests/test_adaln_gate_residual_invariance.py -q -rs
python scripts/validate_adaln_gate_residual.py --backend all --suite strict --require-gpu --output artifacts/adaln-strict.json
python benchmarks/benchmark_adaln_gate_residual.py --backend all --suite qwen-image --output artifacts/adaln-benchmark.json
python -m pytest tests/test_kernel_registry.py rl_engine/tests/test_dispatch.py -q
mkdocs build --strict -f mkdocs.yaml
```

`--require-gpu` 缺任一 GPU 后端必须失败，防止全 skip 变成验收通过。strict suite 包括所有三种真实分辨率、fwd/bwd、batch/launch、trace，结果按子项列出。
现有 gtest 的 tolerance 检查用于回归兼容性，不能替代上述 byte harness。

## 8. Benchmark 与验收

先正确性后性能。比较 CPU oracle 之外的 PyTorch 固定顺序参考、Triton、CUDA；额外列出普通 eager `x + gate*s` 作为用户基线，并标明其低精度中间舍入/反向归约可能不满足本契约。
GPU 内用 CUDA events，预热至少 20 次、正式至少 100 次，排除编译；报告中位数、p95、forward、backward、fwd+bwd 及额外峰值显存。
分开报告公共入口含 layout copy 的时间与 kernel 时间；backward 单测预先准备 saved tensors/dy，不把图构建或累积旧 grad 算进去。
覆盖三种真实分辨率、B=1/2/4、D=3072、两种主 dtype、广播/逐元素 gate，以及小文本 case。
计算有效带宽时公开字节模型，包含 dgate partial 读写；不预先承诺加速比。若退化，定位固定归约开销并给出数据，不能为提速改变舍入树。

本算子 PR 的完成清单：

- [ ] 三后端 fwd/bwd、binding、stub、构建、registry 与可观测 fallback 齐全。
- [ ] FP32 CPU oracle 和 byte harness 独立交付；y/dx/ds/dgate 分项通过。
- [ ] 三种分辨率及 chunk 边界、混合 dtype、实际 gate view 有明确测试。
- [ ] 固定归约、单次最终 cast、禁止 FMA/fast-math/FTZ/atomics 有数值用例及编译证据。
- [ ] batch/launch invariance 通过；token 变化对 dgate 的语义限制有记录。
- [ ] trace 包含实际后端和构建指纹；不能用 fallback 冒充 CUDA/Triton pass。
- [ ] 小图训练梯度与内存/stream 安全检查通过。
- [ ] benchmark、真实环境、机器可读报告、复现命令、文档及相关回归检查齐全。

实施顺序：先固定 oracle/bit harness 和 CPU API，再 CUDA，随后 Triton，最后完成注册、集成、真实尺寸及跨 SM 验证、性能和文档。
当前交付仅此设计稿，上述复选框均保留未完成状态。
