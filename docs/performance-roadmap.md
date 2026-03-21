# FerrisInfer 性能、可用性与平台兼容路线图

## 1. 当前状态判断

FerrisInfer 现在已经具备了一个清晰且可继续演进的最小推理主线：

- Host 侧 `Tensor / Shape / Layout / Model / IO / Tokenizer` 已经成型。
- Runtime 已经明确区分 `prefill` 与 `decode`。
- KV cache 已经接入 session，并且支持多轮对话。
- CPU 路径已经开始做面向真实模型的性能优化，而不只是 reference 正确性实现。
- backend 抽象和 GPU 占位模块已经存在，说明从一开始就没有把系统锁死在 CPU-only。

这说明 FerrisInfer 已经跨过了“能不能跑”的阶段，接下来真正要解决的是：

- 怎么从单 session、单请求、contiguous KV cache 的推理器，演进成真正适合长期优化的高性能运行时。
- 怎么在不破坏现有纯 Rust 主线的前提下，把 CPU 与未来 CUDA 路线统一到同一套 runtime 设计里。

## 1.1 核心目标重申

FerrisInfer 的长期目标不绑定任何单一外部项目，而是稳定推进三条主线：

- 性能
  让 CPU 到 GPU 的推理路径持续优化，覆盖单轮、多轮、批处理与未来多请求场景。
- 高可用
  保证 reference path、optimized path、未来 GPU path 可以长期共存；在资源不足、后端不可用、模型不完全适配时，系统应优雅降级而不是直接失效。
- 平台兼容
  以 Rust 为中心，优先保证 CPU 基线路径在 Windows、Linux、macOS 上稳定可跑，再逐步扩展到 CUDA、Metal、Vulkan、WebGPU 等设备后端。

因此后续所有优化都要同时回答三个问题：

- 是否更快。
- 是否更稳。
- 是否不会把跨平台兼容性做坏。

## 2. 与高性能推理系统的主要差距

如果目标是做一个高性能、高可用、跨平台兼容的 Rust 推理框架，那么瓶颈不只是 kernel，更是运行时结构。vLLM 只是一个参考样本，不是唯一目标。

FerrisInfer 目前和成熟高性能推理系统相比，关键差距主要在以下几层：

- 请求调度层
  现在主要是单 session、单对话顺序执行，还没有 request scheduler、continuous batching、step-level 调度。
- KV cache 管理层
  当前是连续布局 KV cache，适合单会话正确性与基础性能，但不适合高并发复用、前缀共享与细粒度回收。
- 批处理执行层
  当前 prefill / decode 已分开，但还没有“多请求共同 prefill / 共同 decode / 动态插队”的执行器。
- 权重与激活表示层
  现在以 `f32` 为主，后续要进入高性能阶段，必须系统性引入 `f16 / bf16 / int8 / int4` 的数据通路与 kernel。
- 设备执行层
  GPU backend 还只是占位，没有 device memory、stream、event、kernel module、graph capture 等基础设施。
- 性能工程层
  已有 profile 命令，但还没有稳定的 benchmark 套件、热点拆解统计、回归阈值和专门面向多轮/多请求的性能基线。

所以路线判断很明确：

- 第一步不是“立刻上 CUDA”。
- 第一步应该是先把 runtime 与 KV cache 设计推进到能容纳高吞吐系统。
- CUDA 是第二阶段放大器，不是第一阶段替代品。

## 3. 总体演进原则

后续优化建议坚持五条原则。

### 3.1 正确性路径与高性能路径长期共存

必须保留：

- reference path
- optimized CPU path
- future CUDA path

reference path 继续承担：

- 正确性对照
- 小模型验证
- 回归测试基线

optimized path 才承担：

- 真实模型性能
- 多轮延迟
- 吞吐优化

不要让高性能实现吞掉 reference 路径，否则后面做 paged KV、量化、CUDA 时会很难定位错误。

### 3.2 先抽象运行时对象，再写复杂 kernel

高性能系统最值钱的不是单个 matmul kernel，而是这些对象：

- request
- sequence
- scheduler
- block manager
- KV page / block table
- execution batch

这些对象不先稳定，后面无论 CPU 还是 CUDA 都会重写。

### 3.3 Prefill 和 Decode 必须继续走不同优化路线

- prefill 追求吞吐
- decode 追求单步低延迟

这件事后续要继续强化到：

- 不同 batch 组织方式
- 不同 workspace 策略
- 不同调度策略
- 不同 kernel 组合

### 3.4 CPU 先把 runtime 结构跑顺，再上 CUDA

推荐顺序：

1. 在 CPU 上先实现 request scheduler、chunked prefill、paged KV cache 的 reference/optimized 版本。
2. 等运行时结构稳定后，再把同一套对象映射到 CUDA。

这样做的好处是：

- 先把系统复杂度解决一半。
- CUDA 阶段只需要解决“如何把同样的执行计划搬到设备上”。
- 不会把“架构不稳定”和“设备调试困难”叠在一起。

### 3.5 每个阶段都必须有可测量出口

后续每个阶段都应该有明确退出条件，例如：

- 多轮对话第二轮 `sync` 降低多少
- 4/8/16 并发请求吞吐提升多少
- paged KV cache 是否支持前缀共享
- CUDA decode 是否已经快于当前 CPU optimized path

没有可量化出口的优化，最后都会变成持续返工。

## 4. 目标架构

长期建议把 FerrisInfer 的运行时演进成下面这条主线：

1. `ModelRepository`
   负责 host 权重、配置、tokenizer 与 weight layout 元数据。
2. `ExecutorBackend`
   负责 CPU / CUDA 的设备资源、kernel dispatch、memory pool。
3. `RequestScheduler`
   负责接收新请求、切分 prefill / decode 工作、组织执行批次。
4. `SequenceManager`
   负责每条序列的状态、采样、停止条件和 block table。
5. `KvBlockManager`
   负责 KV block/page 分配、引用计数、共享、回收。
6. `ExecutionBatch`
   负责一次 prefill 或 decode tick 的实际执行单元。
7. `Sampler / Streamer`
   负责输出 token、文本增量解码和会话回传。

今天的 `Session` 可以视作未来 `SequenceManager + 单请求 scheduler` 的最小前身。

## 5. 建议分阶段路线

## Phase 1：CPU 高性能基础巩固

目标：让单请求、多轮对话、单 batch CPU 路径达到“结构稳定 + 可持续测量”。

优先项：

- 统一 prefill / append-prefill / decode 的 workspace 生命周期
- 完善 profile 命令，增加更稳定的 benchmark case
- 引入多轮对话基准
- 引入 batch size、prompt length、decode length 三维性能统计
- 继续推进 matmul blocking、attention 热点和复制路径削减

退出条件：

- 单轮与多轮 benchmark 稳定可复现
- 能明确区分首轮 prefill、追加 prompt prefill、decode 的瓶颈

## Phase 2：Runtime 结构升级

目标：从单 session 推理器，演进成具备调度潜力的 runtime。

建议先引入以下对象，但一开始可以只在 CPU 单线程/单请求模式下工作：

- `RequestId`
- `SequenceId`
- `SequenceState`
- `BatchKind`
- `SchedulerTick`
- `SchedulerOutput`

这一步最重要的不是并发，而是把“执行单位”从 `Session` 提升到“批次”和“序列”。

退出条件：

- CLI 虽然仍然看起来像单会话，但 runtime 内部已经可以表达多个 sequence。
- prefill / decode 的组织单位不再直接绑死在单个 session 上。

## Phase 3：KV Cache 抽象升级

目标：为 paged KV cache 做结构准备。

建议把当前 KV cache 进一步拆成两层：

- `KvCacheStorage`
  定义读写接口，不关心底层是 contiguous 还是 paged。
- `ContiguousKvCacheStorage`
  当前实现，继续承担 reference / early optimized 路径。
- `PagedKvCacheStorage`
  未来实现，面向 block/page 组织。

同时引入：

- `KvBlockId`
- `KvBlock`
- `BlockTable`
- `PrefixHandle`

先在 CPU 路径把 block table 跑通，再决定 CUDA 内核接口。

退出条件：

- contiguous 与 paged 两种 storage 可以共用同一套上层 runtime 接口。
- 支持 prefix sharing / block reuse 的最小原型。

## Phase 4：Continuous Batching

目标：学习成熟高性能推理系统最关键的运行时能力。

建议实现顺序：

1. 先做 CPU reference continuous batching。
2. 再做 CPU optimized continuous batching。
3. 最后再把同样的批处理抽象映射到 CUDA。

需要引入的能力：

- step-level scheduler
- 新请求在 decode 周期插入
- chunked prefill
- 按 token block 管理的 KV cache
- batch 内每个 sequence 的独立停止条件

退出条件：

- 多请求吞吐明显高于串行执行
- 相同前缀请求可以复用已有 KV block

## Phase 5：量化与权重布局优化

目标：解决内存带宽和大模型常驻问题。

建议顺序：

- 先做 `f16 / bf16` 权重与激活通路
- 再做 `int8` 权重
- 再做 `int4 / groupwise quantization`
- 最后做 fused dequant matmul

这里要注意：

- 量化不是单独 feature，而是要贯穿 CPU 与 CUDA backend
- 权重布局重排应由 `io/model/backend prepare` 三层协作完成

退出条件：

- 权重 dtype 与执行 dtype 正式解耦
- backend 能显式声明支持哪些量化 kernel

## Phase 6：CUDA Foundation

目标：不是马上做最快 kernel，而是先把 NVIDIA 后端的系统基础设施铺平。

建议拆成六个子模块：

- `CudaDriver`
  最小化封装 device / context / module / stream / event
- `CudaBuffer`
  host <-> device buffer 生命周期与上传下载
- `CudaMemoryPool`
  统一 scratch / activation / KV page 分配
- `CudaKernelRegistry`
  kernel module 装载与参数派发
- `CudaExecutionContext`
  runtime 执行时持有 stream、临时 buffer、graph capture 上下文
- `CudaProfilerHooks`
  记录每次 dispatch、同步和内存拷贝

这一阶段先不追求“比 CPU 快很多”，先追求：

- 生命周期正确
- 错误边界清晰
- 能跑通最小端到端链路

当前这条线已经完成了第一批基础设施：

- driver 动态探测与设备枚举
- primary context retain/release
- device buffer 分配、释放与 host <-> device copy
- CLI 级 probe / smoke 验证

下一步就应该把这层原始 buffer 往上推进到真正可承载 runtime 数据结构的 tensor storage，再开始最小 kernel 路径。

退出条件：

- CUDA backend 能跑通最小 decoder-only 单轮生成
- CPU / CUDA 共享同一套 runtime 主线

## Phase 7：CUDA Kernel 与高性能调度

目标：开始真正往高性能推进。

优先顺序建议是：

1. RMSNorm
2. RoPE
3. Embedding gather
4. Matmul
5. Attention
6. Fused residual + MLP
7. Sampling on device

真正的大头是这几件事：

- paged attention
- fused dequant matmul
- decode 小 batch 低延迟 kernel
- prefill 大 batch 高吞吐 kernel
- graph capture
- stream overlap

退出条件：

- decode 延迟明显优于当前 CPU optimized path
- 多请求吞吐随 batch 增长保持合理提升

## 6. NVIDIA 支持的实现原则

FerrisInfer 的项目原则决定了 NVIDIA 路线不能简单照搬外部框架。

建议坚持以下边界：

- 不依赖第三方深度学习库或线性代数库
- 数学 kernel 由 FerrisInfer 自己定义和实现
- 可以接受把 CUDA driver/runtime 只作为“设备接口”来调用
- 不能把核心推理逻辑外包给 cuBLAS / TensorRT 之类库，否则会偏离项目目标

也就是说，允许的依赖边界应该是：

- 把 CUDA 当作设备接口
- 不把 CUDA 当作推理逻辑实现者

这点需要在项目里长期坚持，否则路线会很快从“从零构建推理框架”滑向“给外部推理库写壳”。

## 6.1 高可用与平台兼容的工程原则

为了避免项目后期变成“某个平台上偶尔跑得很快”的实验代码，建议把下面这些原则固定下来。

- 能力探测优先于硬编码
  backend 是否可用、支持哪些 dtype、是否支持 graph capture、量化 kernel 是否存在，都应在运行时探测，而不是写死假设。
- fallback 优先于失败
  CUDA 不可用时自动退回 CPU；高级优化不可用时退回基础实现；新 storage 未稳定时可以继续走 contiguous KV cache。
- reference 路径长期保留
  任何高性能后端都要有 reference 对照和最小回归集，确保 bug 可以定位。
- 平台测试矩阵前置
  Windows、Linux、macOS 的 CPU 路径要持续测试；GPU 路径则按平台能力逐步接入。
- 性能与稳定性分开验收
  “更快”不等于“可上线”；任何优化都需要同时看正确性、稳定性、资源占用和性能收益。
- 资源压力下优雅降级
  当显存或内存不足时，优先选择缩小 batch、退回低级路径、禁用高级优化，而不是让整个进程崩溃。
## 7. 现阶段最值得立刻做的三件事

如果只看接下来最值当的推进顺序，我建议按下面三件事来。

### 7.1 增加更系统的 benchmark / profiling 基线

先补这些能力：

- 单轮 prefill benchmark
- 多轮追加 prompt benchmark
- 多请求串行 benchmark
- 多请求模拟 continuous batching benchmark
- 关键阶段耗时拆分：tokenize / prefill / append-prefill / decode / sample

这是后面所有优化的地基。

### 7.2 抽象 KV cache storage 与 block table

这是迈向高吞吐推理运行时最关键的一步。

理由很简单：

- 没有 block/page abstraction，就没有 paged KV cache
- 没有 paged KV cache，就没有真正高效的 continuous batching 和 prefix sharing

### 7.3 把 runtime 从 Session 升级为 Scheduler + Sequence

先不需要并发，也不需要异步。

只要先把内部对象改成：

- scheduler 驱动执行
- sequence 持有状态
- batch 作为执行单元

后面 CPU/GPU 两条线都会轻松很多。

## 8. 不建议现在立刻做的事

下面这些事现在不建议抢跑。

- 直接做多 GPU
- 直接做 tensor parallel
- 直接做 speculative decoding
- 直接做复杂 CUDA fused kernel
- 在运行时结构还没稳定前就引入大规模量化分支

这些都很重要，但都建立在 runtime、KV cache、batching 先稳定的前提上。

## 9. 推荐的下一阶段落地顺序

建议下一轮实际编码按这个顺序推进：

1. 已完成：做 benchmark / profiling 基建增强，并把稳定性指标一起纳入基线
2. 已完成：引入 `KvCacheStorage` 抽象
3. 已完成：让当前 contiguous KV cache 成为第一种 storage 实现
4. 已完成：引入 `Request / Sequence / SchedulerTick` 运行时对象，并补上最小 reference scheduler
5. 已完成：在 CPU 上做最小 paged KV cache 原型
6. 已完成：在 CPU 上做最小 continuous batching 原型
7. 已完成：做 chunked prefill reference path，并把它接入 continuous batching 基准与可观测输出
8. 已完成：做 execution batch compaction reference heuristic，包括 decode fairness 与 token-budgeted prefill batching
9. 已完成：把 paged KV prototype 推进到 block table / prefix sharing 的最小原型，并打通 prefix handle / prefix copy 的底层接口
10. 已完成：把 prefix-aware scheduler reference heuristic 做起来，支持 shared-prefix prompt reuse 与 reused token profiling
11. 已完成：引入 PrefixIndex 与元数据级 PrefixBlockManager，并把 scheduler 切到 prefix-index 查询路径
12. 已完成：为 NVIDIA CUDA 引入 driver 动态探测、设备枚举、context / device buffer / host-device copy 基础设施，并接入 CLI 可观测输出与 smoke 验证
13. 已完成：把 CUDA 基础设施从 driver probe 推进到可实际 retain context、分配 device buffer、执行 host-device 往返拷贝的最小 runtime base
14. 下一步：把 block manager / prefix index 从元数据推进到真实 page refcount / page sharing，同时继续把 CUDA 从原始 buffer 推进到 tensor storage 与第一批基础 kernel（zero/fill/copy -> matmul / attention），保持 CPU 跨平台基线不回退

这是当前最稳、也最接近长期目标的路线。













