# FerrisInfer 架构设计

## 1. 设计目标

FerrisInfer 的目标不是做一个“先跑起来再说”的短期 Demo，而是一条长期可推进的纯 Rust LLM 推理路线：

- 从 reference 实现出发，先验证原理与正确性。
- 在 CPU 上逐步演进到高性能，包括 cache-aware、SIMD、多线程与量化。
- 在不推翻现有抽象的前提下，继续扩展到 GPU 后端。

因此首版架构必须同时满足三个要求：

- 现在能落地：支持最小可验证的推理框架骨架。
- 中期能优化：不给未来高性能实现制造结构性阻碍。
- 长期能扩展：CPU 到 GPU 走同一条运行时主线，而不是两个彼此割裂的系统。

## 2. 总体分层

workspace 采用六层结构：

- `ferrisinfer-core`
  主机侧基础抽象：`DType`、`Shape`、`Layout`、`Tensor`、错误类型、执行配置。
- `ferrisinfer-kernel`
  后端与算子层：CPU kernels、后端能力描述，以及未来 GPU backend 的统一入口。
- `ferrisinfer-model`
  模型结构层：`ModelConfig`、架构规格、权重命名约定、decoder-only 模型定义。
- `ferrisinfer-io`
  资产读取层：模型格式、权重映射、tokenizer 资产。
- `ferrisinfer-runtime`
  运行时层：session、KV cache、prefill/decode 生命周期、采样器、执行计划。
- `ferrisinfer-cli`
  命令行入口：调试、验证、基准与后续 demo。

分层原则如下：

- `core/model/io` 负责“数据是什么”。
- `kernel/runtime` 负责“数据怎么跑”。
- `cli` 只负责“如何触发”。

## 3. Host 与 Device 分离

为了支持未来 GPU，FerrisInfer 从第一天起就区分两类对象：

- Host-side assets
  由 `core + model + io` 管理，负责权重、配置、tokenizer、shape 和 layout 的规范化表达。
- Device-side execution resources
  由 `kernel + runtime` 管理，负责把 host 资产转化为 CPU/GPU 可执行资源，并调度算子执行。

这意味着：

- `core::Tensor` 主要表示主机侧规范张量。
- CPU 后端可以直接消费主机侧张量。
- GPU 后端未来通过上传、重排、缓存，把主机侧张量转成设备 buffer。

只要这条边界稳定，未来从 CPU 走向 CUDA、Metal、Vulkan 或 WebGPU 时，就不用改动模型和格式层的核心设计。

## 4. 统一推理主线

端到端推理链路固定为：

1. `io` 读取模型配置、tokenizer 和权重。
2. `model` 校验配置与权重完整性。
3. `kernel` 后端准备执行资源。
4. `runtime` 创建 session 和 KV cache。
5. `tokenizer` 把 prompt 编码为 token。
6. `runtime + kernel` 执行 prefill。
7. `runtime + kernel` 执行 decode loop。
8. `sampler` 选择下一个 token，继续第 7 步。

长期保持不变的是这条主线，变化的是后端能力、算子实现和执行计划。

## 5. Core 层

### 5.1 Tensor

`Tensor` 采用“元数据 + 原始存储”结构：

- `DType`
- `Shape`
- `Layout`
- `Storage`

设计原因：

- 量化权重不能被写死成 `Vec<f32>`。
- 激活值与权重都需要共享一套统一抽象。
- 后续 view、transpose、slice 和设备布局转换必须有位置可挂。

### 5.2 Layout

首版实现可以以 contiguous 为主，但接口必须保留：

- `shape`
- `strides`
- `offset`

这样后续 attention、KV cache view、权重重排和 GPU 布局转换才不会逼着我们重写整个张量层。

### 5.3 Error

项目不依赖 `anyhow/thiserror`，统一使用 `FerrisError + ErrorKind`。需要覆盖：

- 文件读取失败
- 格式解析失败
- shape/layout/dtype 不合法
- 模型配置不一致
- 权重缺失
- 运行时容量溢出
- 后端能力暂未实现

## 6. Kernel 与 Backend 路线

`ferrisinfer-kernel` 不只是 CPU 算子目录，而是整个后端抽象层。

### 6.1 后端能力

后端需要显式描述这些能力：

- 设备类型
- SIMD 支持
- 多线程支持
- 量化 kernel 支持
- 设备内存支持
- graph/fusion/异步提交能力

首版只实现 `CpuBackend`，但接口必须容纳未来的：

- `CudaBackend`
- `MetalBackend`
- `VulkanBackend`
- `WebGpuBackend`

### 6.2 CPU 演进路线

CPU 后端建议分四级推进：

1. `reference`
   单线程、最直白循环，实现正确性基线。
2. `optimized scalar`
   减少临时分配，做 cache-aware blocking 与更合理的布局。
3. `simd`
   针对热点算子接入向量化。
4. `threaded`
   基于标准库线程实现多线程 kernel。

### 6.3 GPU 演进路线

GPU 路线应分成三块设计：

- 设备内存
  权重上传、激活 buffer、KV cache buffer、回传策略。
- kernel dispatch
  matmul、norm、rope、attention、sampling 的设备执行入口。
- execution plan
  prefill 和 decode 在 GPU 上采用不同调度策略。

长期建议的 GPU 推进顺序：

1. 先定义后端接口和资源生命周期。
2. 再实现最小 buffer upload/download 骨架。
3. 接入基础 kernel：`matmul + rms_norm + rope + softmax`。
4. 再打通完整 decoder block。
5. 最后做融合、graph capture、paged KV cache 等高级优化。

## 7. Model 层

模型层统一表达 dense decoder-only Transformer，不为每个模型家族复制整套执行代码。架构差异通过规格控制，例如：

- norm 类型
- attention 布局
- RoPE 参数与缩放策略
- MLP 激活函数与 gated 结构
- 是否 tie embeddings

第一阶段重点覆盖：

- LLaMA-like
- Mistral-like
- Qwen2-like
- Gemma-like

这些模型应尽量复用同一套 decoder-only 执行骨架。

## 8. IO 与 Tokenizer

`io` 层只负责把外部资产映射成内部统一表示。

### 8.1 权重格式

规划两个入口：

- `FerrisSource`
  自有简化格式，优先服务最小测试模型与早期开发。
- `GgufSource`
  面向社区权重兼容，逐步扩展支持范围。

原则是：

- 外部格式差异只停留在 `io` 层。
- 执行层只认统一 `ModelConfig` 与统一权重命名。

### 8.2 Tokenizer

Tokenizer 必须独立建模，长期至少支持：

- BPE
- SentencePiece

不要把 tokenizer 逻辑直接塞进 runtime 或 CLI，否则后面支持多模型会很难清理。

## 9. Runtime 层

运行时层负责把模型、后端和 session 组织成可执行流程。

### 9.1 Session

一个 session 应该持有：

- 模型句柄
- tokenizer 句柄
- backend 执行上下文
- KV cache
- 当前 position
- 采样配置

### 9.2 Prefill / Decode 分离

这两条路径必须从一开始就分开：

- `prefill`
  处理整段 prompt，主要关注吞吐。
- `decode`
  单 token 迭代，主要关注延迟。

CPU 和 GPU 的优化策略都会围绕这条分界线展开。

### 9.3 Execution Plan

长期建议 runtime 显式维护执行计划，至少表达：

- 当前模式：prefill / decode
- 当前后端：cpu / cuda / metal / ...
- 计划步骤：embedding / rope / matmul / attention / mlp / sample
- 中间 buffer 生命周期

这样未来做 CPU/GPU 切换、异步提交、算子融合时会轻松很多。

## 10. 长期实施路线

### Phase A: 地基

- 建立 workspace
- 固定 core/model/runtime/backend 边界
- 把 CPU 到 GPU 的路线写进主设计

### Phase B: Correctness First

- 完成 `Tensor/Shape/Layout`
- 实现 CPU reference kernel
- 打通单层 decoder block 前向

### Phase C: Minimal End-to-End

- 支持多层 decoder-only forward
- 实现最小 tokenizer
- 跑通完整文本生成闭环

### Phase D: CPU High Performance

- cache-aware kernel
- SIMD
- 多线程
- 量化权重与量化 kernel
- KV cache 布局优化

### Phase E: GPU Foundation

- 定义 GPU backend 生命周期
- 实现 host/device buffer 映射
- 接入基础 GPU kernel

### Phase F: GPU End-to-End

- 完整 decoder-only 推理
- GPU KV cache
- prefill/decode 差异化调度

### Phase G: Advanced Optimization

- kernel fusion
- graph capture
- paged KV cache
- 混合精度
- 更复杂模型族

## 11. 当前优先级

虽然长期路线很长，但现在最优先的还是：

1. 把 `core` 的 tensor/layout 做扎实。
2. 把 `kernel` 的 CPU reference path 做对。
3. 把 `runtime` 的 prefill/decode 与 KV cache 抽象定稳。
4. 再接模型格式、tokenizer 与高性能后端。

原因很直接：如果 host 抽象、reference kernel 和运行时主线没有先稳定，未来 GPU 接进来也只会把复杂度和错误一起放大。
