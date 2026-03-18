# FerrisInfer


<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Ferris_the_Crab.svg/200px-Ferris_the_Crab.svg.png" alt="Ferris the Crab" width="120">
  <h3>纯 Rust 从零构建的 LLM 推理框架</h3>
  <p>
    <a href="https://www.rust-lang.org/">
      <img src="https://img.shields.io/badge/Rust-1.75%2B-blue?logo=rust" alt="Rust Version">
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/License-AGPL--3.0-red.svg" alt="License">
    </a>
    <a href="https://github.com/Micro-ATP/FerrisInfer/stargazers">
      <img src="https://img.shields.io/github/stars/Micro-ATP/FerrisInfer?style=social" alt="Stars">
    </a>
  </p>
</div>

## 🦀 项目简介

FerrisInfer 是一个**完全从零开始、不依赖任何第三方深度学习库**的纯 Rust LLM 推理框架。我们的目标是用 Rust 小螃蟹的钳子，从最底层的张量计算开始，一步步夹出一个高性能、内存安全、可移植的大模型推理引擎。

> **⚠️ 声明**：这是一个极限编程探索项目，旨在学习和验证 LLM 推理的全链路实现，不建议用于生产环境（至少目前不建议 😄）。

## ✨ 核心特性

- **100% 纯 Rust**：仅使用 Rust 标准库，无任何外部依赖。
- **从零构建**：手动实现张量数据结构、核心算子、内存管理，每一行代码都可控。
- **内存安全**：Rust 所有权机制保证推理过程无内存泄漏、无数据竞争。
- **极限优化**：后续将支持 SIMD 指令集、多线程并行，追求极致性能。

## 🚀 快速开始

### 前置要求
- 安装 Rust ：[https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

### 构建与运行
1. 克隆仓库：
   ```bash
   git clone https://github.com/your-username/FerrisInfer.git
   cd FerrisInfer
