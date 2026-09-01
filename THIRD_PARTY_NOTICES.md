# 第三方代码、版权与引用说明

## PDEBench

本仓库的数值案例以 PDEBench 为上游软件：

- 原代码仓库：<https://github.com/pdebench/PDEBench>
- 本案例固定的上游提交：`4ff3e3a4aa1561721b5571fa3a048a0a463e0568`
- 论文：Makoto Takamoto 等，*PDEBench: An Extensive Benchmark for Scientific Machine Learning*，NeurIPS 2022 Datasets and Benchmarks Track，<https://arxiv.org/abs/2210.07182>
- 官方数据集 DOI：<https://doi.org/10.18419/darus-2986>

PDEBench 提供本案例所调用的数据生成器、数值求解器、离散算子、初始条件、模型基线和数据接口。本仓库在其上增加可复现环境、YAML 配置、调用封装、兼容层、数据转换、物理诊断、可视化和验收流程。各部分的具体关系见 [`ENVIRONMENT_AND_REPRODUCTION.md`](ENVIRONMENT_AND_REPRODUCTION.md)。

本仓库通常不直接收录完整 PDEBench 源码。运行 `scripts/setup_workspace.sh` 时会从官方仓库克隆源码并检出上述固定提交；`appendix_burgers_fno/patches/pdebench-main-compat.patch` 是应用于该上游源码的兼容补丁，涉及的上游代码及其修改版本仍受下述 MIT License 约束。克隆所得仓库自带原始 `LICENSE.txt`，不得删除其中的版权和许可声明。

本项目是独立的教学与复现案例集，不是 PDEBench 官方发行版，也不代表 PDEBench 作者或其所属机构对本项目作出认可或背书。“PDEBench”名称仅用于准确说明技术来源与兼容对象。

## PDEBench 上游许可证

Except where otherwise stated this code is released under the MIT license.

Copyright 2022 NEC Labs Europe GmbH, Stuttgart University, CSIRO and PDEbench contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

许可证原文来源：<https://github.com/pdebench/PDEBench/blob/main/LICENSE.txt>。

## 学术引用

使用 PDEBench 源码或数据撰写论文、报告及其他研究成果时，请引用其原论文。可复制 [`CITATION.bib`](CITATION.bib) 中的 BibTeX 条目。若使用官方数据集，还应依照 PDEBench 官方仓库的引用说明引用数据集 DOI。

## 其他依赖

本仓库环境文件中列出的 JAX、PyTorch、Clawpack、SciPy、NumPy 等第三方软件分别适用其各自许可证；PDEBench 的 MIT License 不会取代这些依赖的许可证。用户在重新分发环境、依赖、上游源码或数据时，应一并遵守相应条款。
