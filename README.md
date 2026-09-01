# PDEBench 科学计算示范案例集

当前案例集包含三个物理与计算主题互补的主案例，以及两个附录。三个主案例都生成 PDE 数值解，不训练神经网络，因此没有 epoch 或 loss 曲线；Burgers/FNO 附录才包含 50 epoch 的模型训练。

> **来源与署名：** 本案例集建立在 [PDEBench 官方代码仓库](https://github.com/pdebench/PDEBench)及论文 *PDEBench: An Extensive Benchmark for Scientific Machine Learning*（[arXiv:2210.07182](https://arxiv.org/abs/2210.07182)）之上，固定使用上游提交 `4ff3e3a4aa1561721b5571fa3a048a0a463e0568`。PDEBench 原代码版权归 NEC Labs Europe GmbH、Stuttgart University、CSIRO 及 PDEBench contributors 所有，并按 MIT License 发布。本仓库是独立编写的复现案例与调用封装，不是 PDEBench 官方仓库，也未获其作者或所属机构背书。详细的代码边界、版权与许可证说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，论文 BibTeX 见 [CITATION.bib](CITATION.bib)。

面向成果展示和连续阅读的正式版本见：[PDEBench 科学计算案例整合文档](PDEBENCH_INTEGRATED_CASES.md)。该文档从 PDEBench 的功能与应用出发，将三个主案例统一组织为“案例描述—前处理—算法设计与并行优化—后处理”，并只给出可独立执行的前处理、算法运行和后处理命令。各案例独立文档同时保留分阶段命令与一键流水线。

## 获取代码

```bash
git clone https://github.com/shenyun114/pdebench_case.git
cd pdebench_case
export PDEBENCH_CASE_DATA=/home/ubuntu/data  # 可替换为本机容量充足的数据盘
export PDEBENCH_ROOT="$PDEBENCH_CASE_DATA/pdebench-upstream/PDEBench"
```

案例代码可位于任意普通目录；三个案例默认共用 `PDEBENCH_ROOT` 指定的一份固定版本 PDEBench。Conda 环境、HDF5、日志和缓存写入 `PDEBENCH_CASE_DATA`，各案例结果目录彼此独立。

| 内容 | 维度与方程 | 数值/性能重点 | 主要可视化 | 定位 |
|---|---|---|---|---|
| [案例一：二维径向溃坝浅水波](01_radial_dam_break/README.md) | 2D 双曲守恒律 | PyClaw Roe、守恒和三网格自收敛 | 水深、波前、速度、Froude 数、自由液面 GIF | 波动与守恒律 |
| [案例二：二维耦合反应–扩散](02_reaction_diffusion/README.md) | 2D 抛物型双场 | 稀疏 Laplace、RK45、网格一致性 | 双场、反应/扩散平衡、相图、频谱、GIF | 扩散与斑图动力学 |
| [案例三：三维可压缩湍流](03_3d_compressible_turbulence/README.md) | 3D 可压缩 Navier–Stokes | HLLC–MUSCL、JAX CPU、质量和能量诊断 | 三正交切片、等值面、涡量、散度、能谱、GIF | 三维流动与五场数据生成 |
| [附录 A：Gray–Scott 并行数据生成](appendix_gray_scott/README.md) | 2D 自包含扩展 | NumPy/Numba、线程池、参数扫描 | 斑图、时空图、参数相图 | 与案例二同属反应扩散，故降为扩展附录 |
| [附录 B：Burgers/FNO 误差与失败模式](appendix_burgers_fno/README.md) | 1D PDE 学习 | PDEBench FNO1d、50 epoch | rollout、RMSE、守恒与频谱误差 | 模型误差反例 |

## 推荐阅读顺序

1. 从二维浅水波理解守恒量、有限体积通量和波前传播；
2. 用二维反应–扩散理解非守恒双场耦合和斑图尺度；
3. 进入三维可压缩湍流，理解密度、压力、速度、涡量、散度和能谱；
4. 若需要更多参数扫描与 CPU JIT 内容，再阅读 Gray–Scott 附录；
5. 最后把 Burgers/FNO 当作“总体 RMSE 不足以代表物理正确”的误差案例。

## 统一交付规范

三个主案例都按“案例描述—前处理—并行优化与数值执行—后处理”组织，并提供：

- 数据盘 Conda 环境创建命令和 `environment.yml`；
- 固定 PDEBench 提交 `4ff3e3a4aa1561721b5571fa3a048a0a463e0568`；
- YAML 参数配置、前处理/算法运行/后处理命令和一键流水线；
- 大体积 HDF5、日志、JSON/CSV 统一写入 `/home/ubuntu/data`；
- 展示用 PNG/GIF、逐图物理解释和机器可读 PASS/FAIL；
- 三个主案例均提供可在普通 CPU 环境执行的默认复现路径。

[环境与复现补充说明](ENVIRONMENT_AND_REPRODUCTION.md)集中说明源码、数据盘环境和重复运行规则。[复现测试报告](REPRODUCIBILITY_REPORT.md)记录环境、运行路径、实测指标和源码完整性。[PDEBench 支持案例说明](PDEBENCH_SUPPORTED_CASES.md)列出固定提交还能生成的其他 PDE。

## 引用 PDEBench

若在研究、报告或成果中使用本案例集，请按 PDEBench 官方要求同时引用其原论文；可直接使用本仓库的 [`CITATION.bib`](CITATION.bib)：

```bibtex
@inproceedings{PDEBench2022,
  author    = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pfl{\"u}ger, Dirk and Niepert, Mathias},
  title     = {{PDEBench: An Extensive Benchmark for Scientific Machine Learning}},
  year      = {2022},
  booktitle = {36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks},
  url       = {https://arxiv.org/abs/2210.07182}
}
```

本仓库只保存案例层代码、配置、兼容补丁、分析和展示结果；运行脚本会另行克隆官方 PDEBench 源码并固定到上述提交。凡直接来自或修改自 PDEBench 的内容继续适用其 MIT License 和原版权声明，不能因本仓库的整理、补丁或说明而取消原作者署名。

## 一眼看结果

### 二维浅水波：径向重力波与自由液面

动画左侧显示水深，右侧显示速度大小和方向；环形水深波前与速度带同步向外传播。两个面板在完整动画中使用固定色标。

![二维浅水波演化](01_radial_dam_break/results/shallow_water_evolution.gif)

### 二维反应–扩散：两个浓度场形成耦合空间结构

动画同步显示激活场、恢复场和局部反应源，可观察随机高频初值被平滑并逐渐形成相关相区；三个面板均使用固定色标。

![二维反应扩散演化](02_reaction_diffusion/results/reaction_diffusion_evolution.gif)

### 三维可压缩湍流：密度、涡量和压缩/膨胀同步演化

动画使用固定色标显示中心切片上的密度、涡量模和速度散度，分别对应压缩结构、旋转结构及局部压缩/膨胀。

![三维湍流演化](03_3d_compressible_turbulence/results/turbulence_evolution.gif)
