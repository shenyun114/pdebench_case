# 复现测试与参考案例对照报告

本报告记录截至 2026-08-29 的实际执行证据。案例一、二沿用此前从空工作目录完成的复现；案例三重新创建 CUDA JAX 环境，并分别执行 smoke、正式 1/2/4/8 GPU 流水线以及最终版本的空目录 smoke 回归。所有环境、HDF5、原始数组、缓存和日志均位于 `/home/ubuntu/data`。

## 与 `ref-case` 的结构对照

案例文档重点对照 `FEniCS 高性能教程`、`OpenFOAM有限体积法` 等参考文档的章节组织：

| 参考案例标准 | 当前实现 | 状态 |
|---|---|---|
| 术语、背景和问题价值 | 三个主案例均有“案例描述” | 已覆盖 |
| 强形式方程和物理量 | 浅水守恒律、反应–扩散、可压缩质量/动量/能量 | 已覆盖 |
| 初边值条件 | 径向水深、随机双场、傅里叶湍流初值和周期边界 | 已覆盖 |
| 离散算法 | Roe；稀疏 Laplace+RK45；HLLC–MUSCL+中心黏性 | 已覆盖 |
| 从零创建环境 | 独立 `environment.yml`，前缀位于数据盘 | 已实测 |
| 参数配置和规模分档 | default/smoke；3D 另有 highres_128 | 已覆盖 |
| 固定源码版本 | `4ff3e3a4...`，setup 检查提交和干净工作树 | 已实测 |
| 一键命令 | 每个主案例均有 `run_pipeline.sh` | 已覆盖 |
| 日志和机器可读结果 | HDF5、JSON、CSV、pipeline.log | 已覆盖 |
| 图文与动画 | 每图解释物理含义；三个主案例均有 GIF | 已覆盖 |
| 正确性验收 | 形状、有限值、正性、物理量、文件和 PASS/FAIL | 已实测 |
| 网格一致性 | 案例一、二的 32²/64²/128² 研究 | 已覆盖 |
| 并行性能 | 案例三实测 1/2/4/8 GPU；前两例明确不适用 | 已覆盖 |
| 并行模型说明 | 3D 案例明确为样本级 `pmap(vmap)`，不是空间分解 | 已覆盖 |
| 神经网络训练 | 三个主案例没有训练；FNO 仅为附录 B | 边界明确 |

## 测试位置

| 项目 | 路径 |
|---|---|
| 浅水波环境 | `/home/ubuntu/data/pdebench-case-envs/swe` |
| 浅水波完整复现 | `/home/ubuntu/data/pdebench-full-repro/swe` |
| 反应–扩散环境 | `/home/ubuntu/data/pdebench-case-envs/reacdiff` |
| 反应–扩散完整复现 | `/home/ubuntu/data/pdebench-full-repro/reacdiff` |
| 3D CFD 新环境 | `/home/ubuntu/data/pdebench-case-envs/cfd3d` |
| 3D CFD 首次失败证据 | `/home/ubuntu/data/pdebench-cfd3d-smoke/artifacts/logs/dataset.log` |
| 3D CFD smoke PASS | `/home/ubuntu/data/pdebench-cfd3d-smoke-02/artifacts` |
| 3D CFD 正式 PASS | `/home/ubuntu/data/pdebench-cfd3d-formal/artifacts` |
| 3D CFD 最终代码空目录回归 | `/home/ubuntu/data/pdebench-cfd3d-final-smoke/artifacts` |
| Gray–Scott 附录环境 | `/home/ubuntu/data/pdebench-case-envs/gray-scott` |
| Gray–Scott 附录正式复现 | `/home/ubuntu/data/pdebench-full-repro/gray-scott` |

## 案例一：浅水波

- 环境：Python 3.10.20、NumPy 1.26.4、PyTorch 1.13.1、Clawpack 5.9；
- 输出：`h,u,v,hu,hv` 均为 `(101,128,128)`，水深全程为正；
- 质量漂移 `4.79×10⁻¹⁰`，旋转相对 L1 误差 `4.80×10⁻⁴`；
- 32²→64² 相对 L2 从 `0.02015` 降到 `0.01202`，观测阶 `0.745`；
- 自动验收 10 项全部为 `true`，最终 PASS。

## 案例二：反应–扩散

- 环境：Python 3.10.20、NumPy 1.26.4、SciPy 1.15.2；
- 输出：`u,v` 均为 `(101,128,128)` 且全为有限值；
- 场相关系数从 `0.0004226` 增长到 `0.938024`；
- 32²→64² 后，$u$ 相对 L2 从 `0.3608` 降到 `0.1355`，$v$ 从 `0.2424` 降到 `0.09759`；
- 自动验收 9 项全部为 `true`，最终 PASS。

## 案例三：3D 可压缩湍流

### CPU 默认流程

- 后端：JAX 0.4.38 CPU，单个 `CpuDevice`；
- 配置：`configs/cpu.yaml`，不需要 NVIDIA 驱动或 CUDA；
- 输出形状：每场 `1×3×32×32×32`，共五个物理场；
- 全新纯 CPU 环境中的首次 JAX 编译、求解和 NPY 写盘：`42.592 s`；
- HDF5：`1.45 MiB`；
- 质量漂移 `0`，总能量漂移 `8.788×10⁻⁷`；
- 静态图、GIF 和自动验收全部 PASS；
- 独立工作目录：`/home/ubuntu/data/pdebench-cfd3d-cpu-clean-env-test/artifacts`。

### 环境和官方代码兼容

- 新环境：Python 3.10.21、JAX/JAXlib 0.4.38 CUDA 12、NumPy 1.26.4；
- 设备：8 × NVIDIA GeForce RTX 3090 24 GiB，driver 570.153.02；
- `jax.default_backend()` 为 `gpu`，8 张设备全部枚举成功；
- 上游提交保持 `4ff3e3a4aa1561721b5571fa3a048a0a463e0568`，正式运行前后 `git status --short` 均为空。

第一次 smoke 运行在上游 `utils.py` 的 `u.loc[...]` 处失败。该接口与现代 JAX 不兼容，正确的等价更新器是 `u.at[...]`。案例侧增加运行时 `Tracer.loc -> Tracer.at` 别名后重跑成功；没有改动、复制或提交 PDEBench 上游源文件。失败日志被保留，文档没有隐去这一复现过程。

### smoke 测试

- 输出形状：每场 `8×3×32×32×32`；
- 首次 JIT、计算和写盘：`43.330 s`；
- HDF5：`11.42 MiB`；
- 质量漂移 `0`，总能量漂移 `6.40×10⁻⁷`；
- 1/2/4/8 GPU 小任务时间为 `13.056/13.163/13.747/15.603 s`，呈负扩展，验证了过小粒度不适合多 GPU；
- 自动验收 PASS。

最终代码版本又从新的空目录执行一次相同 smoke 流水线，得到 `43.321 s` 的数据阶段和 `13.309/13.366/13.783/15.393 s` 的 1/2/4/8 GPU 计量，并再次完成统一色标 GIF、物理后处理和自动 PASS。正式文档中的性能表仍采用信息量更高的 $48^3$ 正式测试，不用 $24^3$ 回归测试替换。

### 正式数据

- 每个场形状：`8×11×64×64×64`；
- 五场：密度、$v_x$、$v_y$、$v_z$、压力；
- 8 GPU 官方求解、回传和 NPY 写盘：`61.862 s`；
- 合并 HDF5：`363.65 MiB`；
- 终态密度范围 `0.142–2.400`，压力范围 `0.038–3.415`；
- 总质量从 1 保持到 1；总能量相对漂移 `6.495×10⁻⁵`；
- 动能从 `0.5904` 降到 `0.3120`，平均压力从 `0.6000` 升到 `0.7856`；
- GIF 共 11 帧、尺寸 `1200×380`，跨时间使用统一色标；
- 自动验收所有检查为 `true`，最终 PASS。

### 正式多 GPU 结果

固定 8 个样本、$48^3$ 和相同时间区间；每组预热一次，两次计量取中位数：

| GPU | 时间/s | 加速比 | 效率 |
|---:|---:|---:|---:|
| 1 | 21.850 | 1.000 | 1.000 |
| 2 | 17.233 | 1.268 | 0.634 |
| 4 | 15.804 | 1.383 | 0.346 |
| 8 | 16.013 | 1.365 | 0.171 |

结果证明样本级并行有效但很快饱和：4 GPU 为最优点，8 GPU 因每卡只有一个样本以及启动、回传、NPY 写盘开销而轻微退化。报告保留了这一真实结果，没有把样本并行表述为空间域分解或理想线性扩展。

## 时间可变、物理结果可复现的边界

固定提交、随机种子、配置和主依赖版本后，场形状、正性、守恒指标及图像内容应一致；GPU 浮点归约和库小版本可能造成末位差异。墙钟时间受到共享服务器负载、JIT 缓存、GPU 时钟和数据盘 I/O 影响，因此只作为本机证据，不设置严格 PASS 阈值。

案例一、二的网格误差以最细数值解为参考，是 self-convergence/grid consistency，不是解析误差。案例三的 $k^{-5/3}$ 线只是谱斜率参考，$64^3$ 短时可压缩流不能被夸大为充分发展的 Kolmogorov 惯性区。

## 目录重组与重复代码清理

整合前，`PDEBench` 与旧 `PDEBench-54` 的提交和 Git tree 完全相同，重复仓库已清理，只保留 `/home/ubuntu/HW/Case/PDEBench`。原 Gray–Scott 自包含代码此前已迁入案例集；本轮因与二维反应–扩散主案例重合，将目录调整为 `appendix_gray_scott`。三维 CFD 成为新的案例三，直接调用唯一上游仓库。
