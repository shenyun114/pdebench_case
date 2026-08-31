# PDEBench 案例环境与复现补充说明

本文件集中保存三个案例共用的源码位置、环境管理和重复运行说明。面向展示的总文档已经把环境创建、前处理、算法运行和后处理命令分别放入对应案例章节；这里不重复粘贴三套完整求解与出图命令。

## 1. 获取案例代码

案例仓库可以放在任意具有读写权限的位置，无需复制到数据盘：

```bash
git clone https://github.com/shenyun114/pdebench_case.git
cd pdebench_case
export PDEBENCH_CASE_DATA=/home/ubuntu/data  # 可替换为本机容量充足的数据盘
```

案例脚本会将 Conda 环境、固定版本 PDEBench、HDF5、原始 NPY、日志和缓存写入 `PDEBENCH_CASE_DATA`。因此代码仓库占用空间较小，大体积结果不会写入个人文件夹。

## 2. 案例仓库与 PDEBench 上游仓库的关系

`pdebench_case` 不是 PDEBench 源码的替代品。两个仓库分工如下：

- `pdebench_case`：保存可复现配置、上游调用封装、HDF5 转换、物理诊断、可视化和验收代码；
- `pdebench/PDEBench`：提供实际的 PDE 数值求解器、初始条件、离散算子和 Hydra 配置。

用户不需要手动克隆 PDEBench。每个案例执行前处理命令时，`scripts/setup_workspace.sh` 会自动完成：

```text
GitHub: pdebench/PDEBench
        ↓ git clone + 固定到 4ff3e3a4...
$WORK_ROOT/PDEBench
        ↓ 被案例求解脚本导入或执行
HDF5、物理诊断和可视化结果
```

默认工作目录对应为：

| 案例 | 自动下载的 PDEBench 位置 | 实际使用的上游实现 |
|---|---|---|
| 浅水波 | `/home/ubuntu/data/pdebench-swe-staged/PDEBench` | `pdebench/data_gen/src/sim_radial_dam_break.py` 中的 `RadialDamBreak2D` |
| 反应–扩散 | `/home/ubuntu/data/pdebench-reacdiff-staged/PDEBench` | `pdebench/data_gen/src/sim_diff_react.py` 中的 `Simulator` |
| 三维 CFD | `/home/ubuntu/data/pdebench-cfd3d-cpu-staged/PDEBench` | `pdebench/data_gen/data_gen_NLE/CompressibleFluid/CFD_multi_Hydra.py` 及其 JAX 求解组件 |

脚本克隆的是完整上游仓库，但每个案例只执行与自身方程相关的模块。`pdebench_case` 中的三个主案例不调用上游 FNO、U-Net 或 PINN 训练代码。服务器上若已有 `/home/ubuntu/HW/Case/PDEBench`，本复现流程也不会隐式使用它；实际使用的是 `$WORK_ROOT/PDEBench` 中经过提交校验的独立检出。

为使三个案例能够各自独立复现，当前默认会在三个 `WORK_ROOT` 下分别检出一份上游仓库。如果只运行一个案例，只会产生该案例的一份上游工作副本。

## 3. 集中创建三个 CPU 环境

以下命令从仓库根目录执行。只需创建准备运行的案例环境，不要求一次安装全部环境。

二维浅水波：

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p "$PDEBENCH_CASE_DATA/conda-envs"
conda env create \
  --prefix "$PDEBENCH_CASE_DATA/conda-envs/pdebench-swe" \
  -f 01_radial_dam_break/environment.yml
```

二维反应–扩散：

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p "$PDEBENCH_CASE_DATA/conda-envs"
conda env create \
  --prefix "$PDEBENCH_CASE_DATA/conda-envs/pdebench-reacdiff" \
  -f 02_reaction_diffusion/environment.yml
```

三维可压缩湍流 CPU 版本：

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p "$PDEBENCH_CASE_DATA/pdebench-case-envs"
conda env create \
  --prefix "$PDEBENCH_CASE_DATA/pdebench-case-envs/cfd3d-cpu" \
  -f 03_3d_compressible_turbulence/environment-cpu.yml
```

三维 CPU 环境采用 `jax==0.4.38`，不需要 NVIDIA 驱动或 CUDA。固定 PDEBench 提交中的边界函数使用历史 `.loc` 更新接口，案例通过运行时兼容层将其等价映射到现代 JAX 的 `.at`，不修改下载的 PDEBench 源文件、索引或数值公式。

## 4. 分阶段命令之间的关系

每个案例正文均按以下顺序给出命令：

1. 环境创建：首次运行执行一次；
2. 前处理：固定 PDEBench 源码、创建工作目录并保存 `resolved_config.yaml`；
3. 算法运行：读取同一配置并生成 HDF5 或原始 NPY 数值场；
4. 后处理：完成格式转换、物理诊断、PNG/GIF 和自动验收。

前处理、算法运行和后处理应在同一个终端中依次执行，因为后两段会复用 `CASE_DIR`、`WORK_ROOT`、`REPO`、`ART` 和 `CONFIG` 等变量。环境创建完成后不需要每次重复执行。

如果只修改图像样式或诊断方法，可以保留已有 HDF5，直接重新运行对应案例的后处理命令；如果修改网格、时间范围或物理参数，则应使用新的工作目录重新执行数值求解。

## 5. 数据保护与结果验收

案例脚本发现目标目录中已有 HDF5 时会主动退出，以免覆盖已有实验。再次运行时应修改 `WORK_ROOT`，例如：

```bash
export WORK_ROOT="$PDEBENCH_CASE_DATA/pdebench-swe-staged-02"
```

后处理验收不仅检查文件是否存在，还会检查字段形状、有限值、正水深或正密度/压力、守恒漂移、网格误差趋势和非零涡量。三个主案例均已从数据盘新目录按分阶段命令完成复现并输出 PASS，具体路径和数值见[复现测试报告](REPRODUCIBILITY_REPORT.md)。

各案例的完整一键脚本仍保留在独立文档中，适合自动化复现；总文档只展示便于逐步执行和排错的分阶段命令。
