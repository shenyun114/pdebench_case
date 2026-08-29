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

## 2. 集中创建三个 CPU 环境

以下命令从仓库根目录执行。只需创建准备运行的案例环境，不要求一次安装全部环境。

二维浅水波：

```bash
mkdir -p "$PDEBENCH_CASE_DATA/conda-envs"
conda env create \
  --prefix "$PDEBENCH_CASE_DATA/conda-envs/pdebench-swe" \
  -f 01_radial_dam_break/environment.yml
```

二维反应–扩散：

```bash
mkdir -p "$PDEBENCH_CASE_DATA/conda-envs"
conda env create \
  --prefix "$PDEBENCH_CASE_DATA/conda-envs/pdebench-reacdiff" \
  -f 02_reaction_diffusion/environment.yml
```

三维可压缩湍流 CPU 版本：

```bash
mkdir -p "$PDEBENCH_CASE_DATA/pdebench-case-envs"
conda env create \
  --prefix "$PDEBENCH_CASE_DATA/pdebench-case-envs/cfd3d-cpu" \
  -f 03_3d_compressible_turbulence/environment-cpu.yml
```

三维 CPU 环境采用 `jax==0.4.38`，不需要 NVIDIA 驱动或 CUDA。固定 PDEBench 提交中的边界函数使用历史 `.loc` 更新接口，案例通过运行时兼容层将其等价映射到现代 JAX 的 `.at`，不修改下载的 PDEBench 源文件、索引或数值公式。

## 3. 分阶段命令之间的关系

每个案例正文均按以下顺序给出命令：

1. 环境创建：首次运行执行一次；
2. 前处理：固定 PDEBench 源码、创建工作目录并保存 `resolved_config.yaml`；
3. 算法运行：读取同一配置并生成 HDF5 或原始 NPY 数值场；
4. 后处理：完成格式转换、物理诊断、PNG/GIF 和自动验收。

前处理、算法运行和后处理应在同一个终端中依次执行，因为后两段会复用 `CASE_DIR`、`WORK_ROOT`、`REPO`、`ART` 和 `CONFIG` 等变量。环境创建完成后不需要每次重复执行。

如果只修改图像样式或诊断方法，可以保留已有 HDF5，直接重新运行对应案例的后处理命令；如果修改网格、时间范围或物理参数，则应使用新的工作目录重新执行数值求解。

## 4. 数据保护与结果验收

案例脚本发现目标目录中已有 HDF5 时会主动退出，以免覆盖已有实验。再次运行时应修改 `WORK_ROOT`，例如：

```bash
export WORK_ROOT="$PDEBENCH_CASE_DATA/pdebench-swe-staged-02"
```

后处理验收不仅检查文件是否存在，还会检查字段形状、有限值、正水深或正密度/压力、守恒漂移、网格误差趋势和非零涡量。三个主案例均已从数据盘新目录按分阶段命令完成复现并输出 PASS，具体路径和数值见[复现测试报告](REPRODUCIBILITY_REPORT.md)。

各案例的完整一键脚本仍保留在独立文档中，适合自动化复现；总文档只展示便于逐步执行和排错的分阶段命令。
