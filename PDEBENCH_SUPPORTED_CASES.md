# PDEBench 支持的方程与案例选型

PDEBench 不只是 FNO 训练仓库。它同时提供数据生成器、标准 HDF5、FNO/U-Net/PINN 基线、逆问题代码和统一指标。下面按当前固定提交中的实际源码整理。

本案例集中的 Gray–Scott 是自包含扩展案例，并非固定提交里的官方生成器；由于它与官方二维反应–扩散主案例重合，现作为附录保留。新的案例三直接调用固定提交的三维可压缩流求解器，并实测 1/2/4/8 GPU 样本级并行。

## 数据生成案例

| 方程 | 维度 | 主要物理场/参数 | 代码入口 | 视觉表现与成本 |
|---|---:|---|---|---|
| 线性对流 | 1D | 标量 $u$、对流速度 $\beta$ | `data_gen_NLE/AdvectionEq` | 波形平移清楚，成本低，但二维展示能力有限 |
| 黏性 Burgers | 1D | $u$、黏度 $\nu$ | `data_gen_NLE/BurgersEq` | 可看激波陡化与黏性耗散；适合频谱和预测误差，不适合作为视觉主案例 |
| 反应–扩散 | 1D | 标量、扩散/反应参数 | `data_gen_NLE/ReactionDiffusionEq` | 可画传播前沿，成本低 |
| 扩散–吸附 | 1D | 浓度、吸附非线性 | `gen_diff_sorp.py` | 适合多孔介质穿透曲线和守恒分析 |
| 耦合反应–扩散 | 2D | 双场 $u,v$，$D_u,D_v,k$ | `gen_diff_react.py` / `sim_diff_react.py` | 图案丰富；相图、双场 GIF、频谱和尺度分析效果好，已实现为案例二 |
| 径向溃坝浅水方程 | 2D | 水深、速度、动量、重力 | `gen_radial_dam_break.py` / `sim_radial_dam_break.py` | 环形波、三维自由液面、速度矢量和守恒量直观，已实现为案例一 |
| Darcy 流 | 2D | 压力/解场、空间渗透率 | `ReactionDiffusionEq/run_DarcyFlow2D.sh` | 静态问题；适合同时画渗透率与压力/流线，没有时间 GIF |
| 可压缩 Navier–Stokes | 1D/2D/3D | 密度、压力、速度、黏性参数、Mach 数 | `data_gen_NLE/CompressibleFluid` | 可做激波管、随机流、剪切不稳定性和湍流；3D 湍流已实现为案例三 |
| 不可压缩 Navier–Stokes | 2D | 速度、粒子/标量、黏度、外力 | `gen_ns_incomp.py` / `sim_ns_incomp_2d.py` | 可画速度、涡量、流线和粒子输运；依赖 PhiFlow/JAX，默认配置步数很大 |

## 基线模型和任务类型

PDEBench 的 `pdebench/models` 还包含：

- FNO：一维、二维和三维傅里叶神经算子；
- U-Net：网格场的卷积预测基线；
- PINN：通过 DeepXDE 使用方程残差训练；
- forward task：由初始条件预测后续 PDE 场；
- inverse task：从观测反推初始条件或方程参数；
- 统一指标：RMSE、normalized RMSE、守恒误差、边界误差、最大误差和分频段 Fourier RMSE。

## 如何选择下一案例

当前三例已经覆盖二维波动、二维扩散和三维可压缩湍流。如果继续扩展，推荐顺序是：

1. 二维不可压缩 Navier–Stokes：适合增加流线和粒子输运，与当前可压缩案例形成对照，但需要 PhiFlow；
2. 三维爆炸波：适合检验球对称冲击波、激波半径和 Schlieren，但径向叙事与浅水溃坝略有相似；
3. 二维 Darcy：适合说明介质渗透率如何控制压力和通量，属于静态椭圆问题；
4. 一维扩散–吸附：成本低，适合强调质量守恒和突破曲线。

不建议再用小样本 Burgers/FNO 作为展示主案例。它更适合作为“为什么必须检查守恒量和频谱”的负面示范。
