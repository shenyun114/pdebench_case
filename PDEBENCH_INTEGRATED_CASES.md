# PDEBench 科学计算案例：二维浅水波、二维反应–扩散与三维可压缩湍流

## 1. PDEBench 概述

### 1.1 科学机器学习基准

偏微分方程（Partial Differential Equation，PDE）是流体、传热、波动、扩散和材料等科学计算问题的主要数学表达。传统数值方法能够给出可信的高保真解，但在参数反演、实时预测和大规模参数扫描中往往需要反复求解，计算代价较高。科学机器学习试图以数值解为数据基础，学习从初始状态、边界条件和方程参数到后续状态的映射，从而建立计算速度更快的代理模型。

PDEBench 将 PDE 数据生成、模型训练和评价组织为统一基准。其核心对象是由高保真求解器定义的前向传播算子

$$
\mathcal F_\theta:\;v_\theta(t,\mathbf{x})\longmapsto
v_\theta(t+\Delta t,\mathbf{x}),
$$

其中 $v$ 表示状态场，$\theta$ 表示黏性、扩散系数、初边值等物理参数。FNO、U-Net 等数据驱动模型学习近似算子 $\widehat{\mathcal F}_{\theta,\phi}$；逆问题则根据观测到的状态反推未知初值或方程参数。PDEBench 同时提供数值数据生成程序、标准 HDF5 数据、FNO/U-Net/PINN 等基线和统一指标，使不同模型能够在相同数据与评价规则下比较。

PDEBench 覆盖一至三维、稳态与瞬态、线性与非线性问题，包括平流、Burgers、反应–扩散、扩散–吸附、Darcy 流、浅水方程以及可压缩和不可压缩 Navier–Stokes 方程。它们包含间断传播、非线性耦合、多尺度结构、多种边界条件和参数变化，可用于考察代理模型的时间外推、参数泛化、边界处理和物理一致性。

### 1.2 数据、模型与评价

PDEBench 数据通常按“样本—时间—空间—变量”组织，可记为 $N,T,X,Y,Z,V$：$N$ 是样本数，$T$ 是时刻数，$X,Y,Z$ 是空间网格，$V$ 是物理变量数。标准数据覆盖不同空间维度、分辨率、时刻数和变量数；本文根据展示与复现需要，分别采用二维 $128^2$ 和三维 $64^3$ 配置。

单一平均误差不足以判断一个 PDE 预测是否可信。例如，波前位置偏移可能只占少量像素，却会改变传播速度；湍流高波数能量错误也可能被全局均方误差掩盖。PDEBench 因此提供 RMSE、归一化 RMSE、最大误差、守恒量误差、边界误差和低/中/高频 Fourier 误差。本文三个案例进一步把这一原则落实为面向具体方程的诊断：

| 案例 | 主要状态量 | 数值正确性与物理诊断 |
|---|---|---|
| 二维径向溃坝 | $h,u,v,hu,hv$ | 质量、机械能、整体动量、Froude 数、波前、旋转对称性、网格一致性 |
| 二维反应–扩散 | $u,v$ | 反应/扩散强度、相关性、梯度能、相平面、空间频谱、网格一致性 |
| 三维可压缩湍流 | $\rho,v_x,v_y,v_z,p$ | 质量、总能量、动能、平均压力、涡量、速度散度、动能谱、多 GPU 扩展 |

本文展示的是 PDEBench 的**数值求解与数据生成能力**。三个主案例均直接求解控制方程，不训练神经网络，因而不存在 epoch、训练损失或预测误差曲线；生成的时空场可继续作为 FNO、U-Net 或其他代理模型的训练与测试数据。

### 1.3 本文采用的 PDEBench 组件

本文固定使用 PDEBench 提交 `4ff3e3a4aa1561721b5571fa3a048a0a463e0568`。二维浅水波调用官方 `RadialDamBreak2D` 和 PyClaw Roe 求解器；二维反应–扩散调用官方 `Simulator`、稀疏 Laplace 算子和 SciPy 时间积分；三维可压缩湍流调用官方 Hydra 配置与 `CFD_multi_Hydra.py`。案例层只负责参数配置、完整物理场保存、格式转换、诊断、可视化和自动验收，没有改写 PDEBench 的控制方程或数值通量。

案例代码可从 GitHub 直接获取。下文命令默认已经完成以下操作，并用环境变量指定容量充足的数据盘：

```bash
git clone https://github.com/shenyun114/pdebench_case.git
cd pdebench_case
export PDEBENCH_CASE_DATA=/home/ubuntu/data  # 可替换为本机数据盘
```

---

## 2. 案例一：二维径向溃坝浅水波

### 2.1 案例描述

浅水方程描述水平尺度远大于水深时的自由表面流动。通过沿水深方向积分三维流体方程，可用水深和深度平均速度表达海啸、洪水、溃坝、潮汐及城市内涝等长波过程。本案例在平坦底床上设置圆形高水区，瞬时撤去理想挡水边界，观察重力驱动下的径向涌浪和稀疏波。

守恒变量为

$$
\mathbf{q}=(h,hu,hv)^{\mathrm T},
$$

其中 $h$ 为水深，$u,v$ 为两个水平方向的深度平均速度，$hu,hv$ 为单位宽度动量。无底床起伏和摩擦时，二维控制方程为

$$
\frac{\partial h}{\partial t}
+\frac{\partial(hu)}{\partial x}
+\frac{\partial(hv)}{\partial y}=0,
$$

$$
\frac{\partial(hu)}{\partial t}
+\frac{\partial}{\partial x}\left(hu^2+\frac12gh^2\right)
+\frac{\partial(huv)}{\partial y}=0,
$$

$$
\frac{\partial(hv)}{\partial t}
+\frac{\partial(huv)}{\partial x}
+\frac{\partial}{\partial y}\left(hv^2+\frac12gh^2\right)=0.
$$

第一式表达水体质量守恒，后两式表达两个方向的动量守恒；$gh^2/2$ 是静水压力产生的通量。流动状态用 Froude 数

$$
Fr=\frac{\sqrt{u^2+v^2}}{\sqrt{gh}}
$$

判断：$Fr<1$ 为亚临界流，重力扰动能够双向传播；$Fr>1$ 为超临界流。径向溃坝同时包含初始间断、非线性波传播、守恒和轴对称性，是检验双曲守恒律求解器的典型问题，也能为洪水波前识别和快速代理模型提供结构清晰的时空数据。

### 2.2 前处理

计算域为 $[-2.5,2.5]^2$，采用 $128\times128$ 均匀笛卡尔网格，$\Delta x=\Delta y=0.0390625$。网格存储单元平均守恒量，而不是节点上的点值。初始水深为

$$
h(x,y,0)=2\quad\text{当 }r\le0.5,
\qquad
h(x,y,0)=1\quad\text{当 }r>0.5,
$$

其中 $r=\sqrt{x^2+y^2}$，并且 $u(x,y,0)=v(x,y,0)=0$。

圆形间断投影到笛卡尔单元后形成离散初值。四周采用外推边界；在 $t\le1$ 的计算区间内，主波尚未到达外边界，中心波系因而基本不受边界反射影响。配置文件规定 $g=1$、101 个等间隔输出时刻和 $t\in[0,1]$。输出间隔为 0.01，但求解器内部步长由 CFL 稳定性条件决定，二者不能混同。

PDEBench 原保存接口主要面向水深数据，本案例从官方求解器状态中同时提取 $h,u,v,hu,hv$，并写入带 $x,y,t$ 坐标和参数元数据的 HDF5。这样既保留训练数据常用的水深场，也能在后处理中计算动量、机械能、速度与 Froude 数。前处理同时检查网格、时刻、水深正性和官方固定内水深设置，避免配置与实际初值不一致。

主要参数见[默认配置](01_radial_dam_break/configs/default.yaml)。

### 2.3 算法设计与并行优化

PDEBench 的 `RadialDamBreak2D` 使用 Clawpack/PyClaw 有限体积波传播算法。对单元 $(i,j)$ 积分守恒律后，以界面数值通量更新单元平均量：

$$
\mathbf{q}_{ij}^{n+1}=\mathbf{q}_{ij}^{n}
-\frac{\Delta t}{\Delta x}
(\widehat{\mathbf{F}}_{i+1/2,j}-\widehat{\mathbf{F}}_{i-1/2,j})
-\frac{\Delta t}{\Delta y}
(\widehat{\mathbf{G}}_{i,j+1/2}-\widehat{\mathbf{G}}_{i,j-1/2}).
$$

相邻单元共享同一界面通量，因而一个单元的流出量就是相邻单元的流入量。界面处使用 `shallow_roe_with_efix_2D`：Roe 线性化将状态跳跃分解为特征波，entropy fix 避免稀疏波被错误表示为非物理解，MC TVD 限制器抑制间断附近的振荡。时间推进按 CFL 条件自适应选取内部步长。

该 $128^2$ 任务的单次求解约为秒级，官方入口为单进程 PyClaw。本案例不对小网格进行形式化 MPI 拆分，也不报告不适用的并行加速比。计算优化体现在调用已编译的 Roe 内核、一次求解保存全部场、向量化后处理以及用 32²/64²/128² 三网格量化精度—成本关系。大批量生成不同坝高或半径样本时，各样本互不依赖，可进一步在任务层并行。

完整数值、后处理、网格研究和验收由以下命令执行：

```bash
export PDEBENCH_CASE_DATA=/home/ubuntu/data
conda activate "$PDEBENCH_CASE_DATA/conda-envs/pdebench-swe"
cd "$(git rev-parse --show-toplevel)/01_radial_dam_break"
bash scripts/setup_workspace.sh "$PDEBENCH_CASE_DATA/pdebench-swe-demo"
bash scripts/run_pipeline.sh "$PDEBENCH_CASE_DATA/pdebench-swe-demo"
```

### 2.4 后处理与结果分析

#### 2.4.1 水深和自由液面演化

图中各面板使用统一水深色标，白色等值线用于跟踪不同幅值的波面。初始圆形高水柱在释放后分解为向外传播的环形涌浪和向内传播的稀疏波。$t=0.10$ 时间断已经分裂；$t=0.25$ 后中心水位开始明显下降；$t=0.50$ 时高水区转变为环带；$t=1.00$ 时主波前半径约为 1.741，外部区域仍维持初始水深。早期轮廓上轻微的方形印迹来自圆形间断在笛卡尔网格上的离散以及分方向波传播，不代表新的物理不稳定性。

![水深场快照](01_radial_dam_break/results/water_depth_snapshots.png)

三维曲面的竖直坐标是真实水深 $h$。初始垂直侧壁表示水深间断，并非计算域中存在固体圆柱。中间时刻的环形隆起是外行涌浪，末时刻的中心凹陷和外侧环脊体现水体从中心向外重新分配。三维曲面适合辨认自由表面形态，精确的波前位置仍应由径向剖面读取。

![三维自由液面](01_radial_dam_break/results/surface_evolution_3d.png)

#### 2.4.2 速度、Froude 数与径向剖面

上排背景为速度模，箭头给出抽样后的速度方向。高速区最初集中在坝址附近的窄环，随后增宽并外移；圆周各方向的箭头整体向外，符合径向压力梯度驱动。下排为 Froude 数，它还受到局部水深的影响，因此与速度模分布相似但不完全相同。最大速度为 0.5239，最大 $Fr=0.5275<1$，说明全程保持亚临界，没有形成需要按超临界激波解释的区域。后期中心局部向内速度是稀疏波之后的水面回调，并不表示主波反向传播。

![速度和 Froude 数](01_radial_dam_break/results/velocity_and_froude.png)

径向剖面对同一半径圆环上的数值做方位平均，能够削弱笛卡尔网格造成的小幅角向误差。水深剖面右侧的陡变持续向大半径移动，对应外行波前；左侧较缓的斜坡对应稀疏扇。径向速度由正值回到零的位置与水深波前一致，说明两个物理通道的时空索引和传播位置相互吻合。

![径向平均剖面](01_radial_dam_break/results/radial_profiles.png)

#### 2.4.3 守恒、耗散与网格一致性

总质量从 25.79956055 变为 25.79956054，最大相对漂移为 $4.79\times10^{-10}$；整体 $x,y$ 动量仅有舍入级残差，表明圆周各方向的动量能够相互抵消。机械能由 13.69934 降至 13.62710，下降约 0.527%。连续无摩擦方程的机械能应守恒，但 Roe 通量和 TVD 限制器为稳定捕捉间断会引入数值耗散。因此“质量几乎不变而机械能平滑下降”表示稳定格式的耗散代价，而不是边界漏水。最小水深为 0.2832，计算全程保持正水深。

![浅水波守恒诊断](01_radial_dam_break/results/conservation_diagnostics.png)

以 128² 数值解的守恒块平均作为参考，32² 和 64² 末时刻相对 L2 误差分别为 2.015% 和 1.202%，随网格加密单调下降；观测阶约为 0.75。该阶数低于光滑问题的理想高阶并不异常，因为初值和演化波前含有间断，跨间断的全局误差通常只有较低收敛阶。128² 仍是最细数值参考而非解析真解，所以该结果应表述为自收敛或网格一致性证据。

![浅水波网格研究](01_radial_dam_break/results/resolution_study.png)

动画同步展示水深、速度模和方向。水深等色带与速度亮环同步外移，标题中的质量漂移始终保持在验收阈值内，从时间维度确认了波前与动量传播的一致性。

![二维浅水波动态演化](01_radial_dam_break/results/shallow_water_evolution.gif)

---

## 3. 案例二：二维耦合反应–扩散

### 3.1 案例描述

反应–扩散系统描述局部反应与空间扩散共同作用的过程，适用于化学浓度、生态种群、神经兴奋、催化表面和形态发生等问题。扩散倾向于消除空间差异，非线性反应则会放大、限制或耦合局部状态；二者竞争可产生传播波、斑点、条纹或缓慢粗化的相区。

PDEBench 二维模型包含激活场 $u(x,y,t)$ 和恢复场 $v(x,y,t)$：

$$
\frac{\partial u}{\partial t}
=D_u\nabla^2u+u-u^3-k-v,
$$

$$
\frac{\partial v}{\partial t}
=D_v\nabla^2v+u-v.
$$

本案例取 $D_u=10^{-3}$、$D_v=5\times10^{-3}$、$k=5\times10^{-3}$。$v$ 的扩散系数是 $u$ 的 5 倍，因此它更快消除高频波动；$u-u^3$ 中的负三次项限制振幅；$-v$ 与 $u-v$ 完成双场耦合。边界满足齐次 Neumann 条件

$$
\frac{\partial u}{\partial n}=\frac{\partial v}{\partial n}=0,
$$

表示没有物质通过边界扩散。由于局部反应可以改变场的空间平均值，$u,v$ 不是守恒量。本案例的评价重点是反应与扩散的相对强度、场间耦合、空间粗糙度和尺度演化，而不是强行要求“总量不变”。

### 3.2 前处理

计算域为 $[-1,1]^2$，采用 $128\times128$ 均匀网格，$\Delta x=\Delta y=0.015625$。前处理首先按零法向通量条件构造二维五点稀疏 Laplace 矩阵，并修正边界行的对角项，使边界点的离散扩散通量为零。

随机种子 7 生成相互独立的标准正态初始场 $u_0,v_0$。白噪声初值包含从低频到网格截止频率的宽谱扰动，适合观察系统对空间尺度的筛选。为保证网格研究比较的是同一个物理初值，案例先生成 128² 初场，再通过守恒块平均投影到 64² 和 32²；不能仅在不同数组尺寸上重复使用同一随机种子，因为那会得到不同的离散随机场。

两个场展平后拼接为 32,768 维状态向量，积分区间为 $t\in[0,5]$，输出 101 帧。HDF5 保存 $u,v,x,y,t$、扩散系数、反应参数、随机种子、边界和求解器信息。主要参数见[默认配置](02_reaction_diffusion/configs/default.yaml)。

### 3.3 算法设计与并行优化

PDEBench 在规则网格上采用五点差分近似 Laplace 算子：

$$
(\nabla^2u)_{ij}\approx
\frac{u_{i+1,j}-2u_{ij}+u_{i-1,j}}{\Delta x^2}
+\frac{u_{i,j+1}-2u_{ij}+u_{i,j-1}}{\Delta y^2}.
$$

空间离散后，两个场分别满足以下常微分方程：

$$
\frac{d\mathbf{u}}{dt}
=D_uL\mathbf{u}+\mathbf{u}-\mathbf{u}^{3}-k-\mathbf{v},
$$

$$
\frac{d\mathbf{v}}{dt}
=D_vL\mathbf{v}+\mathbf{u}-\mathbf{v},
$$

其中 $L$ 是稀疏 Laplace 矩阵。官方实现将该系统交给 SciPy `solve_ivp` 的 RK45 自适应显式积分器。0.05 是保存帧的间隔，内部步长由局部截断误差控制；随着网格加密，扩散算子的最大特征值增大，显式积分需要更小内部步长。

本案例保持官方单进程算法，使用稀疏矩阵—向量乘法和 NumPy 向量化避免 Python 网格循环。三网格结果用于说明自由度与显式扩散稳定性共同造成的成本增长。若需要生成大量随机样本，可按随机种子在进程或节点间进行样本级并行；这与把单个二维网格做空间域分解是不同的并行层次。

完整流程执行命令为：

```bash
export PDEBENCH_CASE_DATA=/home/ubuntu/data
conda activate "$PDEBENCH_CASE_DATA/conda-envs/pdebench-reacdiff"
cd "$(git rev-parse --show-toplevel)/02_reaction_diffusion"
bash scripts/setup_workspace.sh "$PDEBENCH_CASE_DATA/pdebench-reacdiff-demo"
bash scripts/run_pipeline.sh "$PDEBENCH_CASE_DATA/pdebench-reacdiff-demo"
```

### 3.4 后处理与结果分析

#### 3.4.1 双场形成与耦合状态

每列对应同一时刻，上排为激活场 $u$，下排为恢复场 $v$。$t=0$ 的独立噪声含有大量网格尺度高频分量；到 $t=0.25$，孤立像素首先被扩散消除，其中 $D_v=5D_u$ 使 $v$ 明显更平滑。随后小区域合并为连续正负相区，两个场的空间位置逐渐对应。末时刻 $u$ 的界面更陡、对比度更高，$v$ 则表现为更宽尺度的恢复场。该过程是从随机初值出发的瞬态粗化，不能只根据外观称为已经达到稳态的 Turing 图案。

![反应扩散场快照](02_reaction_diffusion/results/field_snapshots.png)

四个面板依次给出末时刻 $u$、$v$、$u-v$ 和 $R_u=u-u^3-k-v$。其中 $u-v$ 同时是 $v$ 方程的局部反应源：正值推动 $v$ 增大，负值推动 $v$ 减小；$R_u$ 的正负分别表示反应单独作用时对 $u$ 的增加或降低趋势。两个反应源在 $t=5$ 仍有清晰空间结构，说明系统仍在缓慢演化，而非已经完全平衡。

![耦合状态与局部反应](02_reaction_diffusion/results/coupled_state.png)

#### 3.4.2 相平面与机制平衡

相图中的每个点代表一个网格位置的局部状态 $(u,v)$。曲线 $u-u^3-k-v=0$ 与直线 $u-v=0$ 是忽略扩散时的两条反应零流线。初始散点因两场独立而广泛分布；扩散去除极端局部值，反应耦合又将状态拉向零流线附近，后期散点形成沿对角方向的窄带。两条零流线交点为均匀反应平衡 $u=v=-\sqrt[3]{k}\approx-0.171$，空间平均轨迹正向该负值区域移动，但末时刻尚未到达平衡。

![局部相图与零流线](02_reaction_diffusion/results/phase_portrait.png)

图中 RMS 曲线量化方程右端各机制，而不是状态场幅值。初始 $u$ 的扩散/反应 RMS 为 18.198/3.274，$v$ 为 90.536/1.409；高频随机初值产生很大的离散 Laplace 值，且 $v$ 扩散系数更大，因此早期由扩散主导。高频噪声消失后，扩散项迅速下降并与反应项进入相近量级。空间平均值随反应变化并不违反守恒，因为本方程本来含有源项。场幅先下降后部分恢复，反映“扩散消除随机对比度—非线性反应建立有组织结构”的先后过程。

![反应扩散机制平衡](02_reaction_diffusion/results/mechanism_balance.png)

#### 3.4.3 粗糙度、相关性和频谱

$u$ 的梯度能由 4228.26 降至 22.70，$v$ 由 4294.23 降至 3.88，定量对应像素噪声向平滑相区的转变。两场相关系数由 0.0004 增至 0.9380，说明独立初场被耦合动力学锁定为高度相关结构。谱特征尺度方面，$u$ 从 0.0435 增至 0.5838，$v$ 从 0.0441 增至 0.8249；$v$ 的更大尺度与更强扩散一致。末时刻 $u$ 的分布更宽，则说明慢扩散激活场保留了更高图案对比度。

![图案形成诊断](02_reaction_diffusion/results/pattern_diagnostics.png)

二维 FFT 功率按波数半径进行方位平均。随机初场的谱近似宽带；随时间推进，高波数能量下降多个数量级，低波数成分相对增强。频谱向低波数移动与实空间相区增宽是同一过程的两种表示。$v$ 的高频衰减更早、更强，再次验证 $D_v>D_u$ 的低通作用。

![二维空间频谱](02_reaction_diffusion/results/spatial_spectrum.png)

在使用同一投影初值的前提下，32² 到 64² 的末时刻相对 L2 误差由 36.08%/24.24% 降至 13.55%/9.76%（分别对应 $u/v$），观测阶约为 1.41/1.31；相关系数也由 0.9284、0.9357 向 128² 的 0.9380 靠近。运行时间由约 0.040 s、0.404 s 增至 10.294 s，反映二维自由度增长和显式扩散步长约束的共同作用。这里同样是以细网格为参照的网格一致性，不是解析误差。

![反应扩散网格研究](02_reaction_diffusion/results/resolution_study.png)

动画并列展示 $u$、$v$ 和 $u$ 的局部反应源。可以连续观察高频噪声被清除、两场逐渐对齐以及宽尺度相区继续合并的过程，标题中的实时相关系数为图案耦合提供了同步定量指标。

![二维反应扩散动态演化](02_reaction_diffusion/results/reaction_diffusion_evolution.gif)

---

## 4. 案例三：三维可压缩湍流与多 GPU 数据生成

### 4.1 案例描述

可压缩流允许密度随压力和运动发生显著变化，能够描述激波、跨声速气动、燃烧流动和星际气体等问题。三维湍流还包含涡旋拉伸和跨尺度能量传递，状态由多个相互耦合的三维场组成。PDEBench 将可压缩 Navier–Stokes 方程作为高级基准，用于检验代理模型对强非线性、多物理量和小尺度结构的表达能力。

质量、动量和总能量方程为

$$
\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{v})=0,
$$

$$
\rho\left(\frac{\partial\mathbf{v}}{\partial t}
+\mathbf{v}\cdot\nabla\mathbf{v}\right)
=-\nabla p+\eta\nabla^2\mathbf{v}
+\left(\zeta+\frac{\eta}{3}\right)\nabla(\nabla\cdot\mathbf{v}),
$$

$$
\frac{\partial}{\partial t}\left(\epsilon+\frac12\rho|\mathbf{v}|^2\right)
+\nabla\cdot\left[
\left(p+\epsilon+\frac12\rho|\mathbf{v}|^2\right)\mathbf{v}
-\mathbf{v}\cdot\boldsymbol{\sigma}'
\right]=0.
$$

$\rho$ 为质量密度，$\mathbf{v}=(v_x,v_y,v_z)$ 为速度，$p$ 为压力，$\epsilon=p/(\gamma-1)$ 为单位体积内能，$\eta,\zeta$ 分别为剪切和体黏性，$\boldsymbol{\sigma}'$ 为黏性应力。本例采用 $\gamma=5/3$、$\eta=\zeta=10^{-8}$ 和初始马赫数 $M_0=1$。马赫数衡量流速与声速之比；$M_0=1$ 表明压缩和膨胀不可忽略。

涡量

$$
\boldsymbol{\omega}=\nabla\times\mathbf{v}
$$

衡量局部旋转，速度散度 $\nabla\cdot\mathbf{v}$ 的负值/正值分别代表局部压缩/膨胀。动能谱 $E(k)$ 描述不同波数尺度上的速度能量。这些量把三维流场的“旋转性、压缩性和尺度结构”转化为可比较的物理诊断。

### 4.2 前处理

计算域为单位周期立方体，默认采用 $64^3$ 均匀网格和两层 ghost cell。密度与压力初值均匀；速度由有限个随机 Fourier 模态叠加：

$$
\mathbf{v}(\mathbf{x},0)=\sum_m\mathbf{A}_m
\sin(\mathbf{k}_m\cdot\mathbf{x}+\boldsymbol{\phi}_m).
$$

官方程序在 Fourier 空间进行 Helmholtz 分解，削弱速度场的可压缩分量，使初态以近似无散的旋转运动为主，再把速度归一化到指定马赫数。不同 `init_key` 改变模态相位，生成遵循相同统计设置但涡结构位置不同的独立样本。周期边界让穿过一侧的流体从对侧进入，域内没有质量通量损失，适合检查总质量和总能量。

案例通过 Hydra 覆盖网格、样本数、终止时间、保存时刻、随机种子和输出路径。正式配置生成 8 个样本、11 个时刻、$64^3$ 网格及密度、三分量速度、压力共 5 个场；每个场统一为 `[sample,time,x,y,z]`。官方五个 NPY 场随后合并为带坐标、配置、源码提交和字段名的 HDF5。转换阶段还校验各场形状和时间坐标，防止上游多余尾部坐标造成场—时刻错位。

[快速配置](03_3d_compressible_turbulence/configs/smoke.yaml)采用 $32^3$ 验证流程，[正式配置](03_3d_compressible_turbulence/configs/default.yaml)采用 $64^3$ 并执行性能测试，[128³ 配置](03_3d_compressible_turbulence/configs/highres_128.yaml)提供高分辨率扩展入口。128³ 的内存、输出和运行成本显著更高，不属于默认一键测试。

### 4.3 算法设计与并行优化

官方程序对无黏通量采用二阶 HLLC Riemann 求解器，以 MUSCL 和斜率限制器重构界面左右状态。HLLC 分辨左行波、接触波和右行波，MUSCL 在抑制激波附近数值振荡的同时减轻一阶迎风格式对涡结构的过度抹平。时间方向使用二阶预测—校正更新，黏性项采用中心差分，内部步长由三维 CFL 条件自适应确定。

多样本入口的核心并行表达为

```python
pm_evolve = jax.pmap(jax.vmap(evolve, axis_name="j"), axis_name="i")
```

`vmap` 在单张 GPU 内批量推进若干独立样本，`pmap` 再将样本组分配到多张 GPU。每个 `evolve` 都持有完整的 $x\times y\times z$ 网格，因此这是**样本级数据并行**，不是把一个三维网格沿空间方向分块：GPU 之间没有单样本 halo 交换，增加 GPU 只减少每卡样本数，不能缩短单样本的临界计算路径。

固定总工作量为 8 个样本、$48^3$ 和相同时间区间时，每组设备先预热一次，再对两次独立进程计时取中位数：

| GPU 数 | 每卡样本 | 中位时间/s | 加速比 | 并行效率 |
|---:|---:|---:|---:|---:|
| 1 | 8 | 21.850 | 1.000 | 1.000 |
| 2 | 4 | 17.233 | 1.268 | 0.634 |
| 4 | 2 | 15.804 | 1.383 | 0.346 |
| 8 | 1 | 16.013 | 1.365 | 0.171 |

4 GPU 是该固定任务的最短时间点。8 GPU 虽然全部参与计算，但每卡只剩一个 $48^3$ 样本，设备调度、JAX 进程启动、主机回传和串行 NPY 写盘的占比增大，因此较 4 GPU 轻微退化。该结果表明样本并行已经生效，同时也准确给出了当前粒度下的饱和点；它不能被解释为空间域分解的理想强扩展。

![1/2/4/8 GPU 扩展结果](03_3d_compressible_turbulence/results/multi_gpu_scaling.png)

完整正式流程执行命令为：

```bash
export PDEBENCH_CASE_DATA=/home/ubuntu/data
conda activate "$PDEBENCH_CASE_DATA/pdebench-case-envs/cfd3d"
cd "$(git rev-parse --show-toplevel)/03_3d_compressible_turbulence"
bash scripts/run_pipeline.sh \
  "$PDEBENCH_CASE_DATA/pdebench-cfd3d-formal-new" \
  configs/default.yaml
```

正式生成的每个场形状为 `8×11×64×64×64`，官方求解、设备回传和五场 NPY 写盘合计 61.862 s，合并 HDF5 为 363.65 MiB。该时间包含 JAX 编译、初值、计算、回传与 I/O，不等同于纯数值算子时间。

### 4.4 后处理与结果分析

#### 4.4.1 三正交切片与三维等值面

三列分别截取 $x=L/2$、$y=L/2$、$z=L/2$，不是同一二维切片的重复。密度亮区表示压缩形成的高密度流体，暗区表示膨胀低密度区，终态范围为 0.142–2.400；压力范围为 0.038–3.415，高压区大多与高密度结构相邻，符合可压缩流压缩升压的物理关系。速度模表示局部运动强度，其细长结构与密度峰并不完全重合，因为高速平流、旋转和压缩是不同机制。三个方向都有连续而不同的结构，证明流场具有真实三维变化，单一中心切片可能漏掉与该平面不相交的结构。

![密度、压力和速度三正交切片](03_3d_compressible_turbulence/results/orthogonal_slices.png)

左图为终态密度第 90 百分位等值面 $\rho=1.329$，它是高密度压缩区的三维边界，不是固体表面。右图为涡量模第 97 百分位等值面 $|\boldsymbol\omega|=53.90$，片状、弯曲或近似管状区域表示强剪切和局部旋转。等值阈值只保留最强结构，零散小片既可能来自真实小尺度，也受 $64^3$ 分辨率和阈值选择影响，因此不能把每个片段都解释为独立的大尺度涡。

![密度和涡量等值面](03_3d_compressible_turbulence/results/density_vorticity_isosurfaces.png)

#### 4.4.2 守恒、能量转换和压缩性

周期域总质量由 1 保持到 1，末时刻相对漂移为 0，中间约 $10^{-7}$ 的波动来自 float32 归约舍入。总能量相对漂移仅 $6.50\times10^{-5}$，而动能由 0.5904 降至 0.3120、平均压力由 0.6000 升至 0.7856。这组量应联合解释：大尺度运动能量减少的同时，压缩和数值耗散使内能/压力增加，总能量仍近似守恒。显式物理黏性接近零，但 HLLC–MUSCL 捕捉陡峭结构仍需数值耗散，所以动能下降不能全部归因于物理黏性。

RMS 涡量表示旋转活动，RMS 散度表示压缩与膨胀强度。初态虽经 Helmholtz 分解而近似无散，但马赫数为 1 的非线性演化会产生明显密度波和散度，因此不能使用不可压缩假设把 $\nabla\cdot\mathbf{v}$ 视为零。

![三维流动物理诊断](03_3d_compressible_turbulence/results/conservation_and_flow_diagnostics.png)

#### 4.4.3 三维动能谱与时间演化

后处理对三个速度分量进行三维 FFT，并把具有相同整数波数半径的模态能量按球壳求和得到 $E(k)$。初始能量集中在低波数，这是有限个随机 Fourier 模态构造初值的直接结果；到 $t=0.125$ 和 0.25，高波数形成连续能量尾部，说明非线性平流已将结构传递到更小尺度。图中的 $k^{-5/3}$ 仅是 Kolmogorov 斜率参考。由于计算时间短、分辨率为 $64^3$ 且流动可压缩，谱中没有足够宽的惯性区，不能据此宣称已经得到充分发展的 Kolmogorov 湍流。

![三维各向同性动能谱](03_3d_compressible_turbulence/results/kinetic_energy_spectrum.png)

动画从中心 $x$ 平面同步显示密度、涡量模和速度散度。密度面板呈现压缩结构的移动、合并与变形；涡量面板显示旋转结构的增强、拉伸和衰减；散度面板中负值为压缩、正值为膨胀。三个面板在全部 11 帧内使用各自固定色标，因而不同时刻的亮暗可直接比较，避免逐帧自动缩放制造虚假的强度变化。

![密度、涡量和速度散度演化](03_3d_compressible_turbulence/results/turbulence_evolution.gif)

---

## 5. 运行环境与完整复现

### 5.1 源码和数据位置

案例代码通过 GitHub 克隆到任意具有读写权限的位置即可运行，不需要把代码仓库复制到数据盘。建议用 `PDEBENCH_CASE_DATA` 指定容量充足的数据盘；Conda 环境、脚本自动克隆的固定版本 PDEBench、HDF5、原始 NPY、日志和临时缓存均写入该位置。案例脚本在目标数据目录已存在 HDF5 时主动退出，重跑应使用新的工作目录，避免覆盖原结果。

```bash
git clone https://github.com/shenyun114/pdebench_case.git
cd pdebench_case
export PDEBENCH_CASE_DATA=/home/ubuntu/data  # 可替换为本机数据盘
```

### 5.2 创建环境

二维浅水波：

```bash
cd 01_radial_dam_break
mkdir -p "$PDEBENCH_CASE_DATA/conda-envs"
conda env create \
  --prefix "$PDEBENCH_CASE_DATA/conda-envs/pdebench-swe" \
  -f environment.yml
cd ..
```

二维反应–扩散：

```bash
cd 02_reaction_diffusion
conda env create \
  --prefix "$PDEBENCH_CASE_DATA/conda-envs/pdebench-reacdiff" \
  -f environment.yml
cd ..
```

三维可压缩湍流：

```bash
mkdir -p "$PDEBENCH_CASE_DATA/pdebench-case-envs"
conda env create \
  --prefix "$PDEBENCH_CASE_DATA/pdebench-case-envs/cfd3d" \
  -f 03_3d_compressible_turbulence/environment.yml
```

三维环境采用 `jax[cuda12]==0.4.38`。固定 PDEBench 提交中的公共边界函数使用历史 `.loc` 更新接口，案例通过运行时兼容层将其等价映射到现代 JAX 的 `.at`；该处理不改变索引和数值公式，也不修改脚本自动下载的 PDEBench 上游文件。

### 5.3 快速验证和验收依据

三维案例可先执行完整但低分辨率的 smoke 流程：

```bash
conda activate "$PDEBENCH_CASE_DATA/pdebench-case-envs/cfd3d"
cd 03_3d_compressible_turbulence
bash scripts/run_pipeline.sh \
  "$PDEBENCH_CASE_DATA/pdebench-cfd3d-smoke-new" \
  configs/smoke.yaml
```

三个案例的流水线均依次完成数值生成、HDF5 写出、物理诊断、PNG/GIF、网格或并行测试和自动验收。验收不仅检查文件存在，还检查字段形状、有限值、正水深或正密度/压力、守恒漂移、网格误差趋势、非零涡量及 1/2/4/8 GPU 记录。当前交付版本已经在数据盘独立工作目录完成复现，三个主案例均输出 PASS；详细环境、路径和原始数值记录见[复现测试报告](REPRODUCIBILITY_REPORT.md)。

## 6. 总结

三个案例构成了从二维双曲守恒律、二维耦合抛物型方程到三维可压缩流的递进计算链。浅水波案例以波前、守恒和 Froude 数说明有限体积法如何处理间断；反应–扩散案例以双场耦合、机制分解和频谱说明非守恒斑图的形成；三维可压缩湍流案例则同时展示五场数据、旋转/压缩诊断、跨尺度能量分布以及真实的多 GPU 样本并行饱和点。

这些结果既可作为独立的数值模拟示范，也可作为 PDEBench 代理模型实验的真值数据。后续若引入 FNO 或 U-Net，应在常规场误差之外继续保留本文的方程相关诊断：浅水波关注质量和波前，反应–扩散关注相图与频谱，三维流动关注质量/能量、涡量、散度和动能谱。这样才能判断模型不仅“图像相似”，而且保留了关键物理结构。
