# `U_x` 经验邻域修复方案

## 1. 问题重述

这项问题不应再被表述为：

> 理论里的局部邻域 `U_x` 是连续对象，而代码里用了 `kNN`，所以实现错了。

更准确的表述应当是：

1. 理论中的 `U_x \subset \mathcal M_T` 是连续几何邻域；
2. 数值实现本来就只能在离散样本层 `\mathcal Z^T` 上工作；
3. 因此真正可计算的对象，应是
   `\mathcal N_x^{(k)} \subset \mathcal Z^T \cap U_x`
   这样的经验局部样本云；
4. 当前工程的主要问题不是用了 `kNN`，而是很多 active 路径只用了“当前 batch 内的临时近邻图”，使得经验邻域过弱、过噪、过依赖 batch 组成。

也就是说：

> `kNN` 不是理论偏差本身；“batch 内 `kNN` 被直接当成经验邻域本体”才是当前需要修补的层级问题。

## 2. 理论依据

按 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L83)：

1. `\mathcal Z^T` 是离散有效轨迹集合；
2. `\mathcal M_T` 是训练后测度揭示出的连续几何支撑；
3. `\mathcal Z^T` 应被看作 `\mathcal M_T` 上或其邻域中的离散采样层。

按 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L260) 与 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L557)：

1. 固定 `x \in \mathcal M_T` 后，真正进入局部分析的是 `x` 邻域内的局部样本云；
2. `l` 是该局部样本云中的离散索引，而不是全局样本编号；
3. 因而局部响应、局部切空间、局部 jet 拟合都应建立在经验邻域点云上。

按 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L1432)：

1. 连续层：`x \mapsto \Psi_x^c`
2. 几何层：切空间、投影、联络与局部兼容性
3. 离散层：在 `\mathcal Z^T` 中，用 `k` 近邻图、局部正交标架对齐和局部加权最小二乘去近似连续对象

因此，经验 `kNN` 本来就是桥梁的一部分。

## 3. 当前实现的真正问题

当前问题集中在以下三处：

1. [_build_knn_graph](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L547) 仅在当前 batch 内找近邻；
2. [_local_response_jet_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L597) 的局部切空间、局部漂移、局部二阶结构 target 仅由当前 batch 构造；
3. [local_neighbor_smoothness_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L1256) 的邻域平滑也只看当前 batch 签名。

因此，当前系统里真正存在的并不是
`\mathcal N_x^{(k)} \subset \mathcal Z^T \cap U_x`
的经验近似，而更像是：

> `\widehat{\mathcal N}_{x,\mathrm{batch}}^{(k)}`

它只是一个小批量、瞬时、随机的近邻代理。

## 4. 本轮修复目标

本轮不做全局图册，也不做数据集级离线图构建。

本轮只做一件事：

> 把“batch 内临时近邻图”升级为“跨 batch 的经验邻域池 + 当前 batch 局部细化”。

这意味着：

1. 理论对象 `U_x` 仍保持连续意义；
2. 数值对象由 `\mathcal Z^T` 上更稳定的经验邻域池来近似；
3. 当前 batch 退回为经验邻域的一个局部采样块，而不是全部邻域本体。

## 5. 设计原则

### 5.1 不把经验池误写成全局图册

本轮的经验邻域池仍然只是：

1. 离散样本层近似；
2. 短时、有限窗口的经验缓存；
3. 对 `\mathcal N_x^{(k)}` 的工程代理。

它不是：

1. 显式恢复的 `\mathcal M_T`；
2. 全局稳定图册；
3. 数据集级最终近邻结构。

### 5.2 分开几何邻域与平滑邻域

当前 active 路径里至少有两种邻域用途：

1. `response_jet / graph_tau` 需要“几何邻域”
2. `response_smoothness` 需要“平滑邻域”

两者虽然都用 `kNN`，但消费的对象不同：

1. 几何邻域更依赖 `trajectory_point` 与 `response_channels`
2. 平滑邻域更依赖 `signatures` 与 `state / drift / diffusion / log_density`

因此本轮实现里，允许它们共享“跨 batch 经验池”的思想，但不强求完全同一个张量缓存结构。

### 5.3 保持 transport 兼容项的保守回退

当前 `tangent_bundle_compatibility` 里的 `transport` 兼容项依赖当前 batch 内可索引的邻居标架。

若几何邻域来自跨 batch 经验池，则这一项未必总能直接构造。

因此本轮采取保守策略：

1. `response_jet` 的局部几何 target 可以使用跨 batch 经验邻域；
2. 若邻居来自经验池而非当前 batch，则 `transport` 元数据允许退化为空；
3. 此时 `tangent_bundle_compatibility` 回退为签名诱导的局部平滑兼容项。

## 6. 具体实现

### Stage A. 新增经验邻域引用对象

新增两类轻量引用对象：

1. `GeometryNeighborhoodReference`
2. `SmoothnessNeighborhoodReference`

分别承载：

1. 几何邻域所需的 `trajectory_point` 与 `response_channels`
2. 平滑邻域所需的 `signatures` 与平滑字段

并提供：

1. 当前 batch 到引用池的拼接
2. 截断到固定缓存长度
3. detached 存储，避免跨 batch 反传

### Stage B. 让 `kNN` 支持 reference pool

对以下路径增加 `reference_*` 输入：

1. `_build_knn_graph(...)`
2. `_local_response_jet_bundle(...)`
3. `local_neighbor_smoothness_loss(...)`

这样局部邻域就不再只来自当前 batch，而是来自：

1. 当前 batch 自身
2. 经验邻域池

### Stage C. trainer 维护跨 batch 邻域池

在 train/eval 循环中维护两个 detached queue：

1. 几何邻域池
2. 平滑邻域池

每个 batch 结束后更新：

1. 当前 batch 的经验局部点
2. 当前 batch 的响应通道
3. 当前 batch 的平滑签名与平滑字段

下一 batch 进入 `local_measure_loss(...)` 时，将该池作为 reference 输入。

## 7. 关闭标准

只有当下面 5 条满足时，这项问题才算修到当前阶段可关闭：

1. 文档里明确区分 `U_x`、经验样本云 `\mathcal N_x^{(k)}`、batch 内近邻图；
2. `response_jet` 不再只能依赖当前 batch 构造局部几何；
3. `response_smoothness` 不再只能依赖当前 batch 构造局部邻域；
4. train/eval 都有跨 batch 经验邻域池；
5. 新增测试与 smoke 训练通过。

## 8. 明确不在本轮范围内

本轮不处理：

1. 数据集级离线全局近邻图
2. 显式图册拼接
3. 真正的联络离散化
4. 对 `transport` 做跨缓存样本的完整几何并行传输
5. 彻底消除 batch 依赖

这些都属于后续更深层的全局化任务。

## 9. 一句话结论

本轮的正确修法不是“去掉 `kNN`”，而是：

> 承认 `kNN` 是 `\mathcal Z^T` 上近似 `U_x` 的应有桥梁，并把它从“当前 batch 临时图”升级为“跨 batch 的经验局部样本云近似”。
