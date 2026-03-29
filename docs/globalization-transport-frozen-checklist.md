# 全局化第一步：跨参考邻域 transport 冻结清单

## 范围

本轮只处理“全局化”的一个最小而必要的缺口：

1. 理论已经要求局部响应纤维之间要能做跨点比较；
2. 当前实现虽然已有跨 batch 的经验邻域池，但一旦邻居来自 reference pool，`transport` 就退回为空；
3. 因而切向兼容和响应纤维比较，仍然只在纯当前 batch 邻域里是真正成立的。

本轮只修：

> 让 reference 邻域也能参与局部正交标架对齐与离散 transport 构造。

## 理论依据

按 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L1401) 到 [temp-02.md](/home/void0312/AIGC/VH-01/docs/temp-02.md#L1446)：

1. `E=\bigsqcup_x E_x` 上的联络 `\nabla^E` 是跨点比较响应对象的基本结构；
2. 离散实现应通过 `k` 近邻图、局部正交标架对齐和局部加权最小二乘去近似 `\widehat T_x\mathcal M_T`、`\Pi_x` 与 `\nabla^E`；
3. 因而“局部邻域扩到 reference pool”之后，如果没有同步恢复跨邻域标架对齐与 transport，那么全局化仍然只完成了一半。

## 当前实现缺口

当前代码已经做过上一轮修补：

1. `response_jet` 和 `response_smoothness` 都能使用跨 batch 邻域池；
2. 但 [_local_response_jet_bundle](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L970) 里，若邻居来自 reference pool，`transport` 仍直接退化为空；
3. 这意味着 [local_measure_loss](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L2010) 的 `tangent_bundle_compatibility`，在跨参考邻域场景下仍只能退回签名平滑项。

因此，当前“全局化”真正缺的不是更大的池，而是：

1. reference 邻域自己的局部标架；
2. 跨当前 batch 与 reference 邻域的离散正交对齐；
3. 能被训练路径直接消费的 `target_transport`。

## 不在范围内

本轮不处理：

1. 数据集级离线全局图；
2. 显式图册拼接；
3. 真正的联络/connection 学习；
4. 跨邻域高阶 jet 的全局兼容；
5. 彻底消除 batch 依赖。

## 实施项

### A1. 扩展几何 reference

`GeometryNeighborhoodReference` 新增：

1. `tangent_frames`
2. `tangent_frame_valid`

这样 reference pool 不再只缓存点和响应通道，也缓存可供离散 transport 使用的局部正交标架。

### A2. 构造 reference 标架

`build_geometry_neighborhood_reference(...)` 在生成 snapshot 时，同时运行一遍局部 `response_jet`，把该批次的经验切向标架一起缓存下来。

若当前 snapshot 本身不足以稳定估计标架，则保守地标记为 invalid，而不是伪造 transport。

### A3. 恢复跨参考邻域 transport

`_local_response_jet_bundle(...)` 在 reference pool 存在且 reference frame 有效时：

1. 把当前 batch frame 与 reference frame 拼成候选标架池；
2. 对选中的每个邻居都构造 Procrustes 型正交对齐；
3. 输出 `neighbor_idx / neighbor_weights / transport`，即便邻居并非都来自当前 batch。

### A4. 训练路径接线

`local_measure_loss(...)` 的 `tangent_bundle_compatibility` 需要：

1. 在 reference 邻居出现时，也能索引到 reference frame；
2. 使用新的 `target_transport` 做跨邻域兼容损失；
3. 只在 reference frame 不可用时，才退回旧的平滑 fallback。

### A5. 测试

至少覆盖：

1. geometry reference 会缓存 tangent frame；
2. reference 邻居被选中时，`local_measure_targets(...)` 会返回非空 `target_transport`；
3. `local_measure_loss(...)` 在 cross-reference transport 路径下可正常反传；
4. pytest 与 measure-active smoke 训练通过。

## 关闭标准

满足以下条件即可关闭：

1. reference 邻域不再只缓存点和响应通道，也缓存局部 frame；
2. reference 邻居参与时，`target_transport` 不再强制为空；
3. `tangent_bundle_compatibility` 可真正使用跨参考邻域的 transport；
4. 相关 pytest 通过；
5. measure-active smoke 训练通过。
