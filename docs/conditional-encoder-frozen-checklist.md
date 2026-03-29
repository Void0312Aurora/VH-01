# 条件化编码器修复冻结清单

## 1. 冻结说明

本文只服务于一个问题：

1. 理论里允许编码器写成 `z_t \approx E_t(X_{1:T}, c)`；
2. 当前实现中的 [encode_video](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 仍是无条件逐帧编码，条件只进入解码和动力学分支。

自本文起，这个问题的修复范围冻结。
若没有明确改 scope，后续只按本文列出的任务推进，不再临时追加新的独立任务。

冻结时间：

1. 2026-03-29

## 2. 本问题的关闭标准

只有当下面 6 条全部满足时，才能把“条件化编码器缺失”判定为关闭：

1. 代码里存在显式的可选条件化编码入口，形式至少达到 `encode_video(video, cond_embed=None)`。
2. 条件化编码器对整段视频工作，而不是仅在逐帧 CNN 上做浅层拼接。
3. 训练主链与 teacher/measure target 路径在真实条件已知时，默认走条件化编码。
4. query/候选读取/生成观测等“条件未知”场景，仍可走无条件回退编码，不会因此断链。
5. 有固定测试覆盖条件编码前向差异、`forward` 接线、训练 smoke、query/生成 smoke。
6. 默认 smoke 训练能正常结束，新增条件编码不会把现有主链打坏。

## 3. 明确不在本次范围内的事项

下面这些问题虽然重要，但不属于“关闭条件化编码器缺失”的必要条件：

1. 把 query 观测阶段改成逐候选条件重编码。
2. 负条件分支是否也应重编码出 counterfactual latents。
3. `\mu_0 / \mathfrak T_c / \mu_c / Read` 的对象化。
4. `\Psi_x \mapsto I_x^\star` 的最终不变量构造。
5. 条件化编码器与联络、图册兼容性的深层统一。
6. 大规模 checkpoint 兼容性迁移方案。

如果后面要做这些，应该另开任务，不算作这里的派生工作。

## 4. 执行任务

### A1. 冻结实现边界

目标：

1. 明确第一版只实现“可选条件化 + 无条件回退”。
2. 明确 query/观测侧暂不改成候选级重编码。

状态：

1. 已完成

结果：

1. 本文已经冻结第一版实现边界，明确采用“可选条件化 + 无条件回退”。
2. query/观测侧仍保留 `encode_video(video)` 无条件入口，没有改成逐候选重编码。

### A2. 实现条件化编码器

目标：

1. 为 `encode_video` 增加 `cond_embed=None` 可选入口。
2. 引入“无条件基底 + 条件时序残差修正”的编码结构。
3. 条件化编码应依赖整段视频与条件。

状态：

1. 已完成

结果：

1. [encode_video](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 已扩展为 `encode_video(video, cond_embed=None)`。
2. 新增“无条件基底 + 条件时序残差修正”的编码结构，入口见 [mvp.py](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L315)。
3. 条件化编码通过整段 latent 序列上的 temporal conv 与条件门控共同实现，不再是纯逐帧无条件编码。

### A3. 重接训练与 teacher 路径

目标：

1. [forward](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py) 走条件化编码。
2. [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py) 的 teacher/target 路径在条件已知时走条件化编码。
3. 其它主训练入口在条件已知时不再偷走无条件编码。

状态：

1. 已完成

结果：

1. [forward](/home/void0312/AIGC/VH-01/src/vh_mvp/models/mvp.py#L358) 已默认走条件化编码。
2. [local_measure_targets](/home/void0312/AIGC/VH-01/src/vh_mvp/losses/objectives.py#L804) 的 target/teacher 重编码路径已改为 `encode_video(video, cond_embed=source_cond_embed)`。
3. 训练主链在真实条件已知时，默认不再偷走旧的无条件编码入口。

### A4. 保持 query/生成链兼容

目标：

1. query/候选打分、候选 rollout、观测侧 probe 等仍允许直接 `encode_video(video)`。
2. 现有脚本与 smoke 生成链在不传条件时仍可运行。

状态：

1. 已完成

结果：

1. query/候选打分、观测侧 probe、候选 rollout 仍可直接调用 `encode_video(video)`。
2. 现有脚本与生成/读取 smoke 没有因为条件化编码器引入而断链。

### B1. 固定测试覆盖条件编码差异

目标：

1. 测试 `encode_video(video, cond_embed)` 与 `encode_video(video)` 的行为分离。
2. 测试不同条件产生不同编码。

状态：

1. 已完成

结果：

1. 新增 [test_conditional_encoder_frozen_checklist.py](/home/void0312/AIGC/VH-01/tests/test_conditional_encoder_frozen_checklist.py#L1)。
2. 测试覆盖了 `encode_video(video)`、`encode_video(video, cond_embed)` 以及不同条件下编码差异。

### B2. 固定测试覆盖 `forward` 与 teacher 入口

目标：

1. 测试 `forward(...).latents` 与显式条件编码对齐。
2. 测试 target/teacher 路径会在条件已知时走条件化编码。

状态：

1. 已完成

结果：

1. 固定测试已覆盖 `forward(...).latents` 与显式条件编码一致。
2. 固定测试已覆盖 target/teacher 路径会在条件已知时重新走条件化编码。

### B3. 训练 smoke

目标：

1. 默认 smoke 配置训练 2 个 epoch，确认主链正常。

状态：

1. 已完成

结果：

1. `PYTHONPATH=src python scripts/train_mvp.py --config configs/smoke.yaml` 已正常跑完 2 个 epoch。

### B4. query/生成 smoke 与文档收口

目标：

1. query/生成链在无条件回退编码下仍有限并可执行。
2. 本文更新为冻结完成态，并回写审计文档。

状态：

1. 已完成

结果：

1. query/生成 smoke 已由固定测试覆盖，并在无条件回退编码下保持有限输出。
2. 本文已更新为冻结完成态，并已回写审计文档。

## 5. 允许的唯一派生工作

后续允许出现的派生工作只有两种：

1. 为完成 A1-A4、B1-B4 直接产生的 shape / dtype / compile / numerical stability 修复。
2. 为让新增测试通过而做的最小兼容修补。

除此之外，不新增新的独立任务。

## 6. 当前结论

当前这项冻结问题已经关闭。

关闭依据如下：

1. 条件化编码入口已经显式存在。
2. 训练与 target/teacher 路径在条件已知时已接到新编码器。
3. query/观测侧仍保留无条件回退入口。
4. 固定测试、`pytest`、默认 smoke 训练都已经通过。

这只代表“条件化编码器缺失”这一项已关闭，不代表更深层的条件测度与读取理论已经闭合。
