# 外部基线与标准评估口径

## 1. 先说结论

当前仓库的真实数据主线使用的是：

1. `UCF101 subset`
2. `10` 个语义类别
3. `8` 帧
4. `32x32`
5. 条件来自当前工程的离散桥接/语义条件链

因此，**和公开论文做完全 apples-to-apples 的直接数值比较并不存在**。公开 UCF101 结果大多使用：

1. 更标准的全 UCF101 或更大切分；
2. 更高分辨率，例如 `64x64`、`128x128`、`256x256`；
3. 更长视频长度；
4. 更明确的任务设定，例如 `short-video prediction`、`video generation`、`action recognition`。

但我们仍然可以先把**公开常见指标**接入当前项目，至少让评估口径从“纯内部指标”升级为“主流社区常见指标”。

## 2. 公开口径里常见的 UCF101 指标

### 2.1 短视频预测

FAR 的公开模型页给出了 UCF101 `short-video prediction` 的一组标准指标：

1. `PSNR = 25.64`
2. `SSIM = 0.818`
3. `LPIPS = 0.037`
4. `FVD = 194.1`

来源：

1. FAR model zoo（UCF101 short-video prediction）  
   https://huggingface.co/guyuchao/FAR_Models

需要注意，这组结果对应的是它自己的 UCF101 预测设定与更标准的公开评测分辨率，不等同于当前仓库的 `8x32x32` 子集设定。

### 2.2 视频生成

公开 UCF101 视频生成论文通常更强调 `FVD`。例如：

1. FAR README 中 UCF101 generation 直接汇报 `FVD`
2. VideoAR 摘要明确写到：
   - UCF-101 上 FVD 从 `99.5` 提升到 `88.6`

来源：

1. FAR model zoo（UCF101 generation）  
   https://huggingface.co/guyuchao/FAR_Models
2. VideoAR abstract  
   https://arxiv.org/abs/2601.05966

## 3. 为什么当前不把 FVD 当第一优先项

公开实现与常用工具链对 FVD 有一个很现实的约束：通常要求更标准的长视频长度。

例如 `common_metrics_on_video_quality` 明确写到：

1. 可以计算 `FVD / PSNR / SSIM / LPIPS`
2. 但 `FVD` 需要 `frames_num > 10`

来源：

1. https://github.com/JunyaoHu/common_metrics_on_video_quality

而当前主线配置默认是：

1. `seq_len = 8`

因此，**在当前默认真实数据设置上直接引入 FVD，并不能和公开 UCF101 结果形成严肃对比**。

## 4. 当前最合理的第一步

在不重写数据口径的前提下，当前最合理的外部评估升级是：

1. 先在真实数据 checkpoint 上补 `PSNR`
2. 再补 `SSIM`
3. `LPIPS` 作为可选项
4. 等评估口径迁到更标准的视频长度/分辨率后，再补 `FVD`

这也是当前仓库已经开始落地的方案：

1. 新增标准评估脚本 [eval_standard_video_metrics.py](/home/void0312/AIGC/VH-01/scripts/eval_standard_video_metrics.py)
2. 当前先稳定输出 `recon/future` 两组 `MSE + PSNR + SSIM`

## 5. 真正需要记住的边界

当前跑出来的 `PSNR/SSIM`：

1. 是**标准指标**
2. 但不是**对公开 UCF101 SOTA 的直接同口径对比**

要做到后者，后面至少还需要统一：

1. 数据切分
2. 分辨率
3. 帧长
4. 任务定义（prediction / generation / recognition）
5. FVD/LPIPS 的实现与采样协议
