# 第一章：射影矩阵自编程实现

本项目实现斜拍图片的透视矫正，并对比基础 DLT、Hartley 归一化 DLT 和 OpenCV 工程基线。

## 0. 文档导航

| 文档 | 用途 | 什么时候看 |
|---|---|---|
| `第一章加分作业.md` | 老师的作业原文 | 先看，明确要求 |
| **`PyCharm操作复现指南.md`** | **从零到交作业的完整流程**：每期跑什么、运行后出现什么、结果怎么看 | **做作业时照着走** |
| `终端操作复现指南.md` | 同上，但用 PowerShell 命令行而非 IDE | 想用命令行时 |
| **`学习计划.md`** | **三天学习计划**：核心思想的几何直觉、代码阅读顺序、7 个"改一行看结果"的实验、答辩自测题 | **准备理解原理和答辩时** |
| `实验报告.md` | 实验报告正文（已含真值闭环实验的定量结论） | 最后填 |
| `优化计划书.md` | 对现有代码与实验的审计结论 + 分期优化计划（P0、P1 已完成） | 想知道"哪里还有问题、怎么改"时 |
| `data/public/SOURCES.md` | 公开样例图片的来源、作者、授权、引用要求 | 填报告署名时 |

## 1. 作业要求与实现边界

作业要求的是：标定斜拍图片的四个角点，构造射影矩阵的齐次线性方程组，用 SVD 求解 `H`，并输出图像分辨率、角点、像素矩阵和正视结果。

本项目的主算法手写了 DLT 矩阵、点归一化、反归一化和像素逆向映射。SVD 分解调用 NumPy 的成熟数值实现；`educational_svd.py` 额外提供基于 Jacobi 特征分解的学习版实现。OpenCV 只作为图像读取、交互选点和工程基线，不作为主算法求解器。

### ⚠️ 两个必须知道的坑

**坑 1：`full_matrices=True` 是承重参数，不能删。**

主算法里 `np.linalg.svd(..., full_matrices=True)` 这个非默认参数是必需的。DLT 矩阵是 8×9 的长方阵，只有完整形式的 `V` 才是 9×9，`vh[-1]` 才对应那个恒为零的第 9 个奇异值。改成 NumPy 默认的 `full_matrices=False`，`vh[-1]` 会变成第 8 个右奇异向量，**重投影误差从 2e−09 变成 606.78 像素，且不会报错**。详见 `学习计划.md` 的 1.3 节核心 ④。

**坑 2：四点场景下"重投影误差"是退化指标，不能当误差用。**

4 对点给出 8 个方程，而 `H` 恰有 8 个自由度——恰定问题，拟合残差**在数学上恒等于 0**。实测 basic_dlt / normalized_dlt / opencv 三者的拟合残差为 7.47e−09 / 3.16e−13 / 5.95e−14，相差 5 个数量级但**全都是机器零**。按它排序会得出"基础 DLT 比 OpenCV 差 10 万倍"的荒谬结论。

判断矫正质量要看**留出点转移误差**（`transfer_rmse_vs_truth_px`）：在源四边形内部取一组不参与拟合的网格点，比较估计矩阵与真值矩阵的投影差异。该指标需要真值，由 `generate_sample.py` 写出的 sidecar 提供。

## 2. 安装

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

当前环境如果没有 OpenCV，安装依赖后才能运行图片读取、窗口选点、样例图生成和 OpenCV 对照。

## 3. 准备数据

生成合成示例图（真值精确已知）：

```powershell
python generate_sample.py
```

拉取公开数据集样例到 `data/public/`：

```powershell
python fetch_public_samples.py
```

默认从 Wikimedia Commons 拉 5 张（约 5 MB）；加 `--with-docunet` 可额外拉取 DocUNet 学术数据集的 4 张（同一文档的两个视角，会下载 344 MB 压缩包一次）。每张图片的来源、作者与授权见 `data/public/SOURCES.md`。

自己的手机照片请放入 `data/phone/`。

> 注意：`data/public/*` 与 `data/phone/*` 被 `.gitignore` 排除，换电脑后不会随仓库迁移，需重新执行上面的命令或手动拷贝。`data/public/SOURCES.md` 与 `data/public/corners/*.json` 例外，会被提交。

## 4. 单张图片

交互选点，顺序必须是左上、右上、右下、左下：

```powershell
python run.py --input data/sample/sample_slanted.png --output-dir outputs
```

窗口操作：点击四个角点，按 `Enter` 确认，按 `R` 重选，按 `Esc` 或 `Q` 取消。

合成样本自带真值，可直接复现（不弹窗）：

```powershell
# 自动估计目标尺寸
python run.py --input data/sample/sample_slanted.png --points-from-ground-truth --output-dir outputs

# 固定为真值尺寸，消除宽高比偏差
python run.py --input data/sample/sample_slanted.png --points-from-ground-truth --output-dir outputs --output-size 640 420
```

其他图片若没有 sidecar，用 JSON 固定角点（顺序 TL, TR, BR, BL）：

```json
[[155, 95], [765, 70], [835, 590], [85, 625]]
```

```powershell
python run.py --input data/public/chessboard.jpg --points-file data/public/corners/chessboard.json --output-dir outputs
```

`--output-size W H` 可对任意图片固定目标矩形，用于消除宽高比偏差（见 `实验报告.md` §3.6）。

## 5. 批处理

```powershell
python run.py --input-dir data --batch --output-dir outputs
```

批处理会对每张图片分别交互选点，并按来源和文件名保存结果。`summary.csv` 汇总各方法的指标。

每张图片的结果包括：

- `rectified_basic_dlt.png`、`rectified_normalized_dlt.png`、`rectified_opencv.png`
- `pixels_*.npy`：完整输出像素矩阵
- `metadata.json`：分辨率、角点、`H`、奇异值、各指标与计时

> ⚠️ `summary.csv` 是**覆盖写**的，不是追加。先跑单张再跑批处理会清掉单张的行，需要保留就先复制一份。

## 6. 真值闭环实验

合成样本的真值是精确已知的，`gt_experiment.py` 用它做定量验证：

```powershell
python generate_sample.py                       # 生成样本 + 真值 sidecar
python gt_experiment.py --jitter-repeats 100    # 三部分实验 + 六联图
```

产出：

- `outputs/gt_experiment/results.csv`、`outputs/gt_experiment/report.md`
- `outputs/figures/gt_experiment.png`

三部分内容分别是：实现正确性验证、目标矩形选择的影响、角点抖动扫描。

## 7. 运行测试

```powershell
python -m pytest -q
```

测试覆盖四点精确解、四点以上最小二乘、归一化稳定性、退化点校验、手写双线性插值、教学版 Jacobi SVD、留出点网格与转移误差、指标退化回归、CSV 列合并。

## 8. 数学约定

`H` 将源图像坐标映射到目标正视图坐标：

```text
x_dst ~ H x_src
```

四个角点对应关系为：

```text
source:      TL, TR, BR, BL
destination: (0,0), (W-1,0), (W-1,H-1), (0,H-1)
```

由于齐次矩阵存在整体尺度不确定性，程序不强制令 `h33 = 1`，而是使用 Frobenius 范数进行内部规范化。矩阵比较会先做最佳尺度对齐。

**`destination_corners(W, H)` 里的 `W, H` 是一个自由参数**：四点对应只把四边形映到"你所指定的那个矩形"上，无法确定正确的宽高比。这就是 `实验报告.md` §3.6 讨论的度量不确定性——选错宽高比会带来约 95 像素的真实几何误差，远大于角点标定误差。

## 9. 实验建议

1. **先做真值闭环实验**（第 6 节）——这是唯一能给出定量结论的实验，成本最低。
2. 记录目标矩形取"自动估计"与"真值"两种情况下转移误差的差异（预期相差 10 个数量级）。
3. 扫描角点抖动 σ，观察误差是否与 σ 近似成正比。
4. 在 n>4 与不同坐标量级下比较基础 DLT 与归一化 DLT（注意：四点时两者给出相同的解，对照是空的）。
5. 记录手写正视图与 OpenCV 正视图的 MAE（预期约 1e−04，属于实现一致性验证）。
6. 比较示例图、公开图片和手机照片在角点选择难度与视觉效果上的差异。

> 消融实验采用**配对设计**（每档重复 300 次，95% 置信区间），判显著要求 `|t| > 1.96` **且** `|配对差| > 1e−6 px`。完整数值见 `outputs/ablation/report.md`。
>
> 早期版本曾用非配对随机化，在 n = 4 处报出"归一化更差"的方向错误结论；该缺陷已修复，详见 `实验报告.md` §9.6。
