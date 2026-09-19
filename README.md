# 射影矩阵自编程实现与透视矫正

给定一张斜拍平面物体的照片和它的四个角点，求出把该平面矫正成正视图的射影矩阵 `H`。

主算法全部自己实现：DLT 方程矩阵的构造、点归一化与反归一化、齐次方程的 SVD 求解、从目标像素反查源坐标的逆向映射、以及双线性插值。OpenCV 只用在图像读写、窗口选点和作为对照的工程基线，不参与主算法的求解。除 NumPy 与 OpenCV 外没有其他依赖。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python generate_sample.py                     # 生成合成斜拍图与真值 sidecar
python gt_experiment.py --jitter-repeats 100  # 真值闭环实验，产出定量结论
python visualize_experiments.py --output-dir outputs --ablation-repeats 300
python run.py --input-dir data --batch --no-interactive --output-dir outputs
```

最后一条会把 13 张真实图片一次跑完，不需要任何鼠标操作。全部结果落在 `outputs/`，实验报告是 `射影矩阵自编程实现与透视矫正实验报告.md`。

## 作业要求与交付物对照

| 作业要求 | 在本项目中的实现 | 在哪里看到 |
|---|---|---|
| 用 OpenCV 等库做图像预处理 | `io_utils.read_image` 经 `np.fromfile` + `cv2.imdecode` 读图，支持中文路径 | — |
| 标定斜拍图片的四个角点 | OpenCV 窗口交互选点，可存成 JSON 复用 | `point_picker.py`、各图片目录下的 `corners/*.json` |
| 构造射影矩阵的齐次线性方程组 | 手写 DLT 矩阵，每对点贡献两个方程 | `homography.build_dlt_matrix` |
| 用 SVD 求解 `H` | 手写求解流程，取 `V` 最后一行；另有 Jacobi 特征分解的教学版实现 | `homography.solve_homography`、`educational_svd.py` |
| 输出图像分辨率 | 每次运行的元数据 | `outputs/**/metadata.json` |
| 输出四个角点 | 点在图上取、存进元数据与角点 JSON | `outputs/**/metadata.json` |
| 输出正图片的像素矩阵 | 完整的目标图像素矩阵，`*.npy` | `outputs/**/pixels_*.npy` |
| 输出矫正后的正视图 | 三种方法各一张 PNG | `outputs/**/rectified_*.png` |

图片统一按长边缩放到 1600 px 后处理，原图与缩放规则记录在 `data/public/SOURCES.md`。

## 文档导航

| 文档 | 内容 |
|---|---|
| `第一章作业.md` | 老师给的作业原文 |
| `射影矩阵自编程实现与透视矫正实验报告.md` | 实验报告正文，含全部结果表 |
| `实验遇到的问题及处理方法.md` | 实验过程中实际遇到的 13 个问题及其原因和处理方式 |
| `data/public/SOURCES.md` | 公开样例图片的来源、作者、授权与引用要求 |

## 目录结构

```text
homography.py            DLT 矩阵、基础/归一化求解、投影、拟合残差、留出点转移误差
warping.py               手写逆向映射与双线性插值（按行分条带处理），OpenCV 对照
validation.py            角点与对应点有效性检查
point_picker.py          交互选点，可导出角点 JSON
io_utils.py              图像读写、真值 sidecar、汇总表合并写入
run.py                   单张与批处理入口
generate_sample.py       生成合成斜拍图与真值 sidecar
gt_experiment.py         真值闭环实验
visualize_experiments.py 消融实验与出图
paired_vs_unpaired.py    量化非配对设计造成的标准误放大（报告 §9.6 的数据来源）
educational_svd.py       基于 Jacobi 特征分解的教学版 SVD
fetch_public_samples.py  拉取公开数据集样例
tests/                   44 条测试
data/sample/             合成样本及其真值
data/public/             9 张公开样例 + corners/ 角点
data/phone/              4 张自拍照片 + corners/ 角点
outputs/                 运行产物（汇总表、逐图结果、配图）
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 准备数据

```powershell
python generate_sample.py            # 合成样本，真值精确已知
python fetch_public_samples.py       # 公开样例，默认 5 张约 5 MB
python fetch_public_samples.py --with-docunet   # 额外 4 张，需下载 344 MB 压缩包
```

自己的照片放进 `data/phone/`，角点放在同级的 `corners/` 目录下：

```powershell
python point_picker.py --input data/phone/my_photo.jpg   # 点选一次，写入 corners/my_photo.json
```

> `data/public/*` 与 `data/phone/*` 下的图片被 `.gitignore` 排除，换机器后需要用上面的命令重新拉取或手工拷贝。角点 JSON 例外，它们会随仓库提交，因此把照片放回原位即可无交互重跑。

## 单张图片

交互选点，顺序为左上、右上、右下、左下。点击四角后按 `Enter` 确认，`R` 重选，`Esc` 或 `Q` 取消。

```powershell
python run.py --input data/sample/sample_slanted.png --output-dir outputs
```

合成样本自带真值，可以不弹窗直接跑：

```powershell
# 目标矩形由程序按对边长度自动估计
python run.py --input data/sample/sample_slanted.png --points-from-ground-truth --output-dir outputs

# 目标矩形固定为真值尺寸，消除宽高比偏差
python run.py --input data/sample/sample_slanted.png --points-from-ground-truth --output-dir outputs --output-size 640 420
```

没有真值 sidecar 的图片用角点 JSON 固定，数组顺序为 TL, TR, BR, BL：

```json
[[155, 95], [765, 70], [835, 590], [85, 625]]
```

```powershell
python run.py --input data/public/chessboard.jpg --points-file data/public/corners/chessboard.json --output-dir outputs
```

`--output-size W H` 可以对任意图片固定目标矩形，用来消除宽高比偏差，背景见报告 §3.6。单张模式的结果按 `outputs/single/<图片名>/` 归置。

## 批处理

```powershell
python run.py --input-dir data --batch --no-interactive --output-dir outputs
```

批处理逐张查找角点文件，找到就直接求解，找不到就跳过并在控制台说明原因。查找顺序是：

1. `--points-dir <目录>` 指定的目录，给了这个参数就只找这一处；
2. 图片所在目录下的 `corners/<图片名>.json`；
3. `<input-dir>/corners/<图片名>.json`。

`data/public/` 与 `data/phone/` 都按第 2 条存放，因此上面这条命令不需要额外参数。结果按来源分别落在 `outputs/public/`、`outputs/phone/` 下，`summary.csv` 汇总全部指标。

每张图片产出：

- `rectified_basic_dlt.png`、`rectified_normalized_dlt.png`、`rectified_opencv.png`
- `pixels_*.npy`：完整的目标图像素矩阵
- `metadata.json`：分辨率、角点、`H`、奇异值、各项指标与计时

> `summary.csv` 是合并写入的：已有的行会保留，只有"同一来源 + 同一图片 + 同一方法 + 同一目标矩形"才会被本次运行替换。这样单张运行、批处理、以及同一张图的两个目标尺寸可以共存于一份汇总表。要从零开始加 `--overwrite-summary`。

## 真值闭环实验

合成样本的真值精确已知，`gt_experiment.py` 用它做定量验证：

```powershell
python generate_sample.py                       # 生成样本与真值 sidecar
python gt_experiment.py --jitter-repeats 100    # 三组实验 + 六联图
```

产出 `outputs/gt_experiment/results.csv`、`outputs/gt_experiment/report.md` 与 `outputs/figures/gt_experiment.png`。三组实验分别是实现正确性验证、目标矩形选择的影响、以及角点抖动扫描。

## 消融实验

```powershell
python visualize_experiments.py --output-dir outputs --ablation-repeats 300
```

扫描目标点噪声、对应点数量和坐标尺度三个变量，每档重复 300 次并报告 95% 置信区间。采用配对设计：同一次抽样中两种方法拟合同一份数据，比较配对差。判显著要求 `|t| > 1.96` 且 `|配对差| > 1e−6 px`，只满足前者不算。完整数值在 `outputs/ablation/report.md`。

## 运行测试

```powershell
python -m pytest -q
```

共 44 条，覆盖四点精确解与超定最小二乘、归一化稳定性、退化点校验、手写双线性插值与分块逐位一致性、大图内存上限、教学版 Jacobi SVD、留出点网格与转移误差、指标退化回归、汇总表合并规则、以及中文路径的图像读写往返。

## 数学约定

`H` 把源图像坐标映射到目标正视图坐标：

```text
x_dst ~ H x_src
```

四角对应关系为：

```text
source:      TL, TR, BR, BL
destination: (0,0), (W-1,0), (W-1,H-1), (0,H-1)
```

齐次矩阵存在整体尺度不确定性，程序不强制 `h33 = 1`，而是用 Frobenius 范数做内部规范化；矩阵比较前先做最佳尺度对齐。

`destination_corners(W, H)` 里的 `W` 和 `H` 是一个自由参数。四点对应只能把四边形映到"你所指定的那个矩形"上，无法确定正确的宽高比，这就是报告 §3.6 讨论的度量不确定性。选错宽高比会带来约 95 像素的真实几何误差，量级远大于角点标定误差。

## 两个容易踩的地方

第一，不要显式把 SVD 写成 `full_matrices=False`。DLT 矩阵是 8×9，只有完整形式的 `V` 是 9×9，`vh[-1]` 才对应那个恒为零的奇异值。改成 `False` 后 `vh[-1]` 变成第 8 行，重投影误差从 2e−09 涨到 606.78 像素，而且不会报错。顺带说明，`np.linalg.svd` 的默认值本来就是 `True`，所以删掉这个参数不影响结果，危险的是把它改成 `False`；代码里写全是为了显式，不是为了覆盖默认值。

第二，四点场景下的重投影误差不能当作精度指标。4 对点给出 8 个方程，而 `H` 恰好有 8 个自由度，属于恰定问题，拟合残差在数学上恒为 0。实测三个方法分别是 7.47e−09、3.16e−13、5.95e−14，相差 5 个数量级但都是机器零，按它排序会得出"基础 DLT 比 OpenCV 差十万倍"这种明显错误的结论。判断矫正质量要用留出点转移误差：在源四边形内部取一组不参与拟合的网格点，比较估计矩阵与真值矩阵在这些点上的投影差异。该指标需要真值，由 `generate_sample.py` 写出的 sidecar 提供。

## 误差来源

按量级排序，合成样本上的实测结果是：

| 误差来源 | 量级 | 说明 |
|---|---:|---|
| 目标矩形选择 | ~95 px | 输出宽高比必须先验给定，角点标定得再准也无法补救 |
| 角点标定 | ~0.93 × σ | σ = 1 px 时约 0.93 px，可用更多点做最小二乘抑制 |
| 数值条件 | 1e−09 ~ 1e−13 px | 归一化 DLT 把条件数从 1.13e6 降到 3.88 |
| 图像重采样 | < 1 灰度级 | 与 OpenCV 的 MAE 在 1.2e−04 ~ 0.38 之间 |
