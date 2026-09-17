# `data/public/` 数据来源与授权说明

本目录存放用于射影矫正实验的公开数据样例。所有图片均为公开来源，逐张记录来源、作者与授权，供作业引用与答辩说明使用。

**重要：本目录下的图片被 `.gitignore` 排除**（`data/public/*`），不会进入版本库。换电脑后请用项目根目录的 `fetch_public_samples.py` 重新拉取，或手动复制本目录。

---

## 一、图片来源一览

| 文件 | 来源 | 作者 | 授权 | 原始分辨率 | 本目录分辨率 |
|---|---|---|---|---|---|
| `chessboard.jpg` | Wikimedia Commons | Tiia Monto | CC BY-SA 4.0 | 3704×2864 | 1600×1238 |
| `book_cover.jpg` | Wikimedia Commons | Wayne Brezinka | CC0 | 2233×1533 | 1600×1098 |
| `flipchart_venn.jpg` | Wikimedia Commons | Markus Bärlocher | Public domain | 2476×3220 | 1230×1600 |
| `flipchart_table.jpg` | Wikimedia Commons | Markus Bärlocher | Public domain | 2560×3532 | 1160×1600 |
| `menu_board.jpg` | Wikimedia Commons | Peachyeung316 | CC BY-SA 4.0 | 2784×3712 | 1200×1600 |
| `docunet_29_view1.jpg` | DocUNet benchmark | Ke Ma 等（石溪大学 / Megvii） | 学术数据集，需引用 | 3024×4032 | 1200×1600 |
| `docunet_29_view2.jpg` | DocUNet benchmark | 同上 | 学术数据集，需引用 | 3024×4032 | 1200×1600 |
| `docunet_30_view1.jpg` | DocUNet benchmark | 同上 | 学术数据集，需引用 | 3024×4032 | 1200×1600 |
| `docunet_30_view2.jpg` | DocUNet benchmark | 同上 | 学术数据集，需引用 | 3024×4032 | 1200×1600 |

所有图片已按长边 1600 px 缩放（作业原文允许"适当降低分辨率来简化运算"），JPEG 质量 92。原始分辨率为下载源的实际尺寸。

## 二、逐张说明

### Wikimedia Commons（5 张）

| 文件 | 内容 | 为什么选它 | 文件页 |
|---|---|---|---|
| `chessboard.jpg` | 草地上的白桌，桌面棋盘 | 规则网格纹理，矫正是否正确一眼可判；网格是最强的可视化判据 | [File:Chessboard 2.jpg](https://commons.wikimedia.org/wiki/File:Chessboard_2.jpg) |
| `book_cover.jpg` | 一本化学教材的封面 | CC0 授权最宽松；书脊与封面构成明确的平面矩形 | [File:Book-cover.jpg](https://commons.wikimedia.org/wiki/File:Book-cover.jpg) |
| `flipchart_venn.jpg` | 翻页板上的 Venn 图 | 高对比线条 + 手写文字，可检验文字是否被拉直 | [File:Flipchart Motivation03.jpg](https://commons.wikimedia.org/wiki/File:Flipchart_Motivation03.jpg) |
| `flipchart_table.jpg` | 翻页板上的 "To do - Liste" 表格 | 表格横线可直接观察是否保持水平 | [File:Flipchart ToDo-Liste.jpg](https://commons.wikimedia.org/wiki/File:Flipchart_ToDo-Liste.jpg) |
| `menu_board.jpg` | 香港街边黑色 A 字菜单牌 | 真实户外场景，斜拍角度明显，背景杂乱更接近"随手拍" | [File:A black menu board from Kau Yee Hong Kong style restaurant.jpg](https://commons.wikimedia.org/wiki/File:A_black_menu_board_from_Kau_Yee_Hong_Kong_style_restaurant.jpg) |

授权说明：CC BY-SA 4.0 要求署名并以相同方式共享；CC0 与 Public domain 无附加要求。作业报告中建议按上表署名。

### DocUNet benchmark（4 张）

`docunet_29_*` 与 `docunet_30_*` 各是**同一份文档的两个不同视角**。这一点对本作业有额外价值：除了"四边形 → 矩形"的矫正，还可以额外做"两张照片之间的单应性估计"这一组实验，并且能用两个视角互相交叉验证。

DocUNet 的原始数据里还有平板扫描仪扫描的**无畸变真值**（`scan.zip`），如果需要做像素级定量对比（本作业目前缺的就是这个），可以从该数据集补齐。注意 DocUNet 面向的是**文档去弯折（dewarping）**，其中相当一部分照片存在纸张卷曲或折叠，**不满足平面假设**——这些样本正好可以用来演示本作业方法的适用边界。

- 数据集主页：https://www3.cs.stonybrook.edu/~cvl/docunet.html
- 原始照片包：http://vision.cs.stonybrook.edu/~kema/docwarp/original.zip （344 MB，含 130 张，均为同一文档的两个视角）
- 扫描真值包：http://vision.cs.stonybrook.edu/~kema/docwarp/scan.zip （416 MB）

**引用要求**（使用该数据集必须引用）：

> Ke Ma, Zhixin Shu, Xue Bai, Jue Wang, Dimitris Samaras.
> *DocUNet: Document Image Unwarping via A Stacked U-Net.*
> Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

## 三、角点种子值

`corners/` 子目录存放 5 张图片的**种子角点**（JSON 数组，顺序为 TL, TR, BR, BL），均由自动检测生成并**逐张渲染验证过**：

| 种子文件 | 状态 |
|---|---|
| `chessboard.json` | ✅ 已验证，四边形准确套住棋盘 |
| `docunet_29_view1.json` | ✅ 已验证，准确套住纸张 |
| `docunet_29_view2.json` | ✅ 已验证 |
| `docunet_30_view1.json` | ✅ 已验证 |
| `docunet_30_view2.json` | ✅ 已验证 |

以下 4 张**没有**种子角点，需要在 `run.py` 窗口里手动点选：

| 文件 | 原因 |
|---|---|
| `menu_board.jpg` | 黑板与深色背景对比不足，自动检测失败；需手动点选 |
| `book_cover.jpg` | 书本几乎占满画面，透视量很小 |
| `flipchart_venn.jpg` | 翻页板几乎占满画面，接近正视 |
| `flipchart_table.jpg` | 同上 |

后三张虽然不适合作为"斜拍矫正"的主样本，但可以作为**近正视对照样本**使用：对接近正视的输入做矫正，输出应该几乎不变。这本身是一个有价值的对照（可以检验矫正流程不会凭空引入形变）。

种子角点的用法（单张、非交互）：

```powershell
python run.py --input data/public/chessboard.jpg `
              --points-file data/public/corners/chessboard.json `
              --output-dir outputs
```

> 批处理模式 `--input-dir` 目前不支持逐张指定角点文件（`--points-file` 仅对 `--input` 生效），因此这几张样例目前只能逐张跑，或在窗口里逐张点选。这一限制已记录在 `优化计划书.md` 的 C5 条目中。

## 四、数据量与用途分配

| 来源 | 张数 | 报告中的位置 |
|---|---|---|
| `data/sample/sample_slanted.png` | 1（程序生成，含精确真值） | 合成数据、真值闭环实验 |
| `data/public/`（本目录） | 9 | 报告 §8.2 / §11.2 "public" 行 |
| `data/phone/` | 待补充（自行拍摄） | 报告 §8.2 / §11.2 "phone" 行 |

共 5.4 MB，不构成版本库负担；但因 `.gitignore` 规则不会提交，换机需重新拉取。

## 五、重新拉取

```powershell
# 仅 Commons 部分（约 5 MB，快）
python fetch_public_samples.py

# 同时拉取 DocUNet 部分（会下载 344 MB 压缩包，之后只解压需要的 4 张）
python fetch_public_samples.py --with-docunet
```
