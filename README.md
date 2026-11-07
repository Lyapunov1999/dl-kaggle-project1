# CIFAR-10 训练实验

## 运行

先安装 [requirements.txt](./requirements.txt) 中的依赖。训练使用 [train_images/](./train_images/) 的图片和原始 [train_labels.csv](./train_labels.csv)，不会下载数据或修改标签文件。原始 CSV 的 `id,filename,label` 按行绑定，图片路径直接使用 `filename`；也兼容旧的 `id,label` 格式。

```powershell
python train.py --smoke-test
python train.py --epochs 10 --batch-size 256 --lr 0.001
```

默认使用 Adam 优化器，学习率为 `0.001`，权重衰减为 `0.0005`，保留余弦退火学习率调度。可以用 `--seed`、`--weight-decay`、`--workers` 等参数调整实验。完整参数见 `python train.py --help`。

模型保留残差结构，每个阶段各含 2 个残差块，通道数为 `(48, 96, 192, 320)`，共有 **4,947,770** 个可训练参数，符合项目不超过 5,000,000 个参数的要求。

## 每次实验的记录

每次正式训练都会创建一个独立的 `runs/<UTC时间>_<随机编号>/` 目录，里面有：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 命令、超参数、模型与数据配置、设备和软件版本 |
| `metrics.csv` | 每轮训练与验证的 loss、accuracy、学习率、耗时和最佳轮标记 |
| `train.log` | 终端上看到的训练进度与错误信息 |
| `summary.json` | 是否完成、最佳验证准确率、最佳轮数和权重路径 |
| `best_resnet.pt` | 验证准确率最高时的模型权重，以及该次实验的配置 |
| `split.json` | 实际训练/验证 ID、划分种子、原始 CSV 及样本映射 SHA-256 |

`metrics.csv` 可以直接用 Excel 打开，用 `epoch` 作横轴绘制 loss 和 accuracy 曲线。准确率以 0 到 1 的小数记录，例如 `0.85` 代表 85%。

默认权重保存在本次实验目录中，因此多次运行不会相互覆盖。需要指定位置时使用 `--output`；需要更换实验目录时使用 `--runs-dir`。`runs/` 已被 Git 忽略，请自行备份需要长期保存的实验结果。

代码按职责分为 `train.py`（入口和训练循环）、`model.py`（网络结构）、`data.py`（变换与数据切分）、`experiment.py`（实验记录）。

## 预测与提交

必须明确指定本次要使用的模型路径。将下面路径替换为训练完成后打印的最佳权重路径：

```powershell
python predict.py --checkpoint runs/<run_id>/best_resnet.pt --sample-submission sample_submission.csv --output submission.csv
```

预测从权重配置恢复归一化参数，按 [sample_submission.csv](./sample_submission.csv) 的 ID 和行序输出，遇到重复、缺失或额外 ID 会报错。终端显示模型路径、epoch 和保存的验证准确率；提交文件旁的 `submission.csv.metadata.json` 记录模型、模板和提交文件的 SHA-256，便于核对实际提交用了哪次训练。新脚本不再固定使用某个历史运行的模型。

## 数据核查

```powershell
python audit_data.py
python -m unittest discover -s tests -v
```

[audit_data.py](./audit_data.py) 使用已有的 [data/cifar-10-batches-py/](./data/cifar-10-batches-py/) 做本地像素和训练标签核对，并写入 [outputs/data_audit.json](./outputs/data_audit.json)。它不下载数据、不使用官方测试标签，也不修改任何输入。

本次检查的结论和限制见 [AUDIT_REPORT.md](./AUDIT_REPORT.md)。随机划分的 validation 只衡量训练来源下的泛化能力；独立来源测试集的 Kaggle 分数可能明显更低。
