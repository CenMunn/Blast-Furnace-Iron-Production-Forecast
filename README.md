# 高炉铁次预测：王玮数据工程交付

本仓库收录王玮负责的数据读取与清洗、按月扩展窗口划分、G7 铁次节奏特征、G8 变料状态时间特征，以及生成后的 Parquet 数据和验证代码，供石朔冰接入模型、组内复核与复现。

当前时间特征交付版本为 `data_v4_timeaware.g7_g8.v1`：G7 **40 列**、G8 **133 列**，共 **173 列**。全训练集 2754 行、test A 657 行、test B 548 行。该目录是独立的 G7/G8 特征增量；需要全量模型输入时，按 `sample_id` 与相同 Fold、split 的原 data_v3 特征表一对一关联，不能只依赖行号。

## 目录与文件用途

```text
.
├── README.md
├── requirements.txt
├── src/                         # 数据处理、特征、构建和验证源码
├── tests/                       # 单元测试、集成测试、破坏性验证夹具
├── raw/                         # 原始初赛数据、字段字典及结果模板
└── data_v4_timeaware/           # 本次已生成的交付，共19个文件
    ├── folds/
    │   ├── fold_1_train.parquet # fold_1 至 fold_5，分别有train/valid
    │   ├── fold_1_valid.parquet
    │   └── full_train.parquet   # 2754行全训练数据
    ├── test_a.parquet           # 657行，无目标标签
    ├── test_b.parquet           # 548行，无目标标签
    ├── manifests/
    │   ├── feature_dictionary.md # 逐列特征字典
    │   ├── build_config.json     # 列顺序、Fold定义、6套填补状态与构建配置
    │   ├── source_manifest.json  # 六份来源CSV的SHA-256、行数和时间范围
    │   └── leakage_audit.md      # 交付验证审计结果
    └── reports/
        ├── schedule_feature_stats.csv    # G7，3个上下文×40列=120行
        └── burden_time_feature_stats.csv # G8，3个上下文×133列=399行
```

`src/` 同时保留上一步 data_v3 的处理模块和测试所需依赖。`raw/` 已包含原始初赛数据，共10个文件，包括6份来源CSV、`data_dictionary.xlsx`、`result_template.csv`及两份原始README。重建和来源哈希验证使用该目录；读取现成Parquet、运行测试夹具不需要加载原始数据。

## 环境准备

验证环境为 Python 3.14；依赖版本见 `requirements.txt`。在仓库根目录执行：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

以下命令均从仓库根目录执行。Linux/macOS 可用对应Python解释器建立虚拟环境并执行相同的 `python -m ...` 命令。

## 直接读取交付数据

Parquet 是保留字段类型的列式数据文件，本项目使用 Snappy 压缩；用 pandas 和 pyarrow 即可读取。

```python
from pathlib import Path
import numpy as np
import pandas as pd
from src.timeaware_contract import G7_FEATURE_NAMES, G8_FEATURE_NAMES

root = Path("data_v4_timeaware")
train = pd.read_parquet(root / "folds/fold_1_train.parquet")
valid = pd.read_parquet(root / "folds/fold_1_valid.parquet")
test_a = pd.read_parquet(root / "test_a.parquet")
features = [*G7_FEATURE_NAMES, *G8_FEATURE_NAMES]
X_train = train.loc[:, features].to_numpy(dtype=np.float32)
X_valid = valid.loc[:, features].to_numpy(dtype=np.float32)
y_train = train.loc[:, ["tap_iron", "tap_time_len"]]
assert X_train.shape[1] == 173
assert np.isfinite(X_train).all() and np.isfinite(X_valid).all()
assert train["sample_id"].is_unique
```

所有文件前置字段依次为 `sample_id, reference_time, tap_no, spout_no, fold, split`，随后按注册表顺序排列173个模型特征。训练/验证文件在模型特征前另含 `tap_iron`（出铁量）、`tap_time_len`（出铁时长）和 `target_rate = tap_iron / tap_time_len`；test A/B 不含这三个目标字段。模型特征统一为有限 `float32`，不能将ID、时间或目标列混入特征矩阵。

## 五折月份划分

当前版本按2024年月份采用扩展窗口，每折训练时刻严格早于验证时刻，不随机打乱。Fold编号为1—5；full_train/test A/test B的 `fold` 标记为0。

| Fold | 训练月份 | 验证月份 | 训练行数 | 验证行数 |
|---|---|---|---:|---:|
| 1 | 3—6月 | 7月 | 1181 | 310 |
| 2 | 3—7月 | 8月 | 1491 | 313 |
| 3 | 3—8月 | 9月 | 1804 | 288 |
| 4 | 3—9月 | 10月 | 2092 | 333 |
| 5 | 3—10月 | 11月 | 2425 | 329 |

## 数据清洗与时间可见性

- 读取时统一时间类型，校验必需字段、样本ID全局唯一性、整数标识、训练目标有限性；非法关键字段直接报错。
- 每折仅用当前训练行的原始特征拟合中位数；NaN/Inf按训练中位数填补，训练全缺失列填0。全训练集另外拟合一套状态供full_train/test A/test B共用。输出不做标准化，填补状态记录在配置中。
- G7只使用严格早于当前 `reference_time` 的公开铁次时间；同一时刻的样本互不可见，计数窗口为 `[T-window, T)`，滚动统计排除当前gap。验证块内更早样本的公开时刻可更新后续G7，目标标签不参与计算。test A上下文为train+test A，test B为train+test A+test B，均只向过去查询。
- G8按 `cal_time, _original_row` 稳定排序，同时间戳保留原文件最后一行。本次来源有1条重复时间戳记录。每字段独立判断真实变化，非有限输入不覆盖已有状态；连续相同状态计原始事件，不计有效变化。
- G8默认回看48小时、每2小时一点，共25个旧到新网格点。查询采用 `cal_time <= grid_time <= reference_time`，前向保持状态；均值按实际持续时间加权，趋势以真实小时计算。

G7覆盖开铁间隔lag、历史分布与趋势、时间窗铁次数、同铁口间隔、长间隔标记、分钟分数。G8覆盖5个变料字段的当前状态、状态年龄、2/4/8/12/24/48小时lag、差值、时间加权均值、真实时间趋势、相邻变化幅度、有效变化次数及全局事件统计。精确列名与顺序见 [特征字典](data_v4_timeaware/manifests/feature_dictionary.md) 和 [构建配置](data_v4_timeaware/manifests/build_config.json)。

## 从原始数据重建

仓库已包含如下 `raw/` 原始目录，可直接执行后面的重建命令，也可以传入其他绝对路径：

```text
raw/
├── train/
│   ├── train_samples.csv
│   ├── operation_hourly.csv
│   ├── burden_change.csv
│   └── tap_history_train.csv
└── test/
    ├── test_a_samples.csv
    └── test_b_samples.csv
```

运行构建、再运行验证（构建命令本身不会自动执行独立验证）：

```powershell
python -B -m src.build_data_v4_timeaware ./raw ./rebuild_timeaware
python -B -m src.validate_data_v4_timeaware ./raw ./rebuild_timeaware
```

请使用新的输出目录。构建先写入待验证审计，只有验证全部通过后 `leakage_audit.md` 才会写入通过结论。官方数据默认校验test A=657行、test B=548行；小型夹具可传 `--no-official-counts`。

```powershell
python -B -m src.reproducibility_timeaware ./data_v4_timeaware ./rebuild_timeaware
```

比较器按文件集合及SHA-256比较全部19个产物。两次输出都完成验证后再比较；依赖版本和源码路径下既有data_v3哨兵文件的存在状态会进入元数据。在仅源码的新克隆中，旧哨兵通常不存在，因此元数据哈希可能与原交付不同，应结合具体差异判断，不能把路径环境导致的元数据变化当作模型特征变化。

## 源码入口

| 文件 | 用途 |
|---|---|
| `src/data_io.py`、`src/timeaware_contract.py` | 来源目录解析、字段/类型/ID校验、特征注册表与上下文 |
| `src/schedule_features.py` | `ScheduleFeatureBuilder`：G7铁次节奏特征 |
| `src/burden_state_features.py` | `BurdenStateFeatureBuilder`：G8因果状态、网格、统计 |
| `src/timeaware_preprocessing.py` | 折内填补、原始G7/G8装配 |
| `src/splits.py` | `StrictExpandingMonthSplit`：五折月份划分 |
| `src/build_data_v4_timeaware.py` | 端到端生成19个交付文件 |
| `src/timeaware_reports.py`、`src/writers.py` | 字典、统计CSV和Snappy Parquet写出 |
| `src/validate_data_v4_timeaware.py` | 来源哈希、Schema、折内状态与核心时间边界复核 |
| `src/reproducibility_timeaware.py` | 两次构建文件与哈希比较 |
| `src/build_data_v3.py`、`src/preprocessing.py` | 上一阶段data_v3构建及预处理逻辑 |

## 测试与已有验证证据

快速运行本次相关的30项测试，或运行完整78项回归：

```powershell
python -B -m unittest tests.test_schedule_features tests.test_burden_state_features tests.test_timeaware_no_future tests.test_timeaware_schema -v
python -B -m unittest discover -s tests -v
```

2026-10-30提交前复测：完整78项测试全部通过（189.661秒）；待上传数据通过2833项交付检查，50个样本的1700个核心值独立复算，G7/G8边界违规为0、非有限特征为0。历史交付还进行了两次重建，19个文件哈希一致。来源及审计详见 [source_manifest.json](data_v4_timeaware/manifests/source_manifest.json) 和 [leakage_audit.md](data_v4_timeaware/manifests/leakage_audit.md)。

验证器对填补中位数重新计算，但该步骤复用了原始特征生成接口；核心G7/G8边界才使用单独的手算逻辑。Schema与有限值检查覆盖所有交付文件，核心手算为抽样检查，不能理解为173列每个值都由另一套实现完整重算。

本仓库交付数据工程工作，模型训练、消融实验、排行榜分数及最终提交由后续协作流程完成。
