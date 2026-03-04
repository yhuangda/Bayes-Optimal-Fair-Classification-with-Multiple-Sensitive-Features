**This is the code for paper:**

**Yang, Y., Huang, Y., & Chang, X. (2025). Bayes-Optimal Fair Classification with Multiple Sensitive Features. arXiv preprint arXiv:2505.00631.** (Accepted by AAAI 2026)

## 支持的数据集
- `adult` - Adult Income 数据集（单敏感属性）
- `adult_multi` - Adult Income 数据集（多敏感属性）
- `compas` - COMPAS 数据集（单敏感属性）
- `compas_multi` - COMPAS 数据集（多敏感属性）

## 支持的方法
- `Bayes_fair` - 后处理贝叶斯公平分类器（随机森林）
- `Bayes_fair_logit` - 后处理贝叶斯公平分类器（逻辑回归）
- `Bayes_fair_inprocess` - 处理中贝叶斯公平分类器（随机森林）
- `Bayes_fair_inprocess_logit` - 处理中贝叶斯公平分类器（逻辑回归）

## 支持的公平性概念
- `DP` - Demographic Parity (人口统计均等)
- `EOpp` - Equality of Opportunity (机会均等)
- `AP` - Accuracy Parity (准确率均等)
- `PredEqual` - Predictive Equality (预测均等)

## 支持的公平性类型
- `MD` - Mean Difference (均值差)
- `MR` - Min Ratio (最小比率)

## 项目结构
```
Bayes_Public/
├── bayesfair.py          # 主运行脚本
├── multi_trial.py        # 批量实验脚本
├── conf/
│   └── base.yaml         # 配置文件
├── utils/
│   └── methods.py        # 核心方法和数据加载函数
├── data/
│   ├── adult/            # Adult 数据集
│   └── compas/           # COMPAS 数据集
├── logs/                 # 运行日志
└── save_results/         # 实验结果
```

## 使用方法

### 单次运行
```bash
python bayesfair.py --config-name base dataset=adult method=Bayes_fair parameters.fair_notion=DP parameters.type=MD
```

### 批量运行
使用 `multi_trial.py` 可以运行所有组合的实验：
```bash
python multi_trial.py
```

这将运行以下所有组合：
- 4个数据集 × 4个方法 × 4个公平性概念 × 2个公平性类型 = 128个实验

### 修改配置
编辑 `conf/base.yaml` 文件来修改默认参数：
- `dataset`: 选择数据集
- `method`: 选择方法
- `parameters.fair_notion`: 选择公平性概念
- `parameters.type`: 选择公平性类型（MD/MR）
- `parameters.allow_sens`: 是否在模型中使用敏感属性
- `parameters.c`: 成本参数
- `parameters.fair_tolerance`: 公平性容忍度
- `parameters.max_lambda`: 最大lambda值
- `parameters.grid_interval`: 网格搜索间隔

## 依赖
见 `requirements.txt`

## 注意事项
1. 确保数据文件夹 `data/adult/` 和 `data/compas/` 存在且包含正确的数据文件
2. 日志文件会保存在 `logs/` 文件夹中
3. 实验结果会保存在 `save_results/` 文件夹中，按公平性概念/类型/数据集/方法组织

