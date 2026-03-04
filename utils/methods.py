
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import Counter
from itertools import product
from sklearn.metrics import confusion_matrix
import pickle
import xgboost as xgb
import folktables


### 读取数据集
import datasets
import pandas as pd
import os
import urllib.request

import random
import torch

import logging
logger = logging.getLogger(__name__)



 
def load_adult(data_dir, sensitive_attrs=['Sex'], remove_sensitive_attr=False):
  features = [
      "Age", "Workclass", "fnlwgt", "Education", "Education-Num",
      "Martial Status", "Occupation", "Relationship", "Race", "Sex",
      "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target"
  ]

  # Download data
  train_path = f"{data_dir}/adult.data"
  test_path = f"{data_dir}/adult.test"
  if any([not os.path.exists(p) for p in [train_path, test_path]]):
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        train_path)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        test_path)

  original_train = pd.read_csv(train_path,
                               names=features,
                               sep=r"\s*,\s*",
                               engine="python",
                               na_values="?")
  original_test = pd.read_csv(test_path,
                              names=features,
                              sep=r"\s*,\s*",
                              engine="python",
                              na_values="?",
                              skiprows=1)
  original = pd.concat([original_train, original_test])
  
  original.drop(["fnlwgt"], inplace=True, axis=1)   ## 移除了 fnlwgt 列。这列表示每条记录的采样权重，但对模型训练和公平性分析可能并无直接贡献，因此将其移除。

  # Binarize class labels, and remove it from the input data
  labels_original = original[["Target"
                             ]].replace("<=50K.",
                                        "<=50K").replace(">50K.", ">50K")  ## 替换了 <=50K. 为 <=50K，以及 >50K. 为 >50K，统一标签的格式。
  original.drop(["Target"], inplace=True, axis=1)  ##Target 列作为预测目标，从特征中移除。

  groups = original[sensitive_attrs[0]]  #假设敏感属性包含多个列（如 ['race', 'sex']）代码通过循环将这些属性组合为一个新的列 groups，值以逗号分隔，例如："White, Male" 或 "Black, Female"。
  for attribute in sensitive_attrs[1:]:
    groups = np.add(np.add(groups, ", "), original[attribute])

  
  # Encode labels and groups
  label_names, labels = np.unique(labels_original, return_inverse=True)   # np.unique 用于获取标签和组的类别名称（label_names 和 group_names）
  group_names, groups = np.unique(groups, return_inverse=True)            # 同时返回每条记录的数值化编码（labels 和 groups），方便后续用于模型输入或分析


  ## 如果 remove_sensitive_attr 为 True，则从数据集中移除所有敏感属性列。使用正则表达式查找敏感属性列名并删除。这是为了模拟模型在训练时不直接使用敏感信息，但可以评估其对公平性的间接影响。
  if remove_sensitive_attr:
    for sensitive_attr in sensitive_attrs:
      original.drop(columns=list(original.filter(regex=f'^{sensitive_attr}')),
                    inplace=True)  
  # Encode categorical columns
  data = pd.get_dummies(original)
  labels = labels.ravel()
  groups = groups.ravel()
  return data, labels, label_names, groups, group_names



import os
import urllib.request
import numpy as np
import pandas as pd

def load_adult_multi(data_dir, sensitive_attrs=['Race', 'Sex'], remove_sensitive_attr=False):
    features = [
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num",
        "Martial Status", "Occupation", "Relationship", "Race", "Sex",
        "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target"
    ]

    # Download data
    train_path = f"{data_dir}/adult.data"
    test_path = f"{data_dir}/adult.test"
    if any([not os.path.exists(p) for p in [train_path, test_path]]):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            train_path)
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
            test_path)

    original_train = pd.read_csv(train_path,
                                 names=features,
                                 sep=r"\s*,\s*",
                                 engine="python",
                                 na_values="?")
    original_test = pd.read_csv(test_path,
                                names=features,
                                sep=r"\s*,\s*",
                                engine="python",
                                na_values="?",
                                skiprows=1)
    original = original_test #pd.concat([original_train, original_test])

    original.index = range(len( original ))

    original.drop(["fnlwgt"], inplace=True, axis=1)  # 移除了 fnlwgt 列

    # Binarize class labels, and remove it from the input data
    labels_original = original[["Target"]].replace("<=50K.", "<=50K").replace(">50K.", ">50K")
    original.drop(["Target"], inplace=True, axis=1)

    # 统计 Race 特征的频率，选择最多的两个类别
    race_counts = original['Race'].value_counts()
    top_races = race_counts.head(2).index.tolist()  # 确保 top_races 是列表 选择频率最高的两个类别

    # 过滤掉 Race 列中不在 top_races 的数据
    original = original[original['Race'].isin(top_races)]
    labels_original = labels_original.loc[original.index]  # 同步标签

    # 创建组合敏感特征（例如 Race 和 Sex 的组合）

    groups = original[sensitive_attrs[0]]  # 首先取 Race 列作为组
    for attribute in sensitive_attrs[1:]:  # 然后通过迭代将其他敏感特征（如 Sex）组合进来
        groups = np.add(np.add(groups, ", "), original[attribute])

    # 编码 labels 和 groups
    label_names, labels = np.unique(labels_original, return_inverse=True)
    group_names, groups = np.unique(groups, return_inverse=True)

    # 如果 remove_sensitive_attr 为 True，则从数据集中移除所有敏感属性列
    if remove_sensitive_attr:
        for sensitive_attr in sensitive_attrs:
            original.drop(columns=list(original.filter(regex=f'^{sensitive_attr}')), inplace=True)

    # 对分类特征进行 one-hot 编码
    data = pd.get_dummies(original)
    labels = labels.ravel()
    groups = groups.ravel()

    logger.info(f"{data.shape},{labels.shape},{groups.shape}")
    return data, labels, label_names, groups, group_names



def load_compas(data_dir, remove_sensitive_attr=False):
  data_path = f"{data_dir}/compas-scores-two-years.csv"
  if not os.path.exists(data_path):
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
        data_path)

  df = pd.read_csv(data_path)

  # select features for analysis
  df = df[[
      'age', 'c_charge_degree', 'race', 'sex', 'priors_count',
      'days_b_screening_arrest', 'is_recid', 'c_jail_in', 'c_jail_out'
  ]]

  # ix is the index of variables we want to keep.

  # Remove entries with inconsistent arrest information.
  ix = df['days_b_screening_arrest'] <= 30
  ix = (df['days_b_screening_arrest'] >= -30) & ix

  # remove entries entries where compas case could not be found.
  ix = (df['is_recid'] != -1) & ix

  # remove traffic offenses.
  ix = (df['c_charge_degree'] != "O") & ix

  # trim dataset
  df = df.loc[ix, :]

  # create new attribute "length of stay" with total jail time.
  df['length_of_stay'] = (
      pd.to_datetime(df['c_jail_out']) -
      pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)

  # drop 'c_jail_in' and 'c_jail_out'
  # drop columns that won't be used
  dropCol = ['c_jail_in', 'c_jail_out', 'days_b_screening_arrest']
  df.drop(dropCol, inplace=True, axis=1)

  # keep only African-American and Caucasian
  df = df.loc[df['race'].isin(['African-American', 'Caucasian']), :]

  # reset index
  df.reset_index(inplace=True, drop=True)

  # Binarize class labels, and remove it from the input data
  labels_original = df["is_recid"].replace(0, "No").replace(1, "Yes")
  df.drop(["is_recid"], inplace=True, axis=1)

  # Encode labels and groups
  label_names, labels = np.unique(labels_original, return_inverse=True)
  group_names, groups = np.unique(df["race"], return_inverse=True)

  if remove_sensitive_attr:
    df.drop(columns=["race"], inplace=True)

  # Encode categorical columns
  data = pd.get_dummies(df)

  return data, labels, label_names, groups, group_names





def load_compas_multi(data_dir, remove_sensitive_attr=False):
    data_path = f"{data_dir}/compas-scores-two-years.csv"
    if not os.path.exists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
            data_path)

    df = pd.read_csv(data_path)

    # select features for analysis
    df = df[[
        'age', 'c_charge_degree', 'race', 'sex', 'priors_count',
        'days_b_screening_arrest', 'is_recid', 'c_jail_in', 'c_jail_out'
    ]]

    # ix is the index of variables we want to keep.

    # Remove entries with inconsistent arrest information.
    ix = df['days_b_screening_arrest'] <= 30
    ix = (df['days_b_screening_arrest'] >= -30) & ix

    # remove entries entries where compas case could not be found.
    ix = (df['is_recid'] != -1) & ix

    # remove traffic offenses.
    ix = (df['c_charge_degree'] != "O") & ix

    # trim dataset
    df = df.loc[ix, :]

    # create new attribute "length of stay" with total jail time.
    df['length_of_stay'] = (
        pd.to_datetime(df['c_jail_out']) -
        pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)

    # drop 'c_jail_in' and 'c_jail_out'
    # drop columns that won't be used
    dropCol = ['c_jail_in', 'c_jail_out', 'days_b_screening_arrest']
    df.drop(dropCol, inplace=True, axis=1)

    # keep only African-American and Caucasian
    df = df.loc[df['race'].isin(['African-American', 'Caucasian']), :]

    # create a new sensitive feature combining 'race' and 'sex'
    df['sensitive_feature'] = df['race'] + '_' + df['sex']

    # reset index
    df.reset_index(inplace=True, drop=True)

    # Binarize class labels, and remove it from the input data
    labels_original = df["is_recid"].replace(0, "No").replace(1, "Yes")
    df.drop(["is_recid"], inplace=True, axis=1)

    # Encode labels and groups
    label_names, labels = np.unique(labels_original, return_inverse=True)
    
    # Use the sensitive feature as the group
    group_names, groups = np.unique(df["sensitive_feature"], return_inverse=True)

    if remove_sensitive_attr:
        df.drop(columns=["race", "sex", "sensitive_feature"], inplace=True)
    else:
        df.drop(columns=[ "race", "sex"], inplace=True)
        pass

    # Encode categorical columns
    data = pd.get_dummies(df)

    return data, labels, label_names, groups, group_names



class WeightedClassifier:
    def __init__(self, model_type="gbm"):
        """
        初始化分类器
        :param model_type: 选择分类模型, "logistic"（逻辑回归） 或 "gbm"（梯度提升树）
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None  # 之后初始化
    
    def _process_sample_weight(self, y, sample_weight):
        """
        处理负的 sample_weight: 反转 y，并取 sample_weight 绝对值
        :param y: 目标变量
        :param sample_weight: 权重向量
        :return: 处理后的 y 和 sample_weight
        """
        if sample_weight is not None and np.any(sample_weight < 0):
            original_y = y.copy()  # 备份原始 y
            y = np.where(sample_weight < 0, 1 - y, y)  # 反转负权重样本的标签
            sample_weight = np.abs(sample_weight)  # 取权重的绝对值
        
            # 检查反转后的 y 是否只有一个类别
            unique_classes = np.unique(y)  # 只考虑有正权重的样本
            if len(unique_classes) == 1:
                # 随机选择一个样本，并恢复其原始标签和权重
                neg_indices = np.where(original_y != y)[0]  # 找到被反转的样本
                if len(neg_indices) > 0:
                    idx = np.random.choice(neg_indices)  # 随机选一个
                    y[idx] = original_y[idx]  # 还原标签
                    sample_weight[idx] = -sample_weight[idx]  # 还原权重（负值变回去）
        
        sample_weight = np.maximum(sample_weight, 1e-6)  
        return y, sample_weight


    def fit(self, X, y, sample_weight=None):
        """
        训练分类器
        """
        # 归一化特征
        X_scaled = self.scaler.fit_transform(X)

        # 处理负的 sample_weight
        y, sample_weight = self._process_sample_weight(y, sample_weight)

        # 检查 y 是否仍然有至少两个类别
        unique_classes = np.unique(y[sample_weight > 0])  # 只考虑有权重的样本
        if len(unique_classes) < 2:
            raise ValueError("y contains only one class after sample_weight adjustment.")

        # 选择模型
        if self.model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000)
        elif self.model_type == "gbm":
            self.model = GradientBoostingClassifier()
        else:
            raise ValueError("Unsupported model type. Choose 'logistic' or 'gbm'.")

        # 训练模型
        self.model.fit(X_scaled, y, sample_weight=sample_weight)

    def predict(self, X):
        """
        进行预测
        :param X: 测试数据
        :return: 预测结果
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call `fit` first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X, y):
        """
        评估模型
        :param X: 测试数据
        :param y: 真实标签
        :return: 准确率和 F1 分数
        """
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred)
        }



def set_all_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def eval_fair(Y, S, Y_hat, fairness_notion="DP", diff_or_ratio="diff"):
    """
    Calculate a specific fairness metric for multiple sensitive groups.

    Parameters:
        Y (np.ndarray): True labels.
        S (np.ndarray): Sensitive attribute (categorical or multi-group).
        Y_hat (np.ndarray): Predicted labels.
        fairness_notion (str): The fairness notion to calculate ("Demographic Parity", 
                               "Equality of Opportunity", "Equalized Odds", "Predictive Parity").
        diff_or_ratio (str): Whether to calculate "diff" (difference) or "ratio" (proportion).

    Returns:
        float: The calculated fairness metric value (difference or ratio).
    """
    unique_groups = np.unique(S)

    # Compute metrics for the whole population
    tp_total = np.sum((Y_hat == 1) & (Y == 1))
    fp_total = np.sum((Y_hat == 1) & (Y == 0))
    tn_total = np.sum((Y_hat == 0) & (Y == 0))
    fn_total = np.sum((Y_hat == 0) & (Y == 1))

    tpr_total = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0  # True Positive Rate
    fpr_total = fp_total / (fp_total + tn_total) if (fp_total + tn_total) > 0 else 0  # False Positive Rate
    dp_total = np.sum(Y_hat) / len(Y_hat)  # Demographic Parity

    acc_total = np.sum( Y_hat ==  Y  ) / len(Y_hat)

    # Initialize metrics for each group
    dp_rates = []
    tpr_rates = []
    fpr_rates = []
    acc_rates = []

    for group in unique_groups:
        # Mask for the current group
        group_mask = (S == group)
        
        

        # True positive, false positive, true negative, false negative
        tp = np.sum((Y_hat == 1) & (Y == 1) & group_mask)
        fp = np.sum((Y_hat == 1) & (Y == 0) & group_mask)
        tn = np.sum((Y_hat == 0) & (Y == 0) & group_mask)
        fn = np.sum((Y_hat == 0) & (Y == 1) & group_mask)

        # Metrics for this group
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        dp = np.sum(Y_hat[group_mask]) / np.sum(group_mask)  # Demographic Parity
        acc = (tp + tn) / (tp + tn + fp + fn)

        tpr_rates.append(tpr)
        fpr_rates.append(fpr)
        dp_rates.append(dp)
        acc_rates.append(acc)


    # Select metric list based on fairness notion
    if fairness_notion == "DP":
        metric_list = dp_rates
        metric_population = dp_total
    elif fairness_notion == "EOpp":
        metric_list = tpr_rates
        metric_population = tpr_total
    elif fairness_notion == "EOdds":
        # Combine TPR and FPR
        metric_list = tpr_rates + fpr_rates
        metric_population = tpr_total + fpr_total
    elif fairness_notion == "PredEqual":
        metric_list = fpr_rates
        metric_population = fpr_total
    elif fairness_notion == "AP":
        metric_list = acc_rates
        metric_population = acc_total
    else:
        raise ValueError("Invalid fairness_notion. Choose from 'DP: Demographic Parity', 'EOpp: Equality of Opportunity', 'EOdds: Equalized Odds', or 'PredEqual: Predictive Equality'.")

    # Calculate pairwise differences or ratios
    def calculate_pairwise_metric(metric_list, diff_or_ratio):
        max_value = 0

        if diff_or_ratio == "diff":
            value = max(metric_list) - min(metric_list)
        elif diff_or_ratio == "ratio":
            value = (min(metric_list) / metric_population if metric_population != 0 else 0 )
        max_value = max(max_value, value)
        return max_value

    # # Calculate the desired fairness metric
    result = calculate_pairwise_metric(metric_list, diff_or_ratio)
    return result


def group_accuracy(PY_X, Y, S):
    """
    根据 S 的组别，计算每组的准确率。
    
    参数:
    - PY_X: ndarray, shape (n_samples, n_classes), 预测的类别概率
    - Y: ndarray, shape (n_samples,), 真实标签
    - S: ndarray, shape (n_samples,), 敏感属性
    
    返回:
    - group_acc: dict, 每组的准确率
    """
    # 将概率转换为类别预测
    preds = np.argmax(PY_X, axis=1)  # 假设类别标签从 0 开始

    # 获取每个组的唯一值
    unique_groups = np.unique(S)
    
    # 存储每组的准确率
    group_acc = {}
    
    for group in unique_groups:
        # 获取当前组的索引
        group_idx = S == group
        
        # 当前组的真实标签和预测
        group_Y = Y[group_idx]
        group_preds = preds[group_idx]
        
        # 计算准确率
        accuracy = np.mean(group_Y == group_preds)
        group_acc[group] = accuracy
    
    return group_acc


class PostProcessingFairClassifier:
    def __init__(self, cost_param, fairness_tolerance, fair_type, diff_or_ratio = 'diff', allow_sens = False, grid_interval = 0.05):
        self.cost_param = cost_param
        self.fairness_tolerance = fairness_tolerance
        self.allow_sens = allow_sens
        self.fair_type = fair_type
        self.diff_or_ratio = diff_or_ratio
        self.grid_interval = grid_interval
        self.base_model_Y_given_X_S = None   # P(Y=1 | X=x)
        self.base_model_YS_given_X = None # P(E_{z,m} | X=x)
        self.group_probs = None      # P(E_{z,m})


    def compute_p_s(self, S):
        """
        Compute P(S = s) for all unique values of S.

        Parameters:
            S (array-like): Array of sensitive attribute values.

        Returns:
            dict: Probability distribution of P(S = s).
        """
        S = np.array(S)
        count_s = Counter(S)
        total = len(S)
        p_s = {s: count / total for s, count in count_s.items()}
        return p_s

    def compute_p_s_given_y(self, S, Y):
        """
        Compute P(S = s | Y = y) for all combinations of S and Y.

        Parameters:
            S (array-like): Array of sensitive attribute values.
            Y (array-like): Array of target labels.

        Returns:
            dict: Nested dictionary representing P(S = s | Y = y).
        """
        # Combine S and Y for joint distribution
        joint = list(zip(S, Y))
        count_joint = Counter(joint)
        count_y = Counter(Y)
        
        p_s_given_y = {}
        for (s, y), count in count_joint.items():
            if y not in p_s_given_y:
                p_s_given_y[y] = {}
            p_s_given_y[y][s] = count / count_y[y]
        
        return p_s_given_y
    
    def compute_p_y_given_s(self, S, Y):
        """
        Compute P(Y = y | S = s) for all combinations of S and Y.

        Parameters:
            S (array-like): Array of sensitive attribute values.
            Y (array-like): Array of target labels.

        Returns:
            dict: Nested dictionary representing P(Y = y | S = s).
        """
        # Combine S and Y for joint distribution
        joint = list(zip(S, Y))
        count_joint = Counter(joint)
        count_s = Counter(S)
        
        p_y_given_s = {}
        for (s, y), count in count_joint.items():
            if s not in p_y_given_s:
                p_y_given_s[s] = {}
            p_y_given_s[s][y] = count / count_s[s]
        
        return p_y_given_s

    def get_specific_p_s(self,  s):
        """
        Get P(S = s) for a specific value of S.

        Parameters:
            p_s (dict): Dictionary from compute_p_s.
            s: Specific value of S.

        Returns:
            float: The probability P(S = s).
        """
        return self.p_s.get(s, None)  # Return 0 if the value of s is not in the dictionary


    def get_specific_group_prob(self, m,t):
        """
        Get P(Y = t, S = m) for specific values of t and m.

        Parameters:
            group_probs (dict): Dictionary of group probabilities from estimate_group_probs.
            t: Specific value of Y.
            m: Specific value of S.

        Returns:
            float: The probability P(Y = t, S = m), or 0 if not found.
        """
        return self.group_probs.get(( m, t), None)  # Return 0 if (t, m) is not in the dictionary


    def get_specific_p_s_given_y(self, s, y):
        """
        Get P(S = s | Y = y) for specific values of s and y.

        Parameters:
            p_s_given_y (dict): Nested dictionary from compute_p_s_given_y.
            s: Specific value of S.
            y: Specific value of Y.

        Returns:
            float: The probability P(S = s | Y = y).
        """
        return self.p_s_given_y.get(y, {}).get(s, None)

    def get_specific_p_y_given_s(self, s, y):
        """
        Get P(Y = y | S = s) for specific values of y and s.

        Parameters:
            p_y_given_s (dict): Nested dictionary from compute_p_y_given_s.
            y: Specific value of Y.
            s: Specific value of S.

        Returns:
            float: The probability P(Y = y | S = s).
        """
        return self.p_y_given_s.get(s, {}).get(y, None)    
    
    def estimate_base_model_Y_given_X_S(self, X, S, Y):
        """
        Estimate η(x) = P(Y = 1 | X = x) using a standard classifier.
        """
        if self.allow_sens:
            X_combined = np.hstack((X, S.reshape(-1, 1)))  # 添加 S 作为一列
            self.base_model_Y_given_X_S =  xgb.XGBClassifier()  #RandomForestClassifier()  #LogisticRegression()  #
            self.base_model_Y_given_X_S.fit(X_combined, Y)
            pred = self.base_model_Y_given_X_S.predict(X_combined)
        else: 
            self.base_model_Y_given_X_S =   xgb.XGBClassifier()  #RandomForestClassifier() #LogisticRegression()  #
            self.base_model_Y_given_X_S.fit(X, Y)
            pred = self.base_model_Y_given_X_S.predict(X)
        logger.info(f"P(Y | X, S) or P(Y | X) =   {np.sum(pred == Y)/ len(Y)}, {np.sum(pred == Y)}, {len(Y)}" )




    def estimate_base_model_YS_given_X(self, X, S, Y):
        """
        Estimate group membership probabilities P(S, Y | X = x) 
        """
        joint_labels = [(s, y) for s, y in zip(S, Y)]
        joint_labels = pd.Series(joint_labels)

        # Use pandas.factorize to encode
        encoded_labels, unique_combinations = pd.factorize(joint_labels)
        self.gamma_sy_combination = unique_combinations.tolist()

        logger.info(f"self.gamma_sy_combination = {self.gamma_sy_combination}")

        # Train a classifier to predict joint labels based on X
        self.base_model_YS_given_X = xgb.XGBClassifier()  # RandomForestClassifier() #LogisticRegression() #
        self.base_model_YS_given_X.fit(X, encoded_labels)

        # Predict probabilities for all samples
        predicted_encoded_labels = self.base_model_YS_given_X.predict(X)

        # Compute accuracy by comparing predicted and true joint labels
        gamma_acc = np.sum( predicted_encoded_labels == encoded_labels ) / len(encoded_labels)
        
        logger.info(f"P(S, Y | X = x) acc =, {gamma_acc},   {np.sum( predicted_encoded_labels == encoded_labels )}, {len(encoded_labels)}" )


    def estimate_group_probs(self, S, Y):
        """
        Calculate P(E_{z,m}) = P(Y = z, S = m) directly from the data.
        """
        unique_combinations, counts = np.unique(list(zip(S, Y)), axis=0, return_counts=True)
        total_samples = len(Y)
        self.group_probs = {tuple(comb): count / total_samples for comb, count in zip(unique_combinations, counts)}
        return self.group_probs

    def predict(self, X, S,  lambda_set, Y = None):
        """
        Construct the fair classifier based on the decision rule.
        """

        Lam_M = sum(lambda_set.values())

        if self.diff_or_ratio == 'diff':
            tau = 1
        elif self.diff_or_ratio == 'ratio':
            tau = self.fairness_tolerance
        else:
            print("please use valid fairness type (diff or ratio)")

        if self.allow_sens:
            XS = np.hstack([X, S.reshape(-1, 1)])
            PY_XS = self.base_model_Y_given_X_S.predict_proba(XS) # Batch prediction

            # 计算每组的准确率
            if Y is not None:
                group_acc = group_accuracy(PY_XS, Y, S)


            if self.fair_type == 'DP':
                lambda_m_p = np.array([lambda_set[m]/self.get_specific_p_s(m)  for m in S]) 
                fairness_term = lambda_m_p  -   tau * Lam_M  
            elif self.fair_type == 'EOpp':
                lambda_m = np.array([lambda_set[m] for m in S])
                bmt = np.array([self.get_specific_p_s_given_y(m, 1) for m in S])
                P_m_yt = np.array([self.get_specific_group_prob(m, 1) for m in S])
                fairness_term = (lambda_m  - tau * Lam_M * bmt) * PY_XS[:,1] / P_m_yt
            elif self.fair_type == 'PredEqual':
                lambda_m = np.array([lambda_set[m] for m in S])
                bmt = np.array([self.get_specific_p_s_given_y(m, 0) for m in S])
                P_m_yt = np.array([self.get_specific_group_prob(m, 0) for m in S])
                fairness_term = (lambda_m  - tau * Lam_M * bmt) * PY_XS[:,0] / P_m_yt
            elif self.fair_type == 'AP':
                amt = np.array([  1/self.get_specific_p_s(m) for m in S])
                lambda_m = np.array([lambda_set[m] for m in S])
                fairness_term = ( amt * lambda_m - tau * Lam_M ) * (  PY_XS[:,0] -  PY_XS[:,1]  )
            else:
                logger.info('Error! Please use a valid fairness type (i.e., DP, EOpp, PredEqual, AP)')
            
            thresholds = self.cost_param +  fairness_term
            predictions = (PY_XS[:,1] > thresholds).astype(int)
            
        else:

            PY_X = self.base_model_Y_given_X_S.predict_proba(X) # Batch prediction
            PYS_X = self.base_model_YS_given_X.predict_proba(X)    # Batch group probabilities
            fairness_term = np.zeros(len(X))

            # 计算每组的准确率
            if Y is not None:
                group_acc = group_accuracy(PY_X, Y, S)

            if self.fair_type == 'DP':
                for m in np.unique(S):
                    lambda_m =  lambda_set[m] 
                    bmt = self.get_specific_p_s(m) 
                    ind_pos, ind_neg = self.gamma_sy_combination.index((m, 1)), self.gamma_sy_combination.index((m, 0))
                    PS_X = PYS_X[:,ind_neg] + PYS_X[:,ind_pos] 
                    fairness_term += (lambda_m  -   tau * bmt * Lam_M  ) * PS_X / bmt
            elif self.fair_type == 'EOpp':
                for m in np.unique(S):
                    lambda_m =  lambda_set[m] 
                    bmt = self.get_specific_p_s_given_y(m, 1)
                    ind_pos  = self.gamma_sy_combination.index((m, 1))
                    P_m_yt = self.get_specific_group_prob(m, 1)
                    fairness_term += (lambda_m  - tau * Lam_M * bmt) * PYS_X[:,ind_pos]  / P_m_yt
            elif self.fair_type == 'PredEqual':
                for m in np.unique(S):
                    lambda_m =  lambda_set[m] 
                    bmt = self.get_specific_p_s_given_y(m, 0)  
                    ind_neg  = self.gamma_sy_combination.index((m, 0))
                    P_m_yt = self.get_specific_group_prob(m, 0)
                    fairness_term += (lambda_m  - tau * Lam_M * bmt) *  PYS_X[:,ind_neg] / P_m_yt
            elif self.fair_type == 'AP':
                for m in np.unique(S):
                    lambda_m =  lambda_set[m] 
                    amt = self.get_specific_p_y_given_s(m,0)
                    P_m_yt = self.get_specific_group_prob(m, 0)
                    ind_neg  = self.gamma_sy_combination.index((m, 0))
                    fairness_term += (amt * lambda_m  - tau * Lam_M * P_m_yt) *  PYS_X[:,ind_neg] / P_m_yt


                    amt = self.get_specific_p_y_given_s(m,1)
                    P_m_yt = self.get_specific_group_prob(m, 1)
                    ind_pos  = self.gamma_sy_combination.index((m, 1))
                    fairness_term += (amt * lambda_m  - tau * Lam_M * P_m_yt) *  PYS_X[:,ind_pos] / P_m_yt
            else:
                logger.info('Error! Please use a valid fairness type (i.e., DP, EOpp, PredEqual, AP)')
            
            thresholds = self.cost_param +  fairness_term
            predictions = (PY_X[:,1] > thresholds).astype(int)
        return np.array(predictions)


    def fit(self, X, S, Y, fairness_metrics_eval, grid_values, valid_ratio = 0.3 ):
        """
        Train the fair classifier via post-processing.
        """
        # Split the training data into training and validation sets
        X_train, X_val, S_train, S_val, Y_train, Y_val = train_test_split(X, S, Y, test_size=valid_ratio) #, random_state=42

        self.p_s = self.compute_p_s(S_train)
        self.p_s_given_y = self.compute_p_s_given_y(S_train, Y_train)
        self.p_y_given_s = self.compute_p_y_given_s(S_train, Y_train)
        self.group_probs = self.estimate_group_probs(S_train, Y_train)
        
        self.estimate_base_model_Y_given_X_S(X_train, S_train, Y_train)

        
        if not self.allow_sens:
            self.estimate_base_model_YS_given_X(X_train, S_train, Y_train)


        # Step 2: Initialize grid search
        # Define grid search range for tradeoff_param

        # 确保 group_ids 是 Python 的原生类型
        group_ids = [int(group) for group in np.unique(S)]

        # 确保 param_range 是 Python 的原生类型
        param_range = [float(value) for value in np.linspace(-grid_values, grid_values, 2 * int(grid_values / self.grid_interval) + 1)]

        # 使用 product 创建候选参数组合
        candidate_params = list(product(param_range, repeat=len(group_ids)))

        # Best parameters and scores
        best_tradeoff_param = None
        best_fairness_violation = float('inf')
        best_accuracy = 0

        record = []

        for params in candidate_params:
            # Assign tradeoff_param
            self.tradeoff_param = {group: param for group, param in zip(group_ids, params)}

            # Validate on the validation set
            Y_pred = self.predict(X_val, S_val, self.tradeoff_param )

            # Compute fairness metrics and accuracy
            fairness_violation = fairness_metrics_eval(Y_val,  S_val, Y_pred,  fairness_notion= self.fair_type, diff_or_ratio= self.diff_or_ratio )
            accuracy = np.mean(Y_pred == Y_val)

            
            # Update the best parameters if fairness is satisfied and accuracy improves
            
            if self.diff_or_ratio == 'diff':
                if fairness_violation <= self.fairness_tolerance:
                    record.append([self.tradeoff_param,  fairness_violation, accuracy])
                    if accuracy > best_accuracy or (accuracy == best_accuracy and fairness_violation < best_fairness_violation):
                        best_tradeoff_param = self.tradeoff_param
                        best_fairness_violation = fairness_violation
                        best_accuracy = accuracy
            else:
                if fairness_violation >=  1 - self.fairness_tolerance:
                    record.append([self.tradeoff_param,  fairness_violation, accuracy])
                    if accuracy > best_accuracy or (accuracy == best_accuracy and fairness_violation >= best_fairness_violation):
                        best_tradeoff_param = self.tradeoff_param
                        best_fairness_violation = fairness_violation
                        best_accuracy = accuracy
            
        
        # Assign the best tradeoff_param back to the model
        if best_tradeoff_param is not None:
            self.tradeoff_param = best_tradeoff_param
            logger.info(f"Best tradeoff_param: {best_tradeoff_param}")
            logger.info(f"Fairness violation: {best_fairness_violation}, Accuracy: {best_accuracy}")
        else:
            logger.info("No valid tradeoff_param found that satisfies fairness constraints.")
        return record


class PostProcessingFairClassifier_logit:
    def __init__(self, cost_param, fairness_tolerance, fair_type, diff_or_ratio = 'diff', allow_sens = False, grid_interval = 0.05):
        self.cost_param = cost_param
        self.fairness_tolerance = fairness_tolerance
        self.allow_sens = allow_sens
        self.fair_type = fair_type
        self.diff_or_ratio = diff_or_ratio
        self.grid_interval = grid_interval
        self.base_model_Y_given_X_S = None   # P(Y=1 | X=x)
        self.base_model_YS_given_X = None # P(E_{z,m} | X=x)
        self.group_probs = None      # P(E_{z,m})


    def compute_p_s(self, S):
        """
        Compute P(S = s) for all unique values of S.

        Parameters:
            S (array-like): Array of sensitive attribute values.

        Returns:
            dict: Probability distribution of P(S = s).
        """
        S = np.array(S)
        count_s = Counter(S)
        total = len(S)
        p_s = {s: count / total for s, count in count_s.items()}
        return p_s

    def compute_p_s_given_y(self, S, Y):
        """
        Compute P(S = s | Y = y) for all combinations of S and Y.

        Parameters:
            S (array-like): Array of sensitive attribute values.
            Y (array-like): Array of target labels.

        Returns:
            dict: Nested dictionary representing P(S = s | Y = y).
        """
        # Combine S and Y for joint distribution
        joint = list(zip(S, Y))
        count_joint = Counter(joint)
        count_y = Counter(Y)
        
        p_s_given_y = {}
        for (s, y), count in count_joint.items():
            if y not in p_s_given_y:
                p_s_given_y[y] = {}
            p_s_given_y[y][s] = count / count_y[y]
        
        return p_s_given_y
    
    def compute_p_y_given_s(self, S, Y):
        """
        Compute P(Y = y | S = s) for all combinations of S and Y.

        Parameters:
            S (array-like): Array of sensitive attribute values.
            Y (array-like): Array of target labels.

        Returns:
            dict: Nested dictionary representing P(Y = y | S = s).
        """
        # Combine S and Y for joint distribution
        joint = list(zip(S, Y))
        count_joint = Counter(joint)
        count_s = Counter(S)
        
        p_y_given_s = {}
        for (s, y), count in count_joint.items():
            if s not in p_y_given_s:
                p_y_given_s[s] = {}
            p_y_given_s[s][y] = count / count_s[s]
        
        return p_y_given_s

    def get_specific_p_s(self,  s):
        """
        Get P(S = s) for a specific value of S.

        Parameters:
            p_s (dict): Dictionary from compute_p_s.
            s: Specific value of S.

        Returns:
            float: The probability P(S = s).
        """
        return self.p_s.get(s, None)  # Return 0 if the value of s is not in the dictionary


    def get_specific_group_prob(self, m,t):
        """
        Get P(Y = t, S = m) for specific values of t and m.

        Parameters:
            group_probs (dict): Dictionary of group probabilities from estimate_group_probs.
            t: Specific value of Y.
            m: Specific value of S.

        Returns:
            float: The probability P(Y = t, S = m), or 0 if not found.
        """
        return self.group_probs.get(( m, t), None)  # Return 0 if (t, m) is not in the dictionary


    def get_specific_p_s_given_y(self, s, y):
        """
        Get P(S = s | Y = y) for specific values of s and y.

        Parameters:
            p_s_given_y (dict): Nested dictionary from compute_p_s_given_y.
            s: Specific value of S.
            y: Specific value of Y.

        Returns:
            float: The probability P(S = s | Y = y).
        """
        return self.p_s_given_y.get(y, {}).get(s, None)

    def get_specific_p_y_given_s(self, s, y):
        """
        Get P(Y = y | S = s) for specific values of y and s.

        Parameters:
            p_y_given_s (dict): Nested dictionary from compute_p_y_given_s.
            y: Specific value of Y.
            s: Specific value of S.

        Returns:
            float: The probability P(Y = y | S = s).
        """
        return self.p_y_given_s.get(s, {}).get(y, None)    
    
    def estimate_base_model_Y_given_X_S(self, X, S, Y):
        """
        Estimate η(x) = P(Y = 1 | X = x) using a standard classifier.
        """
        if self.allow_sens:
            X_combined = np.hstack((X, S.reshape(-1, 1)))  # 添加 S 作为一列
            self.base_model_Y_given_X_S =   LogisticRegression()  #
            self.base_model_Y_given_X_S.fit(X_combined, Y)
            pred = self.base_model_Y_given_X_S.predict(X_combined)
        else: 
            self.base_model_Y_given_X_S =    LogisticRegression()  #
            self.base_model_Y_given_X_S.fit(X, Y)
            pred = self.base_model_Y_given_X_S.predict(X)
        logger.info(f"P(Y | X, S) or P(Y | X) =   {np.sum(pred == Y)/ len(Y)}, {np.sum(pred == Y)}, {len(Y)}" )




    def estimate_base_model_YS_given_X(self, X, S, Y):
        """
        Estimate group membership probabilities P(S, Y | X = x) 
        """
        joint_labels = [(s, y) for s, y in zip(S, Y)]
        joint_labels = pd.Series(joint_labels)

        # Use pandas.factorize to encode
        encoded_labels, unique_combinations = pd.factorize(joint_labels)
        self.gamma_sy_combination = unique_combinations.tolist()

        # Train a classifier to predict joint labels based on X
        self.base_model_YS_given_X =  LogisticRegression() #
        self.base_model_YS_given_X.fit(X, encoded_labels)

        # Predict probabilities for all samples
        predicted_encoded_labels = self.base_model_YS_given_X.predict(X)

        # Compute accuracy by comparing predicted and true joint labels
        gamma_acc = np.sum( predicted_encoded_labels == encoded_labels ) / len(encoded_labels)
        
        logger.info(f"P(S, Y | X = x) acc =, {gamma_acc},   {np.sum( predicted_encoded_labels == encoded_labels )}, {len(encoded_labels)}" )


    def estimate_group_probs(self, S, Y):
        """
        Calculate P(E_{z,m}) = P(Y = z, S = m) directly from the data.
        """
        unique_combinations, counts = np.unique(list(zip(S, Y)), axis=0, return_counts=True)
        total_samples = len(Y)
        self.group_probs = {tuple(comb): count / total_samples for comb, count in zip(unique_combinations, counts)}
        return self.group_probs

    def predict(self, X, S,  lambda_set, Y = None):
        """
        Construct the fair classifier based on the decision rule.
        """

        Lam_M = sum(lambda_set.values())

        if self.diff_or_ratio == 'diff':
            tau = 1
        elif self.diff_or_ratio == 'ratio':
            tau = self.fairness_tolerance
        else:
            print("please use valid fairness type (diff or ratio)")

        if self.allow_sens:
            XS = np.hstack([X, S.reshape(-1, 1)])
            PY_XS = self.base_model_Y_given_X_S.predict_proba(XS) # Batch prediction

            # 计算每组的准确率
            if Y is not None:
                group_acc = group_accuracy(PY_XS, Y, S)


            if self.fair_type == 'DP':
                lambda_m_p = np.array([lambda_set[m]/self.get_specific_p_s(m)  for m in S]) 
                fairness_term = lambda_m_p  -   tau * Lam_M  
            elif self.fair_type == 'EOpp':
                lambda_m = np.array([lambda_set[m] for m in S])
                bmt = np.array([self.get_specific_p_s_given_y(m, 1) for m in S])
                P_m_yt = np.array([self.get_specific_group_prob(m, 1) for m in S])
                fairness_term = (lambda_m  - tau * Lam_M * bmt) * PY_XS[:,1] / P_m_yt
            elif self.fair_type == 'PredEqual':
                lambda_m = np.array([lambda_set[m] for m in S])
                bmt = np.array([self.get_specific_p_s_given_y(m, 0) for m in S])
                P_m_yt = np.array([self.get_specific_group_prob(m, 0) for m in S])
                fairness_term = (lambda_m  - tau * Lam_M * bmt) * PY_XS[:,0] / P_m_yt
            elif self.fair_type == 'AP':
                amt = np.array([  1/self.get_specific_p_s(m) for m in S])
                lambda_m = np.array([lambda_set[m] for m in S])
                fairness_term = ( amt * lambda_m - tau * Lam_M ) * (  PY_XS[:,0] -  PY_XS[:,1]  )
            else:
                logger.info('Error! Please use a valid fairness type (i.e., DP, EOpp, PredEqual, AP)')
            
            thresholds = self.cost_param +  fairness_term
            predictions = (PY_XS[:,1] > thresholds).astype(int)
            
        else:

            PY_X = self.base_model_Y_given_X_S.predict_proba(X) # Batch prediction
            PYS_X = self.base_model_YS_given_X.predict_proba(X)    # Batch group probabilities
            fairness_term = np.zeros(len(X))

            # 计算每组的准确率
            if Y is not None:
                group_acc = group_accuracy(PY_X, Y, S)

            if self.fair_type == 'DP':
                for m in np.unique(S):
                    lambda_m =  lambda_set[m] 
                    bmt = self.get_specific_p_s(m) 
                    ind_pos, ind_neg = self.gamma_sy_combination.index((m, 1)), self.gamma_sy_combination.index((m, 0))
                    PS_X = PYS_X[:,ind_neg] + PYS_X[:,ind_pos] 
                    fairness_term += (lambda_m  -   tau * bmt * Lam_M  ) * PS_X / bmt
            elif self.fair_type == 'EOpp':
                for m in np.unique(S):
                    lambda_m =  lambda_set[m] 
                    bmt = self.get_specific_p_s_given_y(m, 1)
                    ind_pos  = self.gamma_sy_combination.index((m, 1))
                    P_m_yt = self.get_specific_group_prob(m, 1)
                    fairness_term += (lambda_m  - tau * Lam_M * bmt) * PYS_X[:,ind_pos]  / P_m_yt
            elif self.fair_type == 'PredEqual':
                for m in np.unique(S):
                    lambda_m =  lambda_set[m] 
                    bmt = self.get_specific_p_s_given_y(m, 0)  
                    ind_neg  = self.gamma_sy_combination.index((m, 0))
                    P_m_yt = self.get_specific_group_prob(m, 0)
                    fairness_term += (lambda_m  - tau * Lam_M * bmt) *  PYS_X[:,ind_neg] / P_m_yt
            elif self.fair_type == 'AP':
                for m in np.unique(S):
                    lambda_m =  lambda_set[m] 
                    amt = self.get_specific_p_y_given_s(m,0)
                    P_m_yt = self.get_specific_group_prob(m, 0)
                    ind_neg  = self.gamma_sy_combination.index((m, 0))
                    fairness_term += (amt * lambda_m  - tau * Lam_M * P_m_yt) *  PYS_X[:,ind_neg] / P_m_yt


                    amt = self.get_specific_p_y_given_s(m,1)
                    P_m_yt = self.get_specific_group_prob(m, 1)
                    ind_pos  = self.gamma_sy_combination.index((m, 1))
                    fairness_term += (amt * lambda_m  - tau * Lam_M * P_m_yt) *  PYS_X[:,ind_pos] / P_m_yt
            else:
                logger.info('Error! Please use a valid fairness type (i.e., DP, EOpp, PredEqual, AP)')
            
            thresholds = self.cost_param +  fairness_term
            predictions = (PY_X[:,1] > thresholds).astype(int)
        return np.array(predictions)


    def fit(self, X, S, Y, fairness_metrics_eval, grid_values, valid_ratio = 0.3 ):
        """
        Train the fair classifier via post-processing.
        """
        # Split the training data into training and validation sets
        X_train, X_val, S_train, S_val, Y_train, Y_val = train_test_split(X, S, Y, test_size=valid_ratio) #, random_state=42

        self.p_s = self.compute_p_s(S_train)
        self.p_s_given_y = self.compute_p_s_given_y(S_train, Y_train)
        self.p_y_given_s = self.compute_p_y_given_s(S_train, Y_train)
        self.group_probs = self.estimate_group_probs(S_train, Y_train)
        
        self.estimate_base_model_Y_given_X_S(X_train, S_train, Y_train)

        
        if not self.allow_sens:
            self.estimate_base_model_YS_given_X(X_train, S_train, Y_train)


        # Step 2: Initialize grid search
        # Define grid search range for tradeoff_param

        # 确保 group_ids 是 Python 的原生类型
        group_ids = [int(group) for group in np.unique(S)]

        # 确保 param_range 是 Python 的原生类型
        param_range = [float(value) for value in np.linspace(-grid_values, grid_values, 2 * int(grid_values / self.grid_interval) + 1)]

        # 使用 product 创建候选参数组合
        candidate_params = list(product(param_range, repeat=len(group_ids)))

        # Best parameters and scores
        best_tradeoff_param = None
        best_fairness_violation = float('inf')
        best_accuracy = 0

        record = []

        for params in candidate_params:
            # Assign tradeoff_param
            self.tradeoff_param = {group: param for group, param in zip(group_ids, params)}

            # Validate on the validation set
            Y_pred = self.predict(X_val, S_val, self.tradeoff_param )

            # Compute fairness metrics and accuracy
            fairness_violation = fairness_metrics_eval(Y_val,  S_val, Y_pred,  fairness_notion= self.fair_type, diff_or_ratio= self.diff_or_ratio )
            accuracy = np.mean(Y_pred == Y_val)

            
            # Update the best parameters if fairness is satisfied and accuracy improves
            
            if self.diff_or_ratio == 'diff':
                if fairness_violation <= self.fairness_tolerance:
                    record.append([self.tradeoff_param,  fairness_violation, accuracy])
                    if accuracy > best_accuracy or (accuracy == best_accuracy and fairness_violation < best_fairness_violation):
                        best_tradeoff_param = self.tradeoff_param
                        best_fairness_violation = fairness_violation
                        best_accuracy = accuracy
            else:
                if fairness_violation >=  1 - self.fairness_tolerance:
                    record.append([self.tradeoff_param,  fairness_violation, accuracy])
                    if accuracy > best_accuracy or (accuracy == best_accuracy and fairness_violation >= best_fairness_violation):
                        best_tradeoff_param = self.tradeoff_param
                        best_fairness_violation = fairness_violation
                        best_accuracy = accuracy
            
        
        # Assign the best tradeoff_param back to the model
        if best_tradeoff_param is not None:
            self.tradeoff_param = best_tradeoff_param
            logger.info(f"Best tradeoff_param: {best_tradeoff_param}")
            logger.info(f"Fairness violation: {best_fairness_violation}, Accuracy: {best_accuracy}")
        else:
            logger.info("No valid tradeoff_param found that satisfies fairness constraints.")
        return record



def normalize_weights(weights):
    """
    将样本权重归一化到 [-1, 1] 的范围。
    :param weights: 样本权重 (numpy array 或 list)
    :return: 归一化后的权重 (numpy array)
    """
    weights = np.array(weights, dtype=np.float32)
    
    # 找到最大值和最小值
    max_val = np.max(weights)
    min_val = np.min(weights)
    
    # 避免除以零
    if max_val == min_val:
        return np.zeros_like(weights)
    
    # 归一化到 [-1, 1] 范围
    normalized_weights = 2 * (weights - min_val) / (max_val - min_val) - 1
    
    return normalized_weights

class InProcessingFairClassifier:
    def __init__(self, cost_param, fairness_tolerance, fair_type, diff_or_ratio = 'diff', allow_sens = False):
        self.cost_param = cost_param
        self.fairness_tolerance = fairness_tolerance
        self.allow_sens = allow_sens
        self.fair_type = fair_type
        self.diff_or_ratio = diff_or_ratio
        self.base_model_Y_given_X_S = None   # P(Y=1 | X=x)
        self.base_model_YS_given_X = None # P(E_{z,m} | X=x)
        self.group_probs = None      # P(E_{z,m})

        self.pred_model = None



    def compute_p_s(self, S):
        """
        Compute P(S = s) for all unique values of S.

        Parameters:
            S (array-like): Array of sensitive attribute values.

        Returns:
            dict: Probability distribution of P(S = s).
        """
        S = np.array(S)
        count_s = Counter(S)
        total = len(S)
        p_s = {s: count / total for s, count in count_s.items()}
        return p_s

    def compute_p_s_given_y(self, S, Y):
        """
        Compute P(S = s | Y = y) for all combinations of S and Y.

        Parameters:
            S (array-like): Array of sensitive attribute values.
            Y (array-like): Array of target labels.

        Returns:
            dict: Nested dictionary representing P(S = s | Y = y).
        """
        # Combine S and Y for joint distribution
        joint = list(zip(S, Y))
        count_joint = Counter(joint)
        count_y = Counter(Y)
        
        p_s_given_y = {}
        for (s, y), count in count_joint.items():
            if y not in p_s_given_y:
                p_s_given_y[y] = {}
            p_s_given_y[y][s] = count / count_y[y]
        
        return p_s_given_y
    
    def compute_p_y_given_s(self, S, Y):
        """
        Compute P(Y = y | S = s) for all combinations of S and Y.

        Parameters:
            S (array-like): Array of sensitive attribute values.
            Y (array-like): Array of target labels.

        Returns:
            dict: Nested dictionary representing P(Y = y | S = s).
        """
        # Combine S and Y for joint distribution
        joint = list(zip(S, Y))
        count_joint = Counter(joint)
        count_s = Counter(S)
        
        p_y_given_s = {}
        for (s, y), count in count_joint.items():
            if s not in p_y_given_s:
                p_y_given_s[s] = {}
            p_y_given_s[s][y] = count / count_s[s]
        
        return p_y_given_s

    def get_specific_p_s(self,  s):
        """
        Get P(S = s) for a specific value of S.

        Parameters:
            p_s (dict): Dictionary from compute_p_s.
            s: Specific value of S.

        Returns:
            float: The probability P(S = s).
        """
        return self.p_s.get(s, None)  # Return 0 if the value of s is not in the dictionary


    def get_specific_group_prob(self, m,t):
        """
        Get P(Y = t, S = m) for specific values of t and m.

        Parameters:
            group_probs (dict): Dictionary of group probabilities from estimate_group_probs.
            t: Specific value of Y.
            m: Specific value of S.

        Returns:
            float: The probability P(Y = t, S = m), or 0 if not found.
        """
        return self.group_probs.get(( m, t), 0)  # Return 0 if (t, m) is not in the dictionary


    def get_specific_p_s_given_y(self, s, y):
        """
        Get P(S = s | Y = y) for specific values of s and y.

        Parameters:
            p_s_given_y (dict): Nested dictionary from compute_p_s_given_y.
            s: Specific value of S.
            y: Specific value of Y.

        Returns:
            float: The probability P(S = s | Y = y).
        """
        return self.p_s_given_y.get(y, {}).get(s, None)

    def get_specific_p_y_given_s(self, s, y):
        """
        Get P(Y = y | S = s) for specific values of y and s.

        Parameters:
            p_y_given_s (dict): Nested dictionary from compute_p_y_given_s.
            y: Specific value of Y.
            s: Specific value of S.

        Returns:
            float: The probability P(Y = y | S = s).
        """
        return self.p_y_given_s.get(s, {}).get(y, None)    
    
    def estimate_base_model_Y_given_X_S(self, X, S, Y):
        """
        Estimate η(x) = P(Y = 1 | X = x) using a standard classifier.
        """
        if self.allow_sens:
            X_combined = np.hstack((X, S.reshape(-1, 1)))  # 添加 S 作为一列
            self.base_model_Y_given_X_S = xgb.XGBClassifier()  #RandomForestClassifier()  #LogisticRegression()  #
            self.base_model_Y_given_X_S.fit(X_combined, Y)
            pred = self.base_model_Y_given_X_S.predict(X_combined)
        else: 
            self.base_model_Y_given_X_S = xgb.XGBClassifier()  #RandomForestClassifier() #LogisticRegression()  #
            self.base_model_Y_given_X_S.fit(X, Y)
            pred = self.base_model_Y_given_X_S.predict(X)
        logger.info(f"P(Y | X, S) or P(Y | X) =   {np.sum(pred == Y)/ len(Y)}, {np.sum(pred == Y)}, {len(Y)}" )


    def estimate_base_model_YS_given_X(self, X, S, Y):
        """
        Estimate group membership probabilities P(S, Y | X = x) 
        """
        joint_labels = [(s, y) for s, y in zip(S, Y)]
        joint_labels = pd.Series(joint_labels)

        # Use pandas.factorize to encode
        encoded_labels, unique_combinations = pd.factorize(joint_labels)
        self.gamma_sy_combination = unique_combinations.tolist()

        # Train a classifier to predict joint labels based on X
        self.base_model_YS_given_X = xgb.XGBClassifier()  # RandomForestClassifier() #LogisticRegression() #
        self.base_model_YS_given_X.fit(X, encoded_labels)

        # Predict probabilities for all samples
        predicted_encoded_labels = self.base_model_YS_given_X.predict(X)

        # Compute accuracy by comparing predicted and true joint labels
        gamma_acc = np.sum( predicted_encoded_labels == encoded_labels ) / len(encoded_labels)
        
        logger.info(f"P(S, Y | X = x) acc =, {gamma_acc},   {np.sum( predicted_encoded_labels == encoded_labels )}, {len(encoded_labels)}" )


    def estimate_group_probs(self, S, Y):
        """
        Calculate P(E_{z,m}) = P(Y = z, S = m) directly from the data.
        """
        unique_combinations, counts = np.unique(list(zip(S, Y)), axis=0, return_counts=True)
        total_samples = len(Y)
        self.group_probs = {tuple(comb): count / total_samples for comb, count in zip(unique_combinations, counts)}
        logger.info(f"self.group_probs = {self.group_probs}")
        return self.group_probs

    def predict(self, X ):
        """
        Predict using the fair classifier.
        """
        """
        Construct the fair classifier based on the decision rule.
        """

        predictions = self.pred_model.predict(X)
     
        return predictions #np.array(predictions)


    def compute_base_rates(self, X, S, Y):

        self.p_s = self.compute_p_s(S)
        self.p_s_given_y = self.compute_p_s_given_y(S, Y)
        self.p_y_given_s = self.compute_p_y_given_s(S, Y)
        self.group_probs = self.estimate_group_probs(S, Y)
        self.estimate_base_model_Y_given_X_S(X, S, Y)

        logger.info(f"self.p_s = {self.p_s}")
        if not self.allow_sens:
            self.estimate_base_model_YS_given_X(X, S, Y)



    def fit(self, X, S, Y, group_ids, params, fairness_metrics_eval , valid_ratio ):
        """
        Train the fair classifier via post-processing.
        """
        # Split the training data into training and validation sets
        X_train, X_val, S_train, S_val, Y_train, Y_val = train_test_split(X, S, Y, test_size=valid_ratio)  #,  random_state=42

        # Assign tradeoff_param
        self.tradeoff_param = {group: param for group, param in zip(group_ids, params)}
        Lam_M = sum(self.tradeoff_param.values())
        if self.diff_or_ratio == 'diff':
            tau = 1
        elif self.diff_or_ratio == 'ratio':
            tau = self.fairness_tolerance
        else:
            print("please use valid fairness type (diff or ratio)")

        def compute_Q(self, X, S, Y, lambda_set):
            if self.allow_sens:
                XS = np.hstack([X, S.reshape(-1, 1)])
                PY_XS = self.base_model_Y_given_X_S.predict_proba(XS) # Batch prediction

                if self.fair_type == 'DP':
                    lambda_m_p = np.array([lambda_set[m]/self.get_specific_p_s(m)  for m in S]) 
                    fairness_term = lambda_m_p  -   tau * Lam_M  

                elif self.fair_type == 'EOpp':
                    lambda_m = np.array([lambda_set[m] for m in S])
                    bmt = np.array([self.get_specific_p_s_given_y(m, 1) for m in S])
                    P_m_yt = np.array([self.get_specific_group_prob(m, 1) for m in S])
                    fairness_term = (lambda_m  - tau * Lam_M * bmt) * PY_XS[:,1] / P_m_yt
                elif self.fair_type == 'PredEqual':
                    lambda_m = np.array([lambda_set[m] for m in S])
                    bmt = np.array([self.get_specific_p_s_given_y(m, 0) for m in S])
                    P_m_yt = np.array([self.get_specific_group_prob(m, 0) for m in S])
                    fairness_term = (lambda_m  - tau * Lam_M * bmt) * PY_XS[:,0] / P_m_yt
                elif self.fair_type == 'AP':
                    amt = np.array([  1/self.get_specific_p_s(m) for m in S])
                    lambda_m = np.array([lambda_set[m] for m in S])
                    fairness_term = ( amt * lambda_m - tau * Lam_M ) * (  PY_XS[:,0] -  PY_XS[:,1]  )
                else:
                    logger.info('Error! Please use a valid fairness type (i.e., DP, EOpp, PredEqual, AP)')
                
            else:
                PYS_X = self.base_model_YS_given_X.predict_proba(X)    # Batch group probabilities
                fairness_term = np.zeros(len(X))
                if self.fair_type == 'DP':
                    for m in np.unique(S):
                        lambda_m =  lambda_set[m] 
                        bmt = self.get_specific_p_s(m) 
                        ind_pos, ind_neg = self.gamma_sy_combination.index((m, 1)), self.gamma_sy_combination.index((m, 0))
                        PS_X = PYS_X[:,ind_neg] + PYS_X[:,ind_pos] 
                        fairness_term += (lambda_m  -   tau * bmt * Lam_M  ) * PS_X / bmt
                elif self.fair_type == 'EOpp':
                    for m in np.unique(S):

                        lambda_m =  lambda_set[m] 
                        bmt = self.get_specific_p_s_given_y(m, 1)
                        ind_pos  = self.gamma_sy_combination.index((m, 1))
                        P_m_yt = self.get_specific_group_prob(m, 1)


                        fairness_term += (lambda_m  - tau * Lam_M * bmt) * PYS_X[:,ind_pos]  / P_m_yt
                elif self.fair_type == 'PredEqual':
                    for m in np.unique(S):
                        
                        lambda_m =  lambda_set[m] 
                        bmt = self.get_specific_p_s_given_y(m, 0)  
                        ind_neg  = self.gamma_sy_combination.index((m, 0))
                        P_m_yt = self.get_specific_group_prob(m, 0)
                        fairness_term += (lambda_m  - tau * Lam_M * bmt) *  PYS_X[:,ind_neg] / P_m_yt
                elif self.fair_type == 'AP':
                    for m in np.unique(S):
                        lambda_m =  lambda_set[m] 
                        amt = self.get_specific_p_y_given_s(m,0)
                        P_m_yt = self.get_specific_group_prob(m, 0)
                        ind_neg  = self.gamma_sy_combination.index((m, 0))
                        fairness_term += (amt * lambda_m  - tau * Lam_M * P_m_yt) *  PYS_X[:,ind_neg] / P_m_yt


                        amt = self.get_specific_p_y_given_s(m,1)
                        P_m_yt = self.get_specific_group_prob(m, 1)
                        ind_pos  = self.gamma_sy_combination.index((m, 1))
                        fairness_term += (amt * lambda_m  - tau * Lam_M * P_m_yt) *  PYS_X[:,ind_pos] / P_m_yt
                else:
                    logger.info('Error! Please use a valid fairness type (i.e., DP, EOpp, PredEqual, AP)')

            return fairness_term
        

        if self.diff_or_ratio == 'diff':
            Q =   compute_Q(self, X_train, S_train, Y_train, self.tradeoff_param)
        elif self.diff_or_ratio == 'ratio':
            Q = - compute_Q(self, X_train, S_train, Y_train, self.tradeoff_param)

        sample_weights= (1- 2 * Y_train) * ( self.cost_param + Q ) + Y_train
        sample_weights = np.array(sample_weights)

        # Train a cost-sensitive Logistic Regression model
        

        if self.allow_sens:
            XS = np.hstack([X_train, S_train.reshape(-1, 1)])
            XS_val = np.hstack([X_val, S_val.reshape(-1, 1)])

            self.pred_model  = WeightedClassifier()  #(input_dim= XS.shape[1] , learning_rate=0.01)
            self.pred_model.fit(XS, Y_train, sample_weights)
            Y_pred = self.predict(XS )
            Y_pred_val = self.predict(XS_val )
        else:
            self.pred_model  = WeightedClassifier()  #(input_dim= X_train.shape[1] , learning_rate=0.01)
            self.pred_model.fit(X_train, Y_train, sample_weights)
            Y_pred = self.predict(X_train )
            Y_pred_val = self.predict(X_val )


        fairness_violation = fairness_metrics_eval(Y_train,  S_train, Y_pred,  fairness_notion= self.fair_type, diff_or_ratio= self.diff_or_ratio )
        accuracy = np.mean(Y_pred == Y_train)



        # # Compute fairness metrics and accuracy
        fairness_violation = fairness_metrics_eval(Y_val,  S_val, Y_pred_val,  fairness_notion= self.fair_type, diff_or_ratio= self.diff_or_ratio )
        accuracy = np.mean(Y_pred_val == Y_val)


        record =[self.tradeoff_param,  fairness_violation, accuracy]
        # Update the best parameters if fairness is satisfied and accuracy improves
        
        
    
        return record




class InProcessingFairClassifier_logit:
    def __init__(self, cost_param, fairness_tolerance, fair_type, diff_or_ratio = 'diff', allow_sens = False):
        self.cost_param = cost_param
        self.fairness_tolerance = fairness_tolerance
        self.allow_sens = allow_sens
        self.fair_type = fair_type
        self.diff_or_ratio = diff_or_ratio
        self.base_model_Y_given_X_S = None   # P(Y=1 | X=x)
        self.base_model_YS_given_X = None # P(E_{z,m} | X=x)
        self.group_probs = None      # P(E_{z,m})

        self.pred_model = None



    def compute_p_s(self, S):
        """
        Compute P(S = s) for all unique values of S.

        Parameters:
            S (array-like): Array of sensitive attribute values.

        Returns:
            dict: Probability distribution of P(S = s).
        """
        S = np.array(S)
        count_s = Counter(S)
        total = len(S)
        p_s = {s: count / total for s, count in count_s.items()}
        return p_s

    def compute_p_s_given_y(self, S, Y):
        """
        Compute P(S = s | Y = y) for all combinations of S and Y.

        Parameters:
            S (array-like): Array of sensitive attribute values.
            Y (array-like): Array of target labels.

        Returns:
            dict: Nested dictionary representing P(S = s | Y = y).
        """
        # Combine S and Y for joint distribution
        joint = list(zip(S, Y))
        count_joint = Counter(joint)
        count_y = Counter(Y)
        
        p_s_given_y = {}
        for (s, y), count in count_joint.items():
            if y not in p_s_given_y:
                p_s_given_y[y] = {}
            p_s_given_y[y][s] = count / count_y[y]
        
        return p_s_given_y
    
    def compute_p_y_given_s(self, S, Y):
        """
        Compute P(Y = y | S = s) for all combinations of S and Y.

        Parameters:
            S (array-like): Array of sensitive attribute values.
            Y (array-like): Array of target labels.

        Returns:
            dict: Nested dictionary representing P(Y = y | S = s).
        """
        # Combine S and Y for joint distribution
        joint = list(zip(S, Y))
        count_joint = Counter(joint)
        count_s = Counter(S)
        
        p_y_given_s = {}
        for (s, y), count in count_joint.items():
            if s not in p_y_given_s:
                p_y_given_s[s] = {}
            p_y_given_s[s][y] = count / count_s[s]
        
        return p_y_given_s

    def get_specific_p_s(self,  s):
        """
        Get P(S = s) for a specific value of S.

        Parameters:
            p_s (dict): Dictionary from compute_p_s.
            s: Specific value of S.

        Returns:
            float: The probability P(S = s).
        """
        return self.p_s.get(s, None)  # Return 0 if the value of s is not in the dictionary


    def get_specific_group_prob(self, m,t):
        """
        Get P(Y = t, S = m) for specific values of t and m.

        Parameters:
            group_probs (dict): Dictionary of group probabilities from estimate_group_probs.
            t: Specific value of Y.
            m: Specific value of S.

        Returns:
            float: The probability P(Y = t, S = m), or 0 if not found.
        """
        return self.group_probs.get(( m, t), 0)  # Return 0 if (t, m) is not in the dictionary


    def get_specific_p_s_given_y(self, s, y):
        """
        Get P(S = s | Y = y) for specific values of s and y.

        Parameters:
            p_s_given_y (dict): Nested dictionary from compute_p_s_given_y.
            s: Specific value of S.
            y: Specific value of Y.

        Returns:
            float: The probability P(S = s | Y = y).
        """
        return self.p_s_given_y.get(y, {}).get(s, None)

    def get_specific_p_y_given_s(self, s, y):
        """
        Get P(Y = y | S = s) for specific values of y and s.

        Parameters:
            p_y_given_s (dict): Nested dictionary from compute_p_y_given_s.
            y: Specific value of Y.
            s: Specific value of S.

        Returns:
            float: The probability P(Y = y | S = s).
        """
        return self.p_y_given_s.get(s, {}).get(y, None)    
    
    def estimate_base_model_Y_given_X_S(self, X, S, Y):
        """
        Estimate η(x) = P(Y = 1 | X = x) using a standard classifier.
        """
        if self.allow_sens:
            X_combined = np.hstack((X, S.reshape(-1, 1)))  # 添加 S 作为一列
            self.base_model_Y_given_X_S = LogisticRegression()  #RandomForestClassifier()  #LogisticRegression()  #
            self.base_model_Y_given_X_S.fit(X_combined, Y)
            pred = self.base_model_Y_given_X_S.predict(X_combined)
        else: 
            self.base_model_Y_given_X_S = LogisticRegression()  #RandomForestClassifier() #LogisticRegression()  #
            self.base_model_Y_given_X_S.fit(X, Y)
            pred = self.base_model_Y_given_X_S.predict(X)
        logger.info(f"P(Y | X, S) or P(Y | X) =   {np.sum(pred == Y)/ len(Y)}, {np.sum(pred == Y)}, {len(Y)}" )


    def estimate_base_model_YS_given_X(self, X, S, Y):
        """
        Estimate group membership probabilities P(S, Y | X = x) 
        """
        joint_labels = [(s, y) for s, y in zip(S, Y)]
        joint_labels = pd.Series(joint_labels)

        # Use pandas.factorize to encode
        encoded_labels, unique_combinations = pd.factorize(joint_labels)
        self.gamma_sy_combination = unique_combinations.tolist()

        # Train a classifier to predict joint labels based on X
        self.base_model_YS_given_X = LogisticRegression()  # RandomForestClassifier() #LogisticRegression() #
        self.base_model_YS_given_X.fit(X, encoded_labels)

        # Predict probabilities for all samples
        predicted_encoded_labels = self.base_model_YS_given_X.predict(X)

        # Compute accuracy by comparing predicted and true joint labels
        gamma_acc = np.sum( predicted_encoded_labels == encoded_labels ) / len(encoded_labels)
        
        logger.info(f"P(S, Y | X = x) acc =, {gamma_acc},   {np.sum( predicted_encoded_labels == encoded_labels )}, {len(encoded_labels)}" )


    def estimate_group_probs(self, S, Y):
        """
        Calculate P(E_{z,m}) = P(Y = z, S = m) directly from the data.
        """
        unique_combinations, counts = np.unique(list(zip(S, Y)), axis=0, return_counts=True)
        total_samples = len(Y)
        self.group_probs = {tuple(comb): count / total_samples for comb, count in zip(unique_combinations, counts)}
        logger.info(f"self.group_probs = {self.group_probs}")
        return self.group_probs

    def predict(self, X ):
        """
        Predict using the fair classifier.
        """
        """
        Construct the fair classifier based on the decision rule.
        """

        predictions = self.pred_model.predict(X)
     
        return predictions #np.array(predictions)


    def compute_base_rates(self, X, S, Y):

        self.p_s = self.compute_p_s(S)
        self.p_s_given_y = self.compute_p_s_given_y(S, Y)
        self.p_y_given_s = self.compute_p_y_given_s(S, Y)
        self.group_probs = self.estimate_group_probs(S, Y)
        self.estimate_base_model_Y_given_X_S(X, S, Y)

        logger.info(f"self.p_s = {self.p_s}")
        if not self.allow_sens:
            self.estimate_base_model_YS_given_X(X, S, Y)



    def fit(self, X, S, Y, group_ids, params, fairness_metrics_eval , valid_ratio ):
        """
        Train the fair classifier via post-processing.
        """
        # Split the training data into training and validation sets
        X_train, X_val, S_train, S_val, Y_train, Y_val = train_test_split(X, S, Y, test_size=valid_ratio)  #,  random_state=42

        # Assign tradeoff_param
        self.tradeoff_param = {group: param for group, param in zip(group_ids, params)}
        Lam_M = sum(self.tradeoff_param.values())
        if self.diff_or_ratio == 'diff':
            tau = 1
        elif self.diff_or_ratio == 'ratio':
            tau = self.fairness_tolerance
        else:
            print("please use valid fairness type (diff or ratio)")

        def compute_Q(self, X, S, Y, lambda_set):
            if self.allow_sens:
                XS = np.hstack([X, S.reshape(-1, 1)])
                PY_XS = self.base_model_Y_given_X_S.predict_proba(XS) # Batch prediction

                if self.fair_type == 'DP':
                    lambda_m_p = np.array([lambda_set[m]/self.get_specific_p_s(m)  for m in S]) 
                    fairness_term = lambda_m_p  -   tau * Lam_M  

                elif self.fair_type == 'EOpp':
                    lambda_m = np.array([lambda_set[m] for m in S])
                    bmt = np.array([self.get_specific_p_s_given_y(m, 1) for m in S])
                    P_m_yt = np.array([self.get_specific_group_prob(m, 1) for m in S])
                    fairness_term = (lambda_m  - tau * Lam_M * bmt) * PY_XS[:,1] / P_m_yt
                elif self.fair_type == 'PredEqual':
                    lambda_m = np.array([lambda_set[m] for m in S])
                    bmt = np.array([self.get_specific_p_s_given_y(m, 0) for m in S])
                    P_m_yt = np.array([self.get_specific_group_prob(m, 0) for m in S])
                    fairness_term = (lambda_m  - tau * Lam_M * bmt) * PY_XS[:,0] / P_m_yt
                elif self.fair_type == 'AP':
                    amt = np.array([  1/self.get_specific_p_s(m) for m in S])
                    lambda_m = np.array([lambda_set[m] for m in S])
                    fairness_term = ( amt * lambda_m - tau * Lam_M ) * (  PY_XS[:,0] -  PY_XS[:,1]  )
                else:
                    logger.info('Error! Please use a valid fairness type (i.e., DP, EOpp, PredEqual, AP)')
                
            else:
                PYS_X = self.base_model_YS_given_X.predict_proba(X)    # Batch group probabilities
                fairness_term = np.zeros(len(X))
                if self.fair_type == 'DP':
                    for m in np.unique(S):
                        lambda_m =  lambda_set[m] 
                        bmt = self.get_specific_p_s(m) 
                        ind_pos, ind_neg = self.gamma_sy_combination.index((m, 1)), self.gamma_sy_combination.index((m, 0))
                        PS_X = PYS_X[:,ind_neg] + PYS_X[:,ind_pos] 
                        fairness_term += (lambda_m  -   tau * bmt * Lam_M  ) * PS_X / bmt
                elif self.fair_type == 'EOpp':
                    for m in np.unique(S):

                        lambda_m =  lambda_set[m] 
                        bmt = self.get_specific_p_s_given_y(m, 1)
                        ind_pos  = self.gamma_sy_combination.index((m, 1))
                        P_m_yt = self.get_specific_group_prob(m, 1)


                        fairness_term += (lambda_m  - tau * Lam_M * bmt) * PYS_X[:,ind_pos]  / P_m_yt
                elif self.fair_type == 'PredEqual':
                    for m in np.unique(S):
                        
                        lambda_m =  lambda_set[m] 
                        bmt = self.get_specific_p_s_given_y(m, 0)  
                        ind_neg  = self.gamma_sy_combination.index((m, 0))
                        P_m_yt = self.get_specific_group_prob(m, 0)
                        fairness_term += (lambda_m  - tau * Lam_M * bmt) *  PYS_X[:,ind_neg] / P_m_yt
                elif self.fair_type == 'AP':
                    for m in np.unique(S):
                        lambda_m =  lambda_set[m] 
                        amt = self.get_specific_p_y_given_s(m,0)
                        P_m_yt = self.get_specific_group_prob(m, 0)
                        ind_neg  = self.gamma_sy_combination.index((m, 0))
                        fairness_term += (amt * lambda_m  - tau * Lam_M * P_m_yt) *  PYS_X[:,ind_neg] / P_m_yt


                        amt = self.get_specific_p_y_given_s(m,1)
                        P_m_yt = self.get_specific_group_prob(m, 1)
                        ind_pos  = self.gamma_sy_combination.index((m, 1))
                        fairness_term += (amt * lambda_m  - tau * Lam_M * P_m_yt) *  PYS_X[:,ind_pos] / P_m_yt
                else:
                    logger.info('Error! Please use a valid fairness type (i.e., DP, EOpp, PredEqual, AP)')

            return fairness_term
        

        if self.diff_or_ratio == 'diff':
            Q =   compute_Q(self, X_train, S_train, Y_train, self.tradeoff_param)
        elif self.diff_or_ratio == 'ratio':
            Q = - compute_Q(self, X_train, S_train, Y_train, self.tradeoff_param)

        sample_weights= (1- 2 * Y_train) * ( self.cost_param + Q ) + Y_train
        sample_weights = np.array(sample_weights)

        # Train a cost-sensitive Logistic Regression model
        

        if self.allow_sens:
            XS = np.hstack([X_train, S_train.reshape(-1, 1)])
            XS_val = np.hstack([X_val, S_val.reshape(-1, 1)])

            self.pred_model  = WeightedClassifier()  #(input_dim= XS.shape[1] , learning_rate=0.01)
            self.pred_model.fit(XS, Y_train, sample_weights)
            Y_pred = self.predict(XS )
            Y_pred_val = self.predict(XS_val )
        else:
            self.pred_model  = WeightedClassifier()  #(input_dim= X_train.shape[1] , learning_rate=0.01)
            self.pred_model.fit(X_train, Y_train, sample_weights)
            Y_pred = self.predict(X_train )
            Y_pred_val = self.predict(X_val )


        fairness_violation = fairness_metrics_eval(Y_train,  S_train, Y_pred,  fairness_notion= self.fair_type, diff_or_ratio= self.diff_or_ratio )
        accuracy = np.mean(Y_pred == Y_train)



        # # Compute fairness metrics and accuracy
        fairness_violation = fairness_metrics_eval(Y_val,  S_val, Y_pred_val,  fairness_notion= self.fair_type, diff_or_ratio= self.diff_or_ratio )
        accuracy = np.mean(Y_pred_val == Y_val)


        record =[self.tradeoff_param,  fairness_violation, accuracy]
        # Update the best parameters if fairness is satisfied and accuracy improves
        
        
    
        return record


