import sklearn, sklearn.linear_model
import pickle
import pandas as pd
import os
from argparse import Namespace
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

from utils.methods import (
    PostProcessingFairClassifier,
    PostProcessingFairClassifier_logit,
    InProcessingFairClassifier,
    InProcessingFairClassifier_logit,
    set_all_seed,
    load_adult,
    eval_fair,
    load_adult_multi,
    load_compas,
    load_compas_multi
)

from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
from collections import Counter

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s -%(module)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %p",
    level=10,
)

logger = logging.getLogger(__name__)



timestart = datetime.now().timestamp()
stamp = str(datetime.fromtimestamp( timestart ).strftime("%Y%m%d%H%M%S"))


import time
# 记录开始时间
start_time = time.time()

def run_trial(cfg):
    ### _config 包含了实验的相关配置，如数据集、方法名称、批次大小、学习率等。
    _config = Namespace(
        project_name=f"Bayes-{cfg.dataset}-{cfg.parameters.fair_notion}-{cfg.parameters.type}",
        seed=cfg.seed,
        dataset=cfg.dataset,
        method=cfg.method,
    )

    logger.info(f"using the dataset {cfg.dataset}")

    set_all_seed(cfg.seed)

    if cfg.parameters.type == "MD":
        MD_or_MR = 'diff'
    elif cfg.parameters.type == "MR":
        MD_or_MR = 'ratio'

    scaler = sklearn.preprocessing.StandardScaler()

    # Data loading for required datasets
    if cfg.dataset == "adult":
        data_dir = "data/adult"
        (inputs_df, labels, label_names, groups, group_names) = load_adult(data_dir, remove_sensitive_attr= True )
        inputs = np.array(inputs_df.values, dtype=np.float32)
        inputs = scaler.fit_transform(inputs)

    if cfg.dataset == "adult_multi":
        data_dir = "data/adult"
        (inputs_df, labels, label_names, groups, group_names) = load_adult_multi(data_dir, remove_sensitive_attr= True )
        inputs = np.array(inputs_df.values, dtype=np.float32)
        inputs = scaler.fit_transform(inputs)

        logger.info( f"groups = {groups[0:10]}")
        logger.info( f"labels = {labels[0:10]}")

    if cfg.dataset == "compas":
        data_dir = "data/compas"
        (inputs_df, labels, label_names, groups, group_names) = load_compas(data_dir, remove_sensitive_attr= True )
        inputs = np.array(inputs_df.values, dtype=np.float32)
        inputs = scaler.fit_transform(inputs)

    if cfg.dataset == "compas_multi":
        data_dir = "data/compas"
        (inputs_df, labels, label_names, groups, group_names) = load_compas_multi(data_dir, remove_sensitive_attr= True )
        inputs = np.array(inputs_df.values, dtype=np.float32)
        inputs = scaler.fit_transform(inputs)

        logger.info( f"groups = {groups[0:10]}")
        logger.info( f"labels = {labels[0:10]}")

    # 创建一个 DataFrame
    df_summary = pd.DataFrame({'group': groups, 'label': labels})

    # 使用 crosstab 来统计每个 group 和 label 的样本量
    group_label_counts = pd.crosstab(df_summary['group'], df_summary['label'])
    logger.info(f"group_label_counts:\n{group_label_counts}")

    # Implement required methods
    if cfg.method == 'Bayes_fair':
        logger.info(f"start the method = {cfg.method}")
        X_train, X_test, S_train, S_test, Y_train, Y_test = train_test_split(inputs , groups , labels , test_size= cfg.parameters.test_ratio, random_state= cfg.seed)

        # 将 S 和 Y 组合为单一类别标签
        sy_train = np.array([f"{s}_{y}" for s, y in zip(S_train, Y_train)])
        # 查看平衡后的类别分布
        logger.info(f"Unbalanced class distribution:{Counter(sy_train)}")

        post_fair_classifier = PostProcessingFairClassifier(cost_param= cfg.parameters.c , fairness_tolerance= cfg.parameters.fair_tolerance, fair_type = cfg.parameters.fair_notion , diff_or_ratio = MD_or_MR, allow_sens = cfg.parameters.allow_sens, grid_interval =  cfg.parameters.grid_interval )

        records_train = post_fair_classifier.fit(X_train, S_train, Y_train, fairness_metrics_eval = eval_fair, grid_values = cfg.parameters.max_lambda, valid_ratio = cfg.parameters.valid_ratio )

        # 在测试集上评估（使用验证集的最佳参数）
        lambda_M_i = post_fair_classifier.tradeoff_param  # 直接使用验证集上的最佳参数
        predictions = post_fair_classifier.predict(X_test, S_test, lambda_M_i, Y_test)

        fairness_violation = eval_fair(Y_test,  S_test, predictions,  fairness_notion= cfg.parameters.fair_notion , diff_or_ratio= MD_or_MR )
        accuracy = np.mean(Y_test == predictions)

        records = [[lambda_M_i, 0, 0, fairness_violation, accuracy]]  # train_fair 和 train_acc 这里不再需要，设为0
        logger.info(f"Test set - Fairness violation: {fairness_violation}, Accuracy: {accuracy}")

    elif cfg.method == 'Bayes_fair_logit':
        logger.info(f"start the method = {cfg.method}")
        X_train, X_test, S_train, S_test, Y_train, Y_test = train_test_split(inputs , groups , labels , test_size= cfg.parameters.test_ratio, random_state= cfg.seed)

        # 将 S 和 Y 组合为单一类别标签
        sy_train = np.array([f"{s}_{y}" for s, y in zip(S_train, Y_train)])
        # 查看平衡后的类别分布
        logger.info(f"Unbalanced class distribution:{Counter(sy_train)}")

        post_fair_classifier = PostProcessingFairClassifier_logit(cost_param= cfg.parameters.c , fairness_tolerance= cfg.parameters.fair_tolerance, fair_type = cfg.parameters.fair_notion , diff_or_ratio = MD_or_MR, allow_sens = cfg.parameters.allow_sens, grid_interval =  cfg.parameters.grid_interval )

        records_train = post_fair_classifier.fit(X_train, S_train, Y_train, fairness_metrics_eval = eval_fair, grid_values = cfg.parameters.max_lambda, valid_ratio = cfg.parameters.valid_ratio )

        # 在测试集上评估（使用验证集的最佳参数）
        lambda_M_i = post_fair_classifier.tradeoff_param  # 直接使用验证集上的最佳参数
        predictions = post_fair_classifier.predict(X_test, S_test, lambda_M_i, Y_test)

        fairness_violation = eval_fair(Y_test,  S_test, predictions,  fairness_notion= cfg.parameters.fair_notion , diff_or_ratio= MD_or_MR )
        accuracy = np.mean(Y_test == predictions)

        records = [[lambda_M_i, 0, 0, fairness_violation, accuracy]]  # train_fair 和 train_acc 这里不再需要，设为0
        logger.info(f"Test set - Fairness violation: {fairness_violation}, Accuracy: {accuracy}")

    elif cfg.method == 'Bayes_fair_inprocess':
        logger.info(f"start the method = {cfg.method}")
        X_train, X_test, S_train, S_test, Y_train, Y_test = train_test_split(inputs , groups , labels , test_size=cfg.parameters.test_ratio, random_state= cfg.seed)


        post_fair_classifier = PostProcessingFairClassifier(cost_param= cfg.parameters.c , fairness_tolerance= cfg.parameters.fair_tolerance, fair_type = cfg.parameters.fair_notion , diff_or_ratio = MD_or_MR, allow_sens = cfg.parameters.allow_sens, grid_interval =  cfg.parameters.grid_interval )

        records_train = post_fair_classifier.fit(X_train, S_train, Y_train, fairness_metrics_eval = eval_fair, grid_values = cfg.parameters.max_lambda, valid_ratio = cfg.parameters.valid_ratio )


        # 在测试集上评估 post-processing（使用验证集的最佳参数）
        lambda_M_i = post_fair_classifier.tradeoff_param  # 直接使用验证集上的最佳参数
        predictions = post_fair_classifier.predict(X_test, S_test, lambda_M_i)

        fairness_violation = eval_fair(Y_test,  S_test, predictions,  fairness_notion= cfg.parameters.fair_notion , diff_or_ratio= MD_or_MR )
        accuracy = np.mean(Y_test == predictions)
        logger.info(f"Post-processing on test set - Fairness violation: {fairness_violation}, Accuracy: {accuracy}")

        # In-processing: 使用验证集的最佳参数
        in_fair_classifier = InProcessingFairClassifier(cost_param= cfg.parameters.c , fairness_tolerance= cfg.parameters.fair_tolerance, fair_type = cfg.parameters.fair_notion , diff_or_ratio = MD_or_MR, allow_sens = cfg.parameters.allow_sens )

        group_ids = np.unique(S_train)
        params = tuple(lambda_M_i.values())

        records = []
        in_fair_classifier.compute_base_rates(X_train, S_train, Y_train)
        records_train_i = in_fair_classifier.fit(X_train, S_train, Y_train, group_ids, params, fairness_metrics_eval = eval_fair , valid_ratio = cfg.parameters.valid_ratio )

        lambda_M_in,  train_fair, train_acc = records_train_i

        if cfg.parameters.allow_sens:
            XS_test_array = np.hstack((X_test, S_test.reshape(-1, 1)))
            predictions = in_fair_classifier.predict(XS_test_array)
        else:
            predictions = in_fair_classifier.predict(X_test)

        fairness_violation = eval_fair(Y_test,  S_test, predictions,  fairness_notion= cfg.parameters.fair_notion , diff_or_ratio= MD_or_MR )
        accuracy = np.mean(Y_test == predictions)
        records.append( [ lambda_M_in,  train_fair, train_acc,  fairness_violation, accuracy ] )

        logger.info(f"In-processing on test set - Best tradeoff_param (from validation): {lambda_M_i}")
        logger.info(f"In-processing on test set - Fairness violation: {fairness_violation}, Accuracy: {accuracy}")

    elif cfg.method == 'Bayes_fair_inprocess_logit':
        logger.info(f"start the method = {cfg.method}")
        X_train, X_test, S_train, S_test, Y_train, Y_test = train_test_split(inputs , groups , labels , test_size=cfg.parameters.test_ratio, random_state= cfg.seed)


        post_fair_classifier = PostProcessingFairClassifier_logit(cost_param= cfg.parameters.c , fairness_tolerance= cfg.parameters.fair_tolerance, fair_type = cfg.parameters.fair_notion , diff_or_ratio = MD_or_MR, allow_sens = cfg.parameters.allow_sens, grid_interval =  cfg.parameters.grid_interval )

        records_train = post_fair_classifier.fit(X_train, S_train, Y_train, fairness_metrics_eval = eval_fair, grid_values = cfg.parameters.max_lambda, valid_ratio = cfg.parameters.valid_ratio )

        # 在测试集上评估 post-processing（使用验证集的最佳参数）
        lambda_M_i = post_fair_classifier.tradeoff_param  # 直接使用验证集上的最佳参数
        predictions = post_fair_classifier.predict(X_test, S_test, lambda_M_i)

        fairness_violation = eval_fair(Y_test,  S_test, predictions,  fairness_notion= cfg.parameters.fair_notion , diff_or_ratio= MD_or_MR )
        accuracy = np.mean(Y_test == predictions)
        logger.info(f"Post-processing on test set - Fairness violation: {fairness_violation}, Accuracy: {accuracy}")

        # In-processing: 使用验证集的最佳参数
        in_fair_classifier = InProcessingFairClassifier_logit(cost_param= cfg.parameters.c , fairness_tolerance= cfg.parameters.fair_tolerance, fair_type = cfg.parameters.fair_notion , diff_or_ratio = MD_or_MR, allow_sens = cfg.parameters.allow_sens )

        group_ids = np.unique(S_train)
        params = tuple(lambda_M_i.values())

        records = []
        in_fair_classifier.compute_base_rates(X_train, S_train, Y_train)
        records_train_i = in_fair_classifier.fit(X_train, S_train, Y_train, group_ids, params, fairness_metrics_eval = eval_fair , valid_ratio = cfg.parameters.valid_ratio )

        lambda_M_in,  train_fair, train_acc = records_train_i

        if cfg.parameters.allow_sens:
            XS_test_array = np.hstack((X_test, S_test.reshape(-1, 1)))
            predictions = in_fair_classifier.predict(XS_test_array)
        else:
            predictions = in_fair_classifier.predict(X_test)

        fairness_violation = eval_fair(Y_test,  S_test, predictions,  fairness_notion= cfg.parameters.fair_notion , diff_or_ratio= MD_or_MR )
        accuracy = np.mean(Y_test == predictions)
        records.append( [ lambda_M_in,  train_fair, train_acc,  fairness_violation, accuracy ] )

        logger.info(f"In-processing on test set - Best tradeoff_param (from validation): {lambda_M_i}")
        logger.info(f"In-processing on test set - Fairness violation: {fairness_violation}, Accuracy: {accuracy}")


    model_dir = os.path.join( cfg.log_dir , cfg.parameters.fair_notion, cfg.parameters.type, cfg.dataset, cfg.method)

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "result-" + stamp)
    config_path = os.path.join(model_dir, "config-" + stamp)

    if len(records) != 0:
        # 保存到二进制文件
        with open( model_path, "wb") as f:
            pickle.dump(records, f)
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)

        logger.info(f"Record saved to {model_path} , and  {config_path} ")





@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig):
    logging.info("Starting run...")
    logging.info(f"using parameters:\n {OmegaConf.to_yaml(cfg)}")

    run_trial(cfg)

    logger.info("---------------------------------------------------")
    logger.info("finished!")
    logger.info("---------------------------------------------------")

    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    elapsed_time = end_time - start_time
    logging.info(f"代码运行时间: {elapsed_time:.6f} 秒")

if __name__ == "__main__":
    main()
