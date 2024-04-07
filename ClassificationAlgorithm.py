from copy import deepcopy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb



def self_xgb_construction_classifier(metabolites, x_imposeds, x_miss_types, threshold_i, threshold_ii):
    """
        xgboost分类器：
        连续空缺值、最大值、最小值、平均值、中位数、代谢物缺失率、代谢物、代谢物浓度(高中低)
    :param metabolites: 代谢物名称
    :param x_imposed: 生成的缺失数据
    :param x_miss_type: 生成数据的缺失类型
    :param threshold_i: 低浓度阈值
    :param threshold_ii: 高浓度阈值
    :return: 分类器, 验证集准确率
    """
    metabolites = metabolites.tolist()
    train_data = []
    for x_imposed, x_miss_type in zip(x_imposeds, x_miss_types):
        # 按照平均丰度排序
        mean_concentrations = np.nanmean(x_imposed, axis=1)

        sorted_metabolites = [x for _, x in sorted(zip(mean_concentrations, metabolites))]

        # 根据阈值划分代谢物变量组
        low_abundance_metabolites = sorted_metabolites[:threshold_i]
        mid_abundance_metabolites = sorted_metabolites[threshold_i:threshold_ii]
        high_abundance_metabolites = sorted_metabolites[threshold_ii:]

        for i in range(x_imposed.shape[0]):
            tmp_mis_prop = np.count_nonzero(np.isnan(x_imposed[i])) / len(x_imposed[i])
            for j in range(x_imposed.shape[1]):
                if not np.isnan(x_imposed[i][j]):
                    continue
                tmp_train_data = []
                sample_nan_cnt = 0
                tmp_j = j
                while tmp_j >= 0 and np.isnan(x_imposed[i][tmp_j]):
                    tmp_j -= 1
                    sample_nan_cnt += 1
                tmp_j = j+1
                while tmp_j < x_imposed.shape[1] and np.isnan(x_imposed[i][tmp_j]):
                    tmp_j += 1
                    sample_nan_cnt += 1
                metabolites_nan_cnt = 0
                tmp_i = i
                while tmp_i >= 0 and np.isnan(x_imposed[tmp_i][j]):
                    tmp_i -= 1
                    metabolites_nan_cnt += 1
                tmp_i = i + 1
                while tmp_i < x_imposed.shape[0] and np.isnan(x_imposed[tmp_i][j]):
                    tmp_i += 1
                    metabolites_nan_cnt += 1
                tmp_train_data.append(sample_nan_cnt)
                tmp_train_data.append(metabolites_nan_cnt)
                tmp_train_data.append(np.nanmax(x_imposed[i]))
                tmp_train_data.append(np.nanmin(x_imposed[i]))
                tmp_train_data.append((np.nanmedian(x_imposed[i])))
                tmp_train_data.append((np.nanmean(x_imposed[i])))
                tmp_train_data.append(tmp_mis_prop)
                tmp_train_data.append(metabolites[i])
                if metabolites[i] in low_abundance_metabolites:
                    tmp_train_data.append(1)
                elif metabolites[i] in mid_abundance_metabolites:
                    tmp_train_data.append(2)
                elif metabolites[i] in high_abundance_metabolites:
                    tmp_train_data.append(3)
                tmp_train_data.append(x_miss_type[i][j])
                train_data.append(tmp_train_data)

    # 将数据集拆分为训练集和验证集
    train_data = np.array(train_data)
    train_data[train_data == "MNAR"] = 0
    train_data[train_data == "MCAR"] = 1
    train_data = train_data.astype('float')

    X_train, X_test, y_train, y_test = train_test_split(train_data[:, :9], train_data[:, 9], test_size=0.2)

    # 构建DMatrix数据格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 设置XGBoost分类器的参数
    params = {
        'objective': 'binary:logistic',  # 二分类问题
        'eval_metric': 'logloss',  # 使用对数损失函数
        'eta': 0.1,  # 学习率
        'max_depth': 3,  # 决策树的最大深度
        'min_child_weight': 1,  # 叶子节点最小权重
        'gamma': 0,  # 控制节点分裂的最小损失减少量
        'subsample': 1,  # 每个决策树使用的样本比例
        'colsample_bytree': 1,  # 每个决策树使用的特征比例
        'seed': 42  # 随机种子
    }

    # 训练XGBoost分类器
    num_rounds = 100  # 迭代次数
    model = xgb.train(params, dtrain, num_rounds)

    y_pred = model.predict(dtest)
    y_pred = [round(value) for value in y_pred]  # 将预测概率转换为类别标签

    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy


def self_xgb_prediction_type(metabolites, data, classifier, threshold_i, threshold_ii):
    """
    :param metabolites: 代谢物名称
    :param data: 缺失数据
    :param classifier: 分类器
    :param threshold_i: 中低浓度阈值
    :param threshold_ii: 高中浓度阈值
    :return: 原始数据准确率,原始数据预测类型
    """
    # 按照平均丰度排序
    mean_concentrations = np.nanmean(data, axis=1)
    metabolites = metabolites.tolist()
    sorted_metabolites = [x for _, x in sorted(zip(mean_concentrations, metabolites))]

    # 根据阈值划分代谢物变量组
    low_abundance_metabolites = sorted_metabolites[:threshold_i]
    mid_abundance_metabolites = sorted_metabolites[threshold_i:threshold_ii]
    high_abundance_metabolites = sorted_metabolites[threshold_ii:]

    x_test = []
    for i in range(data.shape[0]):
        tmp_mis_prop = np.count_nonzero(np.isnan(data[i])) / len(data[i])
        for j in range(data.shape[1]):
            if not np.isnan(data[i][j]):
                continue
            tmp_test_data = []
            nan_cnt = 0
            tmp_j = j
            while tmp_j >= 0 and np.isnan(data[i][tmp_j]):
                tmp_j -= 1
                nan_cnt += 1
            tmp_j = j + 1
            while tmp_j < data.shape[1] and np.isnan(data[i][tmp_j]):
                tmp_j += 1
                nan_cnt += 1
            metabolites_nan_cnt = 0
            tmp_i = i
            while tmp_i >= 0 and np.isnan(data[tmp_i][j]):
                tmp_i -= 1
                metabolites_nan_cnt += 1
            tmp_i = i + 1
            while tmp_i < data.shape[0] and np.isnan(data[tmp_i][j]):
                tmp_i += 1
                metabolites_nan_cnt += 1
            tmp_test_data.append(nan_cnt)
            tmp_test_data.append(metabolites_nan_cnt)
            tmp_test_data.append(np.nanmax(data[i]))
            tmp_test_data.append(np.nanmin(data[i]))
            tmp_test_data.append((np.nanmedian(data[i])))
            tmp_test_data.append((np.nanmean(data[i])))
            tmp_test_data.append(tmp_mis_prop)
            tmp_test_data.append(metabolites[i])
            if metabolites[i] in low_abundance_metabolites:
                tmp_test_data.append(1)
            elif metabolites[i] in mid_abundance_metabolites:
                tmp_test_data.append(2)
            elif metabolites[i] in high_abundance_metabolites:
                tmp_test_data.append(3)
            x_test.append(tmp_test_data)

    x_test = np.array(x_test, dtype='float')
    dtest = xgb.DMatrix(x_test)
    y_pred = classifier.predict(dtest)
    y_pred = [round(value) for value in y_pred]  # 将预测概率转换为类别标签
    y_pred = np.array(np.array(y_pred), dtype='str')
    y_pred[y_pred == '0'] = "MNAR"
    y_pred[y_pred == '1'] = "MCAR"

    target_data_type = deepcopy(data)
    target_data_type = target_data_type.astype("str")
    target_cnt = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i][j]):
                target_data_type[i][j] = y_pred[target_cnt]
                target_cnt += 1
            else:
                target_data_type[i][j] = "O"
    return target_data_type
