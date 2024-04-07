import random
import numpy as np
import pandas as pd



# 获取csv数据
def get_csv_data(data_url):
    data = pd.read_csv(data_url, header=None)
    class_label = data.iloc[0:2]
    metabolites = data.iloc[2:, 0]
    rest_of_data = data.iloc[2:, 1:]
    return class_label, metabolites, rest_of_data


# 获取excel数据
def get_excel_data(data_url):
    data = pd.read_excel(data_url, header=None)
    class_label = data.iloc[0:2]
    metabolites = data.iloc[2:, 0]
    rest_of_data = data.iloc[2:, 1:]
    return class_label, metabolites, rest_of_data


def nan_permutation(x):
    new_data = []
    max_num = 0
    for item in x:
        nan_data = []
        tmp_data = []
        nan_count = 0
        for val in item:
            if np.isnan(val):
                nan_data.append(val)
                nan_count += 1
            else:
                tmp_data.append(val)
        new_data.append(tmp_data + nan_data)
        max_num = max(max_num, nan_count)
    return new_data, len(x[0]) - max_num


# 获取完整数据子集
def get_complete_subset_data(data):
    data, max_num = nan_permutation(data)
    data = pd.DataFrame(data).iloc[:, :max_num]
    return data


def MM_Generate(metabolites, data, mis_prop, threshold_i, threshold_ii, threshold_iii, low_missing_rate):
    total_num = data.size
    mnar_percentage_low = low_missing_rate * mis_prop
    mnar_percentage_mid = 0.5 * low_missing_rate * mis_prop
    mcar_percentage_all = (1 - 1.5 * low_missing_rate) * mis_prop

    # 缺失数量
    low_richness_loss = round(mnar_percentage_low * total_num)
    mid_richness_loss = round(mnar_percentage_mid * total_num)
    rest_richness_loss = round(mcar_percentage_all * total_num)

    # 按照平均丰度排序
    data = data.values.astype(float)
    mean_concentrations = np.mean(data, axis=1)
    metabolites = metabolites.tolist()
    sorted_metabolites = [x for _, x in sorted(zip(mean_concentrations, metabolites))]

    # 根据阈值划分代谢物变量组
    index_i = round(len(sorted_metabolites) * threshold_iii)
    index_j = round(len(sorted_metabolites) * (threshold_ii + threshold_iii))

    low_abundance_metabolites = sorted_metabolites[:index_i]
    mid_abundance_metabolites = sorted_metabolites[index_i:index_j]

    # 记录原始缺失类型
    data_target = np.full(data.shape, "O", dtype='object')

    # 针对低丰度的代谢物变量生成MNAR缺失值 80%
    mnar_low_richness_loss = round(low_richness_loss * 0.8)
    low_abundance_metabolites_index = []
    tmp_num = mnar_low_richness_loss % len(mid_abundance_metabolites)
    for metabolite in low_abundance_metabolites:
        num_missing = round(mnar_low_richness_loss / len(low_abundance_metabolites))
        if tmp_num > 0:
            num_missing += 1
            tmp_num -= 1
        indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing-1)[:num_missing]
        kk = metabolites.index(metabolite)
        data[kk][indices_missing] = np.nan
        data_target[kk][indices_missing] = "MNAR"
        tmp = np.setdiff1d(np.where(data[kk])[0], indices_missing)
        low_abundance_metabolites_index.append(tmp + kk * data[kk].shape[0])


    # 针对低丰度的代谢物变量生成MCAR缺失值 20%
    low_abundance_metabolites_index = [item for sublist in low_abundance_metabolites_index for item in sublist]
    mcar_low_richness_loss = low_richness_loss - mnar_low_richness_loss
    if len(low_abundance_metabolites_index) <= mcar_low_richness_loss:
        return [], []
    indices_missing = random.sample(low_abundance_metabolites_index, mcar_low_richness_loss)
    data.flat[indices_missing] = np.nan
    data_target.flat[indices_missing] = "MCAR"

    # 针对中等丰度的代谢物变量生成MNAR缺失值 80%
    mnar_mid_richness_loss = round(mid_richness_loss * 0.8)
    mid_abundance_metabolites_index = []
    tmp_num = mnar_mid_richness_loss % len(mid_abundance_metabolites)
    for metabolite in mid_abundance_metabolites:
        num_missing = round(mnar_mid_richness_loss / len(mid_abundance_metabolites))
        if tmp_num > 0:
            num_missing += 1
            tmp_num -= 1
        indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing-1)[:num_missing]
        kk = metabolites.index(metabolite)
        data[kk][indices_missing] = np.nan
        data_target[kk][indices_missing] = "MNAR"
        tmp = np.setdiff1d(np.where(data[kk])[0], indices_missing)
        mid_abundance_metabolites_index.append(tmp + kk * data[kk].shape[0])

    # 针对中等丰度的代谢物变量生成MNAR缺失值 20%
    mid_abundance_metabolites_index = [item for sublist in mid_abundance_metabolites_index for item in sublist]
    mcar_mid_richness_loss = mid_richness_loss - mnar_mid_richness_loss
    if len(mid_abundance_metabolites_index) <= mcar_mid_richness_loss :
        return [], []
    indices_missing = random.sample(mid_abundance_metabolites_index, mcar_mid_richness_loss)
    data.flat[indices_missing] = np.nan
    data_target.flat[indices_missing] = "MCAR"

    # 针对剩余的代谢物变量生成随机缺失值
    indices_missing = np.random.choice(range(data.size), size=rest_richness_loss, replace=False)
    data.flat[indices_missing] = np.nan
    data_target.flat[indices_missing] = "MCAR"

    return data, data_target


def merge_heading(class_label, metabolites, data):
    tmp_data = np.insert(data, 0, metabolites.tolist(), axis=1)
    df = pd.DataFrame(tmp_data)
    df = pd.concat([class_label, df], axis=0).reset_index(drop=True)
    return df

