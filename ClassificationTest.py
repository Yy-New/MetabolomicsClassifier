import os

import numpy as np

from ClassificationAlgorithm import self_xgb_construction_classifier, self_xgb_prediction_type
from PSOAlgorithm import pso_xyz_gradient
from Util import get_csv_data, get_complete_subset_data, MM_Generate, merge_heading

file_name = "ST000118_20.0%.csv"
class_label, metabolites, missing_data = get_csv_data(f"./Data/{file_name}")
missing_data = missing_data.astype("float").values
prop = np.sum(np.isnan(missing_data)) / missing_data.size
x_complete = get_complete_subset_data(missing_data)
# 计算粒子群算法搜索出的结果 循环10次求平均值
gradient_x, gradient_y, gradient_z, low_mis = pso_xyz_gradient(metabolites, x_complete, missing_data, prop, 10)

index_i = round(len(x_complete) * gradient_z)
index_j = round(len(x_complete) * (gradient_y + gradient_z))
x_imposeds = []
x_miss_types = []
for i in range(5):
    x_imposed, x_miss_type = MM_Generate(metabolites, x_complete, prop, gradient_x, gradient_y, gradient_z, low_mis)
    x_imposeds.append(x_imposed)
    x_miss_types.append(x_miss_type)

self_classifier, self_train_accuracy = self_xgb_construction_classifier(metabolites, x_imposeds, x_miss_types, index_i, index_j)

print(self_train_accuracy)

save_result_path = "./Result/"
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)

self_target_type = self_xgb_prediction_type(metabolites, missing_data, self_classifier, index_i, index_j)
self_target_type = merge_heading(class_label, metabolites, self_target_type)
self_target_type.to_csv(save_result_path + file_name, header=None, index=False)
