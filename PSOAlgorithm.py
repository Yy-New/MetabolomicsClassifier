import random
import numpy as np


def MM_generate_miss_prpo(metabolites, data, mis_prop, threshold_i, threshold_ii, threshold_iii, low_missing_rate):
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

    # 针对低丰度的代谢物变量生成MNAR缺失值 80%
    mnar_low_richness_loss = round(low_richness_loss * 0.8)
    low_abundance_metabolites_index = []
    tmp_num = mnar_low_richness_loss % len(low_abundance_metabolites)
    for metabolite in low_abundance_metabolites:
        num_missing = round(mnar_low_richness_loss / len(low_abundance_metabolites))
        if tmp_num > 0:
            num_missing += 1
            tmp_num -= 1
        if len(data[metabolites.index(metabolite)]) < num_missing:
            return []
        indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing-1)[:num_missing]
        kk = metabolites.index(metabolite)
        data[kk][indices_missing] = np.nan
        tmp = np.setdiff1d(np.where(data[kk])[0], indices_missing)
        low_abundance_metabolites_index.append(tmp + kk * data[kk].shape[0])

    # 针对低丰度的代谢物变量生成MCAR缺失值 20%
    low_abundance_metabolites_index = [item for sublist in low_abundance_metabolites_index for item in sublist]
    mcar_low_richness_loss = low_richness_loss - mnar_low_richness_loss
    if len(low_abundance_metabolites_index) <= mcar_low_richness_loss:
        return []
    indices_missing = random.sample(low_abundance_metabolites_index, mcar_low_richness_loss)
    data.flat[indices_missing] = np.nan

    # 针对中等丰度的代谢物变量生成MNAR缺失值 80%
    mnar_mid_richness_loss = round(mid_richness_loss * 0.8)
    mid_abundance_metabolites_index = []
    tmp_num = mnar_mid_richness_loss % len(mid_abundance_metabolites)
    for metabolite in mid_abundance_metabolites:
        num_missing = round(mnar_mid_richness_loss / len(mid_abundance_metabolites))
        if tmp_num > 0:
            num_missing += 1
            tmp_num -= 1
        if len(data[metabolites.index(metabolite)]) < num_missing:
            return []
        indices_missing = np.argpartition(data[metabolites.index(metabolite)], num_missing-1)[:num_missing]
        kk = metabolites.index(metabolite)
        data[kk][indices_missing] = np.nan
        tmp = np.setdiff1d(np.where(data[kk])[0], indices_missing)
        mid_abundance_metabolites_index.append(tmp + kk * data[kk].shape[0])

    # 针对中等丰度的代谢物变量生成MNAR缺失值 20%
    mid_abundance_metabolites_index = [item for sublist in mid_abundance_metabolites_index for item in sublist]
    mcar_mid_richness_loss = mid_richness_loss - mnar_mid_richness_loss
    if len(mid_abundance_metabolites_index) <= mcar_mid_richness_loss:
        return []
    indices_missing = random.sample(mid_abundance_metabolites_index, mcar_mid_richness_loss)
    data.flat[indices_missing] = np.nan

    # 针对剩余的代谢物变量生成随机缺失值
    if rest_richness_loss <= 0:
        return []
    indices_missing = np.random.choice(range(data.size), size=rest_richness_loss, replace=False)
    data.flat[indices_missing] = np.nan
    data_miss_prpo = []
    for item in data:
        data_miss_prpo.append(np.count_nonzero(np.isnan(item)) / len(item))
    return data_miss_prpo



def evaluate(individual, metabolites, data, mis_prop, orl_miss_prpo, item_k):
    x, y, z, tmp_mis = individual
    constraint = x + y + z - 1
    if constraint != 0:
        return np.inf
    if z <= mis_prop * tmp_mis or y <= mis_prop * tmp_mis * 0.5 or x <= mis_prop * (1 - 1.5 * tmp_mis):
        return np.inf
    euclidean_distance = 0
    for i in range(item_k):
        tmp_miss_prpo = MM_generate_miss_prpo(metabolites, data, mis_prop, x, y, z, tmp_mis)
        if len(tmp_miss_prpo) != len(orl_miss_prpo):
            return np.inf
        euclidean_distance += np.linalg.norm(np.array(orl_miss_prpo) - np.array(tmp_miss_prpo))
    return euclidean_distance / item_k


def initialize_particles(n_particles, n_dimensions, bounds):
    particles = []
    for _ in range(n_particles):
        valid_particle = False
        while not valid_particle:
            particle = np.random.uniform(bounds[:, 0], bounds[:, 1], (1, n_dimensions))[0]
            sum_x = particle[0] + particle[1]
            if 1 - sum_x > 0:
                valid_particle = True
                particle[2] = 1 - sum_x
        particles.append(particle)
    return np.array(particles)



def pso_search(objective_func, n_particles, n_dimensions, max_iterations, bounds,
               metabolites, data, mis_prop, orl_miss_prpo, item_k):
    # 初始化粒子位置和速度
    particles = initialize_particles(n_particles, n_dimensions, bounds)
    velocities = np.zeros((n_particles, n_dimensions))

    # 初始化个体最优位置和适应度
    best_positions = particles.copy()
    best_fitnesses = np.array([objective_func(individual, metabolites, data, mis_prop, orl_miss_prpo, item_k)
                               for individual in particles])

    # 寻找全局最优位置和适应度
    global_best_index = np.argmin(best_fitnesses)
    global_best_position = best_positions[global_best_index].copy()
    global_best_fitness = best_fitnesses[global_best_index]

    # 迭代更新
    for iteration in range(max_iterations):
        for i in range(n_particles):
            # 更新粒子速度和位置
            velocities[i] = velocities[i] + random.random() * (best_positions[i] - particles[i]) + random.random() * \
                            (global_best_position - particles[i])
            particles[i] = particles[i] + velocities[i]

            # 对超出边界的粒子位置进行修正
            particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])

            # 计算适应度
            fitness = objective_func(particles[i], metabolites, data, mis_prop, orl_miss_prpo, item_k)

            # 更新个体最优位置和适应度
            if fitness < best_fitnesses[i]:
                best_positions[i] = particles[i].copy()
                best_fitnesses[i] = fitness

                # 更新全局最优位置和适应度
                if fitness < global_best_fitness:
                    global_best_position = particles[i].copy()
                    global_best_fitness = fitness

    return global_best_position, global_best_fitness


def pso_xyz_gradient(metabolites, data, orl_data, mis_prop, item_k):
    orl_miss_prpo = []
    for item in orl_data:
        orl_miss_prpo.append(np.count_nonzero(np.isnan(item)) / len(item))

    # 设置搜索参数
    n_particles = 100
    n_dimensions = 4
    max_iterations = 300
    bounds = [[0, 1]] * (n_dimensions - 1)
    bounds.append([0.3, 0.8])
    bounds = np.array(bounds)
    # 运行PSO算法
    best_position, best_fitness = pso_search(evaluate, n_particles, n_dimensions, max_iterations, bounds,
                                             metabolites, data, mis_prop, orl_miss_prpo, item_k)

    # 获取最优解
    x, y, z, tmp_mis = best_position
    return x, y, z, tmp_mis
