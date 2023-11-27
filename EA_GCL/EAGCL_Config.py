# -*- coding: utf-8 -*-
# @Author : Houping Yue
# @Time : 2022/11/26 18:47
# @Email : houping.yue@yahoo.com

import copy
import math
import os

from EAGCL_Settings import *
from util.tools import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 2022

args = Settings()


def build_dict(dict_path):

    element_dict = {}
    with open(dict_path, 'r') as file_object:
        elements = file_object.readlines()
    for e in elements:
        e = e.strip().split('\t')
        element_dict[e[1]] = int(e[0])
    return element_dict


def build_mixed_sequences(path, dict_A, dict_B, dict_U):

    with open(path, 'r') as file_object:
        mixed_sequences = []
        lines = file_object.readlines()
        for e in lines:
            e = e.strip().split('\t')
            temp_sequence = []
            user_id = e[0]
            temp_sequence.append(dict_U[user_id])
            for item in e[1:]:
                if item in dict_A:
                    temp_sequence.append(dict_A[item])
                else:
                    temp_sequence.append(dict_B[item] + len(dict_A))
            mixed_sequences.append(temp_sequence)
    return mixed_sequences


def data_generation(mixed_sequence, dict_A):
    data_outputs = []
    for sequence_index in mixed_sequence:
        temp = []
        seq_A, seq_B = [], []
        len_A, len_B = 0, 0
        uid = sequence_index[0]
        seq_A.append(uid)
        seq_B.append(uid)
        for item_id in sequence_index[1:]:
            if item_id < len(dict_A):
                seq_A.append(item_id)
                len_A += 1
            else:
                seq_B.append(item_id - len(dict_A))
                len_B += 1
        target_A = seq_A[-1]
        target_B = seq_B[-1]
        seq_A.pop()
        seq_B.pop()
        temp.append(seq_A)
        temp.append(seq_B)
        temp.append(len_A - 1)
        temp.append(len_B - 1)
        temp.append(target_A)
        temp.append(target_B)
        data_outputs.append(temp)

    return data_outputs


def build_matrix_form(train_data):
    matrix_UA, matrix_AU, matrix_UB, matrix_BU, matrix_A_neighbors, matrix_B_neighbors, matrix_U \
        = [], [], [], [], [], [], []
    matrix_sub_UA_1, matrix_sub_AU_1, matrix_sub_A_neighbors_1, \
    matrix_sub_UA_2, matrix_sub_AU_2, matrix_sub_A_neighbors_2 = [], [], [], [], [], []
    sum_matrices = []
    sub_matrix_1, sub_matrix_2 = [], []

    for data_unit in train_data:
        seq_A = data_unit[0]
        seq_B = data_unit[1]
        uid = int(seq_A[0])
        items_A = [int(i) for i in seq_A[1:]]
        items_B = [int(j) for j in seq_B[1:]]

        items_A_sub1 = copy.deepcopy(items_A)
        items_A_sub2 = copy.deepcopy(items_A)

        perturbed_time = math.ceil(args.alpha * len(items_A))
        random.seed(seed)
        for i in range(0, perturbed_time):
            index = random.randint(0, len(items_A_sub1))
            items_A_sub1[index - 1] = 0
        random.seed(seed + 1)
        for i in range(0, perturbed_time):
            index = random.randint(0, len(items_A_sub2))
            items_A_sub2[index - 1] = 0

        for item_A in items_A:
            matrix_UA.append([uid, item_A])
            matrix_AU.append([item_A, uid])

        for sub1_A in items_A_sub1:
            matrix_sub_UA_1.append([uid, sub1_A])
            matrix_sub_AU_1.append([sub1_A, uid])
        for sub2_A in items_A_sub2:
            matrix_sub_UA_2.append([uid, sub2_A])
            matrix_sub_AU_2.append([sub2_A, uid])

        for item_B in items_B:
            matrix_UB.append([uid, item_B])
            matrix_BU.append([item_B, uid])

        for item_index_A in range(0, len(items_A) - 1):
            item_temp_A = items_A[item_index_A]
            next_item_A = items_A[item_index_A + 1]
            matrix_A_neighbors.append([item_temp_A, item_temp_A])
            matrix_A_neighbors.append([item_temp_A, next_item_A])
        for sub_index_A1 in range(0, len(items_A_sub1) - 1):
            sub_temp_A = items_A_sub1[sub_index_A1]
            next_sub_A = items_A_sub1[sub_index_A1 + 1]
            matrix_sub_A_neighbors_1.append([sub_temp_A, sub_temp_A])
            matrix_sub_A_neighbors_1.append([sub_temp_A, next_sub_A])
        for sub_index_A2 in range(0, len(items_A_sub2) - 1):
            sub_temp_A = items_A_sub2[sub_index_A2]
            next_sub_A = items_A_sub2[sub_index_A2 + 1]
            matrix_sub_A_neighbors_2.append([sub_temp_A, sub_temp_A])
            matrix_sub_A_neighbors_2.append([sub_temp_A, next_sub_A])

        for item_index_B in range(0, len(items_B) - 1):
            item_temp_B = items_B[item_index_B]
            next_item_B = items_B[item_index_B + 1]
            matrix_B_neighbors.append([item_temp_B, item_temp_B])
            matrix_B_neighbors.append([item_temp_B, next_item_B])

        matrix_U.append([uid, uid])

    matrix_U_A = duplicate_matrix(matrix_UA)
    matrix_U_B = duplicate_matrix(matrix_UB)
    matrix_A_U = duplicate_matrix(matrix_AU)
    matrix_B_U = duplicate_matrix(matrix_BU)
    matrix_A_neighbor = duplicate_matrix(matrix_A_neighbors)
    matrix_B_neighbor = duplicate_matrix(matrix_B_neighbors)
    matrix_U_U = duplicate_matrix(matrix_U)

    matrix_sub_U_A_1 = duplicate_matrix(matrix_sub_UA_1)
    matrix_sub_U_A_2 = duplicate_matrix(matrix_sub_UA_2)
    matrix_sub_A_U_1 = duplicate_matrix(matrix_sub_AU_1)
    matrix_sub_A_U_2 = duplicate_matrix(matrix_sub_AU_2)
    matrix_sub_A_neighbor_1 = duplicate_matrix(matrix_sub_A_neighbors_1)
    matrix_sub_A_neighbor_2 = duplicate_matrix(matrix_sub_A_neighbors_2)

    sum_matrices.append(np.array(matrix_U_A))
    sum_matrices.append(np.array(matrix_U_B))
    sum_matrices.append(np.array(matrix_A_U))
    sum_matrices.append(np.array(matrix_B_U))
    sum_matrices.append(np.array(matrix_A_neighbor))
    sum_matrices.append(np.array(matrix_B_neighbor))
    sum_matrices.append(np.array(matrix_U_U))

    sub_matrix_1.append(np.array(matrix_sub_U_A_1))
    sub_matrix_1.append(np.array(matrix_U_B))
    sub_matrix_1.append(np.array(matrix_sub_A_U_1))
    sub_matrix_1.append(np.array(matrix_B_U))
    sub_matrix_1.append(np.array(matrix_sub_A_neighbor_1))
    sub_matrix_1.append(np.array(matrix_B_neighbor))
    sub_matrix_1.append(np.array(matrix_U_U))

    sub_matrix_2.append(np.array(matrix_sub_U_A_2))
    sub_matrix_2.append(np.array(matrix_U_B))
    sub_matrix_2.append(np.array(matrix_sub_A_U_2))
    sub_matrix_2.append(np.array(matrix_B_U))
    sub_matrix_2.append(np.array(matrix_sub_A_neighbor_2))
    sub_matrix_2.append(np.array(matrix_B_neighbor))
    sub_matrix_2.append(np.array(matrix_U_U))

    return sum_matrices, sub_matrix_1, sub_matrix_2


def build_laplace(sum_matrices, dict_A, dict_B, dict_U):

    laplace_matrices = []

    dimension_A = len(dict_A)
    dimension_B = len(dict_B)
    dimension_U = len(dict_U)
    dimension_sum = dimension_A + dimension_U + dimension_B

    inverse_matrix_A_A = matrix2inverse(sum_matrices[4], row_pre=0, col_pre=0, len_all=dimension_sum)

    inverse_matrix_A_U = matrix2inverse(sum_matrices[2], row_pre=0, col_pre=dimension_A, len_all=dimension_sum)
    inverse_matrix_U_A = matrix2inverse(sum_matrices[0], row_pre=dimension_A, col_pre=0, len_all=dimension_sum)
    inverse_matrix_U_U = matrix2inverse(sum_matrices[6], row_pre=dimension_A, col_pre=dimension_A,
                                        len_all=dimension_sum)
    inverse_matrix_U_B = matrix2inverse(sum_matrices[1], row_pre=dimension_A, col_pre=dimension_A + dimension_U,
                                        len_all=dimension_sum)
    inverse_matrix_B_U = matrix2inverse(sum_matrices[3], row_pre=dimension_A + dimension_U, col_pre=dimension_A,
                                        len_all=dimension_sum)
    inverse_matrix_B_B = matrix2inverse(sum_matrices[5], row_pre=dimension_A + dimension_U,
                                        col_pre=dimension_A + dimension_U, len_all=dimension_sum)

    laplace_matrices.append(inverse_matrix_U_A)
    laplace_matrices.append(inverse_matrix_U_B)
    laplace_matrices.append(inverse_matrix_A_U)
    laplace_matrices.append(inverse_matrix_B_U)
    laplace_matrices.append(inverse_matrix_A_A)
    laplace_matrices.append(inverse_matrix_B_B)
    laplace_matrices.append(inverse_matrix_U_U)
    laplace_list = [adj.tocoo() for adj in laplace_matrices]

    return laplace_list


def get_batches(input_data, batch_size, padding_num_A, padding_num_B, isTrain):
    uid_all, seq_A_list, seq_B_list, len_A_list, len_B_list, target_A_list, target_B_list = [], [], [], [], [], [], []
    num_batches = int(len(input_data) / batch_size)

    if isTrain is True:
        random.shuffle(input_data)

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        batch = input_data[start_index:start_index + batch_size]
        uid, seq_A, seq_B, len_A, len_B, target_A, target_B = batch_to_input(batch=batch, padding_num_A=padding_num_A,
                                                                             padding_num_B=padding_num_B)
        uid_all.append(uid)
        seq_A_list.append(seq_A)
        seq_B_list.append(seq_B)

        len_A_list.append(len_A)
        len_B_list.append(len_B)

        target_A_list.append(target_A)
        target_B_list.append(target_B)

    return list((uid_all, seq_A_list, seq_B_list, len_A_list, len_B_list, target_A_list, target_B_list, num_batches))


def batch_to_input(batch, padding_num_A, padding_num_B):
    uid, seq_A, seq_B, len_A, len_B, target_A, target_B = [], [], [], [], [], [], []
    for data_index in batch:
        len_A.append(data_index[2])
        len_B.append(data_index[3])
    maxlen_A = max(len_A)
    maxlen_B = max(len_B)
    i = 0
    for data_index in range(len(batch)):
        uid.append(batch[data_index][0][0])
        seq_A.append(batch[data_index][0][1:] + [padding_num_A] * (maxlen_A - len_A[i]))
        seq_B.append(batch[data_index][1][1:] + [padding_num_B] * (maxlen_B - len_B[i]))
        target_A.append(batch[data_index][4])
        target_B.append(batch[data_index][5])
        i += 1

    return np.array(uid), np.array(seq_A), np.array(seq_B), np.array(len_A).reshape(len(len_A), 1), np.array(
        len_B).reshape(len(len_B), 1), np.array(target_A), np.array(target_B)
