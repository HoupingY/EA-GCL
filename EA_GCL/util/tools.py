# -*- coding: UTF-8 -*-
# Author: JinyuZ1996
# Created at: 2022/7/31 11:14

# coding: utf-8

import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

random.seed(2022)
np.random.seed(2022)


def duplicate_matrix(matrix_in):
    frame_temp = pd.DataFrame(matrix_in, columns=['row', 'column'])
    frame_temp.duplicated()
    frame_temp.drop_duplicates(inplace=True)
    return frame_temp.values.tolist()


def matrix2inverse(array_in, row_pre, col_pre, len_all):
    matrix_rows = array_in[:, 0] + row_pre
    matrix_columns = array_in[:, 1] + col_pre
    matrix_value = [1.] * len(matrix_rows)
    inverse_matrix = sp.coo_matrix((matrix_value, (matrix_rows, matrix_columns)),
                                   shape=(len_all, len_all))
    return inverse_matrix


def reorder_list(org_list, order):
    new_list = np.array(org_list)
    new_list = new_list[order]
    return new_list


def convert_sp_mat_to_sp_tensor(matrices_in):
    coo = matrices_in.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

# def l2_loss(*params):
#     return tf.add_n([tf.nn.l2_loss(w) for w in params])
