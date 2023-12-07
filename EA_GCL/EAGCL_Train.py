# -*- coding: utf-8 -*-
# @Author : Houping Yue
# @Time : 2022/11/26 18:49
# @Email : houping.yue@yahoo.com

from EAGCL_Evaluate import *
from EAGCL_Module import *

args = Settings()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # log
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num  # 0 为cuda


def train_module(sess, GCN_net, batches_train):
    # 首先定义接受变量的缓冲区
    uid_all, seq_A_all, seq_B_all, len_A_all, len_B_all, target_A_all, target_B_all, train_batch_num = (
        batches_train[0], batches_train[1], batches_train[2], batches_train[3],
        batches_train[4], batches_train[5], batches_train[6], batches_train[7])

    shuffled_batch_indexes = np.random.permutation(train_batch_num)
    avg_loss_joint = 0
    avg_loss_ssl = 0

    for batch_index in shuffled_batch_indexes:
        uid = uid_all[batch_index]
        seq_A = seq_A_all[batch_index]
        seq_B = seq_B_all[batch_index]
        len_A = len_A_all[batch_index]
        len_B = len_B_all[batch_index]
        target_A = target_A_all[batch_index]
        target_B = target_B_all[batch_index]

        train_loss_joint, ssl_loss, _ = GCN_net.train_gcn(sess=sess, uid=uid, seq_A=seq_A, seq_B=seq_B, len_A=len_A,
                                                          len_B=len_B,
                                                          target_A=target_A, target_B=target_B,
                                                          learning_rate=args.learning_rate,
                                                          dropout_rate=args.dropout_rate, keep_prob=args.keep_prob)
        avg_loss_joint += train_loss_joint
        avg_loss_ssl += ssl_loss

    rec_loss = avg_loss_joint / train_batch_num
    rec_loss_ssl = avg_loss_ssl / train_batch_num
    return rec_loss, rec_loss_ssl


def evaluate_module(sess, GCN_net, test_batches, test_len):
    uid_all, seq_A_all, seq_B_all, len_A_all, len_B_all, target_A_all, target_B_all, test_batch_num \
        = (test_batches[0], test_batches[1], test_batches[2], test_batches[3], test_batches[4],
           test_batches[5], test_batches[6], test_batches[7])

    return evaluate_ratings(sess=sess, GCN_net=GCN_net, uid=uid_all, seq_A=seq_A_all, seq_B=seq_B_all,
                            len_A=len_A_all, len_B=len_B_all,
                            target_A=target_A_all, target_B=target_B_all,
                            test_batch_num=test_batch_num, test_length=test_len)
