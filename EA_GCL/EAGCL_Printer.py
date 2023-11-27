# -*- coding: utf-8 -*-
# @Author : Houping Yue
# @Time : 2022/11/26 18:49
# @Email : houping.yue@yahoo.com

import logging


def print_rec_message(epoch, rec_loss_joint, ssl_loss, rec_pre_time):
    print('Epoch {} - Training Loss: {:.5f}, SSL Loss:{:.5f} - Training time: {:.3}'.format(epoch, rec_loss_joint,
                                                                                            ssl_loss,
                                                                                            rec_pre_time))
    logging.info(
        'Epoch {} - Training Loss: {:.5f}, SSL Loss:{:.5f} - Training time: {:.3}'.format(epoch, rec_loss_joint,
                                                                                          ssl_loss,
                                                                                          rec_pre_time))


def print_recommender_train(epoch, RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A, RC_5_B, RC_10_B,
                            RC_20_B, MRR_5_B, MRR_10_B, MRR_20_B, NDCG_10_A, NDCG_10_B, rec_test_time):
    print(
        "Evaluate on Domain-A, Epoch %d : RC10 = %.4f, MRR10 = %.4f, NDCG10 = %.4f" % (
            epoch, RC_10_A, MRR_10_A, NDCG_10_A))
    print(
        "Evaluate on Domain-B, Epoch %d : RC10 = %.4f, MRR10 = %.4f, NDCG10 = %.4f" % (
            epoch, RC_10_B, MRR_10_B, NDCG_10_B))


    print("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, rec_test_time))
    logging.info("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, rec_test_time))
