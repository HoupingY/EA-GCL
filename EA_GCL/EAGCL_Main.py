# -*- coding: utf-8 -*-
# @Author : Houping Yue
# @Time : 2022/11/26 18:48
# @Email : houping.yue@yahoo.com

from time import time

from EAGCL_Config import *
from EAGCL_Printer import *
from EAGCL_Train import *

np.seterr(all='ignore')
args = Settings()

if __name__ == '__main__':

    print("Loading dictionary for data generation...")
    dict_A = build_dict(dict_path=args.path_dict_A)
    dict_B = build_dict(dict_path=args.path_dict_B)
    dict_U = build_dict(dict_path=args.path_dict_U)
    num_items_A = len(dict_A)
    num_items_B = len(dict_B)
    num_users = len(dict_U)

    print("Loading the mixed data from datasets...")
    mixed_seq_train = build_mixed_sequences(path=args.path_train, dict_A=dict_A, dict_B=dict_B, dict_U=dict_U)
    mixed_seq_test = build_mixed_sequences(path=args.path_test, dict_A=dict_A, dict_B=dict_B, dict_U=dict_U)

    if args.fast_running is True:

        print("Warning: It's under a fast running setting.")
        mixed_seq_train = mixed_seq_train[:int(len(mixed_seq_train) * args.fast_ratio)]

    print("Transforming the data...")
    train_data = data_generation(mixed_sequence=mixed_seq_train, dict_A=dict_A)
    test_data = data_generation(mixed_sequence=mixed_seq_test, dict_A=dict_A)

    print("Building the matrix-form CDS graphs...")
    original_CDS_graphs, aug_graphs_1, aug_graphs_2 = build_matrix_form(train_data)

    print("Building the laplace list of the original CDS graphs...")
    laplace_origin = build_laplace(original_CDS_graphs, dict_A, dict_B, dict_U)
    if args.ssl_task:
        print("Building the laplace list of the perturbed CDS graphs...")
        laplace_aug_1 = build_laplace(aug_graphs_1, dict_A, dict_B, dict_U)
        laplace_aug_2 = build_laplace(aug_graphs_2, dict_A, dict_B, dict_U)
    else:
        laplace_aug_1 = laplace_origin
        laplace_aug_2 = laplace_origin

    print("Transforming the input_graphs...")
    ori_cds_graphs = sum(laplace_origin)
    if args.ssl_task:
        aug_views_1 = sum(laplace_aug_1)
        aug_views_2 = sum(laplace_aug_2)
    else:
        aug_views_1 = ori_cds_graphs
        aug_views_2 = ori_cds_graphs

    print("Generating the batches for training and testing...")
    train_batches = get_batches(input_data=train_data, batch_size=args.batch_size, padding_num_A=args.padding_int,
                                padding_num_B=args.padding_int, isTrain=True)
    test_batches = get_batches(input_data=test_data, batch_size=args.batch_size, padding_num_A=args.padding_int,
                               padding_num_B=args.padding_int, isTrain=False)
    print("Initializing the GCL-based Cross-domain Recommender.")
    recommender = GCL_Module(num_items_A=num_items_A, num_items_B=num_items_B, num_users=num_users,
                             origin_CDS_graphs=ori_cds_graphs,
                             aug_CDS_graphs_1=aug_views_1, aug_CDS_graphs_2=aug_views_2)

    print("Start training:")
    with tf.Session(graph=recommender.graph, config=recommender.config) as sess:

        recommender.sess = sess
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        best_score_domain_A = -1
        best_score_domain_B = -1

        for epoch in range(args.epochs):

            rec_pre_begin_time = time()
            rec_loss, ssl_loss = train_module(sess=sess, GCN_net=recommender, batches_train=train_batches)
            rec_pre_time = time() - rec_pre_begin_time

            epoch_to_print = epoch + 1
            print_rec_message(epoch=epoch_to_print, rec_loss_joint=rec_loss, ssl_loss=ssl_loss,
                              rec_pre_time=rec_pre_time)

            if epoch_to_print % args.verbose == 0:
                rec_test_begin_time = time()
                [RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A,
                 RC_5_B, RC_10_B, RC_20_B, MRR_5_B, MRR_10_B, MRR_20_B,
                 NDCG_5_A, NDCG_10_A, NDCG_20_A, NDCG_5_B, NDCG_10_B, NDCG_20_B] = \
                    evaluate_module(sess=sess, GCN_net=recommender, test_batches=test_batches, test_len=len(test_data))
                rec_test_time = time() - rec_test_begin_time

                print_recommender_train(epoch_to_print, RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A, RC_5_B,
                                        RC_10_B, RC_20_B, MRR_5_B, MRR_10_B, MRR_20_B, NDCG_10_A, NDCG_10_B,
                                        rec_test_time)

                if RC_5_A >= best_score_domain_A or RC_5_B >= best_score_domain_B:
                    best_score_domain_A = RC_5_A
                    best_score_domain_B = RC_5_B
                    saver.save(sess, args.checkpoint, global_step=epoch_to_print, write_meta_graph=False)
                    print("Recommender performs better, saving current model....")

            train_batches = get_batches(input_data=train_data, batch_size=args.batch_size,
                                        padding_num_A=args.padding_int,
                                        padding_num_B=args.padding_int, isTrain=True)

        print("End.")

