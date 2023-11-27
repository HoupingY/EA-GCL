# -*- coding: utf-8 -*-
# @Author : Houping Yue
# @Time : 2022/11/26 18:48
# @Email : houping.yue@yahoo.com

import os

from EAGCL_Settings import *
from util.tools import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 2022
np.random.seed(seed)
tf.set_random_seed(seed)

args = Settings()


class GCL_Module:
    def __init__(self, num_items_A, num_items_B, num_users, origin_CDS_graphs, aug_CDS_graphs_1, aug_CDS_graphs_2):

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.num_items_A = num_items_A
        self.num_items_B = num_items_B
        self.num_users = num_users

        self.ori_cds_graphs = origin_CDS_graphs
        self.aug_cds_graphs_1 = aug_CDS_graphs_1
        self.aug_cds_graphs_2 = aug_CDS_graphs_2

        self.embedding_size = args.embedding_size
        self.num_folded = args.num_folded
        self.ssl_temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg
        self.ssl_mode = args.ssl_mode
        self.ssl_task = args.ssl_task
        self.beta = args.beta
        self.layer_size = args.layer_size
        self.weight_size = eval(self.layer_size)
        self.is_training = True
        self.ex_on = args.ex_on

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.uid, self.seq_A, self.seq_B, self.len_A, self.len_B, self.target_A, self.target_B, self.learning_rate, \
                self.dropout_rate, self.keep_prob = self.get_inputs()

            with tf.name_scope('encoder'):
                self.all_weights = self.weight_initializer()

                self.graph_embedding_A, self.graph_embedding_u, self.graph_embedding_B, self.sub_embedding_u1, \
                self.sub_embedding_u2, self.sub_embedding_B1, self.sub_embedding_B2 = \
                    self.node_rep_learning(self.num_items_A, self.num_users, self.num_items_B, self.ori_cds_graphs,
                                           self.aug_cds_graphs_1, self.aug_cds_graphs_2, self.embedding_size)

                self.seq_emb_B_output, self.seq_emb_A_output = \
                    self.seq_rep_learning(self.uid, self.seq_A, self.seq_B, self.dropout_rate, self.graph_embedding_A,
                                          self.graph_embedding_u, self.graph_embedding_B)
            with tf.name_scope('loss'):
                self.ssl_loss = tf.constant(0)
                self.loss_A, self.loss_B, self.pred_A, self.pred_B = self.create_cross_entropy_loss(self.target_A,
                                                                                                    self.target_B)
                if self.ssl_task is True:
                    self.ssl_loss = self.calculate_ssl_loss()
                    self.joint_loss = self.loss_A + self.loss_B + self.ssl_loss
                else:
                    self.joint_loss = self.loss_A + self.loss_B
            with tf.name_scope('optimizer'):
                self.op = self.loss_optimizer(self.joint_loss, self.learning_rate)

    def get_inputs(self):
        uid = tf.placeholder(dtype=tf.int32, shape=[None, ], name='uid')
        seq_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_A')
        seq_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_B')
        len_A = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='len_A')
        len_B = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='len_B')
        target_A = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_A')
        target_B = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_B')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        return uid, seq_A, seq_B, len_A, len_B, target_A, target_B, learning_rate, dropout_rate, keep_prob

    def weight_initializer(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.num_users, self.embedding_size]),
                                                    name='user_embedding')
        all_weights['item_embedding_A'] = tf.Variable(
            initializer([self.num_items_A, self.embedding_size]),
            name='item_embedding_A')
        all_weights['item_embedding_B'] = tf.Variable(
            initializer([self.num_items_B, self.embedding_size]),
            name='item_embedding_B')
        self.layers_plus = [self.embedding_size] + self.weight_size
        all_weights['W_gc'] = tf.get_variable('W_gc', [self.layers_plus[0], self.layers_plus[1]],
                                              tf.float32, initializer)
        all_weights['b_gc'] = tf.get_variable('b_gc', [self.layers_plus[1]], tf.float32,
                                              tf.zeros_initializer())
        all_weights['W_bi'] = tf.get_variable('W_bi', [self.layers_plus[0], self.layers_plus[1]],
                                              tf.float32, initializer)
        all_weights['b_bi'] = tf.get_variable('b_bi', [self.layers_plus[1]], tf.float32,
                                              tf.zeros_initializer())

        if self.ssl_task is True:
            all_weights['sub1_W_gc'] = tf.get_variable('sub1_W_gc', [self.layers_plus[0], self.layers_plus[1]],
                                                       tf.float32, initializer)
            all_weights['sub1_b_gc'] = tf.get_variable('sub1_b_gc', [self.layers_plus[1]], tf.float32,
                                                       tf.zeros_initializer())
            all_weights['sub1_W_bi'] = tf.get_variable('sub1_W_bi', [self.layers_plus[0], self.layers_plus[1]],
                                                       tf.float32, initializer)
            all_weights['sub1_b_bi'] = tf.get_variable('sub1_b_bi', [self.layers_plus[1]], tf.float32,
                                                       tf.zeros_initializer())
            all_weights['sub2_W_gc'] = tf.get_variable('sub2_W_gc', [self.layers_plus[0], self.layers_plus[1]],
                                                       tf.float32, initializer)
            all_weights['sub2_b_gc'] = tf.get_variable('sub2_b_gc', [self.layers_plus[1]], tf.float32,
                                                       tf.zeros_initializer())
            all_weights['sub2_W_bi'] = tf.get_variable('sub2_W_bi', [self.layers_plus[0], self.layers_plus[1]],
                                                       tf.float32, initializer)
            all_weights['sub2_b_bi'] = tf.get_variable('sub2_b_bi', [self.layers_plus[1]], tf.float32,
                                                       tf.zeros_initializer())
        if self.ex_on is True:
            all_weights['W_bal_A'] = tf.Variable(initializer([self.embedding_size, self.weight_size[0]]),
                                                 dtype=tf.float32)
            all_weights['b_bal_A'] = tf.Variable(initializer([1, self.weight_size[0]]), dtype=tf.float32)
            all_weights['h_bal_A'] = tf.Variable(tf.ones([self.weight_size[0], 1]), dtype=tf.float32)

            all_weights['W_bal_B'] = tf.Variable(initializer([self.embedding_size, self.weight_size[0]]),
                                                 dtype=tf.float32)
            all_weights['b_bal_B'] = tf.Variable(initializer([1, self.weight_size[0]]), dtype=tf.float32)
            all_weights['h_bal_B'] = tf.Variable(tf.ones([self.weight_size[0], 1]), dtype=tf.float32)

            all_weights['W_ssl_B'] = tf.Variable(initializer([self.embedding_size, self.weight_size[0]]),
                                                 dtype=tf.float32)
            all_weights['b_ssl_B'] = tf.Variable(initializer([1, self.weight_size[0]]), dtype=tf.float32)
            all_weights['h_ssl_B'] = tf.Variable(tf.ones([self.weight_size[0], 1]), dtype=tf.float32)

        return all_weights

    def unfold_cds(self, folded_graphs):
        graphs_data = []
        fold_depth = (folded_graphs.shape[0]) // self.num_folded
        for fold_index in range(self.num_folded):
            start = fold_index * fold_depth
            if fold_index == self.num_folded - 1:
                end = folded_graphs.shape[0]
            else:
                end = (fold_index + 1) * fold_depth
            graphs_data.append(convert_sp_mat_to_sp_tensor(folded_graphs[start:end]))
        return graphs_data

    def node_rep_learning(self, num_items_A, num_users, num_items_B, ori_cds_graphs, aug_sub_graphs_1, aug_sub_graphs_2,
                          embedding_size):

        ori_graph_input = self.unfold_cds(ori_cds_graphs)

        aug_graph_input_1 = self.unfold_cds(aug_sub_graphs_1)
        aug_graph_input_2 = self.unfold_cds(aug_sub_graphs_2)

        ego_embeddings = tf.concat([self.all_weights['item_embedding_A'], self.all_weights['user_embedding'],
                                    self.all_weights['item_embedding_B']], axis=0)
        all_embeddings = [ego_embeddings]

        temp_ori = []

        for index in range(args.num_folded):
            temp_ori.append(tf.sparse_tensor_dense_matmul(ori_graph_input[index], ego_embeddings))

        side_embeddings = tf.concat(temp_ori, 0)
        sum_embeddings = tf.nn.leaky_relu(
            tf.matmul(side_embeddings, self.all_weights['W_gc']) + self.all_weights['b_gc'])
        bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
        bi_embeddings = tf.nn.leaky_relu(tf.matmul(bi_embeddings, self.all_weights['W_bi']) + self.all_weights['b_bi'])
        ego_embeddings = sum_embeddings + bi_embeddings
        all_embeddings += [tf.math.l2_normalize(ego_embeddings, axis=1)]


        graph_ebd_A, graph_ebd_user, graph_ebd_B = [], [], []
        split_gebd_items_A, split_gebd_users, split_gebd_items_B = \
            tf.split(all_embeddings[0], [num_items_A, num_users, num_items_B], 0)

        split_gebd_items_A = tf.reshape(split_gebd_items_A, [num_items_A, 1, embedding_size])
        split_gebd_users = tf.reshape(split_gebd_users, [num_users, 1, embedding_size])
        split_gebd_items_B = tf.reshape(split_gebd_items_B, [num_items_B, 1, embedding_size])

        graph_ebd_A += [split_gebd_items_A]
        graph_ebd_user += [split_gebd_users]
        graph_ebd_B += [split_gebd_items_B]

        graph_ebd_A = tf.concat(graph_ebd_A, -1)
        graph_ebd_user = tf.concat(graph_ebd_user, -1)
        graph_ebd_B = tf.concat(graph_ebd_B, -1)

        graph_ebd_A = tf.reduce_mean(graph_ebd_A, axis=1)
        graph_ebd_user = tf.reduce_mean(graph_ebd_user, axis=1)
        graph_ebd_B = tf.reduce_mean(graph_ebd_B, axis=1)
        print(graph_ebd_A)
        print(graph_ebd_user)
        print(graph_ebd_B)

        if self.ssl_task is True:
            ego_embeddings_sub1 = ego_embeddings
            ego_embeddings_sub2 = ego_embeddings
            all_embeddings_sub1 = [ego_embeddings_sub1]
            all_embeddings_sub2 = [ego_embeddings_sub2]

            temp_aug1, temp_aug2 = [], []

            for index in range(args.num_folded):
                temp_aug1.append(tf.sparse_tensor_dense_matmul(aug_graph_input_1[index], ego_embeddings_sub1))
                temp_aug2.append(tf.sparse_tensor_dense_matmul(aug_graph_input_2[index], ego_embeddings_sub2))

            side_embeddings_sub1 = tf.concat(temp_aug1, 0)
            side_embeddings_sub2 = tf.concat(temp_aug2, 0)
            sum_embeddings_sub_1 = tf.nn.leaky_relu(
                tf.matmul(side_embeddings_sub1, self.all_weights['sub1_W_gc']) + self.all_weights['sub1_b_gc'])
            sum_embeddings_sub_2 = tf.nn.leaky_relu(
                tf.matmul(side_embeddings_sub2, self.all_weights['sub2_W_gc']) + self.all_weights['sub2_b_gc'])

            bi_embeddings_sub_1 = tf.multiply(ego_embeddings_sub1, side_embeddings_sub1)
            bi_embeddings_sub_2 = tf.multiply(ego_embeddings_sub2, side_embeddings_sub2)

            bi_embeddings_sub_1 = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings_sub_1, self.all_weights['sub1_W_bi']) + self.all_weights['sub1_b_bi'])
            bi_embeddings_sub_2 = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings_sub_2, self.all_weights['sub2_W_bi']) + self.all_weights['sub2_b_bi'])

            ego_embeddings_sub1 = sum_embeddings_sub_1 + bi_embeddings_sub_1
            ego_embeddings_sub2 = sum_embeddings_sub_2 + bi_embeddings_sub_2

            all_embeddings_sub1 += [tf.math.l2_normalize(ego_embeddings_sub1, axis=1)]
            all_embeddings_sub2 += [tf.math.l2_normalize(ego_embeddings_sub2, axis=1)]


            sub_usr_ebd1, sub_item_ebd1 = [], []
            _, split_gebd_users, split_gebd_items_B = \
                tf.split(all_embeddings_sub1[0], [num_items_A, num_users, num_items_B], 0)

            split_gebd_users = tf.reshape(split_gebd_users, [num_users, 1, embedding_size])
            split_gebd_items_B = tf.reshape(split_gebd_items_B, [num_items_B, 1, embedding_size])

            sub_usr_ebd1 += [split_gebd_users]
            sub_item_ebd1 += [split_gebd_items_B]

            sub_usr_ebd1 = tf.concat(sub_usr_ebd1, -1)
            sub_item_ebd1 = tf.concat(sub_item_ebd1, -1)

            sub_usr_ebd1 = tf.reduce_mean(sub_usr_ebd1, axis=1)
            sub_item_ebd1 = tf.reduce_mean(sub_item_ebd1, axis=1)

            print(sub_usr_ebd1)
            print(sub_item_ebd1)


            sub_usr_ebd2, sub_item_ebd2 = [], []
            _, split_gebd_users, split_gebd_items_B = \
                tf.split(all_embeddings_sub2[0], [num_items_A, num_users, num_items_B], 0)

            split_gebd_users = tf.reshape(split_gebd_users, [num_users, 1, embedding_size])
            split_gebd_items_B = tf.reshape(split_gebd_items_B, [num_items_B, 1, embedding_size])

            sub_usr_ebd2 += [split_gebd_users]
            sub_item_ebd2 += [split_gebd_items_B]

            sub_usr_ebd2 = tf.concat(sub_usr_ebd2, -1)
            sub_item_ebd2 = tf.concat(sub_item_ebd2, -1)

            sub_usr_ebd2 = tf.reduce_mean(sub_usr_ebd2, axis=1)
            sub_item_ebd2 = tf.reduce_mean(sub_item_ebd2, axis=1)

            print(sub_usr_ebd2)
            print(sub_item_ebd2)
        else:
            sub_item_ebd1 = graph_ebd_B
            sub_item_ebd2 = graph_ebd_B
            sub_usr_ebd1 = graph_ebd_user
            sub_usr_ebd2 = graph_ebd_user

        return graph_ebd_A, graph_ebd_user, graph_ebd_B, sub_usr_ebd1, sub_usr_ebd2, sub_item_ebd1, sub_item_ebd2

    def seq_rep_learning(self, uid, seq_A, seq_B, dropout_rate, graph_embedding_A, graph_embedding_U,
                         graph_embedding_B):
        with tf.variable_scope('sequence_encoder'):
            item_ebd_A = tf.nn.embedding_lookup(graph_embedding_A, seq_A)
            user_ebd = tf.nn.embedding_lookup(graph_embedding_U, uid)
            item_ebd_B = tf.nn.embedding_lookup(graph_embedding_B, seq_B)
            if self.ex_on:
                seq_embed_B_state = self.ex_mlp(seq_ebd_A=item_ebd_A, seq_ebd_B=item_ebd_B, isSparseOne=True)
                seq_embed_A_state = self.ex_mlp(seq_ebd_A=item_ebd_A, seq_ebd_B=item_ebd_B, isSparseOne=False)
            else:
                seq_embed_B_state = tf.reduce_max(item_ebd_B, 1)
                seq_embed_A_state = tf.reduce_max(item_ebd_A, 1)
            seq_ebd_B = tf.concat([seq_embed_B_state, user_ebd], axis=1)
            seq_ebd_A = tf.concat([seq_embed_A_state, user_ebd], axis=1)

            seq_ebd_B_output = tf.layers.dropout(seq_ebd_B, rate=dropout_rate,
                                                 training=tf.convert_to_tensor(self.is_training))
            seq_ebd_A_output = tf.layers.dropout(seq_ebd_A, rate=dropout_rate,
                                                 training=tf.convert_to_tensor(self.is_training))

            print(seq_ebd_B)
            print(seq_ebd_A)

        return seq_ebd_B_output, seq_ebd_A_output

    def create_cross_entropy_loss(self, target_A, target_B):

        with tf.variable_scope('cross_loss'):
            concat_output = tf.concat([self.seq_emb_B_output, self.seq_emb_A_output], axis=-1)
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output, self.keep_prob)
            pred_A = tf.layers.dense(concat_output, self.num_items_A, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            loss_A = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_A, logits=pred_A), name='loss_A')

            concat_output = tf.concat([self.seq_emb_A_output, self.seq_emb_B_output], axis=-1)
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output, self.keep_prob)
            pred_B = tf.layers.dense(concat_output, self.num_items_B, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            loss_B = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_B, logits=pred_B), name='loss_B')

        return loss_A, loss_B, pred_A, pred_B

    def loss_optimizer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                            grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        return train_op

    def train_gcn(self, sess, uid, seq_A, seq_B, len_A, len_B, target_A, target_B, learning_rate,
                  dropout_rate, keep_prob):

        feed_dict = {self.uid: uid, self.seq_A: seq_A, self.seq_B: seq_B,
                     self.len_A: len_A, self.len_B: len_B,
                     self.target_A: target_A, self.target_B: target_B, self.learning_rate: learning_rate,
                     self.dropout_rate: dropout_rate, self.keep_prob: keep_prob}

        return sess.run([self.joint_loss, self.ssl_loss, self.op], feed_dict)

    def evaluate_gcn(self, sess, uid, seq_A, seq_B, len_A, len_B, target_A, target_B, learning_rate, dropout_rate,
                     keep_prob):
        feed_dict = {self.uid: uid, self.seq_A: seq_A, self.seq_B: seq_B,
                     self.len_A: len_A, self.len_B: len_B,
                     self.target_A: target_A, self.target_B: target_B,
                     self.learning_rate: learning_rate, self.dropout_rate: dropout_rate,
                     self.keep_prob: keep_prob}
        return sess.run([self.pred_A, self.pred_B], feed_dict)

    def calculate_ssl_loss(self):
        ssl_loss_user = 0
        ssl_loss_item = 0

        if self.ssl_mode in ['user_side', 'both_side']:
            user_emb1 = tf.nn.embedding_lookup(self.sub_embedding_u1, self.uid)
            user_emb2 = tf.nn.embedding_lookup(self.sub_embedding_u2, self.uid)

            normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
            normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)
            normalize_all_user_emb2 = tf.nn.l2_normalize(self.sub_embedding_u2, 1)
            pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
            ttl_score_user = tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_a=False,
                                       transpose_b=True)

            pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
            ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / self.ssl_temp), axis=1)

            ssl_loss_user = -tf.reduce_sum(tf.log(pos_score_user / ttl_score_user))

        if self.ssl_mode in ['item_side', 'both_side']:
            item_emb1 = tf.nn.embedding_lookup(self.sub_embedding_B1, self.seq_B)
            item_emb2 = tf.nn.embedding_lookup(self.sub_embedding_B2, self.seq_B)

            if self.ex_on:
                seq_emb_1 = self.ssl_ex_unit(seq_ebd_B=item_emb1)
                seq_emb_2 = self.ssl_ex_unit(seq_ebd_B=item_emb2)
            else:
                seq_emb_1 = tf.reduce_max(item_emb1, 1)
                seq_emb_2 = tf.reduce_max(item_emb2, 1)

            normalize_seq_emb1 = tf.nn.l2_normalize(seq_emb_1, 1)
            normalize_seq_emb2 = tf.nn.l2_normalize(seq_emb_2, 1)
            normalize_all_item_emb2 = tf.nn.l2_normalize(self.sub_embedding_B2, 1)
            pos_score_item = tf.reduce_sum(tf.multiply(normalize_seq_emb1, normalize_seq_emb2), axis=1)
            ttl_score_item = tf.matmul(normalize_seq_emb1, normalize_all_item_emb2, transpose_a=False,
                                       transpose_b=True)

            pos_score_item = tf.exp(pos_score_item / self.ssl_temp)
            ttl_score_item = tf.reduce_sum(tf.exp(ttl_score_item / self.ssl_temp), axis=1)

            ssl_loss_item = -tf.reduce_sum(tf.log(pos_score_item / ttl_score_item))

        if self.ssl_mode == 'user_side':
            ssl_loss = self.ssl_reg * ssl_loss_user * 0.5
        elif self.ssl_mode == 'item_side':
            ssl_loss = self.ssl_reg * ssl_loss_item * 0.5
        else:
            ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item) * 0.5

        return ssl_loss

    def ex_mlp(self, seq_ebd_A, seq_ebd_B, isSparseOne=False):
        with tf.variable_scope('ex_mlp'):
            if isSparseOne:
                shape_0_B = tf.shape(seq_ebd_B)[0]
                shape_1_B = tf.shape(seq_ebd_B)[1]

                self.mlp_output_B = tf.matmul(tf.reshape(seq_ebd_B, [-1, self.embedding_size]),
                                              self.all_weights['W_bal_B']) + self.all_weights['b_bal_B']
                mlp_output_B = tf.nn.tanh(self.mlp_output_B)
                d_trans_B = tf.reshape(tf.matmul(mlp_output_B, self.all_weights['h_bal_B']), [shape_0_B, shape_1_B])
                d_trans_B = tf.exp(d_trans_B)
                mask_index_B = tf.reduce_sum(self.len_B, 1)
                mask_mat_B = tf.sequence_mask(mask_index_B, maxlen=shape_1_B, dtype=tf.float32)
                d_trans_B = mask_mat_B * d_trans_B
                exp_sum_B = tf.reduce_sum(d_trans_B, 1, keepdims=True)
                exp_sum_B = tf.pow(exp_sum_B, tf.constant(self.beta, tf.float32, [1]))

                score_B = tf.expand_dims(tf.div(d_trans_B, exp_sum_B), 2)

                return tf.reduce_sum(score_B * seq_ebd_B, 1)
            else:
                shape_0_A = tf.shape(seq_ebd_A)[0]
                shape_1_A = tf.shape(seq_ebd_A)[1]

                self.mlp_output_A = tf.matmul(tf.reshape(seq_ebd_A, [-1, self.embedding_size]),
                                              self.all_weights['W_bal_A']) + self.all_weights['b_bal_A']
                mlp_output_A = tf.nn.tanh(self.mlp_output_A)
                d_trans_A = tf.reshape(tf.matmul(mlp_output_A, self.all_weights['h_bal_A']), [shape_0_A, shape_1_A])
                d_trans_A = tf.exp(d_trans_A)
                mask_index_A = tf.reduce_sum(self.len_A, 1)
                mask_mat_A = tf.sequence_mask(mask_index_A, maxlen=shape_1_A, dtype=tf.float32)
                d_trans_A = mask_mat_A * d_trans_A
                exp_sum_A = tf.reduce_sum(d_trans_A, 1, keepdims=True)
                exp_sum_A = tf.pow(exp_sum_A, tf.constant(self.beta, tf.float32, [1]))

                score_A = tf.expand_dims(tf.div(d_trans_A, exp_sum_A), 2)

                return tf.reduce_sum(score_A * seq_ebd_A, 1)

    def ssl_ex_unit(self, seq_ebd_B):
        shape_0_B = tf.shape(seq_ebd_B)[0]
        shape_1_B = tf.shape(seq_ebd_B)[1]

        self.mlp_output_B = tf.matmul(tf.reshape(seq_ebd_B, [-1, self.embedding_size]),
                                      self.all_weights['W_ssl_B']) + self.all_weights['b_ssl_B']
        mlp_output_B = tf.nn.tanh(self.mlp_output_B)
        d_trans_B = tf.reshape(tf.matmul(mlp_output_B, self.all_weights['h_ssl_B']), [shape_0_B, shape_1_B])
        d_trans_B = tf.exp(d_trans_B)
        mask_index_B = tf.reduce_sum(self.len_B, 1)
        mask_mat_B = tf.sequence_mask(mask_index_B, maxlen=shape_1_B, dtype=tf.float32)
        d_trans_B = mask_mat_B * d_trans_B
        exp_sum_B = tf.reduce_sum(d_trans_B, 1, keepdims=True)
        exp_sum_B = tf.pow(exp_sum_B, tf.constant(self.beta, tf.float32, [1]))

        score_B = tf.expand_dims(tf.div(d_trans_B, exp_sum_B), 2)

        return tf.reduce_sum(score_B * seq_ebd_B, 1)
