# -*- coding: utf-8 -*-
# @Author : Houping Yue
# @Time : 2022/11/26 18:49
# @Email : houping.yue@yahoo.com

class Settings:
    def __init__(self):
        self.code_path = "https://github.com/HoupingY/EA-GCL"

        '''
                   block 2: the draw_figures parameters for EAGCL_Module.py
        '''

        self.learning_rate = 0.005  # 0.005 for Douban 0.001 for AMAZON
        self.dropout_rate = 0.1
        self.keep_prob = 1 - self.dropout_rate
        self.batch_size = 256
        self.epochs = 50
        self.verbose = 10
        self.gpu_num = '0'

        '''
            block 2: the draw_figures parameters for EAGCL_Module.py
        '''
        self.embedding_size = 16
        self.num_folded = self.embedding_size
        self.layer_size = '[' + str(self.embedding_size) + ']'
        self.padding_int = 0
        self.alpha = 0.2
        self.ssl_temp = 0.5
        self.ssl_mode = 'item_side'
        self.ssl_task = True
        self.beta = 0.7
        self.ssl_reg = 1e-3
        self.ex_on = True

        '''
            block 3: the draw_figures parameters for file paths
        '''
        self.dataset = 'Douban'  # Douban or Amazon
        self.path_train = '../data/' + self.dataset + '/train_data.txt'
        self.path_test = '../data/' + self.dataset + '/test_data.txt'
        self.path_dict_A = '../data/' + self.dataset + '/A_dict.txt'
        self.path_dict_B = '../data/' + self.dataset + '/B_dict.txt'
        self.path_dict_U = '../data/' + self.dataset + '/U_dict.txt'

        self.checkpoint = 'checkpoint/trained_model.ckpt'

        self.fast_running = False
        self.fast_ratio = 0.8


