from __future__ import absolute_import, division, print_function

import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf

from utils import discriminator, encoder, compute_classification_loss

class Model(object):

    def __init__(self, input_size, hidden_size, batch_size, learning_rate, log_dir):
        self.input_tensor = tf.placeholder(tf.float32, [None, 3 * input_size])
        
        ##################################
        with open('params.txt') as f:
            first = f.readline()
            first = first.strip('\n')
            temp = first.split(' ')
            o_p_dim = int(temp[3]);
            s_p_dim = int(temp[4]);
            ln = f.readline()
            for i in range(s_p_dim):
                temp = f.readline()
            sig_2_init = np.zeros((s_p_dim, s_p_dim), np.float32)
            for i in range(s_p_dim):
                temp = f.readline().strip('\n').split(' ')
                for j in range(s_p_dim):
                    sig_2_init[i,j] = float(temp[j])
            
            eig_val , eig_vec = np.linalg.eig(sig_2_init)
            cf = np.sqrt(np.repeat(eig_val,s_p_dim).reshape(s_p_dim,s_p_dim).transpose())
            r_2_init = np.multiply(cf,eig_vec)
            
            sig_3_init = np.zeros((o_p_dim, o_p_dim), np.float32)
            for i in range(o_p_dim):
                temp = f.readline().strip('\n').split(' ')
                for j in range(o_p_dim):
                    sig_3_init[i,j] = float(temp[j])
            
            eig_val , eig_vec = np.linalg.eig(sig_3_init)
            cf = np.sqrt(np.repeat(eig_val,o_p_dim).reshape(o_p_dim,o_p_dim).transpose())
            r_3_init = np.multiply(cf,eig_vec)
            
            a_2_init = np.zeros((s_p_dim, s_p_dim), np.float32)
            for i in range(s_p_dim):
                temp = f.readline().strip('\n').split(' ')
                for j in range(s_p_dim):
                    a_2_init[i,j] = float(temp[j])
            
            a_3_init = np.zeros((s_p_dim, o_p_dim), np.float32)
            for i in range(s_p_dim):
                temp = f.readline().strip('\n').split(' ')
                for j in range(o_p_dim):
                    a_3_init[i,j] = float(temp[j])     
        ###################################
        
        self.r_2 = tf.get_variable('r_2',\
            initializer=r_2_init)
        self.r_3 = tf.get_variable('r_3',\
            initializer=r_3_init)
        self.sigma_2 = tf.matmul(self.r_2, tf.transpose(self.r_2))
        self.sigma_3 = tf.matmul(self.r_3, tf.transpose(self.r_3))
        self.a_2 = tf.get_variable('a_2',\
            initializer=a_2_init)
        self.a_3 = tf.get_variable('a_3',\
            initializer=a_3_init)
        
        with arg_scope([layers.fully_connected], activation_fn=tf.nn.relu):
            with tf.variable_scope("encoder"):
                with tf.variable_scope("encoder_s_t"):
                    self.s_t_minus_1_p = encoder(self.input_tensor[:, :input_size],\
                        hidden_size)
                with tf.variable_scope("encoder_s_t", reuse=True):
                    self.s_t_p = encoder(self.input_tensor[:, input_size:2 * input_size],\
                        hidden_size)
                with tf.variable_scope("encoder_o_t"):
                    self.o_t_p = encoder(self.input_tensor[:, 2 * input_size:],\
                        hidden_size)
                    
            # define classification loss
            y_1 = self.s_t_p - tf.matmul(self.s_t_minus_1_p, self.a_2)
            pos_samples_1 = tf.matmul(tf.random_normal([batch_size, hidden_size],\
                stddev=1.), self.r_2)
            y_2 = self.o_t_p - tf.matmul(self.s_t_p, self.a_3)
            pos_samples_2 = tf.matmul(tf.random_normal([batch_size, hidden_size],\
                stddev=1.), self.r_3)
            
            with tf.variable_scope('discriminator'):
                with tf.variable_scope('d1'):
                    pos_samples_1_pred = discriminator(pos_samples_1)
                with tf.variable_scope('d1', reuse=True):
                    neg_samples_1_pred = discriminator(y_1)
                with tf.variable_scope('d2'):
                    pos_samples_2_pred = discriminator(pos_samples_2)
                with tf.variable_scope('d2', reuse=True):
                    neg_samples_2_pred = discriminator(y_2)
            classification_loss_1 = compute_classification_loss(pos_samples_1_pred, neg_samples_1_pred)
            classification_loss_2 = compute_classification_loss(pos_samples_2_pred, neg_samples_2_pred)
            classification_loss = classification_loss_1 + classification_loss_2
            
            # add summary ops
            tf.summary.scalar('classification_loss', classification_loss)
            tf.summary.scalar('classification_loss_1', classification_loss_1)
            tf.summary.scalar('classification_loss_2', classification_loss_2)

            # define references to params
            encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
            discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                scope='discriminator')
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            # define training steps
            self.learn_rate = self._get_learn_rate(global_step, learning_rate)
            
            # update discriminator
            self.train_discriminator = layers.optimize_loss(classification_loss, \
                    global_step, self.learn_rate * 10, optimizer=lambda lr: \
                    tf.train.MomentumOptimizer(lr, 0.1), variables=\
                    #tf.train.AdamOptimizer(lr), variables=\
                    discriminator_params, update_ops=[])
            
            # update encoder params to fool the discriminator
            self.train_encoder = layers.optimize_loss(-classification_loss, \
                    global_step, self.learn_rate , optimizer=lambda lr: \
                    #tf.train.MomentumOptimizer(lr, 0.9), variables=\
                    tf.train.AdamOptimizer(lr), variables=\
                    encoder_params, update_ops=[])
            
            self.sess = tf.Session()
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(log_dir, \
                self.sess.graph)
            self.sess.run(tf.global_variables_initializer())
            
    def _get_learn_rate(self, global_step, learning_rate):
        
        boundaries = [np.int64(150000), np.int64(250000), np.int64(300000), np.int64(350000) , np.int64(400000)]
        values = [learning_rate, learning_rate/10, learning_rate/100, learning_rate/1000, learning_rate/1000, learning_rate/1000]
        
        return tf.train.piecewise_constant(global_step, boundaries, values)

    def update_params(self, inputs):
        ''' the public method that update all params given a batch of data'''
        global_step = tf.contrib.framework.get_or_create_global_step()
        
        classify_loss_value = self.sess.run(self.train_discriminator, {self.input_tensor: inputs})
        #classify_loss_value = 0
        
        summary, _ = self.sess.run([self.merged, self.train_encoder], {self.input_tensor: inputs})
        #summary = self.sess.run(self.merged, {self.input_tensor: inputs})
        
        return classify_loss_value, summary
        
