from __future__ import absolute_import, division, print_function

import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf

from utils import discriminator, encoder, decoder, compute_classification_loss

class Model(object):

    def __init__(self, input_size, hidden_size, batch_size, learning_rate, log_dir):
        self.input_tensor = tf.placeholder(tf.float32, [None, 3 * input_size])
        self.s_t_p_placeholder = tf.placeholder(tf.float32, [None, hidden_size])
        
        '''
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
            self.sig_2_init = np.zeros((s_p_dim, s_p_dim), np.float32)
            for i in range(s_p_dim):
                temp = f.readline().strip('\n').split(' ')
                for j in range(s_p_dim):
                    self.sig_2_init[i,j] = float(temp[j])
            
            eig_val , eig_vec = np.linalg.eig(self.sig_2_init)
            cf = np.sqrt(np.repeat(eig_val,s_p_dim).reshape(s_p_dim,s_p_dim).transpose())
            self.r_2_init = np.multiply(cf,eig_vec)
            
            self.sig_3_init = np.zeros((o_p_dim, o_p_dim), np.float32)
            for i in range(o_p_dim):
                temp = f.readline().strip('\n').split(' ')
                for j in range(o_p_dim):
                    self.sig_3_init[i,j] = float(temp[j])
            
            eig_val , eig_vec = np.linalg.eig(self.sig_3_init)
            cf = np.sqrt(np.repeat(eig_val,o_p_dim).reshape(o_p_dim,o_p_dim).transpose())
            self.r_3_init = np.multiply(cf,eig_vec)
            
            self.a_2_init = np.zeros((s_p_dim, s_p_dim), np.float32)
            for i in range(s_p_dim):
                temp = f.readline().strip('\n').split(' ')
                for j in range(s_p_dim):
                    self.a_2_init[i,j] = float(temp[j])
            
            self.a_3_init = np.zeros((s_p_dim, o_p_dim), np.float32)
            for i in range(s_p_dim):
                temp = f.readline().strip('\n').split(' ')
                for j in range(o_p_dim):
                    self.a_3_init[i,j] = float(temp[j])     
        ###################################
        '''
        
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
            
            with tf.variable_scope("decoder"):
                with tf.variable_scope("decoder_s_t"):
                    self.output_s_t_minus_1 = decoder(self.s_t_minus_1_p, input_size)
                with tf.variable_scope("decoder_s_t", reuse=True):
                    self.output_s_t = decoder(self.s_t_p,input_size)
                with tf.variable_scope("decoder_s_t", reuse=True):
                    self.s_t_decoded = decoder(self.s_t_p_placeholder, input_size)
                with tf.variable_scope("decoder_o_t"):
                    self.output_o_t = decoder(self.o_t_p,input_size);
            self.output_tensor = tf.concat([self.output_s_t_minus_1, self.output_s_t, self.output_o_t],axis=1)
            
            self.a_2, self.b_2, self.sigma_2, self.a_3, self.b_3, self.sigma_3 = self._MLE_Gaussian_params()
            #self.a_2, self.b_2, self.sigma_2, self.a_3, self.b_3, self.sigma_3 = self._simple_Gaussian_params()
            #self.a_2, self.b_2, self.sigma_2, self.a_3, self.b_3, self.sigma_3 = self._simple_Gaussian_plus_offset_params()
            self.r_2 = tf.cholesky(self.sigma_2)
            self.r_3 = tf.cholesky(self.sigma_3)
            
            #define reconstruction loss
            reconstruction_loss = tf.reduce_mean(tf.norm(self.output_tensor - \
                self.input_tensor, axis=1))
            
            # define classification loss
            y_1 = self.s_t_p - tf.matmul(self.s_t_minus_1_p, self.a_2)
            mvn_1 = tf.contrib.distributions.MultivariateNormalFull(self.b_2, self.sigma_2)
            #mvn_1 = tf.contrib.distributions.MultivariateNormalTrill(self.b_2, scale_tril=self.r_2)
            pos_samples_1 = mvn_1.sample(batch_size)
            
            y_2 = self.o_t_p - tf.matmul(self.s_t_p, self.a_3)
            #mvn_2 = tf.contrib.distributions.MultivariateNormalTriL(self.b_3, scale_tril=self.r_3)
            mvn_2 = tf.contrib.distributions.MultivariateNormalFull(self.b_3, self.sigma_3)
            pos_samples_2 = mvn_2.sample(batch_size)
            
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
            
            # define s_t likelihood
            s_diff = self.s_t_p - tf.matmul(self.s_t_minus_1_p, self.a_2)
            s_t_likelihood = tf.reduce_sum(mvn_1.log_prob(s_diff))
            
            # define o_t likelihood
            o_diff = self.o_t_p - tf.matmul(self.s_t_p, self.a_3)
            o_t_likelihood = tf.reduce_sum(mvn_2.log_prob(o_diff))
              
            self.likelihood = s_t_likelihood + o_t_likelihood
            
            # add summary ops
            tf.summary.scalar('likelihood', self.likelihood)
            tf.summary.scalar('s_t_likelihood', s_t_likelihood)
            tf.summary.scalar('o_t_likelihood', o_t_likelihood)
            tf.summary.scalar('classification_loss', classification_loss)
            tf.summary.scalar('classification_loss_1', classification_loss_1)
            tf.summary.scalar('classification_loss_2', classification_loss_2)
            tf.summary.scalar('reconstruction_loss', reconstruction_loss)

            # define references to params
            encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
            decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
            autoencoder_params = encoder_params + decoder_params
            gaussian_params = [self.a_2, self.a_3, self.r_2, self.r_3]
            discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                scope='discriminator')
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            # define training steps
            self.learn_rate = self._get_learn_rate(global_step, learning_rate)
            
            # update autoencoder params to minimise reconstruction loss
            self.train_autoencoder = layers.optimize_loss(reconstruction_loss, \
                    global_step, self.learn_rate * 0.1, optimizer=lambda lr: \
                    tf.train.AdamOptimizer(lr), variables=\
                    #tf.train.MomentumOptimizer(lr, 0.9), variables=\
                    autoencoder_params, update_ops=[])
                        
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
        
        reconstruction_loss_value = self.sess.run(self.train_autoencoder, \
                {self.input_tensor: inputs})
        
        classify_loss_value = self.sess.run(self.train_discriminator, {self.input_tensor: inputs})
        
        summary, likelihood_value, _ = self.sess.run([self.merged, self.likelihood, self.train_encoder],\
                                                     {self.input_tensor: inputs})
        
        return reconstruction_loss_value, likelihood_value, classify_loss_value, summary
    
    def decode_s_t_p(self,input_s_t_p):
        return self.sess.run(self.s_t_decoded, {self.s_t_p_placeholder: input_s_t_p})
        
    def _simple_Gaussian_params(self):
        dim_Sp = int(self.s_t_p.shape[1])
        dim_Op = int(self.o_t_p.shape[1])
        A_2 = A_3 = Sig_2 = Sig_3 = tf.eye(dim_Sp)
        b_2 = b_3 = tf.zeros(dim_Sp)
        return A_2, b_2, Sig_2, A_3, b_3, Sig_3    
        
    def _simple_Gaussian_plus_offset_params(self):
        dim_Sp = int(self.s_t_p.shape[1])
        dim_Op = int(self.o_t_p.shape[1])
        A_2 = A_3 = Sig_2 = Sig_3 = tf.eye(dim_Sp)
        b_2 = b_3 = tf.ones(dim_Sp)
        return A_2, b_2, Sig_2, A_3, b_3, Sig_3
        
    def _MLE_Gaussian_params(self):
        dim_Sp = int(self.s_t_p.shape[1])
        dim_Op = int(self.o_t_p.shape[1])
        batch_size = self.input_tensor.shape[0]

        sumStSt_1 = tf.reduce_sum(\
                                  tf.matmul(\
                                            tf.reshape(self.s_t_p, [-1,dim_Sp,1]),\
                                            tf.reshape(self.s_t_minus_1_p, [-1,1,dim_Sp])\
                                           ),axis=0\
                                 )
        sumSt_1 = tf.transpose(tf.reduce_sum(self.s_t_minus_1_p, axis=0, keep_dims=True))
        meanSt_1 = tf.transpose(tf.reduce_mean(self.s_t_minus_1_p, axis=0, keep_dims=True))
        sumSt = tf.transpose(tf.reduce_sum(self.s_t_p,axis=0,keep_dims=True))
        meanSt = tf.transpose(tf.reduce_mean(self.s_t_p,axis=0,keep_dims=True))
        sumSt_1St_1 = tf.reduce_sum(\
                                    tf.matmul(\
                                              tf.reshape(self.s_t_minus_1_p, [-1,dim_Sp,1]),\
                                              tf.reshape(self.s_t_minus_1_p, [-1,1,dim_Sp])\
                                             ),axis=0\
                                   )

        A_2 = tf.matmul(\
                        sumStSt_1 - tf.matmul(meanSt, tf.transpose(sumSt_1)),\
                        tf.matrix_inverse(\
                                          sumSt_1St_1 - tf.matmul(meanSt_1, tf.transpose(sumSt_1))
                                         )\
                       )
        b_2 = tf.reshape(meanSt - tf.matmul(A_2, meanSt_1), [-1])

        tmp_ones = tf.reduce_mean(self.s_t_p - self.s_t_p, axis=1, keep_dims=True) + 1
        
        tmp = self.s_t_p - tf.matmul(self.s_t_minus_1_p, tf.transpose(A_2)) - \
                            tf.matmul(tmp_ones,tf.reshape(b_2,[1,-1]))
        Sig_2 = tf.reduce_mean(\
                               tf.matmul(\
                                         tf.reshape(tmp,[-1,dim_Sp,1]),\
                                         tf.reshape(tmp,[-1,1,dim_Sp])\
                                        ),axis=0\
                              )

        sumOtSt = tf.reduce_sum(\
                                tf.matmul(\
                                          tf.reshape(self.o_t_p,[-1,dim_Op,1]),\
                                          tf.reshape(self.s_t_p,[-1,1,dim_Sp])\
                                         ),axis=0\
                               )
        sumStSt = tf.reduce_sum(\
                                tf.matmul(\
                                          tf.reshape(self.s_t_p,[-1,dim_Sp,1]),\
                                          tf.reshape(self.s_t_p,[-1,1,dim_Sp])\
                                         ),axis=0\
                               )
        sumOt = tf.transpose(tf.reduce_sum(self.o_t_p,axis=0,keep_dims=True))
        meanOt = tf.transpose(tf.reduce_mean(self.o_t_p,axis=0,keep_dims=True))

        A_3 = tf.matmul(\
                        sumOtSt - tf.matmul(meanOt, tf.transpose(sumSt)),\
                        tf.matrix_inverse(\
                                          sumStSt - tf.matmul(meanSt, tf.transpose(sumSt))
                                         )\
                       )
        b_3 = tf.reshape(meanOt - tf.matmul(A_3, meanSt), [-1])

        tmp = self.o_t_p - tf.matmul(self.s_t_p, tf.transpose(A_3)) - \
                            tf.matmul(tmp_ones,tf.reshape(b_3,[1,-1]))
        Sig_3 = tf.reduce_mean(\
                               tf.matmul(\
                                         tf.reshape(tmp,[-1,dim_Op,1]),\
                                         tf.reshape(tmp,[-1,1,dim_Op])\
                                        ),axis=0\
                              )
        
        return A_2, b_2, Sig_2, A_3, b_3, Sig_3
        
        

        

'''
            # update Gaussian parameters to maximize likelihood
            self.train_likelihood = layers.optimize_loss(-likelihood, \
                    global_step, self.learn_rate * 0.1, optimizer=lambda lr: \
                    tf.train.AdamOptimizer(lr), variables=\
                    gaussian_params, update_ops=[])
'''