import numpy as np

def tune_KF_params_MLE(s_t_minus_1_train, s_t_train,o_t_train):
    train_T = s_t_train.shape[0]
    s_dim = s_t_train.shape[1]
    o_dim = o_t_train.shape[1]
    sumStSt_1 = np.sum(\
                       np.matmul(\
                                 np.reshape(s_t_train, [-1,s_dim,1]),\
                                 np.reshape(s_t_minus_1_train, [-1,1,s_dim])\
                                ),axis=0\
                      )
    sumSt_1 = np.transpose(np.sum(s_t_minus_1_train, axis=0, keepdims=True))
    sumSt = np.transpose(np.sum(s_t_train,axis=0,keepdims=True))
    sumSt_1St_1 =np.sum(\
                        np.matmul(\
                                  np.reshape(s_t_minus_1_train, [-1,s_dim,1]),\
                                  np.reshape(s_t_minus_1_train, [-1,1,s_dim])\
                                 ),axis=0\
                       )

    A_2 = np.matmul(\
                    sumStSt_1 - (np.matmul(sumSt, np.transpose(sumSt_1)) / train_T),\
                    np.linalg.inv(\
                                  sumSt_1St_1 - (np.matmul(sumSt_1, np.transpose(sumSt_1)) / train_T)
                                 )\
                   )
    b_2 = (sumSt - np.matmul(A_2, sumSt_1)) / train_T

    tmp = s_t_train - np.matmul(s_t_minus_1_train, np.transpose(A_2)) - np.repeat(b_2.reshape([1,-1]),train_T,axis=0)
    Sig_2 = np.mean(\
                    np.matmul(\
                              np.reshape(tmp,[-1,s_dim,1]),\
                              np.reshape(tmp,[-1,1,s_dim])\
                             ),axis=0\
                   )

    sumOtSt = np.sum(\
                     np.matmul(\
                               np.reshape(o_t_train,[-1,o_dim,1]),\
                               np.reshape(s_t_train,[-1,1,s_dim])\
                              ),axis=0\
                    )
    sumStSt = np.sum(\
                     np.matmul(\
                               np.reshape(s_t_train,[-1,s_dim,1]),\
                               np.reshape(s_t_train,[-1,1,s_dim])\
                              ),axis=0\
                    )
    sumOt = np.transpose(np.sum(o_t_train,axis=0,keepdims=True))

    A_3 = np.matmul(\
                    sumOtSt - (np.matmul(sumOt, np.transpose(sumSt)) / train_T),\
                    np.linalg.inv(\
                                  sumStSt - (np.matmul(sumSt, np.transpose(sumSt)) / train_T)
                                 )\
                   )
    b_3 = (sumOt - np.matmul(A_3, sumSt)) / train_T

    tmp = o_t_train - np.matmul(s_t_train, np.transpose(A_3)) - np.repeat(b_3.reshape([1,-1]),train_T,axis=0)
    Sig_3 = np.mean(\
                    np.matmul(\
                              np.reshape(tmp,[-1,o_dim,1]),\
                              np.reshape(tmp,[-1,1,o_dim])\
                             ),axis=0\
                   )
    return A_2, b_2, Sig_2, A_3, b_3, Sig_3