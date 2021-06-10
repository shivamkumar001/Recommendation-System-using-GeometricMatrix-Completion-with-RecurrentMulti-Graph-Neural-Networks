# -*- coding: utf-8 -*-
"""
!pip install h5py
!pip install scipy
!pip install matplotlib
!pip install joblib
!pip install tensorflow
"""
from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import joblib
import scipy.sparse.linalg as la
from scipy.sparse.csgraph import laplacian
import scipy
import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# %matplotlib inline

import numpy as np
filepath = '/content/drive/MyDrive/Research-Practice/DataSet_Matrix.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)
print("Type of Dataset Matrix: ",type(arrays))
print("\nM \n",arrays['M'])
print("\nOtraining\n ",arrays['Otraining'])
print("\nOtest\n ",arrays['Otest'])
print("\nW_movies\n ",arrays['W_movies'])
print("\nW_users ",arrays['W_users'])

def Matlab_File_Load(path, name):
    Read_Matlab = h5py.File(path, 'r')
    data_mat = Read_Matlab[name]
    try:
        if 'ir' in data_mat.keys():
            data = np.asarray(data_mat['data'])
            ir   = np.asarray(data_mat['ir'])
            jc   = np.asarray(data_mat['jc'])
            out  = scipy.sparse.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        out = np.asarray(data_mat).astype(np.float32).T
    Read_Matlab.close()
    return out

# Extract matrices from Matlab Matrix
filename = '/content/drive/MyDrive/Research-Practice/DataSet_Matrix.mat'
M = Matlab_File_Load(filename, 'M')
Training_0 = Matlab_File_Load(filename, 'Otraining')
Testing_0 = Matlab_File_Load(filename, 'Otest')
#Both are sparse matrix 
W_users_Row = Matlab_File_Load(filename, 'W_users') 
W_movies_Col = Matlab_File_Load(filename, 'W_movies')

# See the content and values of matrix
print("Number of 0 in M = ",(M==0).sum())

print("Shape of M = ",M.shape)

print("Number of 0 in Training = ",(Training_0==0).sum())

print("Number of 0 in Test = ",(Testing_0==0).sum())

print("Shape of Training = ",Training_0.shape)

print("Shape of Test = ",Testing_0.shape)

print("Users Shape = ",W_users_Row.shape)

print("Movies Shape = ",W_movies_Col.shape)



np.random.seed(0) # Every time same number will appear
pos_tr_samples = np.where(Training_0)
num_tr_samples = len(pos_tr_samples[0])
list_idx = list(range(num_tr_samples))
np.random.shuffle(list_idx)
idx_data = list_idx[:num_tr_samples//2]
idx_train = list_idx[num_tr_samples//2:]
pos_data_samples = (pos_tr_samples[0][idx_data], pos_tr_samples[1][idx_data])
pos_tr_samples = (pos_tr_samples[0][idx_train], pos_tr_samples[1][idx_train])
Odata = np.zeros(M.shape)
Training_0 = np.zeros(M.shape)
for k in range(len(pos_data_samples[0])):
    Odata[pos_data_samples[0][k], pos_data_samples[1][k]] = 1
    
for k in range(len(pos_tr_samples[0])):
    Training_0[pos_tr_samples[0][k], pos_tr_samples[1][k]] = 1
    
print("Number of data samples: ",  (np.sum(Odata),))
print("Number of training samples: ",  (np.sum(Training_0),))
print("Number of training + data samples: ",  (np.sum(Odata+Training_0),))

# Normalized Laplacian
Lrow = laplacian(W_users_Row, normed=True) # Normalized Laplacian Row
Lcol = laplacian(W_movies_Col, normed=True) # Normalized Laplacian Column

# Using SVD
U, s, V = np.linalg.svd(Odata*M, full_matrices=False) #Singular value decomposition to extract main value for initialization
print(U.shape)
print(s.shape) # We only need s 
print(V.shape)
#print(U)
np.set_printoptions(suppress=True)
print(s[:10])

# Make user and item matrix for Factorised method
rank_W_H = 10 # Rank of User and Movies
partial_s = s[:rank_W_H] # first 10 value of s
partial_S_sqrt = np.diag(np.sqrt(partial_s))
initial_W = np.dot(U[:, :rank_W_H], partial_S_sqrt) # 943*10 . 10*10
initial_H = np.dot(partial_S_sqrt, V[:rank_W_H, :]).T # (10*10 . 10*1682)T = 1682*10
print("Initial User Shape ",initial_W.shape)
print("Initial Items Shape ",initial_H.shape)

# Original training matrix
print("Original Training Matrix Odata*M")
plt.figure(figsize=(20,10))
plt.imshow(Odata*M)
plt.colorbar()

#print(np.amax(np.dot(initial_W,initial_H.T)))
print(np.dot(initial_W,initial_H.T).shape)

# Reconstructed Training Matrix from factorised method
print("Reconstructed Training Matrix initial_W.initial_H.T")
plt.figure(figsize=(20,10))
plt.imshow(np.dot(initial_W, initial_H.T)) # user*movies
plt.colorbar()

# GCNN and RNN Implementation
class Train_test_matrix_completion:
    def frobenius_norm(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        frobenius_norm = tf.sqrt(tensor_sum)
        return frobenius_norm
    
    def mono_conv(self, list_lap, ord_conv, A, W, b):
        feat = []
        for k in range(ord_conv):
            c_lap = list_lap[k]                                          
            c_feat = tf.matmul(c_lap, A, a_is_sparse=False)
            feat.append(c_feat)
        all_feat = tf.concat(feat, 1)
        conv_feat = tf.matmul(all_feat, W) + b
        conv_feat = tf.nn.relu(conv_feat)
        return conv_feat
               
    def compute_cheb_polynomials(self, L, ord_cheb, list_cheb):
        for k in range(ord_cheb):
            if (k==0):
                list_cheb.append(tf.cast(tf.linalg.diag(tf.ones([tf.shape(L)[0],])), 'float32'))
            elif (k==1):
                list_cheb.append(tf.cast(L, 'float32'))
            else:
                list_cheb.append(2*tf.matmul(L, list_cheb[k-1])  - list_cheb[k-2])
    
    def __init__(self, M, Lr, Lc, Odata, Training_0,Testing_0 
                 , initial_W, initial_H,order_chebyshev_col = 5, order_chebyshev_row = 5,num_iterations = 10, gamma=1.0, learning_rate=1e-4, idx_gpu = '/gpu:3'):
        
        self.ord_col = order_chebyshev_col 
        self.ord_row = order_chebyshev_row
        self.num_iterations = num_iterations
        self.n_conv_feat = 32
        
        with tf.Graph().as_default() as g:
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                self.graph = g
                tf.random.set_seed(0)
                with tf.device(idx_gpu):
                    
                        self.Lr = tf.constant(Lr.astype('float32'))
                        self.Lc = tf.constant(Lc.astype('float32'))
                        
                        self.norm_Lr = self.Lr - tf.linalg.diag(tf.ones([Lr.shape[0], ]))
                        self.norm_Lc = self.Lc - tf.linalg.diag(tf.ones([Lc.shape[0], ]))
                        self.list_row_cheb_pol = list()
                        self.compute_cheb_polynomials(self.norm_Lr, self.ord_row, self.list_row_cheb_pol)
                        self.list_col_cheb_pol = list()
                        self.compute_cheb_polynomials(self.norm_Lc, self.ord_col, self.list_col_cheb_pol)
                        
                        self.M = tf.constant(M, dtype=tf.float32)
                        self.Odata = tf.constant(Odata, dtype=tf.float32)
                        self.Training_0 = tf.constant(Training_0, dtype=tf.float32) #training mask
                        self.Testing_0 = tf.constant(Testing_0, dtype=tf.float32) #test mask
                         
                        self.W_conv_W = tf.compat.v1.get_variable("W_conv_W", shape=[self.ord_col*initial_W.shape[1], self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.b_conv_W = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.W_conv_H = tf.compat.v1.get_variable("W_conv_H", shape=[self.ord_row*initial_W.shape[1], self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.b_conv_H = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        
                        self.W_f_u = tf.compat.v1.get_variable("W_f_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.W_i_u = tf.compat.v1.get_variable("W_i_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.W_o_u = tf.compat.v1.get_variable("W_o_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.W_c_u = tf.compat.v1.get_variable("W_c_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.U_f_u = tf.compat.v1.get_variable("U_f_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.U_i_u = tf.compat.v1.get_variable("U_i_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.U_o_u = tf.compat.v1.get_variable("U_o_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.U_c_u = tf.compat.v1.get_variable("U_c_u", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.b_f_u = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_i_u = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_o_u = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_c_u = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        
                        self.W_f_m = tf.compat.v1.get_variable("W_f_m", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.W_i_m = tf.compat.v1.get_variable("W_i_m", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.W_o_m = tf.compat.v1.get_variable("W_o_m", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.W_c_m = tf.compat.v1.get_variable("W_c_m", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.U_f_m = tf.compat.v1.get_variable("U_f_m", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.U_i_m = tf.compat.v1.get_variable("U_i_m", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.U_o_m = tf.compat.v1.get_variable("U_o_m", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.U_c_m = tf.compat.v1.get_variable("U_c_m", shape=[self.n_conv_feat, self.n_conv_feat], initializer=tf.initializers.GlorotUniform())
                        self.b_f_m = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_i_m = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_o_m = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        self.b_c_m = tf.Variable(tf.zeros([self.n_conv_feat,]))
                        
                        self.W_out_W = tf.compat.v1.get_variable("W_out_W", shape=[self.n_conv_feat, initial_W.shape[1]], initializer=tf.initializers.GlorotUniform()) 
                        self.b_out_W = tf.Variable(tf.zeros([initial_W.shape[1],]))
                        self.W_out_H = tf.compat.v1.get_variable("W_out_H", shape=[self.n_conv_feat, initial_H.shape[1]], initializer=tf.initializers.GlorotUniform()) 
                        self.b_out_H = tf.Variable(tf.zeros([initial_H.shape[1],]))

                        self.W = tf.constant(initial_W.astype('float32'))
                        self.H = tf.constant(initial_H.astype('float32'))
                        
                        self.X = tf.matmul(self.W, self.H, transpose_b=True)
                        self.list_X = list()
                        self.list_X.append(tf.identity(self.X))
                        
                        self.h_u = tf.zeros([M.shape[0], self.n_conv_feat])
                        self.c_u = tf.zeros([M.shape[0], self.n_conv_feat])
                        self.h_m = tf.zeros([M.shape[1], self.n_conv_feat])
                        self.c_m = tf.zeros([M.shape[1], self.n_conv_feat])
                        
                        
                        for k in range(self.num_iterations):
                            # It extract features of global vector
                            self.final_feat_users = self.mono_conv(self.list_row_cheb_pol, self.ord_row, self.W, self.W_conv_W, self.b_conv_W)
                            self.final_feat_movies = self.mono_conv(self.list_col_cheb_pol, self.ord_col, self.H, self.W_conv_H, self.b_conv_H)
        
                            self.f_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_f_u) + tf.matmul(self.h_u, self.U_f_u) + self.b_f_u)
                            self.i_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_i_u) + tf.matmul(self.h_u, self.U_i_u) + self.b_i_u)
                            self.o_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_o_u) + tf.matmul(self.h_u, self.U_o_u) + self.b_o_u)
                            
                            self.update_c_u = tf.sigmoid(tf.matmul(self.final_feat_users, self.W_c_u) + tf.matmul(self.h_u, self.U_c_u) + self.b_c_u)
                            self.c_u = tf.multiply(self.f_u, self.c_u) + tf.multiply(self.i_u, self.update_c_u)
                            self.h_u = tf.multiply(self.o_u, tf.sigmoid(self.c_u))
                            
                            self.f_m = tf.sigmoid(tf.matmul(self.final_feat_movies, self.W_f_m) + tf.matmul(self.h_m, self.U_f_m) + self.b_f_m)
                            self.i_m = tf.sigmoid(tf.matmul(self.final_feat_movies, self.W_i_m) + tf.matmul(self.h_m, self.U_i_m) + self.b_i_m)
                            self.o_m = tf.sigmoid(tf.matmul(self.final_feat_movies, self.W_o_m) + tf.matmul(self.h_m, self.U_o_m) + self.b_o_m)
                            
                            self.update_c_m = tf.sigmoid(tf.matmul(self.final_feat_movies, self.W_c_m) + tf.matmul(self.h_m, self.U_c_m) + self.b_c_m)
                            self.c_m = tf.multiply(self.f_m, self.c_m) + tf.multiply(self.i_m, self.update_c_m)
                            self.h_m = tf.multiply(self.o_m, tf.sigmoid(self.c_m))
                            
                            self.delta_W = tf.tanh(tf.matmul(self.c_u, self.W_out_W) + self.b_out_W)
                            self.delta_H = tf.tanh(tf.matmul(self.c_m, self.W_out_H) + self.b_out_H) 
                            
                            self.W += self.delta_W
                            self.H += self.delta_H
                        
                            self.X = tf.matmul(self.W, self.H, transpose_b=True)
                            self.list_X.append(tf.identity(tf.reshape(self.X, [tf.shape(self.M)[0], tf.shape(self.M)[1]])))
                        self.X = tf.matmul(self.W, self.H, transpose_b=True)
                        
                        self.norm_X = 1+4*(self.X-tf.reduce_min(self.X))/(tf.reduce_max(self.X-tf.reduce_min(self.X)))
                        frob_tensor = tf.multiply(self.Training_0 + self.Odata, self.norm_X - M)
                        self.loss_frob = tf.square(self.frobenius_norm(frob_tensor))/np.sum(Training_0+Odata)
                        
                        trace_col_tensor = tf.matmul(tf.matmul(self.X, self.Lc), self.X, transpose_b=True)
                        self.loss_trace_col = tf.linalg.trace(trace_col_tensor)
                        trace_row_tensor = tf.matmul(tf.matmul(self.X, self.Lr, transpose_a=True), self.X)
                        self.loss_trace_row = tf.linalg.trace(trace_row_tensor)
                        
                        self.loss = self.loss_frob + (gamma/2)*(self.loss_trace_col + self.loss_trace_row)
                        
                        self.predictions = tf.multiply(self.Testing_0, self.norm_X - self.M)
                        self.predictions_error = self.frobenius_norm(self.predictions)

                        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
                        
                        self.var_grad = tf.gradients(self.loss, tf.compat.v1.trainable_variables())
                        self.norm_grad = self.frobenius_norm(tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0))

                        config = tf.compat.v1.ConfigProto(allow_soft_placement = True)
                        config.gpu_options.allow_growth = True
                        self.session = tf.compat.v1.Session(config=config)
                        init = tf.compat.v1.initialize_all_variables()
                        self.session.run(init)

# Calling 
Column_Order = 5
Row_Order = 5
learning_obj = Train_test_matrix_completion(M, Lrow.toarray(), Lcol.toarray(), Odata, Training_0, Testing_0, initial_W, initial_H,order_chebyshev_col = Column_Order,order_chebyshev_row = Row_Order,gamma=1e-10,learning_rate=1e-3)
list_training_loss = list()
list_training_norm_grad = list()
list_test_pred_error = list()
list_predictions = list()
list_X = list()
list_training_times = list()
list_test_times = list()
list_grad_X = list()
list_X_evolutions = list()

# If you want to train it again then uncomment below code
"""
# Training & Test Iteration and it's values
num_iter = 0
num_iter_test = 100
num_total_iter_training = 4000
for k in range(num_iter, num_total_iter_training):

    tic = time.time()
    _, current_training_loss, norm_grad, X_grad = learning_obj.session.run([learning_obj.optimizer, learning_obj.loss, 
                                                                                        learning_obj.norm_grad, learning_obj.var_grad]) 
    training_time = time.time() - tic
    list_training_loss.append(current_training_loss)
    list_training_norm_grad.append(norm_grad)
    list_training_times.append(training_time)

    if (np.mod(num_iter, num_iter_test)==0):
        message ="[Training] Iteration = %05i, cost = %3.3f, gradient Normalization = %.2f (%3.2fs)" \
#                                     % (num_iter, list_training_loss[-1], list_training_norm_grad[-1], training_time)
        print(message)
        
        #Testing 
        tic = time.time()
        pred_error, preds, X = learning_obj.session.run([learning_obj.predictions_error, learning_obj.predictions,
                                                                             learning_obj.norm_X]) 
        c_X_evolutions = learning_obj.session.run(learning_obj.list_X)
        list_X_evolutions.append(c_X_evolutions)
        test_time = time.time() - tic
        list_test_pred_error.append(pred_error)
        list_X.append(X)
        list_test_times.append(test_time)
        RMSE = np.sqrt(np.square(pred_error)/np.sum(Testing_0))
        message =  "[Testing] Iteration = %05i, cost = %3.3f, Root Mean Square Error = %3.2f (%3.2fs)" % (num_iter, list_test_pred_error[-1], RMSE, test_time)
        print(message)
        
    num_iter += 1

# Plot the Final Matrix
fig, ax1 = plt.subplots(figsize=(20,10))
ax2 = ax1.twinx()
ax1.plot(np.arange(len(list_training_loss)), list_training_loss, 'g-')
ax2.plot(np.arange(len(list_test_pred_error))*num_iter_test, list_test_pred_error, 'b-')
ax1.set_xlabel("Iteration")
ax1.set_ylabel('Training loss', color='g')
ax2.set_ylabel('Test loss', color='b')

best_iter = (np.where(np.asarray(list_training_loss)==np.min(list_training_loss))[0][0]//num_iter_test)*num_iter_test
best_pred_error = list_test_pred_error[best_iter//num_iter_test]
print("Best predictions at iter: %d (error: %0.3f)"%  (best_iter, best_pred_error))
RMSE = np.sqrt(np.square(best_pred_error)/np.sum(Testing_0))
print("Root Mean Square Error : %f"%  RMSE)
"""

# we have save the resulted matrix and now no need to train it again
#np.save('/content/drive/MyDrive/Research-Practice/X_new.txt', X)
Final_matrix=np.load('/content/drive/MyDrive/Research-Practice/X_new.txt.npy')

np.amax(Final_matrix)

print("Final Matrix of Users and Movies")
plt.figure(figsize=(20,10))
plt.imshow(Final_matrix)
plt.colorbar()

"""# **Recommendation for User**"""

top_n_movies=int(input("Recommend How many top Movies for user: "))
uid=int(input("User Id: "))

movies_id=[]
movies_id.append(sorted(range(len(Final_matrix[uid])), key=lambda i: Final_matrix[uid][i], reverse=True)[:top_n_movies])

import numpy as np
movies_id=np.array(movies_id)
movies_id-=1

print("We will recommend these movies to the user \n")
import pandas as pd;
data = pd.read_csv('/content/drive/MyDrive/Research-Practice/u.item',delimiter='|',header=None)
movies=[]
for elm in movies_id:
  movies.append(data.iloc[elm,1])
print(movies)

print("Mean of all Ratings: ",Final_matrix.mean())

print("Shape of final Matrix: ",Final_matrix.shape)