# Recommendation-System-using-GeometricMatrix-Completion-with-RecurrentMulti-Graph-Neural-Networks
Recommendation System
![](/Recommendation_System/recommendation_system_figure.png)
Loss Function of Training and Test
![](/Recommendation_System/Loss_figure.png)

Output of the code
M 
 
 [[5. 4. 0. ... 5. 0. 0.]
 
 [3. 0. 0. ... 0. 0. 5.]
 
 [4. 0. 0. ... 0. 0. 0.]
 
 ...
 
 [0. 0. 0. ... 0. 0. 0.]
 
 [0. 0. 0. ... 0. 0. 0.]
 
 [0. 0. 0. ... 0. 0. 0.]]

...............................................................................................................................

Otraining
 
 [[1. 1. 0. ... 1. 0. 0.]
 
 [1. 0. 0. ... 0. 0. 1.]
 
 [1. 0. 0. ... 0. 0. 0.]
 
 ...
 
 [0. 0. 0. ... 0. 0. 0.]
 
 [0. 0. 0. ... 0. 0. 0.]
 
 [0. 0. 0. ... 0. 0. 0.]]

..................................................................................................................................

Otest
 
 [[0. 0. 0. ... 0. 0. 0.]
 
 [0. 0. 0. ... 0. 0. 0.]
 
 [0. 0. 0. ... 0. 0. 0.]
 
 ...
 
 [0. 0. 0. ... 0. 0. 0.]
 
 [0. 0. 0. ... 0. 0. 0.]
 
 [0. 0. 0. ... 0. 0. 0.]]

.................................................................................................................................................

W_movies
 
 ['data' 'ir' 'jc']

............................................................................................................................................

W_users  

['data' 'ir' 'jc']

............................................................................................................................................

Number of 0 in M =  1486126

Shape of M =  (943, 1682)

Number of 0 in Training =  1506126

Number of 0 in Test =  1566126

Shape of Training =  (943, 1682)

Shape of Test =  (943, 1682)

Users Shape =  (943, 943)

Movies Shape =  (1682, 1682)

Number of data samples:  (40000.0,)

Number of training samples:  (40000.0,)

Number of training + data samples:  (80000.0,)

(943, 943)

(943,)

(943, 1682)

[268.17402035  110.0910939   99.37433058  77.97280296  75.07883397

72.27461596  70.24918496  64.19805813  62.87750106  61.96979953]

Initial User Shape  (943, 10)

Initial Items Shape  (1682, 10)

Original Training Matrix Odata*M

![](/Recommendation_System/Initial_matrix.png)

(943, 1682)

Reconstructed Training Matrix initial_W.initial_H.T

![](/Recommendation_System/Factorizes_matrix.png)

Final Matrix of Users and Movies

![](/Recommendation_System/final_matrix_figure.png)

............................................................................................................................................

Recommend How many top Movies for user: 4

User Id: 65

We will recommend these movies to the user 


[197             Nikita (La Femme Nikita) (1990)

355                          Client, The (1994)

518    Treasure of the Sierra Madre, The (1948)

185                  Blues Brothers, The (1980)

Name: 1, dtype: object]

............................................................................................................................................

Mean of all Ratings:  3.1963606

Shape of final Matrix:  (943, 1682)
