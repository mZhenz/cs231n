# Experiment

## 实验一

learning_rates = [5e-8, 1e-7, 2e-7]

regularization_strengths = [2.5e4, 5e4, 1e5]

num_iters = 1201

batch_size = 2000

![svm1_1](F:\CS231n\assignment1\figure\svm1_1.PNG)

![svm1_2](F:\CS231n\assignment1\figure\svm1_2.png)

![svm1_3](F:\CS231n\assignment1\figure\svm1_3.PNG)



## 实验二

learning_rates = [2.5e-8, 5e-8, 1e-7, 2e-7, 4e-7]

regularization_strengths = [1.2e4, 2.5e4, 5e4, 1e5, 2e5]

num_iters = 1201

batchsize = 1000

```
lr 2.500000e-08 reg 1.200000e+04 train accuracy: 0.264347 val accuracy: 0.277000
lr 2.500000e-08 reg 2.500000e+04 train accuracy: 0.280571 val accuracy: 0.309000
lr 2.500000e-08 reg 5.000000e+04 train accuracy: 0.326837 val accuracy: 0.336000
lr 2.500000e-08 reg 1.000000e+05 train accuracy: 0.359755 val accuracy: 0.361000
lr 2.500000e-08 reg 2.000000e+05 train accuracy: 0.351122 val accuracy: 0.360000
lr 5.000000e-08 reg 1.200000e+04 train accuracy: 0.310102 val accuracy: 0.310000
lr 5.000000e-08 reg 2.500000e+04 train accuracy: 0.349347 val accuracy: 0.366000
lr 5.000000e-08 reg 5.000000e+04 train accuracy: 0.372224 val accuracy: 0.382000
lr 5.000000e-08 reg 1.000000e+05 train accuracy: 0.363816 val accuracy: 0.380000
lr 5.000000e-08 reg 2.000000e+05 train accuracy: 0.350837 val accuracy: 0.362000
lr 1.000000e-07 reg 1.200000e+04 train accuracy: 0.372327 val accuracy: 0.370000
lr 1.000000e-07 reg 2.500000e+04 train accuracy: 0.384959 val accuracy: 0.396000
lr 1.000000e-07 reg 5.000000e+04 train accuracy: 0.371959 val accuracy: 0.376000
lr 1.000000e-07 reg 1.000000e+05 train accuracy: 0.362163 val accuracy: 0.369000
lr 1.000000e-07 reg 2.000000e+05 train accuracy: 0.350306 val accuracy: 0.364000
lr 2.000000e-07 reg 1.200000e+04 train accuracy: 0.395735 val accuracy: 0.395000
lr 2.000000e-07 reg 2.500000e+04 train accuracy: 0.385571 val accuracy: 0.396000
lr 2.000000e-07 reg 5.000000e+04 train accuracy: 0.374775 val accuracy: 0.383000
lr 2.000000e-07 reg 1.000000e+05 train accuracy: 0.360673 val accuracy: 0.375000
lr 2.000000e-07 reg 2.000000e+05 train accuracy: 0.345653 val accuracy: 0.351000
lr 4.000000e-07 reg 1.200000e+04 train accuracy: 0.394816 val accuracy: 0.387000
lr 4.000000e-07 reg 2.500000e+04 train accuracy: 0.379061 val accuracy: 0.380000
lr 4.000000e-07 reg 5.000000e+04 train accuracy: 0.374265 val accuracy: 0.384000
lr 4.000000e-07 reg 1.000000e+05 train accuracy: 0.346388 val accuracy: 0.352000
lr 4.000000e-07 reg 2.000000e+05 train accuracy: 0.346449 val accuracy: 0.351000
best validation accuracy achieved during cross-validation: 0.396000
```

![svm2_2](F:\CS231n\assignment1\figure\svm2_2.png)

![svm2_3](F:\CS231n\assignment1\figure\svm2_3.png)



## 实验三

learning_rates = [5e-8, 1e-7, 2e-7]

regularization_strengths = [2.5e4, 5e4, 1e5]

num_iters = 1401

batch_size = 1500

```
lr 5.000000e-08 reg 2.500000e+04 train accuracy: 0.364000 val accuracy: 0.375000
lr 5.000000e-08 reg 5.000000e+04 train accuracy: 0.375122 val accuracy: 0.383000
lr 5.000000e-08 reg 1.000000e+05 train accuracy: 0.363245 val accuracy: 0.379000
lr 1.000000e-07 reg 2.500000e+04 train accuracy: 0.386347 val accuracy: 0.392000
lr 1.000000e-07 reg 5.000000e+04 train accuracy: 0.376449 val accuracy: 0.383000
lr 1.000000e-07 reg 1.000000e+05 train accuracy: 0.361633 val accuracy: 0.374000
lr 2.000000e-07 reg 2.500000e+04 train accuracy: 0.385816 val accuracy: 0.396000
lr 2.000000e-07 reg 5.000000e+04 train accuracy: 0.376449 val accuracy: 0.392000
lr 2.000000e-07 reg 1.000000e+05 train accuracy: 0.362102 val accuracy: 0.373000
best validation accuracy achieved during cross-validation: 0.396000
```

![svm3_2](F:\CS231n\assignment1\figure\svm3_2.png)

![svm3_3](F:\CS231n\assignment1\figure\svm3_3.png)