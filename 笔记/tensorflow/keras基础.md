识别时间

背景改黑

tensorboard





Fully Convolutional Networks

FCN对图像进行像素级的分类，从而解决了语义级别的图像分割（semantic segmentation）问题。



```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```



### model.compile 需要三个参数 （损失函数，优化器，评估标准）

####  损失函数 losses

  均方误差

  mean_squared_error(y_true, y_pred)

  绝对值损失函数

  mean_absolute_error(y_true, y_pred)

  mean_absolute_percentage_error(y_true, y_pred)

  mean_squared_logarithmic_error(y_true, y_pred)

  squared_hinge(y_true, y_pred)

  hinge(y_true, y_pred)

  categorical_hinge(y_true, y_pred)

  logcosh(y_true, y_pred)

  categorical_crossentropy(y_true, y_pred)

  sparse_categorical_crossentropy(y_true, y_pred)

  binary_crossentropy(y_true, y_pred)

  kullback_leibler_divergence(y_true, y_pred)

  poisson(y_true, y_pred)

  cosine_proximity(y_true, y_pred)

####  优化器 optimizer

  sgd 随机梯度下降

  keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=**False**)

  keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=**None**, decay=0.0)

  keras.optimizers.Adagrad(lr=0.01, epsilon=**None**, decay=0.0)

  keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=**None**, decay=0.0)

  keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=**None**, decay=0.0, amsgrad=**False**)（神经网络常用）

  keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=**None**, decay=0.0)

  keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=**None**, schedule_decay=0.004)

#### 评估标准Metric

  评价函数和损失函数相似，只不过评价函数的结果不会用于训练过程中。

  binary_accuracy(y_true, y_pred)

  categorical_accuracy(y_true, y_pred)

  sparse_categorical_accuracy(y_true, y_pred)

  top_k_categorical_accuracy(y_true, y_pred, k=5)

  sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)

## 激活函数Activation

使得神经元更好地拟合非线性函数

使输出归一化

```python
Dense(64,activation='tanh')
```

keras.activations.softmax(x, axis=-1)

keras.activations.elu(x, alpha=1.0)

keras.activations.selu(x)

keras.activations.softplus(x)

keras.activations.softsign(x)

keras.activations.relu(x, alpha=0.0, max_value=**None**, threshold=0.0)

keras.activations.tanh(x)

sigmoid(x)

hard_sigmoid(x)

keras.activations.exponential(x)自然指数

keras.activations.linear(x)线性