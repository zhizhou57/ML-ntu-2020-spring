# Regression and concept

**目标**: 找到一个函数$f$，对于给定的一个输入，将它进行分类or预测

## 1.构建模型

一个线性模型：令$y=f(x)=b+\sum \omega_{i} ·x_{i}$，其中$\omega_{}i,b_{}$是模型的参数，$x_{i}$是模型的输入

### 模型修正

#### 1.加入更多高次项

$y=f(x)=b+\sum\omega_{i1} ·x_{i} + \sum \omega_{i2} ·x^{2}_{i}+……$

#### 2.加入01阶跃函数解决多类别问题

$y=\sum\delta_{i}y_{i},\quad \delta{i}=\begin{cases}1 \quad if \quad x=x_{i} \\ 0 \quad else \end{cases}$

<img src="\image\image-20220108212049125.png" alt="image-20220108212049125" style="zoom:67%;" />

## 2.模型的衡量——损失函数

$Loss \quad function \quad L:L(f)=\sum_{k=1}^{n}(\hat y -f(x_{k}))^{2}$，其中$x_{k}$为第k个样本，$\hat y$是真实值

目标是最小化Loss function：

$f^{*}=argmin_{f}L(f) \qquad w^{*},b^{*}=argmin_{w,b}L(w,b)$

### 损失函数修正——正则化

$Loss \quad function \quad L:L(f)=\sum_{k=1}^{n}(\hat y -f(x_{k}))^{2}+\lambda\sum(w_{i})^{2}$

意味着越小的$w_{i}$越好，这是因为，我们希望输出更平滑（也就是对输入不那么敏感）

* 越大的$\lambda$表示我们考虑训练误差更多
* 越小的$\lambda$表示我们考虑让函数更平滑更多

<img src="\image\image-20220108212642883.png" alt="image-20220108212642883" style="zoom:67%;" />

## 3.梯度下降法找最优的参数

参数更新方法：$w^{k+1}=w^{k}-\eta \frac{dL}{dW}|_{w=w^{k}}$

## 4.模型解释——误差

将模型在测试数据上计算误差，误差起源于偏差bias和方差variance

* 偏差Bias表示对目标的接近程度
* 方差variance表示样本的离散程度

### 偏差Bias

较简单的模型，可能的函数集范围较小，可能会出现大偏差

较复杂的模型，可能的函数集范围比较大，偏差往往较小

<img src="\image\image-20220108213551309.png" alt="image-20220108213551309" style="zoom:67%;" />

### 方差variance

较简单的模型，可能的函数集范围较小，方差往往较小（约不容易被样本数据影响）

较复杂的模型，可能的函数集范围比较大，方差往往较大

![image-20220108213820939](\image\image-20220108213820939.png)

### Bias v.s. Variance

大偏差意味着**拟合程度低**

* 重新设计一个更复杂模型，考虑更多的特征

大方差意味着**过拟合**

* 进行正则化
* 获取更多的数据

我们的目标是找到一个函数，同时具有较好偏差与方差

<img src="\image\image-20220108213934458.png" alt="image-20220108213934458" style="zoom:67%;" />

## 5. 模型选择

对不同模型进行比较，在训练集中误差小的模型并不一定就是好模型

### 训练集与验证集划分——n重交叉验证

将数据集随机分为n个集合，其中n-1部分选为训练集，1部分选为测试集，选择平均误差最小的那个模型。将该模型在整个数据集上进行训练。

<img src="\image\image-20220108214550794.png" alt="image-20220108214550794" style="zoom:80%;" />