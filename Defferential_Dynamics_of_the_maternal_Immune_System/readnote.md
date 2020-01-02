# Differential Dynamics of the Maternal Immune System in Healthy Pregnancy and Preeclampsia阅读笔记

原论文的链接：[Differential Dynamics of the Maternal Immune System in Healthy Pregnancy and Preeclampsia](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6584811/# )

# 1.问题的设定

* 作者使用了21 immune cell types作为实验的特征.
* 样本总数是23个，其中为preeclampsia的11个，另外12个是normotensive.

#  2. correlation Network

这里应该是相关关系，作者可能在分析特征间的相关关系。

在每个时间点，测量成对的免疫特征间的spearman coorelation。然后使用t_SNE与i-graph软件进行可视化

# 3. 对免疫细胞的动态特征的参数化表示

$\rho=\frac{FEATURE_{T2}-FEATURE_{T1}}{GA_{T2}-GA_{T1}}$

**用$\rho$去构造feature matrix.**

这里使用的方法： 特征数据是使用前面定义的$\rho$构成的matrix，分类结果是Y, 他是二元分类结果，即（0，1），表示是是子痫前期患者或者不是子痫前期患者。

#  4. 基本思路

* Lasso with L1 penalization.
* 使用cross-validation去选择超参数$\lambda$。

# 5.  Logistic Regression

作者在论文中写道：

$l(\beta)=\sum_{i=1}^{n}logp_{yi}(\rho_i;\beta)$，其中$p_{yi}(\rho_i;\beta)=Pr(C=y_i|P=\rho_i;\beta)$

其中$\beta$是定义的参数，$\rho$是特征，$y_i$是真实结果。这里的$p_{yi}(\rho;\beta)$计算的是在使用参数化$\beta$ ,并且给定$\rho_i$ 的情况下结果是子痫前期或者不是子痫前期的概率。

$l(\beta)$则把所有的样本的对谁概率全部加一起，也就是所有的概率乘积的对数形式，这个是在计算似然函数（likelihood），即我们的目标是要去最大化$l(\beta)$这个函数。

## 5.1 逻辑斯蒂回归

这里我们的问题已经是一个二分类的问题了,也就是逻辑斯蒂回归模型(binomial logistic regression model)。

### 5.2 线性模型

先简单说下线性模型的基本形式，我们沿用作者在论文中符号定义，$f(x)=\beta^T\rho+b$，其中$\beta$是权重参数（weights），b是bias。

### 5.3 线性回归（linear regression）

这里我我们给定来了23个病人的数据集，$D=\{(\rho_1,y_1),(\rho_2,y_2),\cdot\cdot\cdot(\rho_{23},y_{23})\}$,其中的$\rho$是作者定义的特征，y是病人是子痫前期或者不是子痫前期。

线性回归是去学习一个函数，$f(\rho)=\beta_i^T\rho+b_i$, 让我们学到的参数$\beta_i$与$b_i$代入函数后算的的结果能够和真实病人是否是子痫前期尽量接近。

这实际上是去解决一个让真实结果和预测结果尽量最小的最优化问题。

$\beta^*,b^*=argmin_{\beta,b}\sum_{i=1}^{23}(f(\rho_i)-y_i)^2=argmin_{\beta,b}\sum_{i=1}^{23}(y_i-\beta_i^T\rho-b_i)$

### 5.4 逻辑斯蒂回归的基本形式

逻辑斯蒂回归是这样定义的：

$P(C=1|\rho)=\frac{exp(\beta\cdot \rho+b)}{1+exp(\beta\cdot \rho+b)}$

$P(C=0|\rho)=\frac{1}{1+exp(\beta\cdot x+b)}$

这两种情况加起来的概率刚好是1，在后面作者使用的方法中没有使用bias，所以没有b。

### 5.5 Sigmoid函数

这里概率被定义为这样的形式是因为我们需要用Sigmoid函数去把任意的$\rho$转换成一个概率结果，如果转换的概率P大于0.5，我们就说这个病人患有子痫前期，如果概率小于0.5，那么病人就没有子痫前期。概率等于0.5，那么就是算法也不知道怎么办。

Sigmod函数的形式是：

$y=\frac{1}{1+e^{-z}}$,其中的z在我们上面的线性模型就是$\beta^T\rho+b$。

这个函数是下面这样的，他吧结果压缩到了[0,1]之间，所以就可以用结果来表示概率。
![image.png](https://upload-images.jianshu.io/upload_images/15463866-b42b61d2591257a4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



### 5.6 log odds

* 我们把事件发生的几率定义为：$$\frac{p}{1-p}$$

然后对数几率就是：对logistic regression而言，就是$log\frac{P(C=1|\rho)}{1-P(Y=1\rho)}=\beta\rho$

从这个结果我们可以看到结果是子痫前期的对数几率是特征$\rho$的线性函数，或者说，输出是子痫前期的对数几率是由输入$\rho$的线性函数表示的模型，即逻辑斯蒂回归模型。

### 5.7 对最大似然进行推导

根据前面的概率定义，我们用极大似然估计去估计参数模型：

我们对之前设定做一个简化

$P(C=1|\rho)=\frac{exp(\beta\cdot \rho+b)}{1+exp(\beta\cdot \rho+b)}=\pi$

$P(C=0|\rho)=\frac{1}{1+exp(\beta\cdot x+b)}=1- \pi$

似然函数，这个意思就是让每个样本属于其真实标记（是子痫前期，或者不是子痫前期）的概率越大越好：

$\prod_{i=1}^{N}{\pi}^{y_i}_i{[1-\pi]}^{1-y_i}$

注意这里的$y_i$是只有0与1两类的，

当$y_i=0$的时候$\pi^{y_i}_i{[1-\pi]}^{1-y_i}$就简化为了$1-\pi$

当$y_i=1$的时候$\pi^{y_i}_i{[1-\pi]}^{1-y_i}$就简化为了$\pi$

所以上面的复杂连乘运算只是做了一个简化操作。然后我们看看他的对数似然可以表示为什么：

$l(\beta)=\sum_{i=1}^{N}[y_ilog\pi+(1-y_i)log(1-\pi)]$  （这里是对数的乘法转化为加法）

$=\sum_{i=1}^{N}[y_ilog\frac{\pi}{1-\pi}+log(1-\pi)]$    （简单的合并）

$=\sum_{i=1}^{N}[y_i(\beta\cdot\rho)-log(1+exp(\beta\cdot \rho_i))]$  (这里代入了之前的概率的定义$P(C=1|\rho)$与$P(C=0|\rho)$）



在这里基本就和论文中的简化后的损失函数对应上了，只是矩阵乘法要用上转置而已.

$l(\beta)=\sum_{i=1}^{n}[y_i\beta^T\rho_i-log(1+exp(\beta^T\rho_i))]$

**这里我们推导出来的是似然函数，$l(\beta)$他是似然函数，不是损失，损失应该是最大似然的反方向$-l(\beta)$,真实的损失函数应该是：**

$Loss(\beta)=\sum_{i=1}^{n}[-y_i\beta^T\rho_i+log(1+exp(\beta^T\rho_i))]$



# 6.LASSO

既然前面定义了logistic regression的loss function, lasso其实很简单，他是在前面的损失函数基础上加了一个L1的正则项（regularization）。

LASSO的正则项使用的是参数$\beta$的一阶范数之和。

所以结果就是：

$l(\beta)=\sum_{i=1}^{n}[y_i\beta^T\rho_i-log(1+exp(\beta^T\rho_i))]+\lambda \sum_{j=1}^{p}|\beta_j|$

**注意这里作者的符号用错了，他把regularization用p累加,这里起始应该是n**。

**正常的的逻辑斯蒂回归加LASSO应该这样写,其中$\lambda>0$, 按照作者的思路那么$\lambda<0$,同时把后续的最小化损失的过程转换为最大化似然，也能说得通。**

$Loss(\beta)=\sum_{i=1}^{n}[-y_i\beta^T\rho_i+log(1+exp(\beta^T\rho_i))]+\lambda \sum_{j=1}^{n}|\beta_j|$

regularization这一项主要是为了降低过拟合的风险（overfitting），lasso使用的是L1范数，这是比L2范数更容易得到稀疏解（sparse），也就是这样求得的结果$\beta$中会有更少的非零分量。关于稀疏解不是这篇文章的重点，就不说了。

总之，LASSO只是在logistic regression的损失函数的基础上加了个参数的一阶范数而已。让最后得到的参数能够有更好的泛化能力。

# 7.训练目标

既然有了损失函数，我们的目标就是找到一组参数$\beta$，他可以让loss function最小。数学表达式就是：

$argmin_w \ \ \  Loss(\beta)=\sum_{i=1}^{n}[-y_i\beta^T\rho_i+log(1+exp(\beta^T\rho_i))]+\lambda \sum_{j=1}^{n}|\beta_j|$, 其中$\lambda>0$

按照作者的思路，也可以是最大化似然：

$argmax_w \ \ \  l(\beta)=\sum_{i=1}^{n}[y_i\beta^T\rho_i-log(1+exp(\beta^T\rho_i))]+\lambda \sum_{j=1}^{p}|\beta_j|$， 其中$\lambda<0$

**如何去解这个最优化问题的**

样本少的时候是完全有解析解的，作者在论文中应该是用了工具去做这个最优化的计算。他跑了100个迭代，然后用100次迭代优化（还不知道用的工具的优化方法），最后用这100次迭代之后得到的参数$\beta_{final}$去进行测试集的验证。

**个人有点怀疑结果，样本量太少，而且还经过了交叉验证，作者其实没有足够的样本来保证这个模型的泛化能力，也就是说他的结果很可能过拟合特别严重。我们可以进一步考虑用新的样本去检查他的模型的泛化能力，作者后面使用了student t-test去做了测试检验，需要进一步复现下实验结果看看效果如何**



# 8. 关于cross validation

这是将数据及D划分为k个大小相似的互斥子集，可以通过一些分布随机采样。每次用k-1个子集的并集作为训练集，然后留下一个做测试集。这样就有了k个训练集/测试集，然后就可以进行k次训练和测试，最后可以返回k个结果的均值。k的选择其实很重要，我们通常把cross validation 叫做k-fold cross validation作者在论文中没有说明如何选k，可以自选下作者的定义。

# 9. 遗留问题：

还有些其他问题我还需要继续研究下：

1. AUC and interquartile range
2. piecewise regression method
3. How to draw boxplots
4. arcsinh transofrm
5. Spearman correlation analyses
6. t-SNE algorithm
7. i-graph R package
8. how to evaluate it with student t-test
9. check Shapiro-Wilk test and Mann-Whitney test 
10. Use SPSS 12.0 for confounder analysis



