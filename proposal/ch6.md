参考 [A Comparative Study on Different Types of Approaches to Text Categorization](http://www.ijmlc.org/papers/158-C01020-R001.pdf) 和 [Representation and Classification of Text Documents: A Brief Review](https://pdfs.semanticscholar.org/5466/da15feb8e87724576683647fdda66a27195a.pdf) 的 Table 1: Comparative Results Among Different Representation Schemes and Classifiers obtained on Reuters 21578 and 20 Newsgroup Datasets，选取其中以 20 Newsgroup 为数据集、且与本项目待测方法有关的实验结果如下所示：

Results reported by | Representation Scheme | Classifier Used | Micro F1 | Macro F1
--------------|--------------|--------------|--------------|--------------
[[Lan et al., 2009]](https://www-old.comp.nus.edu.sg/~tancl/publications/j2009/PAMI2007-v3.pdf) | VSM with term weighting schemes |  SVM | 0.808 | 0.808
  |   |  K-NN | 0.691 | 0.691

当前有 11314 篇训练文档，7532 篇测试文档。考虑到 k-NN 算法的实现，将在每次实际分类时才计算待分类文档与训练集中所有文档的相似度，再从中选取 k 个最近的样本点，然后计算出待分类文档的标签。尽管训练开销为 0，但单次测试开销较大（与其余方法相比，该方法大约花费 11314 倍于其他方法的运行时），而且当前待测试的文本较多，因此预计总开销较大，故本次项目中舍弃 k-NN。

除了传统上常用的 2 种文本分类器（SVM、朴素贝叶斯）外，还使用了神经网络方法。其中，文献 [[Joachims, T. (1998)]](https://eldorado.tu-dortmund.de/bitstream/2003/2595/1/report23_ps.pdf) 测试了 SVM 和朴素贝叶斯分类器在 Reuters-21578 “ModeApte” 版本上的效果，前者在 $F_{1}$ 上的得分至少超过后者 10%。。在文献 [[Sebastiani, F. (2002)]](https://arxiv.org/pdf/cs/0110053.pdf) 中提到，同样在 Reuters-21578 “ModApte” 上进行测试，SVM 与神经网络的效果没有显著差别。

对于 Doc2Vec 的效果，参考 [Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053v2.pdf) 的 Table 3，在信息检索（information retrieval）的任务中，使用 Doc2Vec 表示的的模型，相对使用 TF-IDF 方法表示的模型，在错误率上有 32% 左右的改善。相应地可认为（使用同一种方法进行训练后的） Doc2Vec 模型相对 TF-IDF 在正确率上有 32% 左右的改善。考虑到这里并不清楚所谓的「信息检索任务」采用的数据集等信息，因此在本项目中保守期待 Doc2Vec 模型相对 TF-IDF 在正确率上有不超过 20% 的改善。

因此，设定基准如下：

文本表示方案 | 分类器 | 微 $F_1$（Micro $F_1$） | 宏 $F_1$（Macro $F_1$）
--------|----------|-------------|---------------
TF-IDF | SVM | 0.80 | 0.80
 | 神经网络 | 0.80 | 0.80
 | 朴素贝叶斯 | 0.70 | 0.70
Doc2Vec | SVM | 0.80~0.96 | 0.80~0.96
 | 神经网络 | 0.80~0.96 | 0.80~0.96
 | 朴素贝叶斯 | 0.70~0.84 | 0.70~0.84

本节中未列出全名的参考文献如下：

```
[Lan et al., 2009] 
Lan, M., Tan, C. L., Su. J., and Lu, Y.2009. Supervised and Traditional Term Weighting Methods for Automatic Text Categorization. IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume: 31 (4), pp. 721 – 735

[Joachims, T. (1998)]
Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. Machine learning: ECML-98, 137-142.

[Sebastiani, F. (2002)]
Sebastiani, F. (2002). Machine learning in automated text categorization. ACM computing surveys (CSUR), 34(1), 1-47.
```





--------------Information above is all I need----------------------------------









参考 [Machine Learning in Automated Text Categorization](https://arxiv.org/pdf/cs/0110053.pdf)，该文章在 7.3 Which text classifier is best? 这一小节的讨论中尝试得出一些结论：

1. 表现最好的学习器：集成学习器， 支持向量机（SVM）$\approx$ 决策树，kNN
2. 次优的学习器：神经网络
3. 表现最差的学习器：朴素贝叶斯

文章随后也提及，上述结论不是绝对的，例如实际使用环境中某写「语境」具备的特征可能与训练语料中的性质大为不同，而不同的分类器对这些性质的响应又不同（It is important to bear in mind that the considerations above are not absolute statements (if there may be any) on the comparative effectiveness of these TC methods. One of the reasons is that a particular applicative context may exhibit very different characteristics from the ones to be found in Reuters, and different classifiers may respond differently to these characteristics）。尽管如此，仍不妨以上述结论与数据为参考之一。

关于 Word2Vec 的性能，[Deeplearning4j 的文档中是这样陈述的](https://deeplearning4j.org/cn/bagofwords-tf-idf)：

> Word2vec很适合对文档进行深入分析，识别文档的内容和内容子集。它的向量表示每个词的上下文，亦即词所在的n-gram。词袋法适合对文档进行总体分类。

估计 Word2Vec 对文档进行总体分类的效果或许不如 TF-IDF。再考虑到[Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053v2.pdf) 的 Table 3 中表明在 Distributed Representations  下的错误率有相对 TF-IDF 情况下 32% 左右的改善，相应可认为正确率有 32% 左右的改善，从而设定标准如下，所有提及的学习器的训练效果将不小于下述基准：

文本表示方案 | 分类器 | 微 $F_1$（Micro $F_1$） | 宏 $F_1$（Macro $F_1$）
--------|----------|-------------|---------------
TF-IDF | 集成学习器 | 0.85 | 0.85
 | SVM | 0.80 | 0.80
 | 决策树 | 0.80 $\pm$ 0.02 | 0.80 $\pm$ 0.02
 | kNN | 0.75 | 0.75
 | 神经网络 | 0.70 | 0.70
 | 朴素贝叶斯 | 0.65 | 0.65
Word2Vec | 集成学习器 | 0.85 $\pm$ 0.10 | 0.85 $\pm$ 0.10
 | SVM | 0.80 $\pm$ 0.10 | 0.80 $\pm$ 0.10
 | 决策树 | 0.80 $\pm$ 0.12 | 0.80 $\pm$ 0.12
 | kNN | 0.75 $\pm$ 0.10 | 0.75 $\pm$ 0.10
 | 神经网络 | 0.70 $\pm$ 0.10 | 0.70 $\pm$ 0.10
 | 朴素贝叶斯 | 0.65 $\pm$ 0.10 | 0.65 $\pm$ 0.10

