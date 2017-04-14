# 毕业项目开题报告：自动文档分类

[TOC]

## 1. 项目背景

在互联网时代，越来越多的信息被上传到互联网上，下至个体上至各类组织，在进行大大小小各种决策时，都不可能忽视从互联网这个渠道获取的信息。面对浩如烟海的信息，要在更短的时间内得到更多、更准确的信息，仅靠人力查看与整理完全不现实；从而，各种相关的工具、技术应时而生，典型的例子包括搜索引擎、邮件过滤器、问答系统、消费者意见与情感分析技术（ref: http://52opencourse.com/222/斯坦福大学自然语言处理第六课-文本分类（text-classification） ）。而这些工具、技术的重要基础之一就是与文本分类相关的技术。

文本分类是基于文本内容将待定文本划分到一个或多个预先定义的类中的方法（ref: http://c.xml.org.cn/blog/uploadfile/20076211443809.PDF ），包括文本表示（预处理、索引、统计、特征表示）、分类器训练、评价与反馈等（ref: http://c.xml.org.cn/blog/uploadfile/20076211443809.PDF ）。文本表示方面的工作，包括词汇层面的独热编码（one-hot representation）、N 元语法（N-gram） 模型，句子、段落层面的词袋模型（Bag-of-words model）（ref: https://zh.wikipedia.org/wiki/词袋模型 ）等，也有词嵌入（Word Embedding）模型（ref: http://forum.yige.ai/thread/70 ）（ref: https://www.zhihu.com/question/32275069 ）（ref: http://weibo.com/3121700831/BsCvWgmPs ）如词汇层面的 Word2Vec，句子、段落层面的 Sentence2Vec、Doc2Vec 等（ref: http://www.cnblogs.com/maybe2030/p/5427148.html ）（ref: http://blog.csdn.net/wangongxi/article/details/51591031 ）；分类器的训练方法则包括 SVM、KNN、贝叶斯、基于有监督学习器的集成学习器等常见的有监督学习方法（ref: http://59.108.48.5/course/mining/12-13spring/参考文献/04-04%20基于机器学习的文本分类技术研究进展.pdf ）（ref: http://c.xml.org.cn/blog/uploadfile/20076211443809.PDF ）；评价方法则包括对于各分类器分类效果的查准率 P、查全率 R、F1 度量，以及对于总体而言的宏观平均（Macroaveraging）（给予每个分类同等权重从而求算术平均值，计算所有分类器的综合效果；用于测量小分类的效果）与对于总体而言的微观平均（Microaveraging）（给予每篇文档同等权重从而求算术平均值，计算每篇文档分类结果的综合效果；用于测量大分类的效果）（ref: http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-text-classification-1.html ）（ref: 周志华西瓜书 2.3.2 ）

本文作者对搜索引擎设计技术与情感计算技术感兴趣，因此选择研究此课题，以便为后续相关研究与设计做准备。

## 2. 问题描述

总的来说，本项目要解决的问题：

> 如何设计一套文本分类模型，能够把 18000 多条新闻较准确地分配到 20 个主题类别中？

问题分解为 3 个部分：

1. 从文本表示的角度看：已知的两种文本表示方法，即词袋模型（Bag-of-Words）方法和词嵌入模型（Word Embedding）方法，如何使用这两种方法为文本建立表示模型？哪一种模型更适合用于建立表示模型？
2. 从训练分类器的角度看：这是一个有监督学习问题。那么对于给定的文本表示模型，应该选择哪种监督学习方法进行训练，分类效果更好？
3. 从评估结合上述 2 个问题后得到的最终模型效果的角度看：经过训练后，哪种文本表示模型与哪种监督学习方法结合后训练得到的分类器的分类效果更好？

## 3. 输入数据

ch3

## 4. 解决办法

1. 【特征抽取与文本表示】从文本数据集中抽取表示文本所需的特征，然后在文本数据集上用这些抽取出的特征重新表示文本数据集
2. 【分类器训练】对已经建立的表示模型，在每个模型上分别使用一些有监督学习方法在训练数据上分别训练出一定数量的分类器。
3. 【性能评估】对上述分类器进行如下评估：
    1. 对于上述每种表示模型：比较同一文本表示模型下不同训练方法的训练效果
    2. 在每种表示模型的语境下各选出分类效果最好的那个分类器，并进行比较

## 5. 评估指标

综合考虑下述 3 个指标：

+ F1：$F_{1} = \frac{2PR}{P+R}$，其中同时考虑了查准率（Precision） $P = \frac{TP}{TP + FP}$ 与查全率（Recall） $R = \frac{TP}{TP + FN}$
+ 训练时间 $t_{train}$：训练分类器达到标准所耗费的时间
+ 分类时间 $t_{test}$：训练出的分类器在测试数据上进行分类所耗费的时间

在最终评估分类器性能时，使用下述公式来综合考虑这 3 个指标：

+ $score(F_{1}, t_{train}, t_{test}) = \frac{F_1}{t_{train} * t_{test}}$

这个指标 $score$（得分） 是我自己设计的，其含义是：

+ $F_{1}$ 放在分子处，值越高，即综合考虑了查准率 $P$ 和查全率 $R$ 的指标得分越高，模型总得分越高，即给予模型越好的评价：我希望训练得到一个在 $P$ 和 $R$ 上效果都不错的分类器
+ $t_{train} * t_{test}$ 放在分子处，即综合考虑了训练时间 $t_{train}$ 和实际分类耗时 $t_{test}$ 的影响：我希望训练得到一个训练速度和实际工作速度都较好的分类器。无论是训练耗时 $t_{train}$ 太长、实际使用时耗时 $t_{test}$ 较短，还是训练耗时 $t_{train}$ 较短、实际使用时耗时 $t_{test}$ 太长，都不是太好的模型。

## 6. 基准模型

参考 [A Comparative Study on Different Types of Approaches to Text Categorization](http://www.ijmlc.org/papers/158-C01020-R001.pdf) 和 [Representation and Classification of Text Documents: A Brief Review](https://pdfs.semanticscholar.org/5466/da15feb8e87724576683647fdda66a27195a.pdf) 的 Table 1: Comparative Results Among Different Representation Schemes and Classifiers obtained on Reuters 21578 and 20 Newsgroup Datasets，选取其中以 20 Newsgroup 为数据集、且与本项目待测方法有关的实验结果如下表：

Results reported by | Representation Scheme | Classifier Used | Micro F1 | Macro F1
--------------|--------------|--------------|--------------|--------------
[Ko et al., 2004] | Vector representation with different weights | Naïve Bayes | 83.00 | 83.30
  |   |  K-NN | 81.04 | 81.20
  |   |  SVM | 86.10 | 86.00
[Tan et al., 2005] | Vector representation |  Naïve Bayes | 0.835 | 0.835
  |   |  K-NN | 0.848 | 0.846
  |   |  SVM | 0.889 | 0.887
[Mubaid and Umair.,2006]  | Vector representation | SVM | 84.62 | 78.19
[Lan et al., 2009] | VSM with term weighting schemes |  SVM | 0.808 | 0.808
  |   |  K-NN | 0.691 | 0.691

表中的参考文献如下：

```
[Ko et al., 2004] 
Ko, Y. J., Park, J., and Seo, J. 2004. Improving text categorization using the importance of sentences. An
International Journal Information Processing and Management, Vol. 40, pp. 65 – 79.

[Tan et al., 2005] 
Songbo, T., Cheng, X., Ghanem, M. M., Wnag, B., and Xu, H. 2005. A novel refinement approach for text
categorization. In the Proceedings of Fourteenth ACM International Conference on Information and Knowledge Management, pp 469 – 476.

[Mubaid and Umair.,2006] 
Mubaid, H. A., and Umair, S. A. 2006. A New Text Categorization Technique Using Distributional Clustering
and Learning Logic. IEEE Transactions on Knowledge and Data Engineering, Vol 18 (9), pp. 1156 – 1165

[Lan et al., 2009] 
Lan, M., Tan, C. L., Su. J., and Lu, Y.2009. Supervised and Traditional Term Weighting Methods for Automatic Text Categorization. IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume: 31 (4), pp. 721 – 735
```

考虑到[文献](http://nmis.isti.cnr.it/sebastiani/Publications/ACMCS02.pdf)在 7.2. Benchmarks for Text Categorization 这一节提到的：

> In general, different sets of experiments may be used for cross-classifier comparison only if the experiments have been performed
> (1) on exactly the same collection (i.e., same documents and same categories);
> (2) with the same “split” between training set and test set;
> (3) with the same evaluation measure and, whenever this measure depends on some parameters (e.g., the utility matrix chosen), with the same parameter values

也就是说，仅当这些实验满足下述条件时，各分类器是可比的：（1）实验数据集相同；（2）在数据集上的划分相同（训练集，测试集）；（3）使用相同的方法测量性能，且每当测量依赖于某些参数时，参数必须取相同值

上述 4 篇文章能满足全部 3 个条件的只有 [Lan et al., 2009] 所进行的部分实验，即 V-A 中的 Figure 3 与 Figure 5 对应实验。参考[该文章](https://www-old.comp.nus.edu.sg/~tancl/publications/j2009/PAMI2007-v3.pdf)，整理表格如下：

Results reported by | Representation Scheme | Classifier Used | Micro F1 | Macro F1
--------------|--------------|--------------|--------------|--------------
[Lan et al., 2009] | VSM with term weighting schemes |  SVM | 0.808 | 0.808
  |   |  K-NN | 0.691 | 0.691

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

## 7. 设计大纲

ch7