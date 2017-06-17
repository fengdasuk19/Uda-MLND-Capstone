参考 [A Comparative Study on Different Types of Approaches to Text Categorization](http://www.ijmlc.org/papers/158-C01020-R001.pdf) 和 [Representation and Classification of Text Documents: A Brief Review](https://pdfs.semanticscholar.org/5466/da15feb8e87724576683647fdda66a27195a.pdf) 的 Table 1: Comparative Results Among Different Representation Schemes and Classifiers obtained on Reuters 21578 and 20 Newsgroup Datasets，选取其中以 20 Newsgroup 为数据集、且与本项目待测方法有关的实验结果如下所示：

Results reported by | Representation Scheme | Classifier Used | Micro F1 | Macro F1
--------------|--------------|--------------|--------------|--------------
[[Lan et al., 2009]](https://www-old.comp.nus.edu.sg/~tancl/publications/j2009/PAMI2007-v3.pdf) | VSM with term weighting schemes |  SVM | 0.808 | 0.808

当前有 11314 篇训练文档，7532 篇测试文档。除了传统上常用的 2 种文本分类器（SVM、朴素贝叶斯）外，还使用了神经网络方法。其中，文献 [[Joachims, T. (1998)]](https://eldorado.tu-dortmund.de/bitstream/2003/2595/1/report23_ps.pdf) 测试了 SVM 和朴素贝叶斯分类器在 Reuters-21578 “ModeApte” 版本上的效果，前者在 $F_{1}$ 上的得分至少超过后者 10%。。在文献 [[Sebastiani, F. (2002)]](https://arxiv.org/pdf/cs/0110053.pdf) 中提到，同样在 Reuters-21578 “ModApte” 上进行测试，SVM 与神经网络的效果没有显著差别。

对于 Doc2Vec 的效果，参考 [Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053v2.pdf) 的 Table 3，在信息检索（information retrieval）的任务中，使用 Doc2Vec 表示的的模型，相对使用 TF-IDF 方法表示的模型，在错误率上有 32% 左右的改善。相应地可认为（使用同一种方法进行训练后的） Doc2Vec 模型相对 TF-IDF 在正确率上有 32% 左右的改善。考虑到这里并不清楚所谓的「信息检索任务」采用的数据集等信息，因此在本项目中保守期待 Doc2Vec 模型相对 TF-IDF 在正确率上有不超过 20% 的改善。

因此，设定基准如下：

本节中未列出全名的参考文献如下：

```
[Lan et al., 2009] 
Lan, M., Tan, C. L., Su. J., and Lu, Y.2009. Supervised and Traditional Term Weighting Methods for Automatic Text Categorization. IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume: 31 (4), pp. 721 – 735

[Joachims, T. (1998)]
Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. Machine learning: ECML-98, 137-142.

[Sebastiani, F. (2002)]
Sebastiani, F. (2002). Machine learning in automated text categorization. ACM computing surveys (CSUR), 34(1), 1-47.
```