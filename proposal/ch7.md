0. 【数据预处理】
    + 由于通过[gensim 所指明的收录 text8 的数据源](http://www.mattmahoney.net/dc/textdata)上提供的[下载 text8 的链接](http://www.mattmahoney.net/dc/text8.zip)指向的文件已经损坏（md5 校验值与网上给的不同）。需要从该数据源下载 [enwik9](http://www.mattmahoney.net/dc/enwik9.zip) 并使用修改过的 Perl 脚本处理 enwik9 后得到 text8（md5 值与 http://www.mattmahoney.net/dc/textdata 给出的一致）（可以从[这里](http://www.mattmahoney.net/dc/textdata#appendixa)找到原始脚本，仅修改 `s/{{[^}]*}}//g;` 为 `s/\{\{[^}]*\}\}//g;`，因为我的 Perl 提示了 `Unescaped left brace in regex is deprecated, passed through in regex; marked by <-- HERE in m/{{ <-- HERE [^}]*}}/ at old_wikifil.pl line 34.`）
    + 清洗的步骤至少包括：
        + 保留常规正文文本、图像说明
        + 丢弃表格、超链接（转为普通文字）、引用、脚注、标记符号（如 `<text ...>`、`</text>`、`#REDIRECT`、`[`、`]`、`{{`、`}}`……）外国语言（英语以外的语言）版本，并将数字用英文拼写出来，将大写字母转换为小写字母等。
    + 经过上述清洗处理后，文本中只包含：由小写字母 a-z 组成的单词、单一空格（将不在 a-z 之间的字符也一律转换为空格）
1. 【特征抽取与文本表示】
    + 使用**TF-IDF** 方法抽取特征，建立表示模型 1
    + 使用**词嵌入**（Word embedding）方法（在这里，具体使用 Word2Vec）抽取特征，建立表示模型 2
    + 考虑到 Word2Vec 的对标是 LSI，可能会使用 LSI 或 LDA 建立表示模型 1 或表示模型 3
2. 【分类器训练】
    + 文本表示模型建模工具
        + gensim
        + scikit-learn
    + 学习算法：对上述 2~3 个模型，在每个模型上分别使用下述有监督学习方法在训练数据上训练出一组分类器：
        + 神经网络
        + 逻辑回归
        + 决策树
        + 支持向量机（SVM）
        + k 近邻（k-NN）
        + 朴素贝叶斯
        + 集成学习
            + 基于上述方法（决策树以外、神经网络以外）的集成学习（AdaBoost）
            + 随机森林
    + 学习工具：
        + tensorflow：用于训练神经网络模型
        + scikit-learn：用于训练下述学习算法
            + [逻辑回归](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
            + [决策树](http://scikit-learn.org/stable/modules/tree.html)
            + [支持向量机（SVM）](http://scikit-learn.org/stable/modules/svm.html)
            + [k 近邻（k-NN）](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
            + [朴素贝叶斯](http://scikit-learn.org/stable/modules/naive_bayes.html)
            + 集成学习
                + 基于上述方法的集成学习（AdaBoost）
                + 随机森林
3. 【性能评估】
    1. 评估流程
        1. 在每个文本表示模型 $m$ 的语境下训练每一个分类器 $c$ 时，就记下分类器 $c$ 被训练到不低于基准要求的水平时所耗费的时间 $t_{train}$
        2. 对于上述每种文本表示模型：对同一种文本表示模型，对测试集文本数据进行分类，并得到 $F_1$ 与实际分类时间 $t_{test}$，比较所有方法的效果
        3. 从每个表示模型对应的所有分类器中选出效果最好的一个分类器 $c(i)$，比较这些分类器的性能
    2. 评估指标：见上述「5. 评估指标」

