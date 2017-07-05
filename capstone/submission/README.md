# README

## 运行时必要配置说明

### 使用的数据集

使用了 *20 Newsgroups* 数据集。请点击[这里](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz)下载。

下载完毕后，请解压到 data/srcdata 下；解压完成后，data 下的目录结构应为：

```
.
├── data
│   ├── records_CSV
│   ├── srcdata
│   │   └── 20news-bydate
│   │       ├── 20news-bydate-test
│   │       └── 20news-bydate-train
│   └── trialdata
```

其中 `20news-bydate-train` 保存训练集数据， `20news-bydate-test` 保存测试集数据。

### 依赖的库

本项目在 conda 4.3.21 环境中创建的虚拟环境中运行正常，至少满足下述依赖：

- ipykernel=4.6.1=py36_0
- ipython=6.1.0=py36_0
- ipython_genutils=0.2.0=py36_0
- jupyter_client=5.0.1=py36_0
- jupyter_core=4.3.0=py36_0
- matplotlib=2.0.2=np112py36_0
- nltk=3.2.4=py36_0
- notebook=5.0.0=py36_0
- numpy=1.12.1=py36_0
- pandas=0.20.2=np112py36_0
- scikit-learn=0.18.1=np112py36_1
- scipy=0.19.0=np112py36_0
- pip:
  - gensim==2.1.0
  - ipython-genutils==0.2.0
  - jupyter-client==5.0.1
  - jupyter-core==4.3.0
  - textblob==0.12.0

要完全复现环境，请查看同一目录下的 `dep.yml` 文件以了解所有依赖关系。

## 如何执行

请打开 `_main_book.ipynb` 文件，首先运行第一个 cell 以完成初始化，包括导入必要的库文件等：

```
%run Experiment-exploration-ALL.init.ipynb
```

随后一路运行到底即可。


## 预计耗时

8 小时（其中以 SVC 上的 GridSearchCV 耗时最长，占 6 h 21 min 10 s）

所有成功运行的结果均可在 notebook 中查看