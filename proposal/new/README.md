## 20 Newsgroup Document Classification

整理未完成，已经整理的部分[请看这里](./notebooks/ExperimentRecords-version-Full.ipynb)。简要说明如下：

1. 旧有的实验结果保存在文件夹 `notebooks` 下
2. `notebook` 下旧有的实验结果，即以 `trial_` 开头的 `.ipynb` 文件，需要进行进一步修改，原因是：原有的这些 `.ipynb` 文件是放在本层，即与文件夹 `data`、`models`、`modules` 平行，同属一个根目录；现在这些 `.ipynb` 文件与上述这些文件夹的子结点才是平行的，因此读取数据、保存数据的路径应稍作更改
    + 迁移到当前实验报告中的部分都已经完成文件夹读写路径的更改工作。尚未迁移的部分还有：
        - [trial_Doc2Vec](./notebooks/trial_Doc2Vec.ipynb)
3. 要获取`text8`，请从根目录取得，即其相对于本级目录的位置：`../../text8`
  + 或通过执行 `sh ../../getText8.sh` 获取（执行该脚本前，需要拥有 `wikifil.pl` 脚本，详情请见 [About the Test Data](http://mattmahoney.net/dc/textdata)）
  + 若有某（些） `.ipynb` 需要 `text8.zip`，请通过自行压缩该文件获得

