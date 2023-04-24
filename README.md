这是本人研究生期间关于轨迹修复与异常检测的开源代码、数据以及相关资料的记录，希望能给后来研究这两个方向的人提供一些帮助，减少一些检索资料的时间

# 轨迹修复

## 相关研究时间线

轨迹修复的研究方法分为两种，基于插值算法的修复和基于深度学习算法的修复，其中插值算法修复比较早，多数用于船舶领域，深度学习算法较晚，主要用于城市中车辆轨迹修复。了解了原理之后就很容易理解这两种算法都有其自己试用的场景，船舶航行通常有固定的航道，而且转向不会太剧烈，轨迹比较平滑，比较适合用一个分段函数去拟合剩余的轨迹点，进而推测出缺失点的位置，但是城市中的车辆轨迹比较复杂，轨迹是不平滑的，无法做到像船舶那样修复，目前的修复研究都是粗粒度的，需要把轨迹点转换成为网格编号，用一组网格编号表示车辆的运动，用序列模型来学习网格编号之间的相关性特征，最后推测缺失位置所在每个网格区域的概率。

### 向量插值

由于向量插值属于比较老的方法，所以基于向量插值的船舶轨迹修复相关论文都没有开源代码也就没有细读的必要，但是相关的向量插值算法基本都被做成工具包了，可以通过python轻易的调用，下面介绍几种常用的插值算法，通过百度或者谷歌应该很容易找到对应的包，例如scipy包

1. 线性插值
2. 多项式插值
3. Hermite插值

最后这篇论文是目前我找到的最新的用于船舶轨迹修复且发在CCF-A类期刊上的论文，可以放在论文中引用

*秦红星,杨茜.改进线性插值的船舶轨迹修复迭代算法[J].计算机辅助设计与图形学学报,2019,31(10):1759-1767.*（期刊论文）

*杨茜. 向量插值在轨迹修复中的算法研究[D].重庆邮电大学,2019.DOI:10.27675/d.cnki.gcydx.2019.000837.*（学位论文）

### 深度学习

基于深度学习的轨迹修复研究是从兴趣点推荐发展起来的，兴趣点推荐就是根据用于的历史轨迹推荐下一步可能去的地点，而轨迹修复是基于历史轨迹推测缺失的轨迹点所在的位置

1. 这篇论文是基于循环神经网络预测用户轨迹，算是轨迹修复领域的最早的一篇纹章了

论文 [Deepmove: Predicting human mobility with attentional recurrent networks](https://dl.acm.org/doi/abs/10.1145/3178876.3186058)

代码 https://github.com/vonfeng/DeepMove

2. 这篇论文算是在deepmove模型的基础上进行的研究，在实验结果部分对比的就是deepmove模型

论文 [Attnmove: History enhanced trajectory recovery via attentional network](https://ojs.aaai.org/index.php/AAAI/article/view/16577)

代码 https://github.com/anonymous1833/AttnMove

3. 这篇论文算是在attnmove模型的基础上进行的研究，在实验结果部分对比的就是attnmove模型，**但是经过仔细研究发现其模型部分缺少了自注意力网络，在作者的学位论文部分又给加上了**

会议论文 [Periodicmove: Shift-aware human mobility recovery with graph neural network](https://dl.acm.org/doi/abs/10.1145/3459637.3482284)

学位论文 *周凡. 基于深度学习的人类移动轨迹补全技术[D].电子科技大学,2022.DOI:10.27005/d.cnki.gdzku.2022.001508.*

代码 https://github.com/mcdragon/PeriodicMove **推荐从此代码开始研究，该代码提供了一部分数据，使用的是Pytroch框架，文件结构比较合理，较为容易复现**

上面三篇论文属于一个系列的，所提出的模型都是在前一个模型上的改进

4. 这篇论文是基于多任务学习的，但是数据给的不全，代码复现也比较困难，在轨迹数据处理部分不像之前的论文映射到网格进行处理，作者的一个创新点使用道路编号加上一个百分比表示此轨迹点在这条道路的百分之几位置处，最终有两个预测任务，需要预测道路编号+百分比。此论文参考价值不大，作者所在的公司有路网资源，其他人很难拿到这种数据。

论文  [Mtrajrec: Map-constrained trajectory recovery via seq2seq multi-task learning](https://dl.acm.org/doi/abs/10.1145/3447548.3467238)

代码 https://github.com/huiminren/MTrajRec

5. 这篇论文和论文4是同一个团队出的，仍然很难复现。同样是结合了路网来进行轨迹修复

论文 [RNTrajRec: Road Network Enhanced Trajectory Recovery with Spatial-Temporal Transformer](https://arxiv.org/abs/2211.13234)

代码 https://github.com/chenyuqi990215/RNTrajRec

6. 这篇论文结合了注意力网络和卡尔曼滤波来轨迹修复，没有提供代码也没有数据，本人能力有限，啥都没看懂

论文 [Deep trajectory recovery with fine-grained calibration using kalman filter](https://ieeexplore.ieee.org/abstract/document/8834829/)

## 本人研究思路

本人同样也是做的城市车辆轨迹修复，而且也用的网格映射的预处理方法，基于PeriodicMove这个模型基础进行的研究，对这个模型做出了以下三点改进

1. 加上了原作者漏掉的自注意力网络
2. 在注意力网络部分结合了DTW相似度来进一步提取历史轨迹相似特征
3. 参考了RoBert模型中的动态mask机制给轨迹修复的训练阶段也引入了动态mask

动态mask参考 https://zhuanlan.zhihu.com/p/360982134

具体的可以看PeriodicMove论文和本仓库的代码

## 开源数据集

1. 微软开源的北京市出租车数据集，包含了大约10000辆出租车一周的轨迹，这个数据集被多个论文使用过

https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/

2. 北京、深圳、成都的出租车数据集，包含了这三个城市大约一个月的轨迹数据，每个城市的样本大约都在10000辆左右

https://github.com/cbdog94/STL

对应的会议论文[STL: Online detection of taxi trajectory anomaly based on spatial-temporal laws](https://link.springer.com/chapter/10.1007/978-3-030-18579-4_45)

对应的学位论文 *程彬. 基于轨迹数据的异常行为实时检测方法研究[D].上海交通大学,2020.DOI:10.27307/d.cnki.gsjtu.2020.001918.*

3. 巴西里约公交车数据集，来自kaggle竞赛，包含了大约5000辆公交车两个月的轨迹数据

https://www.kaggle.com/datasets/igorbalteiro/gps-data-from-rio-de-janeiro-buses



# 轨迹异常检测

## 相关研究时间线

轨迹异常检测最核心的问题时定义**异常**，但是目前没有一个公认的标准和数据集给出什么算是异常轨迹，大多数论文实际做的是出租车绕路检测。由于这个领域没有一个标准的数据集，所以论文之间几乎没有相关性。多数论文都用的是自己的私有的数据集，并且自己来定义或者人工标注什么是**异常**，少数论文将其公开。异常检测的方法总体分为两种，传统算法和深度学习方法。在进行出租车绕路检测时，考虑到数据集基本没有大量已经标注的数据，所以这必定是一个无监督学习的任务，在无监督检测方面，目前深度学习相关的研究很少，而且本人也都没有复现出现应有的效果。

1. 这是一篇软件学报上的综述，基本总结了2017年之前的异常轨迹研究，总结的非常好。参考文献比较老了，但是可以了解下17年之前怎么做异常检测研究的

毛嘉莉, 金澈清, 章志刚, 周傲英. 轨迹大数据异常检测:研究进展及系统框架[J]. 软件学报, 2017, 28(1): 17-34.http://www.jos.org.cn/1000-9825/5151.htm    

2. 这是一个基于RNN模型做的有监督的异常检测，公开了代码和数据集，实际使用价值不大，但是可以参考

论文 [Embedding geographic information for anomalous trajectory detection](https://link.springer.com/article/10.1007/s11280-020-00812-z)

代码 https://github.com/LeeSongt/ATD-RNN

3. 这时一个基于自编码网络做的异常检测研究，虽然发表在A类会议上，但是代码用的tensorflow1.x，代码写的很乱，本人没有跑通

论文 [Online anomalous trajectory detection with deep generative sequence modeling](https://ieeexplore.ieee.org/abstract/document/9101353/)

代码 https://github.com/liuyiding1993/ICDE2020_GMVSAE

4. 这是一个基于交叉集相似度做的异常检测，典型的传统方法，代码很简单，论文也写得比较通俗易懂，可以优先学习下

论文 [Anomalous trajectory detection and classification based on difference and intersection set distance](https://ieeexplore.ieee.org/abstract/document/8963673/)

代码 https://github.com/networkanddatasciencelab/ATDC

5. 这是一个基于极坐标特征和聚类算法做的异常检测，代码和论文都比较简单，可以学习下

论文 [MiPo: How to Detect Trajectory Outliers with Tabular Outlier Detectors](https://www.mdpi.com/2072-4292/14/21/5394)

代码 https://github.com/TimeIsAFriend/trajectory_MiPo

## 本人研究思路

本人在深度学习方面能力有限（这方面容易写论文），没有在无监督的条件下使用深度学习模型做出一点点效果，因此还是使用的传统算法。异常检测的核心是找出少而不同的轨迹，那么最直接的思路是计算当前轨迹与所有轨迹的相似度，当它与大多数轨迹都不相似时就是异常轨迹，考虑到轨迹序列不对齐的问题采用了DTW相似度，此外遍历当前轨迹与所有轨迹计算相似度，尤其是对每一条轨迹都这么做时，时间复杂会很高，通过DBA算法生成平均轨迹优化了时间复杂度。

## 开源数据集

1. 美国旧金山湾区从机场到居民区的轨迹数据，此数据集已经被预处理过，抽取出了其中六条路线，作者未提供原始数据

https://github.com/networkanddatasciencelab/ATDC

相关的论文 [Anomalous trajectory detection and classification based on difference and intersection set distance](https://ieeexplore.ieee.org/abstract/document/8963673/)

2. 葡萄牙波尔图市轨迹数据集，此数据集已经被预处理过，作者未给出预处理的程序

https://github.com/TimeIsAFriend/trajectory_MiPo

相关的论文 [MiPo: How to Detect Trajectory Outliers with Tabular Outlier Detectors](https://www.mdpi.com/2072-4292/14/21/5394)

原始数据集 https://www.kaggle.com/datasets/crailtap/taxi-trajectory

3. 葡萄牙波尔图市轨迹数据集，此数据集已经被预处理过，作者给出了预处理的程序

https://github.com/LeeSongt/ATD-RNN

相关的论文 [Anomalous trajectory detection using recurrent neural network](https://link.springer.com/chapter/10.1007/978-3-030-05090-0_23)

对应的学位论文 *宋礼. 出租车轨迹异常检测可视分析系统的研究与实现[D].北京邮电大学,2019.*

原始数据集 https://www.kaggle.com/datasets/crailtap/taxi-trajectory



# 总结

所有的研究最重要的就是有**数据**，当一篇论文开源了代码和数据时就值得一看（无论是什么级别的期刊或者会议），但是也要注意有的论文只开源了部分代码和部分数据，导致无法复现以及弄懂作者模型的细节，所以在本人毕业之后总结了这样一个关于**轨迹修复与轨迹异常检测**的文档希望能够后人节约点时间。

具体到这两个研究领域，相关的研究算是不温不火，不像NLP或者CV领域随便都是经典的论文、开源代码、公开的标准数据集。这算是有利有弊吧，不好的就是相关资源较少上手比较困难，好的就是竞争没有那么激烈，可以从NLP领域或者CV领域借鉴点东西过来就有了创新点，毕竟处理的都是序列数据。没有大公司来这个领域研究说明这个研究方向大概率也是学术圈的自嗨，实际的应用价值不高，当然一般车辆轨迹模型必然要结合路网来做，只有大公司有这样的资源，能够结合路网做出有一定实际价值的研究。下面说下我的关于这两个研究方向的想法，但是限于资源没有条件做的。

轨迹修复可以参考NLP领域的大模型，先用全国的路网数据来对模型进行训练，然后对于不同城市，使用城市中心的车辆轨迹数据对模型进行微调

轨迹异常检测必要要做成无监督的，可以使用GAN网络学习正常的车辆行驶轨迹分布，当输入异常的轨迹时，生成的轨迹会偏离正常的分布，就可以检测出来。

