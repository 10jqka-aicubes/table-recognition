# 【2-4春季赛】文档图片表格结构识别算法

​	表格作为一种高效的数据组织与展现方法被广泛应用，已成为各类文档中最常见的页面对象。目前很大一部分文档以图片的形式存在，无法直接获取表格信息。人工还原表格既费时又容易出错，因此如何自动并准确地从文档图片中识别出表格成为一个亟待解决的问题。但由于表格大小、种类与样式的复杂多样（例如表格中存在不同的背景填充、不同的行列合并方法、不同的分割线类型等），导致表格识别一直是文档识别领域的研究难点。

- 本代码是该赛题的一个基础demo，仅供参考学习。


- 比赛地址：http://contest.aicubes.cn/	


- 时间：2022-02 ~ 2022-04



## 如何运行Demo

- clone代码


- 准备其他模型

  - 下载模型： [官网](http://contest.aicubes.cn/#/detail?topicId=51) -> 赛题与数据 -> 文本检测demo演示用模型.zip
  - 解压后将文件放在一个目录，预测时使用

- 准备环境

  - cuda10.0以上
  - python3.7以上
  - 安装python依赖

  ```
  python -m pip install -r requirements.txt
  ```

- 准备数据，从[官网](http://contest.aicubes.cn/#/detail?topicId=51)下载数据

  - 解压`train.zip`后将`gt/`和`imgs/`两个目录放在训练数据目录中
  - 解压`test_a.zip` 后将`gt/`和`imgs/`两个目录放在预测目录下

- 调整参数配置，参考[模板项目](https://github.com/10jqka-aicubes/project-demo)的说明

  - `table_recognition/setting.conf`
  - 其他注意下`run.sh`里使用的参数，预测代码中需要指定第二步下载的模型

- 运行

  - 训练

  ```
  bash table_recognition/train/run.sh
  ```

  - 预测

  ```
  bash table_recognition/predict/run.sh
  ```

  - 计算结果指标

  ```
  bash table_recognition/metrics/run.sh
  ```



## 反作弊声明

1）参与者不允许在比赛中抄袭他人作品、使用多个小号，经发现将取消成绩；

2）参与者禁止在指定考核技术能力的范围外利用规则漏洞或技术漏洞等途径提高成绩排名，经发现将取消成绩；

3）在A榜中，若主办方认为排行榜成绩异常，需要参赛队伍配合给出可复现的代码。



## 赛事交流

![同花顺比赛小助手](http://speech.10jqka.com.cn/arthmetic_operation/245984a4c8b34111a79a5151d5cd6024/客服微信.JPEG)