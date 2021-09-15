

## 评价指标计算规则

a. 评价指标

1）单元格F1



b. 评价指标计算流程

1）计算预测的表格内的单元格的相邻关系，假设所有预测的表格内的单元格相邻关系的总数为S

参考论文：

《A Methodology for Evaluating Algorithms for Table Understanding in PDF Documents》

2）将预测的表格区域与ground truth中的表格区域进行映射，取IOU大于0.8的表格区域作为预测正确的表格区域，其余作为预测错误的表格区域

3）在正确的表格区域内，将预测的单元格内的文本区域与ground truth中的单元格文本区域进行映射，取IOU的阈值为0.5和0.6；当单元格内存在多行文本时，所有文本的最小外包矩形作为文本区域，参考下图：

![](../../docs/图片3.jpg)

4）计算单元格相邻关系的precision和recall

注：计算precision时，预测错误的表格区域也要统计在内，即分母为S

参考下图：

![](../../docs/图片1.png)



5）计算单元格F1 = （2 *recall * precision）/ (precision + recall)

6）计算平均F1值， 例如IOU=0.5时，F1值为95%，IOU=0.6时，F1值为90%，则最终的平均F1值为（95% + 90%）/ 2 = 92.5%

7）IOU计算公式：IOU = Area of overlap / Area of Union，即两个区域的交集除以并集

8）参考论文：

《ICDAR 2019 Competition on Table Detection and Recognition (cTDaR)》

《ICDAR 2013 Table Competition》
