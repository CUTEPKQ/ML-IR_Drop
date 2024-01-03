# ML-IR_Drop
第五届EDA设计精英挑战赛——赛题六（[基于机器学习的Soc静态压降预测](https://eda.icisc.cn/download/index?type=2)）
## 任务描述
![image](https://github.com/CUTEPKQ/ML-IR_Drop/assets/126258180/7548fc91-4861-4014-ba0f-a6a3b785b479#pic_center)
通过将输入的文件送入到以golden data为监督训的ML 回归模型，得到最后的IR_Drop report.
## 总体解决思路
- From image to image
![image](https://github.com/CUTEPKQ/ML-IR_Drop/assets/126258180/e259b35c-bca9-4324-94b7-971115ce5587#pic_center)
## 实现方案

#### 框架图
![image](https://github.com/CUTEPKQ/ML-IR_Drop/assets/126258180/e7d6d00f-57bc-4b4d-9e7e-add901363a1f#pic_center)
#### 特征提取
- Min path res (暂未提取）
- Static_IR(抽取网格averge ir_drop)
![image](https://github.com/CUTEPKQ/ML-IR_Drop/assets/126258180/7cb91051-af72-4e50-b4f3-ce005019f94b#pic_center)
- Power 抽取total power(在final test中提取name和count)
![image](https://github.com/CUTEPKQ/ML-IR_Drop/assets/126258180/034582b6-6670-474f-86ef-6e36b28a9dda#pic_center)
- Effective Res 提取vdd_r + gnd_r
![image](https://github.com/CUTEPKQ/ML-IR_Drop/assets/126258180/0987e31e-083e-4998-bfa1-dacd5262e25a#pic_center)
#### 模型训练
- 模型1：FCN模型 ([参考repo](https://github.com/circuitnet/CircuitNet/tree/icisc_2023))
- 模型2：ConvNexV2 + FCN head
##### Trick
* Flost64
* SmoothL1 Loss
#### 模型预测
![image](https://github.com/CUTEPKQ/ML-IR_Drop/assets/126258180/8462bc70-5b9c-4a5d-8367-2f80c09e6fca)

**说明**
代码实现主要基于[circuitnet](https://github.com/circuitnet/CircuitNet/tree/icisc_2023)


