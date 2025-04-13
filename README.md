
# dl_hw1 三层卷积神经网络分类器 - CIFAR10图像分类

从零开始实现的三层卷积神经网络分类器，基于NumPy手动实现反向传播，在CIFAR-10数据集上达到67.21%测试准确率。

---

## 目录
- [项目简介](#项目简介)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [实验结果](#实验结果)
- [如何训练与测试](#如何训练与测试)
- [模型权重下载](#模型权重下载)
- [许可证](#许可证)

---

## 项目简介

本项目手动实现包含卷积层和全连接层的三层神经网络，核心特性包括：

- **纯NumPy实现**：手动推导反向传播，禁用自动微分框架  
- **完整训练流程**：SGD优化器、学习率衰减、交叉熵损失、L2正则化
- **贝叶斯超参数优化**：搜索卷积通道数、隐藏层大小等关键参数
- **模块化设计**：分离模型、训练、测试与参数搜索模块

**数据集**：CIFAR-10（10类，32x32彩色图像）  
**最终测试准确率**：67.21%  
**最高验证准确率**：67.98%（12个训练周期）

---

## 模型架构  

$$
  \text{Input(3072)} \rightarrow \text{Conv(3×3, k=83)} \rightarrow \text{MaxPool(2×2)} \rightarrow \text{FC(799)} \rightarrow \text{ReLU} \rightarrow \text{Output(10)}
$$

- **输入层**：32x32x3 = 3072维
- **卷积层**：3x3卷积核，通道数83，批归一化，步长1，填充2
- **池化层**：2x2最大池化，步长2
- **全连接层**：隐藏单元799，ReLU激活
- **输出层**：10维Softmax输出

**超参数配置**：

| 参数                | 值                  |
|---------------------|---------------------|
| 初始学习率          | $0.09702  $          |
| 学习率衰减周期      | $10 $ epochs          |
| 批量大小            | $128 $               |
| L2正则化系数        | $3.038×10^{-5}$   |
| 训练周期            | $12$                 |

---

## 项目结构  

```bash  
.
├── model_train.py          # 神经网络架构以及训练和调参
├── visualize.py          # 模型权重可视化
```

## 实验结果  

### 训练过程指标

- **训练损失**：从 $1.0443$ 降至 $0.0938$
- **验证损失**：在 $1.10-1.25$ 间波动
- **验证准确率**：从 $59.04\%$ 提升至 $67.98\%$

### 超参数搜索结果（3个训练周期）

| 验证准确率 | 卷积通道 | 隐藏层大小 | L2系数      | 学习率   |
|------------|----------|------------|-------------|----------|
| $64.18\%$     | $83 $      | $799$        | $3.038×10^{-5}$  | $0.09702$  |

---

## 如何训练与测试

### 如何训练
如果要贝叶斯优化进行调参，可以这样。
```python
from model_train import *
import pickle
import os
# 加载数据
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "cifar-10-python")
X_train, y_train, X_test, y_test = load_cifar10(data_dir)
X_train, X_test = preprocess_data(X_train, X_test)

# 划分验证集
val_ratio = 0.1
split_idx = int(len(X_train) * (1 - val_ratio))
X_val, y_val = X_train[split_idx:], y_train[split_idx:]
X_train, y_train = X_train[:split_idx], y_train[:split_idx]

# 贝叶斯优化搜索参数
best_params = bayesian_optimization(X_train, y_train, X_val, y_val,
                                    init_points=3, n_iter=3)

# 用最佳参数训练最终模型
final_model = Model(
    conv_channels=int(best_params['conv_channels']),
    hidden_size=int(best_params['hidden_size']),
    l2_reg=best_params['l2_reg']
)


# 完整训练
final_trainer = Trainer(
    final_model, X_train, y_train, X_val, y_val,
    lr=best_params['lr'],
    batch_size=128,
    l2_reg=best_params['l2_reg']
)
final_trainer.run(epochs=12)
with open(os.path.join(current_dir, str(time.time()) + "besthy.pkl"), "wb") as f:
    pickle.dump(final_model, f)
# 绘制训练曲线
plot_training_curve(final_trainer, save_path=str(time.time()) + 'lacurve.pdf')

# 最终测试评估
test_acc = final_trainer.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
```

如果要直接训练可以这样。

```python
# 加载数据
params = {'conv_channels':32,
            'hidden_size':512,  
            'l2_reg': 0.0001,
            'lr': 0.01}

# 模型
final_model = Model(
    conv_channels=int(params['conv_channels']),
    hidden_size=int(params['hidden_size']),
    l2_reg=params['l2_reg']
)

# 完整训练
final_trainer = Trainer(
    final_model, X_train, y_train, X_val, y_val,
    lr=params['lr'],
    batch_size=128,
    l2_reg=params['l2_reg']
)
final_trainer.run(epochs=12)
with open(os.path.join(current_dir, str(time.time()) + "besthy.pkl"), "wb") as f:
    pickle.dump(final_model, f)
# 绘制训练曲线
plot_training_curve(final_trainer, save_path=str(time.time()) + 'lacurve.pdf')

# 最终测试评估
test_acc = final_trainer.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
```

### 如何测试

根据保存的模型的名称读取后测试。

```python
import numpy
file_name = "1744411485.2150824besthy.pkl"

# 使用二进制读取模式打开文件
with open(os.path.join(current_dir, file_name), 'rb') as f:
    model = pickle.load(f)
logits = model.forward(X_test, training=False)
preds = np.argmax(logits, axis=1)
print("测试集上准确率：", np.mean(preds == y_test)) 
```

## 模型权重下载

训练好的模型权重已上传至百度云：

- **链接**：`https://pan.baidu.com/s/11tFJKlFpcsWrI4DIJeZUug?pwd=6wph`  
