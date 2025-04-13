import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

# 获得文件所在目录
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)


# 数据预处理函数
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data'].astype(np.float32)
    y = np.array(dict[b'labels'])
    return X, y


def load_cifar10(data_dir):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        X, y = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(X)
        train_labels.append(y)

    X_train = np.concatenate(train_data)
    y_train = np.concatenate(train_labels)
    X_test, y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return X_train, y_train, X_test, y_test


# 预处理，归一化
def preprocess_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train.reshape(-1, 3, 32, 32), X_test.reshape(-1, 3, 32, 32)


# 工具函数
# 交叉熵
def cross_entropy_loss(logits, y_true, l2_reg, params):
    probs = softmax(logits)
    loss = -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-8))

    # L2正则化
    l2_loss = 0.0
    for p in params:
        if p.ndim >= 2:  # 只计算权重矩阵的L2
            l2_loss += np.sum(p ** 2)
    loss += 0.5 * l2_reg * l2_loss
    return loss


# im to col 函数，卷积层需要
def im2col(images, kernel_size, stride, pad):
    N, C, H, W = images.shape
    out_h = (H + 2 * pad - kernel_size) // stride + 1
    out_w = (W + 2 * pad - kernel_size) // stride + 1

    img_padded = np.pad(images, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    col = np.zeros((N, C, kernel_size, kernel_size, out_h, out_w))

    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img_padded[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


# col to im 函数，也是卷积层需要
def col2im(col, input_shape, kernel_size, stride, pad):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - kernel_size) // stride + 1
    out_w = (W + 2 * pad - kernel_size) // stride + 1
    col = col.reshape(N, out_h, out_w, C, kernel_size, kernel_size).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


# one-hot 编码
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]


# 激活函数 RELU
def relu(x):
    return np.maximum(0, x), x


# RELU 的后向
def relu_backward(dout, cache):
    dx = dout.copy()
    dx[cache <= 0] = 0
    return dx


# softmax 函数
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


# 卷积层
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, pad=2,
                 activation='relu', use_bn=True, l2_reg=0.0):
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(
            2.0 / (in_channels * kernel_size ** 2))
        self.b = np.zeros(out_channels)
        self.stride = stride
        self.pad = pad
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bn = use_bn
        self.l2_reg = l2_reg

        # 是否使用 batch normalization
        if use_bn:
            self.bn_gamma = np.ones(out_channels)
            self.bn_beta = np.zeros(out_channels)
            self.bn_running_mean = np.zeros(out_channels)
            self.bn_running_var = np.zeros(out_channels)

        self.cache = None

    def batchnorm_forward(self, x, training):
        if training:
            mu = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)
            self.bn_running_mean = 0.9 * self.bn_running_mean + 0.1 * mu.squeeze()
            self.bn_running_var = 0.9 * self.bn_running_var + 0.1 * var.squeeze()
        else:
            mu = self.bn_running_mean.reshape(1, -1, 1, 1)
            var = self.bn_running_var.reshape(1, -1, 1, 1)

        x_hat = (x - mu) / np.sqrt(var + 1e-5)
        out = self.bn_gamma.reshape(1, -1, 1, 1) * x_hat + self.bn_beta.reshape(1, -1, 1, 1)
        return out, (x, x_hat, mu, var)

    def batchnorm_backward(self, dout, cache):
        x, x_hat, mu, var = cache
        N, C, H, W = x.shape

        dgamma = np.sum(dout * x_hat, axis=(0, 2, 3))
        dbeta = np.sum(dout, axis=(0, 2, 3))

        dx_hat = dout * self.bn_gamma.reshape(1, -1, 1, 1)
        dvar = np.sum(dx_hat * (x - mu) * (-0.5) * (var + 1e-5) ** (-1.5), axis=(0, 2, 3))
        dmu = np.sum(dx_hat * (-1 / np.sqrt(var + 1e-5)), axis=(0, 2, 3)) + dvar * np.mean(-2 * (x - mu),
                                                                                           axis=(0, 2, 3))

        dx = dx_hat / np.sqrt(var + 1e-5)
        dx += dvar.reshape(1, -1, 1, 1) * 2 * (x - mu) / (C * H * W)
        dx += dmu.reshape(1, -1, 1, 1) / (C * H * W)

        return dx, dgamma, dbeta

    def forward(self, x, training=True):
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.kernel_size) // self.stride + 1

        col = im2col(x, self.kernel_size, self.stride, self.pad)
        col_W = self.W.reshape(self.W.shape[0], -1).T

        out = col @ col_W + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        if self.use_bn:
            out, bn_cache = self.batchnorm_forward(out, training)
        else:
            bn_cache = None

        if self.activation == 'relu':
            out, act_cache = relu(out)
        else:
            act_cache = None

        self.cache = (x, col, col_W, bn_cache, act_cache)
        return out

    def backward(self, dout):
        x, col, col_W, bn_cache, act_cache = self.cache
        N, C, H, W = x.shape

        if act_cache is not None:
            dout = relu_backward(dout, act_cache)

        if self.use_bn:
            dout, dgamma, dbeta = self.batchnorm_backward(dout, bn_cache)

        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.W.shape[0])

        dcol = dout @ col_W.T
        dW = dout.T @ col
        db = np.sum(dout, axis=0)

        dx = col2im(dcol, x.shape, self.kernel_size, self.stride, self.pad)
        dW = dW.T.reshape(self.W.shape)

        dW += self.l2_reg * self.W
        db += self.l2_reg * self.b

        self.grads = [dW, db]
        if self.use_bn:
            self.grads += [dgamma, dbeta]

        return dx


# 最大池，池化层
class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        col = im2col(x.reshape(N * C, 1, H, W), self.pool_size, self.stride, 0)
        col = col.reshape(-1, self.pool_size ** 2)

        max_idx = np.argmax(col, axis=1)
        out = col[np.arange(col.shape[0]), max_idx]
        out = out.reshape(N, C, out_h, out_w)

        self.cache = (x.shape, max_idx)
        return out

    def backward(self, dout):
        orig_shape, max_idx = self.cache
        N, C, H, W = orig_shape

        dout_flat = dout.transpose(0, 2, 3, 1).flatten()
        dcol = np.zeros((dout_flat.size, self.pool_size ** 2))
        dcol[np.arange(dout_flat.size), max_idx] = dout_flat

        dx = col2im(dcol, (N * C, 1, H, W), self.pool_size, self.stride, 0)
        dx = dx.reshape(orig_shape)
        return dx


# 全连接层
class FCLayer:
    def __init__(self, input_size, output_size, activation='relu', use_bn=True, l2_reg=0.0):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)
        self.activation = activation
        self.use_bn = use_bn
        self.l2_reg = l2_reg

        if use_bn:
            self.bn_gamma = np.ones(output_size)
            self.bn_beta = np.zeros(output_size)
            self.bn_running_mean = np.zeros(output_size)
            self.bn_running_var = np.zeros(output_size)

        self.cache = None

    def batchnorm_forward(self, x, training):
        if training:
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.bn_running_mean = 0.9 * self.bn_running_mean + 0.1 * mu
            self.bn_running_var = 0.9 * self.bn_running_var + 0.1 * var
        else:
            mu = self.bn_running_mean
            var = self.bn_running_var

        x_hat = (x - mu) / np.sqrt(var + 1e-5)
        out = self.bn_gamma * x_hat + self.bn_beta
        return out, (x, x_hat, mu, var)

    def batchnorm_backward(self, dout, cache):
        x, x_hat, mu, var = cache
        N = x.shape[0]

        dgamma = np.sum(dout * x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)

        dx_hat = dout * self.bn_gamma
        dvar = np.sum(dx_hat * (x - mu) * (-0.5) * (var + 1e-5) ** (-1.5), axis=0)
        dmu = np.sum(dx_hat * (-1 / np.sqrt(var + 1e-5)), axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)

        dx = dx_hat / np.sqrt(var + 1e-5)
        dx += dvar * 2 * (x - mu) / N
        dx += dmu / N

        return dx, dgamma, dbeta

    def forward(self, x, training=True):
        out = x @ self.W + self.b

        if self.use_bn:
            out, bn_cache = self.batchnorm_forward(out, training)
        else:
            bn_cache = None

        if self.activation == 'relu':
            out, act_cache = relu(out)
        else:
            act_cache = None

        self.cache = (x, bn_cache, act_cache)
        return out

    def backward(self, dout):
        x, bn_cache, act_cache = self.cache

        if act_cache is not None:
            dout = relu_backward(dout, act_cache)

        if self.use_bn:
            dout, dgamma, dbeta = self.batchnorm_backward(dout, bn_cache)

        dx = dout @ self.W.T
        dW = x.T @ dout
        db = np.sum(dout, axis=0)

        dW += self.l2_reg * self.W
        db += self.l2_reg * self.b

        self.grads = [dW, db]
        if self.use_bn:
            self.grads += [dgamma, dbeta]

        return dx


# 神经网络模型
class Model:
    def __init__(self, conv_channels=32, hidden_size=512, activation='relu', use_bn=True, l2_reg=0.0):
        self.conv = ConvLayer(3, conv_channels, activation=activation, use_bn=use_bn, l2_reg=l2_reg)
        self.pool = MaxPool2D()
        self.fc = FCLayer(conv_channels * 16 * 16, hidden_size, activation=activation, use_bn=use_bn, l2_reg=l2_reg)
        self.fc_out = FCLayer(hidden_size, 10, activation=None, use_bn=False, l2_reg=l2_reg)

    def forward(self, x, training=True):
        x = self.conv.forward(x, training)
        x = self.pool.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc.forward(x, training)
        x = self.fc_out.forward(x, training)
        return x

    def backward(self, dout):
        dout = self.fc_out.backward(dout)
        dout = self.fc.backward(dout)
        dout = dout.reshape(dout.shape[0], -1, 16, 16)
        dout = self.pool.backward(dout)
        dout = self.conv.backward(dout)
        return dout

    def get_params(self):
        params = []
        params.extend([self.conv.W, self.conv.b])
        if self.conv.use_bn:
            params.extend([self.conv.bn_gamma, self.conv.bn_beta])
        params.extend([self.fc.W, self.fc.b])
        if self.fc.use_bn:
            params.extend([self.fc.bn_gamma, self.fc.bn_beta])
        params.extend([self.fc_out.W, self.fc_out.b])
        return params

    def get_grads(self):
        grads = []
        grads.extend(self.conv.grads)
        grads.extend(self.fc.grads)
        grads.extend(self.fc_out.grads)
        return grads


class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val,
                 lr=0.1, batch_size=128, lr_decay=0.5, decay_epochs=10, l2_reg=0.0):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.lr = lr
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.decay_epochs = decay_epochs
        self.l2_reg = l2_reg

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_params = None

    def train_epoch(self, epoch):
        # 学习率衰减
        if (epoch + 1) % self.decay_epochs == 0:
            self.lr *= self.lr_decay

        # 打乱数据
        permutation = np.random.permutation(len(self.X_train))
        self.X_train = self.X_train[permutation]
        self.y_train = self.y_train[permutation]

        # 训练
        for i in range(0, len(self.X_train), self.batch_size):
            X_batch = self.X_train[i:i + self.batch_size]
            y_batch = self.y_train[i:i + self.batch_size]

            logits = self.model.forward(X_batch)
            loss = cross_entropy_loss(logits, y_batch, self.l2_reg, self.model.get_params())

            probs = softmax(logits)
            dout = (probs - one_hot(y_batch)) / self.batch_size
            self.model.backward(dout)

            params = self.model.get_params()
            grads = self.model.get_grads()
            for p, g in zip(params, grads):
                p -= self.lr * g

    def evaluate(self, X, y):
        logits = self.model.forward(X, training=False)
        preds = np.argmax(logits, axis=1)
        return np.mean(preds == y)

    def run(self, epochs=10):
        for epoch in range(epochs):
            self.train_epoch(epoch)

            train_acc = self.evaluate(self.X_train, self.y_train)
            val_acc = self.evaluate(self.X_val, self.y_val)
            train_loss = cross_entropy_loss(
                self.model.forward(self.X_train, training=False),
                self.y_train, self.l2_reg, self.model.get_params()
            )
            val_loss = cross_entropy_loss(
                self.model.forward(self.X_val, training=False),
                self.y_val, self.l2_reg, self.model.get_params()
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = [p.copy() for p in self.model.get_params()]

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
                  f"Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

        return self.best_val_acc


def bayesian_optimization(X_train, y_train, X_val, y_val, init_points=5, n_iter=10):
    # 定义参数空间
    pbounds = {
        'conv_channels': (16, 128),
        'hidden_size': (256, 1024),
        'lr': (0.001, 0.1),
        # 'batch_size': (64, 256),
        'l2_reg': (1e-5, 1e-3)
    }

    best_acc = 0.0
    best_params = None

    def objective(conv_channels, hidden_size, lr, l2_reg):
        nonlocal best_acc, best_params

        # 转换整数参数
        conv_channels = int(round(conv_channels))
        hidden_size = int(round(hidden_size))
        # batch_size = int(round(batch_size))

        # 创建模型和训练器
        model = Model(conv_channels=conv_channels,
                      hidden_size=hidden_size,
                      l2_reg=l2_reg)
        trainer = Trainer(model, X_train, y_train, X_val, y_val,
                          lr=lr, l2_reg=l2_reg)

        # 短期训练快速验证
        current_acc = trainer.run(epochs=3)

        # 保存最佳参数
        if current_acc > best_acc:
            best_acc = current_acc
            best_params = {
                'conv_channels': conv_channels,
                'hidden_size': hidden_size,
                'lr': lr,
                # 'batch_size': batch_size,
                'l2_reg': l2_reg
            }

        return current_acc

    # 运行贝叶斯优化
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )

    return best_params


def plot_training_curve(trainer, save_path='training_curve.png'):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(trainer.train_accs, label='Train Acc')
    plt.plot(trainer.val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, save_path))
    plt.close()


def main():
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


if __name__ == "__main__":
    time1 = time.time()
    main()
    print(time.time() - time1, "消耗时间")
