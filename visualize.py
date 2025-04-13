import matplotlib.pyplot as plt
import os
import time
import matplotlib.pyplot as plt
import pickle
from model_train import *

# 获得文件所在目录
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)


def visualize_model(model, save_dir='./'):
    conv_weights = model.conv.W
    print("Convolutional layer weights shape:", conv_weights.shape)

    conv_weights = (conv_weights - conv_weights.min()) / (conv_weights.max() - conv_weights.min())

    n_filters = 16
    fig = plt.figure(figsize=(10, 8))
    for i in range(n_filters):
        ax = fig.add_subplot(4, 4, i + 1)
        filter_rgb = conv_weights[i].transpose(1, 2, 0)
        ax.imshow(filter_rgb)
        ax.axis('off')
        ax.set_title(f'Filter {i + 1}')
    plt.suptitle('First 16 Conv Filters (3 input channels as RGB)')
    plt.savefig(os.path.join(save_dir, 'conv_filters.png'))
    plt.close()

    fc_weights = model.fc.W
    print("FC layer weights shape:", fc_weights.shape)

    input_channels = model.conv.W.shape[0]
    reshaped_weights = fc_weights.reshape(
        input_channels, 16, 16, -1
    ).transpose(3, 0, 1, 2)

    plt.figure(figsize=(12, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        channel_avg = reshaped_weights[i].mean(axis=0)
        plt.imshow(channel_avg, cmap='viridis')
        plt.axis('off')
        plt.title(f'Hidden Unit {i + 1}')
    plt.suptitle('First 16 Hidden Units Spatial Patterns')
    plt.savefig(os.path.join(save_dir, 'fc_weights.png'))
    plt.close()

    output_weights = model.fc_out.W
    print("Output layer weights shape:", output_weights.shape)

    plt.figure(figsize=(12, 6))
    plt.imshow(output_weights.T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Hidden Units')
    plt.ylabel('Classes')
    plt.title('Output Layer Weights')
    plt.savefig(os.path.join(save_dir, 'output_weights.png'))
    plt.close()


if __name__ == "__main__":
    # 加载保存的模型
    model_path = os.path.join(current_dir, "1744411485.2150824besthy.pkl")  # 修改为你的模型路径
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 可视化参数
    visualize_model(model, save_dir=current_dir)
