import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def parse_txt(path):
    with open(path) as f:
        data = np.array(list(map(lambda x:np.array(x.strip().split()), f.readlines())))
    return data

names = ['ori_yolov7_tiny', 'coupling_yolov7_tiny', 'custom_yolov7_tiny']

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
for i in names:
    data = parse_txt(f'runs/train/{i}/results.txt')
    plt.plot(np.array(data[:, 8], dtype=float), label=i)
plt.xlabel('epoch')
plt.title('precision')
plt.legend()

plt.subplot(2, 2, 2)
for i in names:
    data = parse_txt(f'runs/train/{i}/results.txt')
    plt.plot(np.array(data[:, 9], dtype=float), label=i)
plt.xlabel('epoch')
plt.title('recall')
plt.legend()

plt.subplot(2, 2, 3)
for i in names:
    data = parse_txt(f'runs/train/{i}/results.txt')
    plt.plot(np.array(data[:, 10], dtype=float), label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5')
plt.legend()

plt.subplot(2, 2, 4)
for i in names:
    data = parse_txt(f'runs/train/{i}/results.txt')
    plt.plot(np.array(data[:, 11], dtype=float), label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5:0.95')
plt.legend()

plt.tight_layout()
plt.savefig('curve.png')

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
for i in names:
    data = parse_txt(f'runs/train/{i}/results.txt')
    plt.plot(np.array(data[:, 2], dtype=float), label=i)
plt.xlabel('epoch')
plt.title('train/box_loss')
plt.legend()

plt.subplot(2, 3, 2)
for i in names:
    data = parse_txt(f'runs/train/{i}/results.txt')
    plt.plot(np.array(data[:, 3], dtype=float), label=i)
plt.xlabel('epoch')
plt.title('train/obj_loss')
plt.legend()

plt.subplot(2, 3, 3)
for i in names:
    data = parse_txt(f'runs/train/{i}/results.txt')
    plt.plot(np.array(data[:, 4], dtype=float), label=i)
plt.xlabel('epoch')
plt.title('train/cls_loss')
plt.legend()

plt.subplot(2, 3, 4)
for i in names:
    data = parse_txt(f'runs/train/{i}/results.txt')
    plt.plot(np.array(data[:, -3], dtype=float), label=i)
plt.xlabel('epoch')
plt.title('val/box_loss')
plt.legend()

plt.subplot(2, 3, 5)
for i in names:
    data = parse_txt(f'runs/train/{i}/results.txt')
    plt.plot(np.array(data[:, -2], dtype=float), label=i)
plt.xlabel('epoch')
plt.title('val/obj_loss')
plt.legend()

plt.subplot(2, 3, 6)
for i in names:
    data = parse_txt(f'runs/train/{i}/results.txt')
    plt.plot(np.array(data[:, -1], dtype=float), label=i)
plt.xlabel('epoch')
plt.title('val/cls_loss')
plt.legend()

plt.tight_layout()
plt.savefig('loss_curve.png')