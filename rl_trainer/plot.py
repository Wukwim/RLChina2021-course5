import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

plot_freq = 500
x_min = 0
x_max = 30000
y_min = 0
y_max = 80

# 解决中文显示问题
font1 = {'family': 'SimHei',
         'weight': 'normal',
         'size': 14}
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 16}
labelsize = 13

if os.path.exists('score.txt'):
    f1 = open('score.txt', "r")
    lines1 = f1.readlines()
    a1 = []
    for line in lines1:
        a1.append(float(line))
    # 重组数组
    a10 = np.array(a1)
    print('Episode:', len(a1))

    # 数组舍弃余数重组
    rest1 = len(a10) % plot_freq
    for i in range(rest1):
        a10 = np.delete(a10, len(a10) - rest1)
    len(a1) * plot_freq + rest1,
    a10 = a10.reshape(-1, plot_freq)
    a1 = a10.mean(axis=1)

    plt.figure(1)
    num1 = list(range(len(a1)))
    reward_list = []
    for num in num1:
        num *= plot_freq
        reward_list.append(num)
    plt.axis([0, x_max, y_min, 8])
    # plt.grid(linestyle='-.')
    # plt.xlabel('episodes', font)
    # plt.ylabel('Average reward', font)
    plt.xlabel('episodes', font1)
    plt.ylabel('score', font1)
    a1 = np.array(a1)
    a10 = np.mean(a1)
    plt.title('mean is %.4f' % a10, font)
    print('score is %.6f' % a10)

    plt.plot(reward_list, a1, c='blue', ls='-', marker='', label="MADDPG")
    plt.grid()
    plt.tick_params(labelsize=labelsize)
    # plt.legend(loc='lower right', prop=font)

    x_major_locator = MultipleLocator(5000)
    y_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.savefig('score.png')
    plt.show()


if os.path.exists('reward.txt'):
    f1 = open('reward.txt', "r")
    lines1 = f1.readlines()
    a1 = []
    for line in lines1:
        a1.append(float(line))
    # 重组数组
    a10 = np.array(a1)
    print('Episode:', len(a1))

    # 数组舍弃余数重组
    rest1 = len(a10) % plot_freq
    for i in range(rest1):
        a10 = np.delete(a10, len(a10) - rest1)
    len(a1) * plot_freq + rest1,
    a10 = a10.reshape(-1, plot_freq)
    a1 = a10.mean(axis=1)

    plt.figure(1)
    num1 = list(range(len(a1)))
    reward_list = []
    for num in num1:
        num *= plot_freq
        reward_list.append(num)
    plt.axis([0, x_max, y_min, y_max])
    # plt.grid(linestyle='-.')
    # plt.xlabel('episodes', font)
    # plt.ylabel('Average reward', font)
    plt.xlabel('episodes', font1)
    plt.ylabel('reward', font1)
    a1 = np.array(a1)
    a10 = np.mean(a1)
    plt.title('mean is %.4f' % a10, font)
    print('reward is %.6f' % a10)

    plt.plot(reward_list, a1, c='red', ls='-', marker='', label="MADDPG")
    plt.grid()
    plt.tick_params(labelsize=labelsize)
    # plt.legend(loc='lower right', prop=font)

    x_major_locator = MultipleLocator(5000)
    y_major_locator = MultipleLocator(20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.savefig('reward.png')
    plt.show()


# if os.path.exists('loss.txt'):
#     f1 = open('loss.txt', "r")
#     lines1 = f1.readlines()
#     a1 = []
#     for line in lines1:
#         a1.append(float(line))
#     # 重组数组
#     a10 = np.array(a1)
#     print('Episode:', len(a1))
#
#     # 数组舍弃余数重组
#     rest1 = len(a10) % plot_freq
#     for i in range(rest1):
#         a10 = np.delete(a10, len(a10) - rest1)
#     len(a1) * plot_freq + rest1,
#     a10 = a10.reshape(-1, plot_freq)
#     a1 = a10.mean(axis=1)
#
#     plt.figure(1)
#     num1 = list(range(len(a1)))
#     reward_list = []
#     for num in num1:
#         num *= plot_freq
#         reward_list.append(num)
#     plt.axis([0, x_max, y_min, 500])
#     # plt.grid(linestyle='-.')
#     # plt.xlabel('episodes', font)
#     # plt.ylabel('Average reward', font)
#     plt.xlabel('episodes', font1)
#     plt.ylabel('loss', font1)
#     a1 = np.array(a1)
#     a10 = np.mean(a1)
#     plt.title('mean is %.4f' % a10, font)
#     print('loss is %.6f' % a10)
#
#     plt.plot(reward_list, a1, c='green', ls='-', marker='', label="MADDPG")
#     plt.grid()
#     plt.tick_params(labelsize=labelsize)
#     # plt.legend(loc='lower right', prop=font)
#
#     x_major_locator = MultipleLocator(5000)
#     y_major_locator = MultipleLocator(50)
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(x_major_locator)
#     ax.yaxis.set_major_locator(y_major_locator)
#
#     plt.savefig('loss.png')
#     plt.show()
