import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.signal import stft
from PIL import Image
import os
import cv2
import pdb

CLASS_INFO = {
    # class: [mat_path,     num]
    0: ["IR007_1.mat", 106],
    1: ["IR014_1.mat", 170],
    2: ["IR021_1.mat", 210],
    3: ["B007_1.mat", 119],
    4: ["B014_1.mat", 186],
    5: ["B021_1.mat", 223],
    6: ["OR007@6_1.mat", 131],
    7: ["OR014@6_1.mat", 198],
    8: ["OR021@6_1.mat", 235],
    9: ["normal_1.mat", 98],
}


def readmat(dir_path, idx=0, name="DE_time"):
    print("readmat:\tCLASS_{0} INFO is {1}".format(idx, CLASS_INFO[idx]))
    mat_path = os.path.join(
        dir_path, "{}".format(CLASS_INFO[idx][0])
    )
    mat_key = "X{:03d}_{}".format(CLASS_INFO[idx][1], name)
    data = scio.loadmat(mat_path)[mat_key]  # (samples, 1)
    print("readmat:\tCLASS_{0} has {1} pieces of raw data".format(idx, data.shape[0]))
    return data


def normalize(data, ):
    """ min-max normalize, data shape is (samples, 1) """
    norm_data = (data - data.min()) / (data.max() - data.min())
    return norm_data


def slide_split(data, idx=0, sw_width=1024, sw_steps=300):
    """" slide window split
    sw_width:   slide window width, default 1024
    sw_steps:   slide steps, default 300
    :return :   return split data - (s_num, 1024)
    """
    start = 0
    nums = data.shape[0]
    s_num = (nums - sw_width) // sw_steps  # compute slide times
    data_list = []
    for i in range(s_num):
        data_list.append(data[start: start + sw_width].reshape(-1))
        start += sw_steps
    split_data = np.array(data_list)
    print("slide_split:\tCLASS_{0} has {1} segments of norm data".format(idx, split_data.shape[0]))
    return split_data


def fft_plot(data, idx=0, save=True):
    """ data shape is (1024,) """
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.sans-serif"] = ["SimSun"]
    
    time_plt = plt.subplot(311)
    time_plt.plot(list(range(len(data))), data, "g")
    time_plt.set_ylabel("振幅")
    time_plt.set_xlabel("时间/s")
    time_plt.set_title("时域信号图")

    freq_amp = np.abs(np.fft.fft(data))[:int(1024 / 2)]
    freq_plt = plt.subplot(312)
    # pdb.set_trace()
    freq_list = 12e3 * np.array(range(0, int(1024 / 2))) / 1024
    freq_plt.plot(freq_list, freq_amp, '-')
    freq_plt.set_ylabel("振幅")
    freq_plt.set_xlabel("频率/Hz")
    freq_plt.set_title("频域信号图")

    f, t, Z = stft(data, fs=12000, window="hann", nperseg=50)
    stft_plt = plt.subplot(313)
    stft_plt.pcolormesh(t, f, np.abs(Z), vmin=0, vmax=np.abs(Z).mean() * 10)
    stft_plt.set_ylabel("频率/Hz")
    stft_plt.set_xlabel("时间/s")
    stft_plt.set_title("时频信号图")

    plt.tight_layout()
    # plt.show()
    if save:
        plt.savefig("test_{}.png".format(idx))


def fft_trfm(data, idx=0):
    """
    Input shape :   (s_nums, 1024)
    """
    # NOTE: if you want to plot frequency figure, you should not to normalize the data
    # because normalize data between [0, 1], not preformancing big amplitude
    fft_plot(data[0], idx)  # just for test plot figure
    # pdb.set_trace()
    # TODO: modify the correlated values
    f, t, stft_data = stft(data, fs=12e3, window="hann", nperseg=50)
    return np.abs(stft_data)


def preprocess(dir_path="../"):
    """ preprocess data and return a list:
            [
    idx_0 ->    (fft_data, label)
    idx_1 ->    (fft_data, label)
    ...
    idx_9 ->    (fft_data, label)
            ]
        fft_data -> (s_num, fft_dim)
        label    -> (s_num, )

    """
    data_list = []
    label_list = []
    for i in range(10):
        raw_data = readmat(dir_path=dir_path, idx=i)
        # norm_data = normalize(data=raw_data)          # (samples, 1)
        split_data = slide_split(data=raw_data, idx=i)  # (s_num, 1024)
        fft_data = fft_trfm(data=split_data, idx=i)
        new_fft_data = np.zeros((fft_data.shape[0], 32, 32))
        for j in range(len(fft_data)):
            new_fft_data[j] = cv2.resize(fft_data[i], (32, 32))
        label = np.ones(fft_data.shape[0]) * i # (s_num,)
        data_list.extend(new_fft_data)
        label_list.extend(label)
    return data_list, label_list


if __name__ == "__main__":
    data_list, label_list = preprocess(dir_path="./CRWU")  # dir_path is the data directory path
    # print(np.array(data_list).shape)
    # data_list = normalize(np.array(data_list))
    np.save('data', np.array(data_list))
    np.save('label', np.array(label_list).astype(int))
    data = np.load('label.npy', allow_pickle=True)
    print(data)
