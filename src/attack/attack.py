# Generate a backdoor attack to poison the original dataset
# Spit back out the new poisoned dataset
# Save it to the poisoned directory
import numpy as np

def poison(x_train, y_train, parameter):
    target_label = parameter["target label"]
    num_images = int(parameter["poisoning_rate"] * y_train.shape[0])

    index = np.where(y_train != target_label)
    index = index[0]
    index = index[:num_images]
    x_train[index] = poison_frequency(x_train[index], y_train[index], parameter)
    y_train[index] = target_label
    return x_train

def poison_frequency(x_train, y_train, parameter):
    if x_train.shape[0] == 0:
        return x_train
    
    x_train *= 255.

    # going to the frequency domain
    x_train = DCT(x_train, param["window_size"])  # (idx, ch, w, h)

    # plug trigger frequency
    for i in range(x_train.shape[0]):
        for ch in parameter["channel_list"]:
            for w in range(0, x_train.shape[2], parameter["window_size"]):
                for h in range(0, x_train.shape[3], parameter["window_size"]):
                    for pos in parameter["pos_list"]:
                        x_train[i][ch][w + pos[0]][h + pos[1]] += param["magnitude"]


    x_train = IDCT(x_train, parameter["window_size"])  # (idx, w, h, ch)

    if parameter["YUV"]:
        x_train = YUV2RGB(x_train)

    x_train *= 255.
    x_train = np.clip(x_train, 0, 1)
    return x_train