import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


class Convolution:

    def __init__(self):

        # declare variable
        image_af_conv = []
        total_sum_conv = []

        stride = 1

        self.stride = stride
        self.image_af_conv = image_af_conv
        self.total_sum_conv = total_sum_conv

        # declare input image
        image_length, row, column = map(int, input('이미지의 전체 크기와 shape(행,열)을 입력하세요').split())
        image_bf_conv = list(range(1, image_length + 1))
        self.image_bf_conv = np.reshape(image_bf_conv, (row, column))

        # declare kernel
        kernel_size = input(' \n 커널의 한 변의 크기를 입력하세요')
        kernel_size = int(kernel_size)
        self.kernel = np.ones((kernel_size, kernel_size))

        # input image Padding
        self.pad_image_bf_conv = np.pad(self.image_bf_conv,pad_width=((kernel_size,kernel_size),
                                                        (kernel_size,kernel_size)), mode='constant', constant_values=0)

        print('이미지 크기 \n {}'.format(self.pad_image_bf_conv))
        print(' \n 커널 \n {} \n'.format(self.kernel))

    def conv2d(self):
        cnt_step_col = cnt_step_row = kernel_row = kernel_col = 0

        # extract first nonzero
        start_point = np.transpose(np.nonzero(self.pad_image_bf_conv))

        # whole Convolution
        while cnt_step_row + len(self.kernel) <= len(self.image_bf_conv):

            while cnt_step_col + len(self.kernel[0]) <= len(self.image_bf_conv[0]):
                # image_bf_conv * Kernel done once
                while kernel_row < len(self.kernel):

                    while kernel_col < len(self.kernel[0]):
                        self.total_sum_conv.append(
                            self.pad_image_bf_conv[start_point[0,0]+kernel_row + cnt_step_row, start_point[0,1] +
                                                   kernel_col + cnt_step_col] * self.kernel[kernel_row,kernel_col])
                        kernel_col += 1

                    kernel_col = 0
                    kernel_row += 1

                kernel_row = 0
                cnt_step_col += 1

            cnt_step_col = 0
            cnt_step_row += self.stride

    def feature_map(self):
        # Result of convolution Reshape
        total_sum_conv_row = int(len(self.total_sum_conv) / self.kernel.size)
        total_sum_conv = np.reshape(self.total_sum_conv, (total_sum_conv_row, self.kernel.size))
        self.image_af_conv = np.sum(total_sum_conv, 1)

        image_af_conv = np.reshape(self.image_af_conv, (
            ((len(self.image_bf_conv) - len(self.kernel) + 1) // self.stride),
            ((len(self.image_bf_conv[0]) - len(self.kernel[0]) + 1) // self.stride)))

        self.image_af_conv = torch.Tensor(image_af_conv)
        print('image_af_conv 값 \n{}\n'.format(image_af_conv))

    def relu(self):
        relu = nn.ReLU(inplace=True)
        self.image_af_conv = relu(self.image_af_conv)

        # image_af_conv = result of convolution & activation function
        print('Activate 함수를 지난 후 \n{}'.format(self.image_af_conv))
        print(self.image_af_conv.shape)

    def max_pooling(self):
        pooling_size = input('conv 후의 이미지 사이즈에 걸맞는 pooling Size를 입력하세요')
        pooling_size = int(pooling_size)

        pool = np.array([])
        pool = pool.astype(int)
        max_pool = np.array([])
        max_pool = max_pool.astype(int)
        stride = pooling_size

        # Padding
        af_to_numpy = np.array(self.image_af_conv)
        pad_image_af_conv = np.pad(af_to_numpy, pad_width=((pooling_size, pooling_size), (pooling_size, pooling_size))
                                   , mode='constant', constant_values=0)
        print('pad_image_af_conv 값 \n{}\n'.format(pad_image_af_conv))

        # Stride = pooling_size
        cnt_row = (len(self.image_af_conv) + pooling_size - 1) // pooling_size
        cnt_col = (len(self.image_af_conv[0]) + pooling_size - 1) // pooling_size

        # extract first nonzero
        starting_point = np.transpose(np.nonzero(pad_image_af_conv))
        first_starting_point = starting_point[0]

        # number of row repetition
        for steps_row in range(cnt_row):
            starting_row = steps_row * stride

            # number of col repetition
            for steps_col in range(cnt_col):
                starting_col = steps_col * stride

                # pooling done once , max_pool = result of pooling
                for cnt_one_pool in range(pooling_size):
                    pool = np.append(pool, pad_image_af_conv[first_starting_point[0] + cnt_one_pool+starting_row,
                                           first_starting_point[1]+starting_col:first_starting_point[1] +starting_col+ pooling_size])

                max_pool = np.append(max_pool, np.max(pool))

        # max_pool reshape
        max_pool = np.reshape(max_pool,(cnt_row,cnt_col))
        print('max_pool 값 \n{}\n'.format(max_pool))

    def model(self):
        self.conv2d()
        self.feature_map()
        self.relu()
        self.max_pooling()
        return


hello = Convolution()
hello.model()
