import torch.nn as nn 

# rnn = nn.RNN(input_size, hidden_size) # shell이자 A를 선언하는 박스.
# output, _status = rnn(input_data)

# e.g. hello
# use ont-hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

input_size = 4


# 미리 input_size 를 선언해주어야 한다. 

# hiddin_state : output_size 
# 차원의 출력값을 정해준다.
# 감성분석을 한다면 감정의 갯수를 hidden_size로 정한다.

hidden_size = 2

# sequence length
# rnn의 x0~xt 까지의 갯수가 몇개인가하는 갓. t+1개의 sequence length를 가진다.
# hello를 입력값으로 한다면 sequence length는 5다.
# pytorch는 sequence length를 따로 지정해주지 않더라도, 알아서 인식해준다. good!
# input_data.shape (batch_size, sequence_length , input_size)
# output_data.shape (batch_size, sequenc_length, hidden_size)

import torch
import numpy as np
input_size =4;
hidden_size = 2;

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

input_data_np = np.array(
    [[h, e, l, l, o],
     [e, o, l, l, l],
     [l, l, e, e, l]], dtype = np.float32)


input_data = torch.Tensor(input_data_np)

rnn = torch.nn.RNN(input_size, hidden_size)
outputs, _status = rnn(input_data)
outputs.shape

