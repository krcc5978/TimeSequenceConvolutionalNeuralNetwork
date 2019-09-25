# 画像サイズ
w = 224
h = 224
c = 6

# 軸の数
axis = 4

# RNN/LSTMの中間ユニットの数
hidden_neuron_number = 100

# 時系列の長さ
timesteps = 10

# バッチサイズ
batch_size = 8

# 入力データパス
base_path = 'D:\\data\\optical_flow_data\\testdata\\'
teacher_directory_list = ['move', 'stop', 'pickup', 'putdown']

# 入力モデル
input_model_path = ''

# 入力重み
input_weight_path = ''

# 重みファイル出力場所
output_weight_path = './logs/000/ep003-loss10.074-val_loss1.537.h5'

# 使用するCNNモデル
use_model_name = 'AlexNet'
