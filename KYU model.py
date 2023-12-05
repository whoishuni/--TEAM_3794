import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, ReLU, Flatten, Dense, Add, BatchNormalization, Multiply, GlobalAveragePooling2D, Reshape, Dropout
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras.layers import MultiHeadAttention, LayerNormalization

df1 = open('./CSVs/kyu_train.csv').read().splitlines()
df2 = open('./CSVs/dan_train.csv').read().splitlines()
df = df1 + df2  # 將兩份資料合併

games = [i.split(',', 2)[-1] for i in df]

chars = 'abcdefghijklmnopqrst'
coordinates = {k:v for v,k in enumerate(chars)}

def prepare_input(moves):
    x = np.zeros((19,19,4))
    for move in moves:
        color = move[0]               #獲取落子的color
        column = coordinates[move[2]] #獲取落子的column坐標
        row = coordinates[move[3]]    #獲取落子的row坐標
        '''如果棋子是黑色，x[row, column, 0] 和 x[row, column, 2] 被設為 1。
           如果棋子是白色，x[row, column, 1] 和 x[row, column, 2] 被設為 1。'''
        if color == 'B':
            x[row,column,0] = 1
            x[row,column,2] = 1
        elif color == 'W':
            x[row,column,1] = 1
            x[row,column,2] = 1
            #如果有移動記錄，會標記最後一步棋的位置, 這在第三個通道（索引為3）上完成
            #空白位置的標記：
            # 最後，函數更新了第三個通道（索引為2），將其中的0值替換為1，標記棋盤上的空白位置。
    if moves:
        last_move_column = coordinates[moves[-1][2]]
        last_move_row = coordinates[moves[-1][3]]
        x[last_move_row, last_move_column, 3] = 1
    x[:,:,2] = np.where(x[:,:,2] == 0, 1, 0)
    return x

'''
def augment_data(x):
    """
    返回原始数据以及翻转旋转后的数据
    """
    rotations = [x]

    # 90 degree rotation
    rot_90 = np.rot90(x, 1, (0, 1))
    rotations.append(rot_90)

    # 180 degree rotation
    rot_180 = np.rot90(x, 2, (0, 1))
    rotations.append(rot_180)

    # 270 degree rotation
    rot_270 = np.rot90(x, 3, (0, 1))
    rotations.append(rot_270)

    return rotations
'''
def prepare_label(move): #將棋盤上的落子動作轉換為一個數字標籤
    '''
    輸入: 例如："Bpd"，其中"B"表示黑色，"p"和"d"分別是棋盤上的列和行的坐標。
    輸出: 表示落子的位置，如果落子位置是第一行第一列，則輸出是0；如果是第二行第一列，則輸出是19，以此類推。
    '''
    column = coordinates[move[2]]
    row = coordinates[move[3]]
    return column*19+row

n_moves = sum([len(game.split(',')) for game in games])
n_games = len(games)
print(f"Total Games: {n_games}, Total Moves: {n_moves}")
#games: 所有的遊戲數據，包含各種落子動作序列。
#batch_size: 每次提供的數據批次大小。
#n_sampled_games: 在每次迭代中隨機選取的遊戲數量。

n_sampled_games = 1000

def data_generator(games, batch_size, n_sampled_games=n_sampled_games):
    while True:
        x, y = [], []

        # 随机选择游戏
        sampled_games = np.random.choice(games, n_sampled_games, replace=False)

        for game in sampled_games:
            moves_list = game.split(',')
            for count, move in enumerate(moves_list):
                state = prepare_input(moves_list[:count])
                x.append(state)
                y.append(prepare_label(move))
                if len(x) == batch_size:
                    y_one_hot = tf.one_hot(y, depth=19 * 19).numpy()
                    yield np.array(x), {'policy_output': y_one_hot, 'value_output': y_one_hot[:, 0]}
                    x, y = [], []


'''隨機選擇一些遊戲。
對於每場選擇的遊戲，逐個提取落子動作，然後使用prepare_input函數獲取當前的棋盤狀態，使用prepare_label函數獲取當前的落子位置標籤。
當積累的數據達到指定的batch_size時，就會輸出這批數據，然後清空數據並繼續進行下一批次的收集。
'''


batch_size = 128

# 在呼叫model.fit之前計算steps_per_epoch
average_moves_per_game = n_moves / n_games
# 這一行計算每場遊戲的平均落子次數，其中 n_moves 是所有遊戲中落子的總次數，n_games 是遊戲的總數。
steps_per_epoch = (n_sampled_games * average_moves_per_game) // batch_size #average_moves_per_game 是每場遊戲的平均落子數。


train_games, val_games = train_test_split(games, test_size=0.30)
train_gen = data_generator(train_games, batch_size)
val_gen = data_generator(val_games, batch_size)

def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


def transformer_block(x, d_model, num_heads, ffn_units, dropout=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn_output = Dense(ffn_units, activation="relu")(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)

    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


# 定义带有膨胀卷积和SE注意力机制的残差块
def residual_block(x, filters, dilation_rate=1, l2_reg=1e-4, dropout_rate=0.5):
    skip = x
    x = Conv2D(filters, kernel_size=3, padding='same', dilation_rate=dilation_rate,
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)  # 新增 Dropout 層
    x = Conv2D(filters, kernel_size=3, padding='same', dilation_rate=dilation_rate,
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Add()([skip, x])
    x = squeeze_excite_block(x)
    x = ReLU()(x)
    return x

# 创建模型时，我们可以通过更改residual_block函数调用的dilation_rate参数来更改膨胀率
def create_model():
    inputs = Input(shape=(19, 19, 4))
    x = Conv2D(256, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    dilation_rates = [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4]
    for rate in dilation_rates:
        x = residual_block(x, 256, dilation_rate=rate)

    x = transformer_block(x, d_model=256, num_heads=8, ffn_units=512)

    policy_head = Conv2D(2, kernel_size=1, padding='same')(x)
    policy_head = BatchNormalization()(policy_head)
    policy_head = ReLU()(policy_head)
    policy_head = Flatten()(policy_head)
    policy_head = Dense(19*19, activation='softmax', name='policy_output')(policy_head)
    value_head = Conv2D(1, kernel_size=1, padding='same')(x)
    value_head = BatchNormalization()(value_head)
    value_head = ReLU()(value_head)
    value_head = Flatten()(value_head)
    value_head = Dense(256, activation='relu')(value_head)
    value_head = Dense(1, activation='tanh', name='value_output')(value_head)
    model = Model(inputs=inputs, outputs=[policy_head, value_head])
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mean_squared_error'},
                  metrics={'policy_output': 'accuracy', 'value_output': 'mean_squared_error'})
    return model
'''
策略頭部（policy_head）
策略頭部的作用是輸出一個機率分佈，表示在給定的棋盤狀態下，在每個位置落子的機率。 
它首先通過一個卷積層，然後通過批量標準化（BatchNormalization）和ReLU激活函數。 
之後，輸出被展平並通過一個有361個單元的全連接層（因為19x19的棋盤有361個位置），並使用softmax激活函數。 
這樣，策略頭部輸出一個形狀為(19*19,)的機率分佈，其中每個元素都代表對應位置的落子機率。

2. 價值頭部（value_head）
價值頭部的作用是評估目前棋盤狀態對於目前玩家的優劣。 
它輸出一個介於-1和1之間的標量值，其中1代表當前玩家處於完全贏的狀態，而-1代表完全輸的狀態。 
價值頭部也是透過一個卷積層開始的，然後經過批量標準化和ReLU激活。 
然後，輸出被展平並透過一個有256個單元的全連接層和ReLU激活，再接一個單一單元的全連接層和tanh激活。

策略頭部和價值頭部共享底層的特徵提取網路（包括殘差塊），但是在頭部有各自特定的層。 
策略頭部和價值頭部的組合使得網路能夠在一個統一的框架中同時進行動作選擇和狀態評估
'''
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

model = load_model('./h5/5_model_dan.h5')
'''
def average_model_weights(model_paths):
    # 載入第一個模型
    model1 = load_model(model_paths[0])
    # 載入第二個模型
    model2 = load_model(model_paths[1])

    # 獲取兩個模型的權重
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()

    # 確保兩模型的權重結構相同
    assert len(weights1) == len(weights2), "Models have different number of layers/weights"

    # 平均權重
    averaged_weights = [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]

    # 使用平均權重建立新模型
    averaged_model = create_model()
    averaged_model.set_weights(averaged_weights)
    return averaged_model

# 建立模型並設定平均權重
model_paths = ['./h5/3_model_dan.h5', './h5/3_model_kyu.h5']
model = average_model_weights(model_paths)
'''
model.summary()

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=len(val_games) // batch_size,
    epochs=7,
    callbacks=[early_stopping, reduce_lr_on_plateau]  # 添加 callbacks 到你的 fit 函数
)

model.save('./h5/5_model_kyu.h5')
