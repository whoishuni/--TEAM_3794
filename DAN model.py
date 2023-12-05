import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, ReLU, Flatten, Dense, Add, BatchNormalization, Dropout
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, random_rotation
from keras.regularizers import l2

df1 = open('./CSVs/kyu_train.csv').read().splitlines()
df2 = open('./CSVs/dan_train.csv').read().splitlines()
df = df1 + df2  # 將兩份資料合併
print("Size of df1:", len(df1))
print("Size of df2:", len(df2))

print("Size of combined df:", len(df))

games = [i.split(',', 2)[-1] for i in df]

chars = 'abcdefghijklmnopqrst'
coordinates = {k:v for v,k in enumerate(chars)}


def prepare_input(moves):
    x = np.zeros((19, 19, 5))  # Increase feature count to five
    board_state = np.zeros((19, 19), dtype=int)  # Initialize a 19x19 empty board

    for move in moves:
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]

        if color == 'B':
            x[row, column, 0] = 1
            x[row, column, 3] = 1  # Liberties feature for black stones
        if color == 'W':
            x[row, column, 1] = 1
            x[row, column, 4] = 1  # Liberties feature for white stones

        # Calculate liberties for the group of stones
        group_of_stones = [(row, column)]  # Start with the current stone
        group_liberties = calculate_liberties(board_state, group_of_stones)

        if color == 'B':
            x[row, column, 3] = len(group_liberties)  # Update liberties count for black stones
        if color == 'W':
            x[row, column, 4] = len(group_liberties)  # Update liberties count for white stones

        board_state[row, column] = 1  # Update the board state with the new stone

    if moves:
        last_move_column = coordinates[moves[-1][2]]
        last_move_row = coordinates[moves[-1][3]]
        x[last_move_row, last_move_column, 2] = 1

    x[:, :, 2] = np.where(x[:, :, 2] == 0, 1, 0)

    return x


def calculate_liberties(board, group):
    liberties = set()  # Use a set to avoid duplicate liberties
    for stone in group:
        x, y = stone  # Stone coordinates
        adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        for pos in adjacent_positions:
            if is_valid_position(pos) and is_empty(board, pos):
                liberties.add(pos)

    return liberties


def is_valid_position(pos):
    x, y = pos
    return 0 <= x < 19 and 0 <= y < 19  # Assuming a 19x19 Go board


def is_empty(board, pos):
    x, y = pos
    return board[x, y] == 0  # Assuming 0 represents an empty intersection


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

n_sampled_games = 5000


def data_generator(games, batch_size, n_sampled_games=n_sampled_games):
    while True:
        x, y = [], []

        # 随机选择游戏
        sampled_games = np.random.choice(games, n_sampled_games, replace=False)

        for game in sampled_games:
            moves_list = game.split(',')
            for count, move in enumerate(moves_list):
                board = prepare_input(moves_list[:count])  # 使用更新后的函数
                # 应用数据增强
                board = datagen.random_transform(board)
                x.append(board)
                y.append(prepare_label(move))

                if len(x) == batch_size:
                    y_one_hot = tf.one_hot(y, depth=19 * 19).numpy()
                    yield np.array(x), {'policy_output': y_one_hot, 'value_output': y_one_hot[:, 0]}
                    x, y = [], []

'''隨機選擇一些遊戲。
對於每場選擇的遊戲，逐個提取落子動作，然後使用prepare_input函數獲取當前的棋盤狀態，使用prepare_label函數獲取當前的落子位置標籤。
當積累的數據達到指定的batch_size時，就會輸出這批數據，然後清空數據並繼續進行下一批次的收集。
'''
def custom_rotation(image):
    angle = np.random.choice([90, 180, 270])
    return random_rotation(image, rg=angle, row_axis=0, col_axis=1, channel_axis=2)

# Data Augmentation: Adding custom rotation
datagen = ImageDataGenerator(
    preprocessing_function=custom_rotation,
    #horizontal_flip=True,
    #vertical_flip=True
)

batch_size = 256

# 在呼叫model.fit之前計算steps_per_epoch
average_moves_per_game = n_moves / n_games
# 這一行計算每場遊戲的平均落子次數，其中 n_moves 是所有遊戲中落子的總次數，n_games 是遊戲的總數。
steps_per_epoch = (n_sampled_games * average_moves_per_game) // batch_size #average_moves_per_game 是每場遊戲的平均落子數。


train_games, val_games = train_test_split(games, test_size=0.4)
train_gen = data_generator(train_games, batch_size)
val_gen = data_generator(val_games, batch_size)


def residual_block(x, filters):
    skip = x
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([skip, x])
    x = ReLU()(x)
    return x

def create_model():
    inputs = Input(shape=(19, 19, 5))
    x = Conv2D(256, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    for _ in range(3):
        x = residual_block(x, 256)
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
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

#model = load_model('./h5/4_1_model_dan.h5')
model = create_model()
#model.summary()

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=len(val_games) // batch_size,
    epochs=100,
    callbacks=[early_stopping, reduce_lr_on_plateau]  # 添加 callbacks 到你的 fit 函数
)

model.save('./dan_kyu_1.h5')
