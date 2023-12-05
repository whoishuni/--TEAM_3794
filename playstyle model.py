from keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import (Input, Conv2D, BatchNormalization, ReLU, Add,
                          Multiply, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D)
from keras.preprocessing.image import ImageDataGenerator, random_rotation
from sklearn.metrics import accuracy_score
from keras.layers import DepthwiseConv2D, Concatenate

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Dropout, Activation

from sklearn.utils import shuffle
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from keras.regularizers import l1_l2

# Setting random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
df = open('./CSVs/play_style_train.csv').read().splitlines()
games = [i.split(',')[2:] for i in df]
game_styles = [int(i.split(',')[1]) for i in df]  # 獲取棋風標籤


# 平衡然後打散

def balance_and_shuffle(games, game_styles):
    # get playsyle
    indices_1 = [i for i, style in enumerate(game_styles) if style == 1]
    indices_2 = [i for i, style in enumerate(game_styles) if style == 2]
    indices_3 = [i for i, style in enumerate(game_styles) if style == 3]

    # 平衡數量
    min_len = min(len(indices_1), len(indices_2), len(indices_3))

    indices_1 = np.random.choice(indices_1, min_len, replace=False)
    indices_2 = np.random.choice(indices_2, min_len, replace=False)
    indices_3 = np.random.choice(indices_3, min_len, replace=False)

    # 合併然後打亂進行訓練
    balanced_games = [games[i] for i in indices_1] + [games[i] for i in indices_2] + [games[i] for i in indices_3]
    balanced_styles = [game_styles[i] for i in indices_1] + [game_styles[i] for i in indices_2] + [game_styles[i] for i
                                                                                                   in indices_3]

    balanced_games, balanced_styles = shuffle(balanced_games, balanced_styles, random_state=42)

    return balanced_games, balanced_styles


games, game_styles = balance_and_shuffle(games, game_styles)

# Define coordinates for game moves
chars = 'abcdefghijklmnopqrst'
coordinates = {k: v for v, k in enumerate(chars)}

# Compute class weights for balancing
class_weights = compute_class_weight('balanced', classes=np.unique(game_styles), y=game_styles)
class_weight_dict = dict(enumerate(class_weights))


def visualize_game(moves):
    # 這個函數主要就是顯示出圍棋的棋盤，跟棋子因為當初一直遇到準確度無法提升，
    # 所以一直在思考是不是資料處理出問題，所以設計一個可以顯示棋盤的，這樣視覺化就很容易看出是否有問題
    board = np.zeros((19, 19))
    move_count = 1

    cmap = plt.get_cmap('Reds', len(moves))

    for move in moves:
        if move.startswith('B') or move.startswith('W'):
            color, position = move[0], move[2:4]
            value = 1 if color == 'B' else -1
            column = ord(position[0]) - ord('a')
            row = ord(position[1]) - ord('a')
            board[row, column] = value

            plt.text(column, row, str(move_count), ha='center', va='center', fontsize=10, color=cmap(move_count))
            move_count += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(np.arange(19))
    ax.set_yticks(np.arange(19))
    ax.set_xticklabels(chars[:19])
    ax.set_yticklabels(chars[:19])
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-0.5, 18.5)
    ax.imshow(np.ones_like(board) * 0.8, cmap='YlOrBr', vmin=-1, vmax=1)  # Background color

    for (i, j), value in np.ndenumerate(board):
        if value == 1:  # Black piece
            circle = plt.Circle((j, i), 0.4, color='black')
            ax.add_artist(circle)
        elif value == -1:  # White piece
            circle = plt.Circle((j, i), 0.4, color='white')
            ax.add_artist(circle)

    ax.grid(which='both', color='k', linestyle='-', linewidth=1.5)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()


# visualize_game(games[0]) ＃把註解用掉可以看到完整的棋盤，這是第一局

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


# Prepare input
x = np.array([prepare_input(game) for game in games])
y = np.array(game_styles) - 1
y_hot = tf.one_hot(y, depth=3)

x_train, x_val, y_train, y_val = train_test_split(x, y_hot.numpy(), test_size=0.1, stratify=y_hot.numpy())


def custom_rotation(image):
    angle = np.random.choice([90, 180, 270])
    return random_rotation(image, rg=angle, row_axis=0, col_axis=1, channel_axis=2)

# Data Augmentation: Adding custom rotation
datagen = ImageDataGenerator(
    preprocessing_function=custom_rotation,
    horizontal_flip=True,
    vertical_flip=True
)


def multi_head_attention_block(x, num_heads=5):
    """Generate multi-head attentions from the input x."""
    heads = []
    head_size = x.shape[-1] // num_heads

    for i in range(num_heads):
        qs = Dense(head_size)(x)
        ks = Dense(head_size)(x)
        vs = Dense(head_size)(x)

        head, _ = attention_head(qs, ks, vs)
        heads.append(head)

    concat_heads = Concatenate()(heads)
    return concat_heads


def attention_head(q, k, v):
    qk_dot = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_qk_dot = qk_dot / tf.math.sqrt(dk)
    attention_weights = tf.nn.softmax(scaled_qk_dot, axis=-1)

    return tf.matmul(attention_weights, v), attention_weights


# 深度可分離卷積和多頭注意力的殘差塊
def residual_block(x, filters, kernel_size=3, repetitions=2, num_heads=5):
    """Residual block with depthwise separable convolutions and multi-head attention."""
    for _ in range(repetitions):
        skip = x

        # Depthwise Separable Convolution
        x = DepthwiseConv2D(kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, kernel_size=1, padding='same', activation='relu')(x)

        # Multi-Head Attention
        x = multi_head_attention_block(x, num_heads=num_heads)

        # Matching dimensions of skip and output
        if skip.shape[-1] != x.shape[-1]:
            skip = Conv2D(x.shape[-1], kernel_size=1, padding='same')(skip)

        x = Add()([skip, x])
        x = ReLU()(x)
    return x


def create_model(learning_rate=0.001):
    inputs = Input(shape=(19, 19, 5))
    x = Conv2D(256, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    x = residual_block(x, 256, repetitions=2, num_heads=5)
    x = residual_block(x, 128, repetitions=2, num_heads=5)
    x = residual_block(x, 128, repetitions=2, num_heads=5)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)  # Added Dropout layer
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 指數衰減學習率
def exp_decay_lr(epoch, initial_lr=0.001, decay_factor=0.9, step_size=10):
    lr = initial_lr * (decay_factor ** (epoch // step_size))
    return lr



# 隨機序列數據生成器
class BalancedRandomizedSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.num_classes = len(np.unique(np.argmax(y_set, axis=1)))
        self.indices = np.arange(len(self.x))
        np.random.shuffle(self.indices)  # 一開始打亂
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        # 獲取對應批次索引
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        return batch_x, batch_y

    def on_epoch_end(self):
        # 重洗牌所有索引
        np.random.shuffle(self.indices)
class ClassDistributionCallback(Callback):
    def __init__(self, train_data, val_data):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.params['epochs']}")
        self.print_distribution(self.train_data, '訓練')
        self.print_distribution(self.val_data, '驗證')

    def print_distribution(self, data, name):
        y_argmax = np.argmax(data, axis=1)
        class_counts = np.bincount(y_argmax, minlength=3)  # 確保至少有3個類別
        total = np.sum(class_counts)
        class_percentages = class_counts / total * 100
        print(f"{name} 集: " +
              ", ".join([f"PlayStyle {i + 1} - {percent:.2f}%" for i, percent in enumerate(class_percentages)]))

class ClassWeightAdjuster(Callback):
    def __init__(self, val_data, initial_class_weights, increase_threshold=0.5, decrease_threshold=0.7, max_increase_factor=1.5, min_decrease_factor=0.8):
        super().__init__()
        self.val_data = val_data
        self.class_weights = initial_class_weights
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.max_increase_factor = max_increase_factor
        self.min_decrease_factor = min_decrease_factor

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.val_data
        y_pred = self.model.predict(x_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)

        # 計算混淆矩陣
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)

        # 打印每一類的準確率
        print("\n類別準確率:")
        for i, accuracy in enumerate(class_accuracies):
            print(f"類別 {i+1}: {accuracy:.2f}")

        # 根據表現調整類別權重並打印調整信息
        print("下一輪權重調整:")
        for i in range(len(class_accuracies)):
            original_weight = self.class_weights[i]
            if class_accuracies[i] < self.increase_threshold:
                increase_factor = min(self.max_increase_factor, 1 + (self.increase_threshold - class_accuracies[i]) / self.increase_threshold)
                self.class_weights[i] *= increase_factor
                increase_percentage = (self.class_weights[i] - original_weight) / original_weight * 100
                print(f"類別 {i+1} 的權重增加至 {self.class_weights[i]:.2f} ({increase_percentage:.2f}%)")
            elif class_accuracies[i] > self.decrease_threshold:
                decrease_factor = max(self.min_decrease_factor, 1 - (class_accuracies[i] - self.decrease_threshold) / (1 - self.decrease_threshold))
                self.class_weights[i] *= decrease_factor
                decrease_percentage = (original_weight - self.class_weights[i]) / original_weight * 100
                print(f"類別 {i+1} 的權重減少至 {self.class_weights[i]:.2f} ({decrease_percentage:.2f}%)")

        if not any([class_accuracies[i] < self.increase_threshold for i in range(len(class_accuracies))]) and \
           not any([class_accuracies[i] > self.decrease_threshold for i in range(len(class_accuracies))]):
            print("這一輪無需調整，所有類別學習平衡")

class CSVDataGenerator(Sequence):
    def __init__(self, csv_filepath, batch_size, num_classes, subset_size=None):
        self.csv_filepath = csv_filepath
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.subset_size = subset_size
        self.all_games, self.all_game_styles = self.load_csv_data()
        self.previous_indices = None  # Keep track of the indices from the previous epoch
        self.on_epoch_end()

    def load_csv_data(self):
        with open(self.csv_filepath, 'r', encoding='utf-8') as file:
            df = file.readlines()
        games = [line.strip().split(',')[2:] for line in df[1:]]
        game_styles = [int(line.strip().split(',')[1]) for line in df[1:]]
        return games, game_styles

    def __len__(self):
        return int(np.ceil(min(self.subset_size, len(self.all_games)) / self.batch_size)) if self.subset_size else int(np.ceil(len(self.all_games) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        batch_x = np.array([prepare_input(self.all_games[i]) for i in batch_indices])
        batch_y = np.array([self.all_game_styles[i] - 1 for i in batch_indices])
        return batch_x, tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)

    def on_epoch_end(self):
        # If subset_size is specified and it's smaller than the dataset size, sample without replacement
        if self.subset_size and self.subset_size < len(self.all_games):
            self.indices = np.random.choice(len(self.all_games), self.subset_size, replace=False)
        else:
            self.indices = np.arange(len(self.all_games))
        np.random.shuffle(self.indices)

        # Calculate repeat rate with the previous epoch
        if self.previous_indices is not None:
            repeat_rate = np.intersect1d(self.indices, self.previous_indices).size / self.indices.size
            print(f"Repeat rate with the previous epoch: {repeat_rate:.2%}")
        self.previous_indices = self.indices.copy()


subset_size = 25000

train_generator = CSVDataGenerator(csv_filepath='./CSVs/play_style_train.csv',
                                   batch_size=64,
                                   num_classes=3,
                                   subset_size=subset_size)

val_generator = CSVDataGenerator(csv_filepath='./CSVs/play_style_train.csv',
                                 batch_size=256,
                                 num_classes=3,
                                 subset_size=subset_size)


#model = load_model('./h5/8_playstyle.h5')
model = create_model()
#model.summary()
# 動態調整學習率

# lr_schedule = LearningRateScheduler(cyclic_lr)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = LearningRateScheduler(exp_decay_lr)
class_distribution_cb = ClassDistributionCallback(y_train, y_val)

# 初始化類別權重
initial_class_weights = {0: 1, 1: 1, 2: 1}


class_weight_adjuster = ClassWeightAdjuster((x_val, y_val), initial_class_weights)


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    class_weight=initial_class_weights,
    callbacks=[
        reduce_lr,
        early_stop,
        lr_schedule,
        class_distribution_cb,
        class_weight_adjuster  # 加入自定義的回調函數
    ]
)

model.save('./h5/5_playstyle_1.h5')

y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1) + 1
y_true_classes = np.argmax(y_val, axis=1) + 1

# 計算整體準確度
overall_accuracy = accuracy_score(y_true_classes, y_pred_classes)

# 混淆矩陣
cm = confusion_matrix(y_true_classes, y_pred_classes)

# 類別標籤
labels = ['1', '2', '3']
label_names = ['playstyle 1', 'playstyle 1', 'playstyle 1']

# 計算每個類別的準確度
class_accuracies = cm.diagonal() / cm.sum(axis=1)


plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('prediction category')
plt.ylabel('real category')


for i, class_accuracy in enumerate(class_accuracies):
     plt.text(i + 0.5, len(cm) + 0.5, f"{label_names[i]} Accuracy: {class_accuracy:.2f}",
              ha='center', va='center', fontsize=12)

plt.title(f'overall accuracy {overall_accuracy:.2f}')
plt.show()


report = classification_report(y_true_classes, y_pred_classes, target_names=label_names, digits=2)
print(report)




plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
