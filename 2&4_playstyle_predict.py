import numpy as np
from keras.models import load_model
import csv

# 加载所有模型

models_1 = [
    load_model('./h5/1_playstyle.h5'),
    load_model('./目前最優的playstyle/1_playstyle.h5'),
    load_model('./h5/best_playstyle.h5'),
]

#models_2 = [load_model(f'./h5/{i}_playstyle.h5') for i in range(5, 13)]
#models_2 = [load_model(f'./h5/{i}_playstyle.h5') for i in range(8, 16)]s
""" 
 #    load_model('./優化第二次/1_playstyle.h5'), #    load_model('./h5/transform_playstyle.h5'),

    load_model('./h5/playDCDNN.h5'),偏一
    load_model('./h5/playDCDNN1.h5')篇二三

----------------
    load_model('./h5/8_playstyle.h5'),
    load_model('./h5/9_playstyle.h5'),
    load_model('./h5/10_playstyle.h5'),
    load_model('./h5/11_playstyle.h5'),
    load_model('./h5/12_playstyle.h5'),
    load_model('./h5/13_playstyle.h5'),
    load_model('./h5/14_playstyle.h5'),
    load_model('./h5/15_playstyle.h5'),
"""
models_2 = [
    load_model('./h5/8_playstyle.h5'),
    load_model('./h5/9_playstyle.h5'),
    load_model('./h5/10_playstyle.h5'),
    load_model('./h5/11_playstyle.h5'),
    load_model('./h5/12_playstyle.h5'),
    load_model('./h5/13_playstyle.h5'),
    load_model('./h5/14_playstyle.h5'),
    load_model('./h5/15_playstyle.h5'),
    load_model('./h5/playDCDNN2.h5')

]

chars = 'abcdefghijklmnopqrst'
chartonumbers = {v: k for k, v in enumerate(chars)}


def prepare_input1(moves):
    x = np.zeros((19, 19, 2))
    for move in moves:
        color = move[0]
        column = chartonumbers[move[2]]
        row = chartonumbers[move[3]]
        x[row, column, 0] = 1
    if moves:
        last_move_column = chartonumbers[moves[-1][2]]
        last_move_row = chartonumbers[moves[-1][3]]
        #        x[row, column, 1] = 1
        x[last_move_row, last_move_column, 1] = 1
    return x
def prepare_input2(moves):
    x = np.zeros((19, 19, 4))
    for move in moves:
        color = move[0]
        column = chartonumbers.get(move[2])
        row = chartonumbers.get(move[3])
        if color == 'B':
            x[row, column, 0] = 1
            x[row, column, 2] = 1
        elif color == 'W':
            x[row, column, 1] = 1
            x[row, column, 2] = 1
    if moves:
        last_move = moves[-1]
        if len(last_move) >= 4:
            last_move_column = chartonumbers.get(last_move[2])
            last_move_row = chartonumbers.get(last_move[3])
            if last_move_column is not None and last_move_row is not None:
                x[last_move_row, last_move_column, 3] = 1
    x[:, :, 2] = np.where(x[:, :, 2] == 0, 1, 0)
    return x

# 读取测试数据
df = open('./CSVs/play_style_test_public.csv').read().splitlines()
games_id = [i.split(',', 2)[0] for i in df]
games = [i.split(',', 2)[-1] for i in df]

# 准备测试数据
x_testing1 = np.array([prepare_input1(game.split(',')) for game in games])
x_testing2 = np.array([prepare_input2(game.split(',')) for game in games])

# 对两组模型分别进行预测
predictions_1 = [model.predict(x_testing1) for model in models_1]
predictions_2 = [model.predict(x_testing2) for model in models_2]

# 使用最高機率決策法選擇預測
# 將所有模型的預測結果合併，並對每個類別選擇最高的機率
average_prob_predictions = np.mean(np.concatenate((predictions_1, predictions_2), axis=0), axis=0)

# 选择平均概率最高的类别作为最终预测
final_predictions = np.argmax(average_prob_predictions, axis=1) + 1

# 保存最终预测到CSV
with open('./predict_csv/FinalPredictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for index, prediction in enumerate(final_predictions):
        writer.writerow([games_id[index], prediction])

# 记录每个模型对于1至3类的预测概率到另一CSV
with open('./predict_csv/ModelPredictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入标题行
    header = ['GameID']
    for i in range(len(models_1) + len(models_2)):
        header.extend([f'Model_{i+1}_Class_1', f'Model_{i+1}_Class_2', f'Model_{i+1}_Class_3'])
    writer.writerow(header)

    # 写入每个游戏的预测概率
    for index, _ in enumerate(games_id):
        row = [games_id[index]]
        for prediction in predictions_1 + predictions_2:
            row.extend(prediction[index])
        writer.writerow(row)