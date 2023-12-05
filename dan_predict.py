import numpy as np
from keras.models import Model, load_model
existing_game_ids = set()

chars = 'abcdefghijklmnopqrst'
coordinates = {k: v for v, k in enumerate(chars)}

def prepare_input(moves):
    x = np.zeros((19, 19, 4))
    for move in moves:
        color = move[0]
        position = move[2:4]
        column = coordinates[position[0]]
        row = coordinates[position[1]]
        if color == 'B':
            x[row, column, 0] = 1
            x[row, column, 2] = 1
        elif color == 'W':
            x[row, column, 1] = 1
            x[row, column, 2] = 1
    if moves:
        last_move_position = moves[-1][2:4]
        last_move_column = coordinates[last_move_position[0]]
        last_move_row = coordinates[last_move_position[1]]
        x[last_move_row, last_move_column, 3] = 1
    x[:, :, 2] = np.where(x[:, :, 2] == 0, 1, 0)
    return x

def number_to_char(number):
    number_1, number_2 = divmod(number, 19)
    reverse_coordinates = {v: k for k, v in coordinates.items()}
    return reverse_coordinates[number_1] + reverse_coordinates[number_2]


def top_5_with_probs(prediction):
    # 获取前五个预测及其概率
    top_5_indices = prediction.argsort()[-5:][::-1]
    top_5_probs = [prediction[i] for i in top_5_indices]
    return list(zip(top_5_indices, top_5_probs))


def voting_algorithm(models_predictions):
    final_predictions = []
    for game_predictions in zip(*models_predictions):
        # 存储所有模型的前五预测及其概率
        all_top_5 = [top_5_with_probs(model_pred) for model_pred in game_predictions]

        # 计算加权平均概率
        avg_probs = {}
        for top_5 in all_top_5:
            for position, prob in top_5:
                if position in avg_probs:
                    avg_probs[position] += prob
                else:
                    avg_probs[position] = prob

        avg_probs = {k: v / len(models_predictions) for k, v in avg_probs.items()}

        # 按概率排序并取前五
        sorted_top_5 = sorted(avg_probs.items(), key=lambda item: item[1], reverse=True)[:5]
        final_predictions.append([number_to_char(pos) for pos, _ in sorted_top_5])

    return final_predictions


models = []
#model_paths = ['./h5/model_kyu.h5', './h5/model_dan.h5', './h5/1_model_kyu.h5', './h5/1_model_dan.h5', './h5/2_model_kyu.h5', './h5/2_model_dan.h5', './h5/3_model_dan.h5', './h5/3_model_kyu.h5', './h5/4_model_kyu.h5', './h5/5_model_dan.h5', './h5/5_model_kyu.h5' ]
model_paths = ['./h5/model_kyu.h5', './h5/model_dan.h5', './h5/1_model_kyu.h5', './h5/1_model_dan.h5', './h5/2_model_kyu.h5', './h5/2_model_dan.h5', './h5/3_model_dan.h5', './h5/3_model_kyu.h5', './h5/4_1_model_dan.h5']
for path in model_paths:
    models.append(load_model(path))



df = open('./CSVs/dan_test_public.csv').read().splitlines()
games_id = [i.split(',', 2)[0] for i in df]
games = [i.split(',', 2)[-1] for i in df]

x_testing = []

for game in games:
    moves_list = game.split(',')
    x_testing.append(prepare_input(moves_list))

x_testing = np.array(x_testing)

# 從所有模型獲取預測
all_predictions = []
for model in models:
    policy_predictions, _ = model.predict(x_testing)
    all_predictions.append(policy_predictions)

final_predictions = voting_algorithm(all_predictions)

final_predictions_chars = final_predictions

with open('./predict_csv/FinalPredictions.csv', 'a') as f:
    for game_id, predictions in zip(games_id, final_predictions_chars):
        f.write(f"{game_id},{','.join(predictions)}\n")

