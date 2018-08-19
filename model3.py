from preprocessing import shuffleList, deleteColumn, oneHotEncoding, deleteColumn, preprocessing
from model import separateSet, majority, compare_lists, evaluate_lists, createModel
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import concatenate, Input, BatchNormalization, PReLU
from keras.models import Model, Sequential
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import operator
import random
from math import exp
from pprint import pprint

def createModel3(inputs):
    # inputs
    models = []

    # inputs의 열 별로
    for input in inputs:
        # 행은 각 속성의 분류 개수(예:주간,아간은 2)이고 열은 개수에 맞춰서 설정된 행렬
        model = Input(shape=(len(input[0]),))
        models.append(model)

    # more layers for each one-hot encoding vector
    _models = []
    for i, model in enumerate(models):

        #배치 정규화:입력값이 너무 차이가 나지 않게 입력값 정규화해서 넘겨줌(매 층마다 정규화)
        model = Dense(round(len(inputs[i][0])*2), kernel_initializer='he_normal')(model)  # default node number = 200
        model = BatchNormalization()(model)
        model = Activation('elu')(model)

        # collect refined model
        _models.append(model)

    # merge
    x = concatenate(_models)

    x = Dense(500, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    # x = Dropout(0.2)(x)

    # output
    x = Dense(20)(x)
    x = Activation('softmax')(x)

    return Model(inputs=models, outputs=x)


if __name__ == "__main__":
    _inputs, _outputs, _ = preprocessing('./Kor_Train_교통사망사고정보(12.1~17.6).csv', './test_kor.csv')  # 범주형 데이터 리스트, 사람 수 데이터 리스트, 벡터화 dictionary

    _inputs, _outputs = shuffleList(_inputs, _outputs)  # 데이터 리스트가 고루 섞이도록 _inputs와 _outputs를 함께 섞음
    inputs, outputs, input_test, input_train, input_val, output_test, output_train, output_val\
        = separateSet(_inputs, _outputs)  # 범주형 데이터와 사람 수 데이터를 각각 test, train, validate를 위해 분류

    # _outputs = to_categorical(_outputs, num_classes=20)

    model = createModel3(inputs)

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    # model.summary()

    early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=0)

    # train
    # model의 학습 이력 정보로 train의 loss와 accuracy, val의 loss와 accuracy 값을 받음
    hist = model.fit([np.array(i) for i in input_train], np.array(output_train),
                     epochs=100, batch_size=pow(2, 13),
                     validation_data=([np.array(i) for i in input_val], np.array(output_val)), callbacks=[early_stopping], verbose=0)

    # plot_hist(hist)

    # test
    # model의 성능 평가
    score = model.evaluate([np.array(i) for i in input_test], np.array(output_test), verbose=0)
    # print('complete: %s = %.2f%%' % (model.metrics_names[1], score[1] * 100))

    #predict
    _preds = model.predict([np.array(i) for i in input_test])

    preds = []
    for i, _pred in enumerate(_preds):
        preds.append([])
        for val in _pred:
            # 반올림된 예측값이 0보다 클 경우 preds 리스트에 추가, 음수일경우 0을 추가
            preds[i].append(int(max(0, round(val))))

