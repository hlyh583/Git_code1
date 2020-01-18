import numpy as np
import pandas as pd
import os, shutil, random
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import layers, models, optimizers
from keras.models import Sequential, Model
from keras import applications
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn import  metrics

## 해당 모델은 아래 사이트를 참조하여 만듦
# https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/

## Hand Veins Similarity Check with CNN
# 지문 인식처럼 주인의 손등의 정맥을 학습 후, 해당 손등의 정맥이 주인의 것인지 아닌지를 CNN으로 판별하는 코드
# Recall 값은 1.0, Accuracy는 0.95까지 나옴


## 베이스 폴더 지정
base_dir = 'C:/Users/JKKIM/Desktop/KTH/Pilot'
os.chdir(base_dir)


## 난수값 설정 (이 수치를 바꾸지 않는한 Train, Validation, Test 데이터셋 파일을 동일하게 나눔)
random.seed(12)


## 오늘 날짜 설정 (폴더 이름 설정 때문)
now = datetime.now()


## 전체 데이터 셋 구성 : KTH(50장), 친구1(50장), 친구2(50장)
# KTH Train set : 30장, Validation set : 10장, Test set : 10장
# 친구1 Train set : 15장, Validation set : 5장, Test set : 5장
# 친구2 Train set : 15장, Validation set : 5장, Test set : 5장
kth_train_size = 30
kth_validation_size = 10 + kth_train_size
kth_test_size = 10 + kth_validation_size
others_train_size = int(kth_train_size / 2)
others_validation_size = int(kth_validation_size / 2)
others_test_size = int(kth_test_size / 2)


## 실험대상자 이름 전부를 list에 넣기
name_list = ['hwi', 'jbg', 'kth']


## Train set, Validation set, Test set 폴더 만들기
file_list = []
train_dst_name = base_dir + '/train_0' + str(now.month) + str(now.day)
os.mkdir(train_dst_name)
validation_dst_name = base_dir + '/validation_0' + str(now.month) + str(now.day)
os.mkdir(validation_dst_name)
test_dst_name = base_dir + '/test_0' + str(now.month) + str(now.day)
os.mkdir(test_dst_name)


## Train set, Validation set 각각의 폴더 안에 KTH과 다른 사람들(others)의 이미지를 모아놓는 세부 폴더 생성
train_dst_sub_kth = train_dst_name + '/train_kth'
os.mkdir(train_dst_sub_kth)
validation_dst_sub_kth = validation_dst_name + '/validatioin_kth'
os.mkdir(validation_dst_sub_kth)

train_dst_sub_others = train_dst_name + '/train_others'
os.mkdir(train_dst_sub_others)
validation_dst_sub_others = validation_dst_name + '/validatioin_others'
os.mkdir(validation_dst_sub_others)


## Test set 폴더는 KTH과 다른 사람들의 이미지를 하나의 세부 폴더 안에 모아둠
# 이렇게 해야만 테스트가 가능
test_dst_sub_total = test_dst_name + '/test_total'
os.mkdir(test_dst_sub_total)


## 각각의 세부 폴더 안에 이미지들을 분류하는 함수(Function)
def dir_classification(train_size, validation_size, test_size, train_dst_sub_name, validation_dst_sub_name, test_dst_sub_name) :
    ## Train set 분류
    for f_name in dir_list[:train_size] :
        src = os.path.join(base_dir + dir_name, f_name)
        dst = os.path.join(train_dst_sub_name, f_name)
        shutil.copyfile(src, dst)
    
    ## Validation set 분류
    for f_name in dir_list[train_size:validation_size] :
        src = os.path.join(base_dir + dir_name, f_name)
        dst = os.path.join(validation_dst_sub_name, f_name)
        shutil.copyfile(src, dst)
    
    ## Test set 분류
    for f_name in dir_list[validation_size:test_size] :
        src = os.path.join(base_dir + dir_name, f_name)
        dst = os.path.join(test_dst_sub_name, f_name)
        shutil.copyfile(src, dst)


## KTH, 친구1, 친구2의 이미지가 저장된 폴더로부터 랜덤하게 이미지를 선택하여, Train set, Validation set, Test set 분류
# random.seed(12) 때문에 사실 정말로 랜덤하게 이미지를 선택하는 것은 아님
for i, i_value in enumerate(name_list) :
    dir_name = '/' + i_value + '_binary'
    dir_list = os.listdir(base_dir + dir_name)
    random.shuffle(dir_list)
    file_list.append(dir_list)
    
    if i_value == 'kth' :
        dir_classification(kth_train_size, kth_validation_size, kth_test_size, train_dst_sub_kth, validation_dst_sub_kth, test_dst_sub_total)
    else :
        dir_classification(others_train_size, others_validation_size, others_test_size, train_dst_sub_others, validation_dst_sub_others, test_dst_sub_total)
    

## Train set, Validation set, Test set을 ImageDataGenerator에 집어넣음
train_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dst_name,
    target_size=(320, 240),
    batch_size = 5,
    class_mode = 'binary',
    shuffle=False,
    seed=42
    )

validation_generator = validation_datagen.flow_from_directory(
    validation_dst_name,
    target_size=(320, 240),
    batch_size = 5,
    class_mode = 'binary',
    shuffle=False,
    seed=42
    )

test_generator = test_datagen.flow_from_directory(
    test_dst_name,
    target_size=(320, 240),
    batch_size = 1,
    class_mode = 'binary',
    shuffle=False,
    seed=42
    )


## Train set, Validation set, Test set을 예측할 때 사용할 Steps를 계산
step_size_train = int(train_generator.n//train_generator.batch_size)
step_size_validation = int(validation_generator.n//validation_generator.batch_size)
step_size_test = int(test_generator.n//test_generator.batch_size)


## 이미 학습된 VGG16 Network(imagenet)를 불러오기
datagen1 = ImageDataGenerator(rescale=1. / 255)
model = applications.VGG16(include_top=False, weights='imagenet')


## 이미 학습된 VGG16 Network(imagenet)에 Train set을 넣고 예측한 결과(Trains set의 특징 = 병목 특징)를 npy파일 형태로 저장
train_generator1 = datagen1.flow_from_directory(
    train_dst_name,
    target_size=(320, 240),
    batch_size=5,
    class_mode=None,
    shuffle=False)
bottleneck_features_train = model.predict_generator(train_generator1, step_size_train)
np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


## 이미 학습된 VGG16 Network(imagenet)에 Validation set을 넣고 예측한 결과(Validation set의 특징 = 병목 특징)를 npy파일 형태로 저장
validation_generator1 = datagen1.flow_from_directory(
    validation_dst_name,
    target_size=(320, 240),
    batch_size=5,
    class_mode=None,
    shuffle=False)
bottleneck_features_validation = model.predict_generator(validation_generator1, step_size_validation)
np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)


## Train generator에 들어가 있는 파일들의 이름 및 순서, 그리고 총 몇 개인지를 확인
# 참고로 파일들의 순서를 확인하는 이유는 정답 레이블(train_labels, validation_labels)을 만들기 위함
print(train_generator1.filenames)
print("Total: %d\n" % len(train_generator1.filenames))
print(validation_generator1.filenames)
print("Total: %d" % len(validation_generator1.filenames))


## Train set의 특징(= 병목 특징)이 저장된 npy 파일을 불러옴
train_data = np.load(open('bottleneck_features_train.npy', "rb"))


## 각각의 Train 데이터에 대한 정답 레이블을 만듦
# 0이 30개 있고, 그 다음에 1이 30개 있는 형태
train_labels = np.array([0] * int(train_generator1.n / 2) + [1] * int(train_generator1.n / 2))


## Validation set의 특징(= 병목 특징)이 저장된 npy 파일을 불러옴
validation_data = np.load(open('bottleneck_features_validation.npy', "rb"))


## 각각의 Validation 데이터에 대한 정답 레이블을 만듦
# 0이 10개 있고, 그 다음에 1이 10개 있는 형태
validation_labels = np.array([0] * int(validation_generator1.n / 2) + [1] * int(validation_generator1.n / 2))


## VGG16 Network(imagenet)으로부터 얻은 병목 특징을 소규모 네트워크로 학습
# 최종 결과는 0에서 1 사이의 하나의 값으로 출력(sigmoid)
model = Sequential()
model.add(layers.Flatten(input_shape=train_data.shape[1:]))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))



### 위에 있는 소규모 네트워크 설정
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])


## 위에 있는 소규모 네트워크 학습
history1 = model.fit(train_data, train_labels, epochs=50, batch_size=120,
                     validation_data=(validation_data, validation_labels))


## 계산된 가중치를 저장할 파일(h5)을 생성
top_model_weights_path = 'bottleneck_fc_model.h5'
model.save_weights(top_model_weights_path)


## 모델의 가중치 파일에 대한 경로
weights_path = '../keras/examples/vgg16_weights.h5'





#           --- Fine tuning ---

## Fine tuning을 위해 이미 학습된 VGG16 Network(imagenet)를 다시 불러오고, Input 값을 원래 이미지의 크기(Width: 320, Height: 240, RGB: 3)로 설정
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(320, 240, 3))
print('Model loaded.')


## 다시 한번 소규모 네트워크 구성
top_model = Sequential()
top_model.add(layers.Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(layers.Dense(256, activation='relu'))
top_model.add(layers.Dropout(0.5))
top_model.add(layers.Dense(1, activation='sigmoid'))


## VGG16과 소규모 네트워크에서 학습된 가중치를 다시 불러옴
top_model.load_weights(top_model_weights_path)


## 위에서 설명한 전체 모델에서 데이터가 들어갈 Input과 Output의 형태를 설정
model = Model(input= base_model.input, output= top_model(base_model.output))


## VGG16 윗단(1~15계층) 동결(=안 씀)
for layer in model.layers[:15]:
    layer.trainable = False


### 위에 있는 네트워크 설정
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


## 데이터 증식. 즉, 기존의 이미지를 약간 돌리거나(rotation), 옆으로 이동시키거나(width/height_shift_range), 확대하는(zoom_range) 등의 변형을 가한 데이터를
# 추가로 만듦. 이것은 딥러닝 네트워크가 이미지들의 특징(패턴)을 좀 더 자세히 학습하기 위함임.
train_datagen = ImageDataGenerator(
    rescale= 1./255,
    rotation_range = 2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.1,
    fill_mode = 'constant'
    )

validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dst_name,
    target_size=(320, 240),
    batch_size = 5,
    class_mode = 'binary',
    shuffle=False
    )

validation_generator = validation_datagen.flow_from_directory(
    validation_dst_name,
    target_size=(320, 240),
    batch_size = 5,
    class_mode = 'binary',
    shuffle=False
    )

test_generator = test_datagen.flow_from_directory(
    test_dst_name,
    target_size=(320, 240),
    batch_size = 1,
    class_mode = 'binary',
    shuffle=False,
    seed=42
    )

step_size_train = int(train_generator.n//train_generator.batch_size)
step_size_validation = int(validation_generator.n//validation_generator.batch_size)
step_size_test = int(test_generator.n//test_generator.batch_size)


## 여기서 epochs 값을 더 늘리면(=학습 횟수를 늘리면), 결과가 더 좋아질 것임
history = model.fit_generator(
    train_generator,
    steps_per_epoch=step_size_train,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=step_size_validation
    )


## Test set에 대해서 예측
pred = model.predict_generator(test_generator, steps=step_size_test)


## Accuracy를 설명하면서도 말했지만, 딥러닝 모델의 결과값은 "다른 사람(친구1, 친구2)의 손등일 확률"을 나타냄.
# 그러나 우리 연구에서는 KTH의 손등을 맞추는게 핵심이기 때문에, 딥러닝 모델의 결과값을 "KTH의 손등일 확률"로 바꿈!
pred2 = 1.0 - pred


## "KTH의 손등일 확률"이 50% 이상이면, KTH의 손등이라고 판별
pred2[pred2 >= 0.5] = 1.0


## "KTH의 손등일 확률"이 50% 미만이면, KTH의 손등이 아니라고 판별
pred2[pred2 < 0.5] = 0.0


## "KTH의 손등일 확률"을 소수점 3자리에서 반올림하고, 기존의 Numpy 배열(Array) 형식에서 List 형식으로 변환
pred2 = [round(pred2[i][0], 3) for i in range(len(pred2))]


## Test set의 파일 이름을 List 형식으로 변환
filenames = [test_generator.filenames[i].replace("test_total\\", "").replace('_binary.png', '')[:3] for i in range(len(test_generator.filenames))]


## 결과를 손등의 주인(Filename)이 누구인지와 "KTH의 손등일 확률"를 쌍으로 표현하기 위해 DataFrame 형식(엑셀 형태)으로 변환 및 결과 출력
results2 = pd.DataFrame({"Filename":filenames, "Predictions(KTH일 확률)":pred2})
print(results2)


## 손등의 주인(Filename)이 KTH인 이미지들만 선별하여, 딥러닝 학습 모델이 KTH 손등을 얼마나 잘 예측했는지 보여줌
kth_result = results2[results2['Filename'] == 'kth']
print("KTH임을 맞춘 확률 : %.1f %%" % (sum(kth_result.iloc[:,1]) / len(kth_result.iloc[:,1]) * 100))


## 손등의 주인(Filename)이 다른 사람(others. 즉, hwi나 jbg)인 이미지들만 선별하여, 딥러닝 학습 모델이 KTH 손등을 얼마나 잘 예측했는지 보여줌
others_result = results2[results2['Filename'] != 'kth']
print("KTH이 아님을 맞춘 확률 : %.1f %%" % ((1 - sum(others_result.iloc[:,1]) / len(others_result.iloc[:,1])) * 100))


## 최종 정확도 계산 및 결과 출력
correct = 0
filenames = [test_generator.filenames[i].replace("test_total\\", "").replace('_binary.png', '')[:3] for i in range(len(test_generator.filenames))]
for i, n in enumerate(filenames):
    if n == 'kth' and pred[i][0] <= 0.5:
        correct += 1
    elif n != 'kth' and pred[i][0] > 0.5:
        correct += 1
print("KTH임을 맞춘 확률 : %.1f %%" % (sum(kth_result.iloc[:,1]) / len(kth_result.iloc[:,1]) * 100))
print("KTH이 아님을 맞춘 확률 : %.1f %%" % ((1 - sum(others_result.iloc[:,1]) / len(others_result.iloc[:,1])) * 100))
print("정확도: %.1f %%" % (correct / len(filenames) * 100))


## Precision (정밀도) = TP / (TP + FP)
tp = sum(kth_result.iloc[:,1]) / len(kth_result.iloc[:,1])
tn = (1 - sum(others_result.iloc[:,1]) / len(others_result.iloc[:,1]))
fn = 1.0 - tp
fp = 1.0 - tn
precision = tp / (tp + fp)
print("Precision (정밀도) : %.2f" % precision)


## Recall (재현율) = TP / (TP + FN)
recall = tp / (tp + fn)
print("Recall (재현율) : %.2f" % recall)


## Accuracy (정확도) = ((tp + tn) / (tp + fn + fp + tn))
print("Accuracy (정확도) : %.2f" % ((tp + tn) / (tp + fn + fp + tn)))


## F1 score (Precision과 Recall의 조화평균) = (2 * ((precision * recall) / (precision + recall)))
f1_score = (2 * ((precision * recall) / (precision + recall)))
print("F1 score (Precision과 Recall의 조화평균) : %.2f" % f1_score)


## 학습 정확도(Accuracy)와 손실율(Loss)을 보여주는 그래프(Plot) 생성
# Spyder 4를 사용한다면, Plots 탭에서 그래프를 확인 가능!
# 나중에 결과 보고서를 사용할 때, 해당 그래프를 첨부하면 좋을듯 ^^
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b--', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
































