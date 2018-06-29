
# Chap07.2 - 텐서플로 추상화와 간소화, TFLearn

> **TFLearn**은 [Chap07.1 Estimator](http://excelsior-cjh.tistory.com/157)에서 살펴본 `tf.estimator`와 마찬가지로 텐서플로의 추상화 라이브러리이다. 이번에는 TFLearn에 대해 알아보도록 하자.

 

## 7.3 TFLearn

### 7.3.1 설치

[**TFLearn**](http://tflearn.org/)은 텐서플로에 포함되어 있지 않기 때문에 별도의 설치가 필요하다. Terminal(또는 cmd창)에 `pip` 명령을 이용해 설치할 수 있다.

```bash
pip install tflearn
```

 

### 7.3.2 CNN 

TFLearn은 [Chap07.1 - tf.estimator](http://excelsior-cjh.tistory.com/157)와 유사하지만, TFLearn을 사용하면 조금 더 깔끔하게 모델을 만들 수 있다. [TFLearn.org](http://tflearn.org/)에서는 TFLearn을 다음과 같이 소개하고 있다.

> - Easy-to-use and understand high-level API for implementing deep neural networks, with tutorial and examples.
- Fast prototyping through highly modular built-in neural network layers, regularizers, optimizers, metrics...
- Full transparency over Tensorflow. All functions are built over tensors and can be used independently of TFLearn.
- Powerful helper functions to train any TensorFlow graph, with support of multiple inputs, outputs and optimizers.
- Easy and beautiful graph visualization, with details about weights, gradients, activations and more...
- Effortless device placement for using multiple CPU/GPU.

 

TFLearn에서의 모델 생성은 `regression()`을 사용하여 래핑되고 마무리된다. `regression()`함수에서 손실함수(`loss`) 및 최적화(`optimizer`)를 설정해준다.

그렇다면, TFLearn을 이용해 MNIST 데이터를 분류하는 CNN 모델을 만들어 보도록 하자.


```python
import numpy as np
import tflearn
import tflearn.datasets.mnist as mnist
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# 데이터를 로딩하고 기본적인 변환을 수행
train_x, train_y, test_x, test_y = mnist.load_data(one_hot=True, 
                                                   data_dir='../data')
train_x = train_x.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

# Building the network
CNN = input_data(shape=[None, 28, 28, 1], name='input')
CNN = conv_2d(CNN, 32, 5, activation='relu', regularizer="L2")
CNN = max_pool_2d(CNN, 2)
CNN = local_response_normalization(CNN)
CNN = conv_2d(CNN, 64, 5, activation='relu', regularizer='L2')
CNN = max_pool_2d(CNN, 2)
CNN = local_response_normalization(CNN)
CNN = fully_connected(CNN, 1024, activation=None)
CNN = dropout(CNN, 0.5)
CNN = fully_connected(CNN, 10, activation='softmax')
CNN = regression(CNN, optimizer='adam', learning_rate=0.0001, 
                 loss='categorical_crossentropy', name='target')

# Training the network
model = tflearn.DNN(CNN, tensorboard_verbose=0, 
                    tensorboard_dir='./MNIST_tflearn_board/', 
                    checkpoint_path='./MNIST_tflearn_checkpoints/checkpoint')
model.fit({'input': train_x}, {'target': train_y}, n_epoch=3,
          validation_set=({'input': test_x}, {'target': test_y}),
          snapshot_step=1000, show_metric=True, run_id='convnet_mnist')
```

    Training Step: 2579  | total loss: [1m[32m0.10814[0m[0m | time: 5.414s
    | Adam | epoch: 003 | loss: 0.10814 - acc: 0.9841 -- iter: 54976/55000
    Training Step: 2580  | total loss: [1m[32m0.10413[0m[0m | time: 6.558s
    | Adam | epoch: 003 | loss: 0.10413 - acc: 0.9826 | val_loss: 0.04778 - val_acc: 0.9861 -- iter: 55000/55000
    --
    INFO:tensorflow:/D/dev/LearningTensorFlow/Chap07-TF_Abstractions/MNIST_tflearn_checkpoints/checkpoint-2580 is not in all_model_checkpoint_paths. Manually adding it.
    INFO:tensorflow:/D/dev/LearningTensorFlow/Chap07-TF_Abstractions/MNIST_tflearn_checkpoints/checkpoint-2580 is not in all_model_checkpoint_paths. Manually adding it.


 

위의 코드에서 `tflearn.DNN()`함수는 `tf.estimator.Estimator()`와 비슷한 기능을 하는데, `regression()`으로 래핑된 모델을 인스턴스화하고 만들어진 모델을 전달하는 역할을 한다. 또한 텐서보드(TensorBoard)와 체크포인트(checkpoint) 디렉터리 등을 설정할 수 있다. 모델 적합 연산은 `.fit()` 메서드를 이용해 수행된다. 

모델 적합(`.fit()`), 즉 학습이 완료되면, 다음과 같은 메소드를 이용해 모델을 평가, 예측, 저장 및 불러오기 등을 수행할 수 있다.

| 메서드                         | 설명                                               |
| ------------------------------ | -------------------------------------------------- |
| `evaluate(X, Y, batch_size)`   | 주어진 샘플에서 모델을 평가                        |
| `fit(X, Y, n_epoch)`           | 입력 feature `X`와 타겟 `Y`를 모델에 적용하여 학습 |
| `get_weights(weight_tensor)`   | 변수의 가중치를 반환                               |
| `load(model_file)`             | 학습된 모델 가중치를 불러오기                      |
| `predict(x)`                   | 주어진 `x` 데이터를 모델을 이용해 예측             |
| `save(model_file)`             | 학습된 모델 가중치를 저장                          |
| `set_weights(tensor, weights)` | 주어진 값을 텐서 변수에 할당                       |


```python
# Evaluate the network
evaluation = model.evaluate(X=test_x, Y=test_y, batch_size=128)
print(evaluation)

# Predict
pred = model.predict(test_x)
accuarcy = (np.argmax(pred, 1)==np.argmax(test_y, 1)).mean()
print(accuarcy)
```

    [0.9861]
    0.9861


 

### 7.3.3. RNN

이번에는 TFLearn을 이용해 RNN을 구현해 보도록하자. 구현할 RNN 모델은 영화 리뷰에 대한 감성분석으로, 리뷰에 대해 좋거나/나쁘거나 두 개의 클래스를 분류하는 모델이다. 데이터는 학습 및 테스트 데이터가 각각 25,000개로 이루어진 [IMDb](https://www.imdb.com/interfaces/) 리뷰 데이터를 사용한다.


```python
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# IMDb 데이터셋 로드
(train_x, train_y), (test_x, test_y), _ = imdb.load_data(path='../data/imdb.pkl', 
                                                         n_words=10000,
                                                         valid_portion=0.1)
```

 

위에서 불러온 IMDb 데이터는 각각 다른 시퀀스 길이를 가지고 있으므로 최대 시퀀스 길이를 100으로 하여 `tflearn.data_utils.pad_sequences()`를 사용해 제로 패딩으로 시퀀스의 길이를 맞춰준다.


```python
# Sequence padding and Converting labels to binary vectors
train_x = pad_sequences(train_x, maxlen=100, value=0.)
test_x = pad_sequences(test_x, maxlen=100, value=0.)
train_y = to_categorical(train_y, nb_classes=2)
test_y = to_categorical(test_y, nb_classes=2)
```

 

그런다음, `tflearn.embedding()`으로 벡터 공간으로의 임베딩을 수행한다. 아래의 코드에서 확인할 수 있듯이 각 단어는 128 크기인 벡터에 매핑된다. 이렇게 임베딩된 결과를 `LSTM` layer와 `fully_connected` layer를 추가해 모델을 구성해준다.


```python
# Building a LSTM network
# Embedding
RNN = tflearn.input_data([None, 100])
RNN = tflearn.embedding(RNN, input_dim=10000, output_dim=128)

# LSTM Cell
RNN = tflearn.lstm(RNN, 128, dropout=0.8)
RNN = tflearn.fully_connected(RNN, 2, activation='softmax')
RNN = tflearn.regression(RNN, optimizer='adam', 
                         learning_rate=0.001, loss='categorical_crossentropy')

# Training the network
model = tflearn.DNN(RNN, tensorboard_verbose=0, 
                    tensorboard_dir='./IMDb-tflearn_board/')
model.fit(train_x, train_y, 
          validation_set=(test_x, test_y), 
          show_metric=True, batch_size=32)
```

    Training Step: 7039  | total loss: [1m[32m0.05996[0m[0m | time: 22.310s
    | Adam | epoch: 010 | loss: 0.05996 - acc: 0.9827 -- iter: 22496/22500
    Training Step: 7040  | total loss: [1m[32m0.05474[0m[0m | time: 23.425s
    | Adam | epoch: 010 | loss: 0.05474 - acc: 0.9845 | val_loss: 0.85953 - val_acc: 0.8064 -- iter: 22500/22500
    --



```python
# evaluate the network
evaluation = model.evaluate(test_x, test_y, batch_size=128)
print(evaluation)
```

    [0.8063999997138978]


 

아래의 코드는 위의 코드를 합친 전체 코드이다.


```python
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# IMDb 데이터셋 로드
(train_x, train_y), (test_x, test_y), _ = imdb.load_data(path='../data/imdb.pkl', 
                                                         n_words=10000,
                                                         valid_portion=0.1)


# Sequence padding and Converting labels to binary vectors
train_x = pad_sequences(train_x, maxlen=100, value=0.)
test_x = pad_sequences(test_x, maxlen=100, value=0.)
train_y = to_categorical(train_y, nb_classes=2)
test_y = to_categorical(test_y, nb_classes=2)

# Building a LSTM network
# Embedding
RNN = tflearn.input_data([None, 100])
RNN = tflearn.embedding(RNN, input_dim=10000, output_dim=128)

# LSTM Cell
RNN = tflearn.lstm(RNN, 128, dropout=0.8)
RNN = tflearn.fully_connected(RNN, 2, activation='softmax')
RNN = tflearn.regression(RNN, optimizer='adam', 
                         learning_rate=0.001, loss='categorical_crossentropy')

# Training the network
model = tflearn.DNN(RNN, tensorboard_verbose=0, 
                    tensorboard_dir='./IMDb-tflearn_board/')
model.fit(train_x, train_y, 
          validation_set=(test_x, test_y), 
          show_metric=True, batch_size=32)

# evaluate the network
evaluation = model.evaluate(test_x, test_y, batch_size=128)
print(evaluation)
```

 

### 7.3.4 정리

이번 포스팅에서는 TFLearn을 살펴보고, 이를 이용해 CNN과 RNN을 구현해 보았다. 이 외에도 TFLearn에 대한 사용법 및 예제는 http://tflearn.org/ 에서 확인할 수 있다.
