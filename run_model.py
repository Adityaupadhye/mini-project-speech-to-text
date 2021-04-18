import os
import librosa
import numpy as np

classes = ['down', 'no', 'left', 'go', 'off', 'on', 'right', 'stop', 'up', 'yes']

def preprocessing(new_model, path):
    # global model
    print('pre processing data', type(new_model))
    samples, sample_rate = librosa.load(path, sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    # print(samples, sample_rate, np.shape(samples)
    # samples = samples.reshape(1,124,129)
    # print('shape = ', np.shape(samples))
    # new_model.summary()
    prediction = new_model.predict(samples.reshape(1,8000,1))
    print(prediction[0])
    idx = np.argmax(prediction[0])
    print(idx, classes[idx])

    return classes[idx]


# print(tf.__version__)
# print('hellp')
# preprocessing()  
# model_run()