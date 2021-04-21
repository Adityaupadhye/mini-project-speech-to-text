from flask import Flask, render_template, request, redirect
import sounddevice as sd
import soundfile as sf
import os
import onnxruntime as ort
import librosa
import numpy as np
from flaskwebgui import FlaskUI

application = Flask(__name__)
app = application
ui = FlaskUI(app, maximized=True)
classes = ['down', 'no', 'left', 'go', 'off', 'on', 'right', 'stop' , 'up', 'yes']

# start the ort session
ort_session = ort.InferenceSession('trained_model.onnx')

@app.route('/', methods=['GET','POST'])
def index():
    preds = ''
    if request.method == 'POST':
        print('post')
        preds = record()
    
    return render_template('index.html', preds=preds)


def record():
    sr = 16000  # Sample rate
    seconds = 1  # Duration of recording
    print('recording')
    myrecording = sd.rec(int(seconds * sr), samplerate=sr, channels=1, blocking=True)
    # shape of myrecording (16000,1)
    rec = myrecording.reshape([16000,]) # now shape is (16000,) all in 1 row
    rec = librosa.resample(rec, sr, 8000) # (8000,)
    rec = rec.reshape(1,8000,1) # (1,8000,1)
    # testing realtime 
    inputs = {ort_session.get_inputs()[0].name: rec}
    output = ort_session.run(None, inputs)[0].argmax()
    print('realtime output = ', output, classes[output])
    preds = classes[output]
    return 'prediction = '+preds

# def run_model():
#     print('run model')
#     samples, sample_rate = librosa.load('output.wav', sr=16000)
#     print('samples = ', samples, type(samples), np.shape(samples))
#     samples = librosa.resample(samples, sample_rate, 8000)
#     print('sample shape = ', np.shape(samples))
#     inputs = {ort_session.get_inputs()[0].name: samples.reshape(1,8000,1)}
#     output = ort_session.run(None, inputs)[0].argmax()
#     print(output, classes[output])
#     return classes[output]

# print('done')
# sd.wait()  # Wait until recording is finished
# if os.path.exists('output.wav'):
#     os.remove('output.wav')
#     print('removed')
# else: 
#     print('writing 1st time')
# sf.write('output.wav', myrecording, sr)
# return True


# def play():
#     print('playing')
#     data, fs = sf.read('output.wav')
#     sd.play(data, fs)
#     status = sd.wait()
#     print(status)


# @app.route('/start', methods=['GET','POST'])
# def start():
#     print(record()) 
#     play()
#     preds = run_model()
    
#     return render_template('index.html', preds='prediction = '+preds)
#     return redirect('/')


# @app.route('/predict', methods=['GET','POST'])
# def predict():
#     preds = run_model()
    
#     return render_template('index.html', preds='prediction = '+preds)

if __name__ == '__main__':
    ui.run()
    # app.run(debug=True)