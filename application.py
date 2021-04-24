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

if __name__ == '__main__':
    ui.run()
    # app.run(debug=True)