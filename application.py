from flask import Flask, render_template, request, redirect
import sounddevice as sd
import soundfile as sf
import os
import onnxruntime as ort
import librosa

application = Flask(__name__)
app = application
classes = ['down', 'no', 'left', 'go', 'off', 'on', 'right', 'stop' , 'up', 'yes']

@app.route('/', methods=['GET','POST'])
def index():
    preds = ''
    return render_template('index.html', preds=preds)


def run_model():
    print('run model')
    samples, sample_rate = librosa.load('output.wav', sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ort_session = ort.InferenceSession('trained_model.onnx')
    inputs = {ort_session.get_inputs()[0].name: samples.reshape(1,8000,1)}
    output = ort_session.run(None, inputs)[0].argmax()
    print(output, classes[output])
    return classes[output]

def record():
    sr = 16000  # Sample rate
    seconds = 1  # Duration of recording
    print('recording')
    myrecording = sd.rec(int(seconds * sr), samplerate=sr, channels=1, blocking=True)
    print('done')
    sd.wait()  # Wait until recording is finished
    if os.path.exists('output.wav'):
        os.remove('output.wav')
        print('removed')
    else: 
        print('writing 1st time')
    sf.write('output.wav', myrecording, sr)
    return True


def play():
    print('playing')
    data, fs = sf.read('output.wav')
    sd.play(data, fs)
    status = sd.wait()
    print(status)


@app.route('/start', methods=['GET','POST'])
def start():
    print(record()) 
    play()
    return redirect('/')


@app.route('/predict', methods=['GET','POST'])
def predict():
    preds = run_model()
    
    return render_template('index.html', preds='prediction = '+preds)

if __name__ == '__main__':
    app.run(debug=True)