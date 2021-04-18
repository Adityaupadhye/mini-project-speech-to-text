from flask import Flask, render_template, request, redirect
import sounddevice as sd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
import soundfile as sf
import run_model

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    preds = ''
    return render_template('index.html', preds=preds)


def record():
    sr = 16000  # Sample rate
    seconds = 1  # Duration of recording
    print('recording')
    myrecording = sd.rec(int(seconds * sr), samplerate=sr, channels=1, blocking=True)
    print('done')
    sd.wait()  # Wait until recording is finished
    if os.path.exists('voice/output.wav'):
        os.remove('voice/output.wav')
        print('removed')
    else: 
        print('writing 1st time')
    sf.write('voice/output.wav', myrecording, sr)
    return True

def play():
    print('playing')
    data, fs = sf.read('voice/output.wav')
    sd.play(data, fs)
    status = sd.wait()
    print(status)

@app.route('/start', methods=['GET','POST'])
def start():
    print(record()) 
    play()
    return redirect('/')

new_model = load_model('saved_models/trained_model.h5')
print('loaded model')

@app.route('/predict', methods=['GET','POST'])
def predict():
    preds = run_model.preprocessing(new_model, 'voice/output.wav')
    
    return render_template('index.html', preds='prediction = '+preds)

if __name__ == '__main__':
    app.run(debug=True)