from flask import Flask, render_template, request, redirect
import sounddevice as sd
from playsound import playsound
from keras.models import load_model
# import tensorflow as tf
# from scipy.io.wavfile import write
import soundfile as sf
import os
import run_model

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')


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
    # write('voice/output.wav', sr, myrecording)  # Save as WAV file 

def play():
    print('playing')
    data, fs = sf.read('voice/output.wav')
    sd.play(data, fs)
    status = sd.wait()
    print(status)
    # playsound('voice/output.wav')

@app.route('/start', methods=['GET','POST'])
def start():
    print(record()) 
    play()
    return redirect('/')

print('line before predict')
new_model = load_model('saved_models/trained_model.h5')

@app.route('/predict', methods=['GET','POST'])
def predict():
    preds = run_model.preprocessing(new_model, 'voice/output.wav')
    return render_template('preds.html', preds=preds)

if __name__ == '__main__':
    app.run(debug=True)