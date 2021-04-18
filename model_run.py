import onnxruntime as ort
import librosa

classes = ['down', 'no', 'left', 'go', 'off', 'on', 'right', 'stop', 'up', 'yes']

def run_model():
    print('run model')
    samples, sample_rate = librosa.load('voice/output.wav', sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ort_session = ort.InferenceSession('saved_models/trained_model.onnx')
    inputs = {ort_session.get_inputs()[0].name: samples.reshape(1,8000,1)}
    output = ort_session.run(None, inputs)[0].argmax()
    print(output, classes[output])
    return classes[output]

if __name__ == '__main__':
    print('run')
    run_model()