from flask import Flask, render_template, request,send_file
from scipy.io import wavfile
from speech_Verification import Speaker_Recognition
from speechbrain.pretrained import SepformerSeparation as separator
from keras.models import model_from_json
from IPython.display import Audio
import numpy as np
import base64
import os 
import pandas as pd
import librosa
from json import encoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
#--------------------------------------------------------

ravdess = "/"
ravdess_directory_list = os.listdir(ravdess)
file_emotion = []
file_path = []

for i in ravdess_directory_list:
    actor = os.listdir(ravdess + i)
    for f in actor:
        part = f.split('.')[0].split('-')
        file_emotion.append(int(part[2]))
        file_path.append(ravdess + i + '/' + f)

emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])


ravdess_df = pd.concat([emotion_df, path_df], axis=1)
ravdess_df.Emotions.replace({1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',
                             8:'Surprise'},
                            inplace=True)

data,sr = librosa.load(file_path[0])

emotions1 = {1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Fear', 7: 'Disgust', 8: 'Surprise'}

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)



def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)


def pitch(data,sampling_rate,pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


def feat_ext(data):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr,n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
    return result


def get_feat(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    # normal data
    res1 = feat_ext(data)
    result = np.array(res1)
    #data with noise
    noise_data = noise(data)
    res2 = feat_ext(noise_data)
    result = np.vstack((result, res2))
    #data with stretch and pitch
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = feat_ext(data_stretch_pitch)
    result = np.vstack((result, res3))
    return result

from joblib import Parallel, delayed
import timeit
start = timeit.default_timer()
# Define a function to get features for a single audio file


def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=feat_ext(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,41))
    scaler=StandardScaler()
    i_result = scaler.transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    return final_result


def feat_ext(data):

    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr,n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
    return result

def get_feat(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    # normal data
    res1 = feat_ext(data)
    result = np.array(res1)
    #data with noise
    noise_data = noise(data)
    res2 = feat_ext(noise_data)
    result = np.vstack((result, res2))
    #data with stretch and pitch
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = feat_ext(data_stretch_pitch)
    result = np.vstack((result, res3))
    return result

def predictioninput(path1):

    with open('model.json', 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    # Load the model weights from a .h5 file
    model.load_weights('model.h5')
    res=get_predict_feat(path1)
    predictions=model.predict(res)
    y_pred = encoder.inverse_transform(predictions)
    print(y_pred[0][0])
    return y_pred[0][0]


#--------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/SpeakerVerification', methods=['GET', 'POST'])
def SpeakerVerification():
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        file1.save("a1.wav")
        file2.save("a2.wav")

        score, prediction = Speaker_Recognition("a1.wav", "a2.wav")
        print(prediction)
        value = bool(prediction)
        if value==True:
            prediction= 'Same Person'
        else:
            prediction= 'Diffrent Person'

        return render_template('SpeakerVerification.html',prediction=prediction)

    return render_template('SpeakerVerification.html')

@app.route('/SpeechEnhancment', methods=['GET', 'POST'])
def SpeechEnhancement():
    if request.method == 'POST':
        file1 = request.files['file1']
        file1.save("a1.wav")

        model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='pretrained_models/sepformer-whamr-enhancement')
        enhanced_speech = model.separate_file('a1.wav')

        enhanced_speech = enhanced_speech.squeeze().detach().cpu().numpy()
        enhanced_speech = enhanced_speech / np.max(np.abs(enhanced_speech))

        enhanced_file = "enhanced.wav"
        wavfile.write(enhanced_file, 8000, enhanced_speech.astype('float32'))

        return send_file(enhanced_file, mimetype="audio/wav", as_attachment=True)

    return render_template('SpeechEnhancment.html')


@app.route('/SpeakerEmotion',methods=['GET','POST'])
def SpeakerEmotion():
  if request.method == 'POST': 
    file2 = request.files['file1']
    file2.save("file.wav")
    # Load the model architecture from a .json file
    with open('model.json', 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    # Load the model weights from a .h5 file
    model.load_weights('model.h5')
    output = predictioninput("file.wav")
    # prediction = model.predict(ee)
    return render_template('SpeakerEmotion.html.html', data=output)
  return render_template('SpeakerEmotion.html')

if __name__ == '__main__':
    app.run(debug=True)


