from flask import Flask, render_template, request,send_file
from scipy.io import wavfile
from speech_Verification import Speaker_Recognition
from speechbrain.pretrained import SepformerSeparation as separator
# from keras.models import model_from_json
from IPython.display import Audio
import numpy as np
import base64


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

# def SpeechEnhancemt():
#     if request.method == 'POST':
#         file1 = request.files['file1']
        
#         file1.save("a1.wav")

#         model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='pretrained_models/sepformer-whamr-enhancement')
#         enhanced_speech = model.separate_file('a1.wav') 
#         prediction = Audio(enhanced_speech[:, :].detach().cpu().squeeze(), rate=8000)
#         print(prediction)
#         return render_template('SpeechEnhancment.html',prediction=prediction)

#     return render_template('SpeechEnhancment.html')

# @app.route('/SpeakerEmotion',methods=['GET','POST'])
# def SpeakerEmotion():
#   if request.method == 'POST': 
#     file = request.form['file1']
#     file.save("file.wav")
#     json_file = open('model.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights("model3.h5")
#     print("Loaded model from disk")
#     p = loaded_model.predict(file)
#     return render_template('SpeakerEmotion.html', p= p)
#   return render_template('SpeakerEmotion.html')

if __name__ == '__main__':
    app.run(debug=True)


