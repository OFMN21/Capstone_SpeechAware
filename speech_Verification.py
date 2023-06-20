
from speechbrain.pretrained import SpeakerRecognition



def Speaker_Recognition(v1, v2) :
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    score, prediction = verification.verify_files(v1, v2)
    return score , prediction