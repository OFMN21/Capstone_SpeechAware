from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.dataio.dataio import read_audio



def Speaker_Recognition(v1) :

    model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='pretrained_models/sepformer-whamr-enhancement')
    enhanced_speech = model.separate_file('audios\lwpwlpwlpwl.wav') 

    return score , prediction