from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
import scipy
import librosa
import argparse
import pandas as pd
import numpy as np
import pickle as pkl 
import torch
import torchaudio
import torchvision
from PIL import Image
import os
from joblib import dump
import densenet
####################################################################################
sampling_rate=22050
path= '/content/recording.wav'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
####################################################################################
def extract_spectrogram(clip):
	clip, sr = librosa.load(path, sr=sampling_rate)
	num_channels = 3
	window_sizes = [25, 50, 100]
	hop_sizes = [10, 25, 50]

	specs = []

	for i in range(num_channels):
		window_length = int(round(window_sizes[i]*sampling_rate/1000))
		hop_length = int(round(hop_sizes[i]*sampling_rate/1000))

		clip = torch.Tensor(clip)
		spec = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_fft=2205, win_length=window_length, hop_length=hop_length, n_mels=128)(clip) #Check this otherwise use 2400
		eps = 1e-6
		spec = spec.numpy()
		spec = np.log(spec+ eps)
		spec = np.asarray(torchvision.transforms.Resize((128, 1500))(Image.fromarray(spec)))
		specs.append(spec)
	return specs
#####################################################################################

#####################################################################################
AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_btn.appendChild(t);
//my_p.appendChild(my_btn);
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  gumStream = stream;
  var options = {
    //bitsPerSecond: 8000, //chrome seems to ignore, always 48k
    mimeType : 'audio/webm;codecs=opus'
    //mimeType : 'audio/webm;codecs=pcm'
  };            
  //recorder = new MediaRecorder(stream, options);
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {            
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data); 
    reader.onloadend = function() {
      base64data = reader.result;
      //console.log("Inside FileReader:" + base64data);
    }
  };
  recorder.start();
  };

recordButton.innerText = "Recording... press to stop";

navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);


function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Saving the recording... pls wait!"
  }
}

// https://stackoverflow.com/a/951057
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var data = new Promise(resolve=>{
//recordButton.addEventListener("click", toggleRecording);
recordButton.onclick = ()=>{
toggleRecording()

sleep(2000).then(() => {
  // wait 2000ms for the data to be available...
  // ideally this should use something like await...
  //console.log("Inside data:" + base64data)
  resolve(base64data.toString())

});

}
});
      
</script>
"""

def get_audio():
  display(HTML(AUDIO_HTML))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  
  process = (ffmpeg
    .input('pipe:0')
    .output('pipe:1', format='wav')
    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
  )
  output, err = process.communicate(input=binary)
  
  riff_chunk_size = len(output) - 8
  # Break up the chunk size into four bytes, held in b.
  q = riff_chunk_size
  b = []
  for i in range(4):
      q, r = divmod(q, 256)
      b.append(r)

  # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
  riff = output[:4] + bytes(b) + output[8:]

  sr, audio = wav_read(io.BytesIO(riff))

  return audio, sr
##############################################################################################################  
model = densenet.DenseNet(pretrained=True).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
# to load
checkpoint = torch.load('/content/model_best_diagnostic.pth.tar',map_location=device)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()
#################################################################################################################
class_mapping= ['allergie',
 'amblyopie',
 'astigmatisme hypermétropique',
 'anisometropie',
 'astigmatisme mixte',
 'astigmatisme myopique',
 'chalasion',
 'dacriocystite aigue',
 'décollement de rétine',
 'epiphora',
 'esotropie alternante',
 'exotropie alternante',
 'forte myopie',
 'glaucome',
 'hypermétropie',
 'syndrome de Gougerot-Sjögren',
 'kératite herpétique disciform',
 'kératite herpétique',
 'kératocône',
 'kératopathie du pseudophake',
 'limbo conjonctivite endémique des tropiques',
 'monophtalme',
 'myopie',
 'néovaisseau choroïdien',
 'neuropathie optique',
 'norb',
 'nystagmus',
 'OBVR',
 'oedeme papillaire',
 'opere de chalasion',
 'opere de ptérygion + greffe',
 'opere de ptérygion_ simple',
 'opere de strabisme',
 'opere de trichiasis',
 'OVCR',
 'phaco',
 'presbytie',
 'pseudophakie',
 'ptosis',
 'rétinite pigmentaire',
 'rétinopathie diabétique',
 'rosacée oculaire',
 'sécheresse oculaire',
 'surface',
 'trachome',
 'traumatisme',
 'uvéite antérieure',
 'uvéite intermédiaire',
 'uvéite postérieure',
 'uvéite totale',
 'zona',
 'trichiasis',
 'cataracte',
 'Ptyrégion',
 'Gougerot-Sjögren',
 'supprimer',
 'valider',
 'OTHER']
 #################################################################################################################
 def predict(input, class_mapping):
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
    return predicted
###################################################################################################################
audio, sr = get_audio()
scipy.io.wavfile.write('recording.wav', sr, audio)
sample=extract_spectrogram(path)
input = torch.tensor(sample)
input.unsqueeze_(0)
predicted = predict(input,class_mapping)
print(f"Predicted: '{predicted}'")
