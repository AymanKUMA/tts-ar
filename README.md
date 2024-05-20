# tts-arabic-Darija-pytorch

TTS models (Tacotron2, FastPitch) based on [tts-arabic-pytorch](https://github.com/nipponjo/tts-arabic-pytorch) by nipponjo, trained on Wamid darija dataset, , including the [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan) for direct TTS inference.

Papers:

Tacotron2 | Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions ([arXiv](https://arxiv.org/abs/1712.05884))

FastPitch | FastPitch: Parallel Text-to-speech with Pitch Prediction ([arXiv](https://arxiv.org/abs/2006.06873))

HiFi-GAN  | HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis ([arXiv](https://arxiv.org/abs/2010.05646))

## Required packages:
`torch torchaudio pyyaml`

for tarining 
`librosa matplotlib tensorboard`

## How to train the model

* Clone the [github repository](https://github.com/AymanKUMA/tts-ar.git).
* using the BunnyCLI download the dataset in a seperated folder named `/dataset` outside the `/tts-ar`.

### pre-processing 

* Generate the `orthographic-tarscript` in the file using the [Buckwalter transliteration code](https://github.com/Similar-Intelligence/development-arabic-tts/tree/develop/buckwalter_trans)

#### 1 - Preprocess text: 
- Replace the `orthographic-tarscript` file in the `/tts-ar/data` directory and run the script `preprocess_text.py` in the `/tts-ar/scripts` directory.

#### 2 - Preprocess audio:
- Run the scripts `preprocess_audio.py` and `extract_f0.py` respectfuly

### Configuration:
- Change the configuration files as desired including the restore model, checkpoint directory, batch size, epochs and so on.
- Make sure if the restore model path is the following : `restore_model: ''` download the [FastPitch checkpoint (PyTorch, AMP, LJSpeech-1.1, 22050Hz)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/fastpitch__pyt_ckpt) and save it in the `/tts-ar/models` directory, otherwise load the desired checkpoint into the pretrained directory and change the path in the configuration file.

### Run the training: 
You can Run the training as follows: `python3 train_fp_adv.py --config /configs/wamid_config.py` 

