# Lina-Speech: Gated Linear Attention is a Fast and Parameter-Efficient Learner for text-to-speech synthesis
[[Paper]](https://arxiv.org/abs/2410.23320) [[Audio samples]](https://theodorblackbird.github.io/blog/demo_lina/)

### Authors: Théodor Lemerle, Harrison Vanderbyl, Vaibhav Srivastav, Nicolas Obin, Axel Roebel.

Lina-Speech is a neural codec language model that provides state-of-the-art performances on zero-shot TTS. It replaces self-attention with [Gated Linear Attention](https://arxiv.org/abs/2312.06635), we believe it is a sound choice for audio.
It features: 
- **Voice cloning** with short samples by prompt continuation.
- **High-throughput** : batch inference can go high at no cost on a consumer grade GPU.
- **Initial-State Tuning** (s/o [RWKV](https://github.com/BlinkDL/RWKV-LM) + fast implem by [FLA](https://github.com/sustcsonglin/flash-linear-attention)): fast speaker adaptation by tuning a recurrent state. Save your context window from long prompt !


### Environment setup
```
conda create -n lina python=3.10
conda activate lina

pip install torch==2.5.1
pip install causal-conv1d==1.3.0.post1
pip install -r requirements.txt

ln -s 3rdparty/flash-linear-attention/fla fla
ln -s 3rdparty/encoder encoder
ln -s 3rdparty/decoder decoder

cd 3rdparty/flash-linear-attention
git checkout 739ef15f97cff06366c97dfdf346f2ceaadf05ce
```
### Checkpoints
#### WavTokenizer
You will need this checkpoint of WavTokenizer **and the config file** : [[WavTokenizer-ckpt]](https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/blob/main/wavtokenizer_medium_speech_320_24k.ckpt) [[config file]](https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml)

#### Lina-Speech
Dataset: LibriTTS + LibriTTS-R + MLS-english split (10k hours) + GigaSpeech XL:

169M parameters version trained for 100B tokens: [[Lina-Speech 169M]](https://huggingface.co/lina-speech/all-models/tree/main/lina_gla_gigaspeech_d1024l12_convblind_shortconv_lr2e-4
)

### Inference
See ```InferenceLina.ipynb``` and complete the first cells with the correct checkpoints and config paths.

https://github.com/user-attachments/assets/624288be-73cc-4734-b08e-95792006c7b3

https://github.com/user-attachments/assets/5dde6d53-89a9-4ae3-af46-ae1db3178a11

### Acknowledgments

- The RWKV authors and the community for carrying high-level truly open-source research.
- @SmerkyG for making our life easy at testing cutting edge language model.
- To the [GLA/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) authors for their outstanding work.
- To the [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) authors for releasing such a brilliant speech codec.
- 🤗 for supporting this project.

### Cite
```bib
@misc{lemerle2024linaspeechgatedlinearattention,
      title={Lina-Speech: Gated Linear Attention is a Fast and Parameter-Efficient Learner for text-to-speech synthesis}, 
      author={Théodor Lemerle and Harrison Vanderbyl and Vaibhav Srivastav and Nicolas Obin and Axel Roebel},
      year={2024},
      eprint={2410.23320},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2410.23320}, 
}
```
### Disclaimer

Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.

### IRCAM

This work has been initiated in the [Analysis/Synthesis team of the STMS Laboratory](https://www.stms-lab.fr/team/analyse-et-synthese-des-sons/) at IRCAM, and has been funded by the following project:
- [ANR Exovoices](https://anr.fr/Projet-ANR-21-CE23-0040)

<img align="left" width="150"  src="https://github.com/theodorblackbird/lina-speech/assets/1331899/7391b3c2-ec9a-431e-a090-f2ac5f55026b">
<img align="left" width="150"  src="logo_ircam.jpeg">
<img align="left" width="150" src="https://github.com/theodorblackbird/lina-speech/assets/1331899/74cc1ade-be95-4087-9cc1-83af6d7a54be">
<img align="left" width="150" src="https://github.com/theodorblackbird/lina-speech/assets/1331899/fc0ae259-26ae-451b-8893-80471255479d">


