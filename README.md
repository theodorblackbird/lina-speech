# lina-speech (alpha)

Exploring "linear attention" for text-to-speech.

It predicts audio codec "à la" [MusicGen](https://arxiv.org/abs/2306.05284) : delayed residual vector quantizers so that we do not need multiple models.
Featuring [RWKV](https://github.com/BlinkDL/RWKV-LM), [Mamba](https://github.com/state-spaces/mamba), [Gated Linear Attention](https://github.com/sustcsonglin/flash-linear-attention).

Consider this as highly experimental and subject to many changes in the near future.

### Samples

Mamba model (60M parameters) :

https://theodorblackbird.github.io/blog/demo_alpha/

### Prepare a dataset

Soon

### Training

`python train.py fit -c config/libritts-train-360.yaml`

### Acknowledgment

- The RWKV authors and the community around for carrying high-level truly opensource research.
- @SmerkyG for making my life easy at testing cutting edge language model.
- @lucidrains for its huge and easy to understand codebase.

### IRCAM

This work takes place at IRCAM, and is part of the following project :
[ANR Exovoices](https://anr.fr/Projet-ANR-21-CE23-0040)
