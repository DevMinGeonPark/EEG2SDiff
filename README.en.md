[한국어](README.md) | English

# EEG2SDiff — EEG Representation Learning for Neural Image Decoding

Reconstructing seen or imagined images from brain signals was my undergraduate thesis topic.
This repository holds the front half of that pipeline: the encoder that turns EEG signals into
a latent representation where image classes become separable.

Presentation: oral talk at the Korea Intelligent Information Systems Society, Spring Conference
2023, "Latent diffusion-based neural decoding system for cognitive simulation."

## What this repo covers

The full pipeline has two stages: an EEG encoder and a latent-diffusion image decoder. Only the
first stage, the encoder, is here. The decoder was the second stage of the thesis and lives
outside this repo. Treat this as research code rather than a finished tool.

## How it works

The input is the ThoughtViz EEG dataset (14 channels, 10 object classes). The model, TripleNet,
stacks two LSTM layers (32 to 128), then a dense projection and L2 normalization to produce an
embedding. Training uses a semi-hard triplet loss, pulling same-class EEG together and pushing
different classes apart. For evaluation it runs KMeans over the embeddings and measures cluster
accuracy with Hungarian matching, plus a t-SNE plot to see how the classes separate.

I used metric learning instead of a plain classifier because EEG is noisy and varies a lot
between subjects. An embedding where class structure shows up as distance is a better fit for a
downstream generative decoder to condition on.

## Run

```bash
python lstm_kmean/train.py        # train TripleNet (TF/Keras + tensorflow-addons)
python lstm_kmean/inference.py    # embed test set, KMeans cluster accuracy, t-SNE
```

Stack: TensorFlow/Keras, tensorflow-addons (TripletSemiHardLoss), scikit-learn.
Undergraduate research (2023); the code is kept as it was at the time.
