
# Neural Image Caption Generation with Visual Attention

# Introduction

A model is trained that can generate a descriptive caption for the image provided. The model is able to automatically learn to fix its gaze on salient objects through attention mechanism while generating the corresponding words in the output sequence. The model is trained with teacher-forcing and decoding is done using beam search to get the best possible caption.

The 101 layered Residual Network trained on the ImageNet classification task is used as Encoder and single layer, unidirectional LSTM network is used as the decoder to generate captions.

To evaluate the model's performance on the validation set, I used the automated BiLingual Evaluation Understudy (BLEU) evaluation metric. This evaluates a generated caption against reference caption(s).


# DataSet

The MSCOCO '14 Dataset. You'd need to download the [Training](https://images.cocodataset.org/zips/train2014.zip) and [Validation](https://images.cocodataset.org/zips/val2014.zip) images.

I used [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). This zip file contain the captions. You will also find splits and captions for the Flicker8k and Flicker30k datasets, so feel free to use these instead of MSCOCO if the latter is too large for your computer.


# Dependencies

### Install

The Code is written in [Python 3.7](https://www.python.org/downloads/) . If you don't have Python installed you can find it here. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.The code is written in [PyTorch version 1.2.0](https://pytorch.org/docs/1.2.0/)

To install pip run in the command Line
```bash
python -m ensurepip -- default-pip
```
to upgrade it
```bash
python -m pip install -- upgrade pip setuptools wheel
```
to upgrade Python
```bash
pip install python -- upgrade
```

# Usage

You can [clone]((https://github.com/shasha3493/Neural-Image-Caption-Generation-with-Visual-Attention.git) the repository

### Prerequisites

1. Download the MSCOCO '14 Dataset: [Training](https://images.cocodataset.org/zips/train2014.zip) and [Validation](https://images.cocodataset.org/zips/val2014.zip), unzip and place it under images folder.

2. Download [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip), unzip and place it undercaption_datasets folder

Then run the following commands in order from your working directory in order to reproduce the results.

1. Splits images and captions into train, validation and test set and creates vocabulary
```bash 
python create_input_files.py
```
2. Trains the model
```bash
python train.py
```

The model was trained in stages. I first trained only the Decoder, i.e. without fine-tuning the Encoder, with a batch size of 80. I trained for 20 epochs, and the BLEU-4 score peaked at about 23.25 at the 13th epoch. I used the Adam() optimizer with an initial learning rate of 4e-4.

I continued from the 13th epoch checkpoint allowing fine-tuning of the Encoder with a batch size of 32. The smaller batch size is because the model is now larger because it contains the Encoder's gradients. With fine-tuning, the score rose to 24.47. I used Adam() for the Encoder as well, but with a learning rate of 1e-4, which is a tenth of the default value for this optimizer.

To control batch size, fine tuninig encoder and checkpoint path, corresponding values can be assigned to varibles 'batch_size', 'fine_tune' and 'checkpoint' in [train.py](https://github.com/shasha3493/Neural-Image-Caption-Generation-with-Visual-Attention/blob/master/train.py)

You can download this pretrained model [here](https://github.com/shasha3493/Neural-Image-Caption-Generation-with-Visual-Attention/tree/master/best%20validation%20BLEU's%20checkpoint).

# Results

BLEU-4 Scores on validation and test data can be found [here](https://github.com/shasha3493/Neural-Image-Caption-Generation-with-Visual-Attention/blob/master/results/BLEU-4%20scores/BLEU-4.csv)

Some examples of attention visualization can be found [here](https://github.com/shasha3493/Neural-Image-Caption-Generation-with-Visual-Attention/tree/master/results/Attention%20Visualization)
 


```python

```
