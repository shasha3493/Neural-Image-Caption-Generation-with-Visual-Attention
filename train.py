import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# Data parameters
data_folder = './data'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimensions of decoder hidden state
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# benchmark mode is good whenever your input sizes for your network do not vary. This way,
#  cudnn will look for the optimal set of algorithms for that particular configuration 
# (which takes some time). This usually leads to faster runtime.
# But if your input sizes changes at each iteration, then cudnn will benchmark every time a 
# new size appears, possibly leading to worse runtime performances.
cudnn.benchmark = True  

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for, in case early stopping is not triggered
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 80
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none


def main():
    """
    Training and validation.
    """

    # In Python, global keyword allows you to modify the variable outside of the current scope. 
    # It is used to create a global variable and make changes to the variable in a local context.
    '''
    The basic rules for global keyword in Python are:

    When we create a variable inside a function, it is local by default.
    When we define a variable outside of a function, it is global by default. You don't have to use global keyword.
    We use global keyword to read and write a global variable inside a function.
    Use of global keyword outside a function has no effect.
    
    '''

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        
        '''
        The filter() method constructs an iterator from elements of an iterable for which a function returns true.

        The filter() method takes two parameters:

        function - function that tests if elements of an iterable returns true or false
                    If None, the function defaults to Identity function - which returns false if any elements are false

        iterable - iterable which is to be filtered, could be sets, lists, tuples, or containers of any iterators

        The filter() method returns an iterator that passed the function check for each element in the iterable.

        '''
        
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # If there's no improvement in Bleu score for 20 epochs then stop training
        if epochs_since_improvement == 20:
            break
        
        # If there's no improvement in Bleu score for 8 epochs lower the lr
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training

    train_loader: data loader for training data
    encoder: encoder model
    decoder: decoder model
    criterion: loss fn
    encoder_optimizer: optimizer to update encoder's weights if fine tune
    decoder-optimizer: optimizer to update decoder's weights
    epoch: epoch number

    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    

    start = time.time()

    # Looping over mini-batches
    # imgs: (Batch_size, 3, 256, 256)
    # caps: (Batch_size, max_len + 2 (start and end token))
    # caplens: (Batch_size, 1)
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs) # Output of the encoder
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # imgs: (batch_size, 14,14,2048)
        # scores: (batch_size, max_decode_length, vocab_size)
        # caps_sorted: (batch_size, max_len + 2 (start and end token))
        # decode_lengths: list of length = batch_size
        # alphas: (batch_size, max_decode_lengths, 14*14)
        # sort_ind: (80,)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:] # (batch_size, max_len + 1 (start token))

        # Remove timesteps that we didn't decode at, or are pads
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0] # (n, vocab_size) where n is the total number of tokens which aren't <pad>
        targets= pack_padded_sequence(targets, decode_lengths, batch_first=True)[0] # (n, )

        # Calculate loss
        loss = criterion(scores, targets)

        # Doubly stochastic attention
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,))
def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation

    val_loader: Data loader for validation set
    encoder: encoder model
    decoder: decoder model
    criterion: loss layer

    Returns:
    bleu4: BLEU-4 score

    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():

        # imgs: (Batch_size, 3, 256, 256)
        # caps: (Batch_size, max_len + 2 (start and end token))
        # caplens: (Batch_size, 1)
        # all_caps: (Batch_size, cpi, max_len + 2 (start and end token))

        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # imgs: (batch_size, 14,14,2048)
            # scores: (batch_size, max_decode_length, vocab_size)
            # caps_sorted: (batch_size, max_len + 2 (start and end token))
            # decode_lengths: list of length = batch_size
            # alphas: (batch_size, max_decode_lengths, 14*14)
            # sort_ind: (80,)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[[ref1a], [ref1b], [ref1c]], [[ref2a], [ref2b], [ref2c]]...], hypotheses = [[hyp1], [hyp2], ...]
            # where ref1a, ref1b and ref1c are three captions of 1st image  
            # We have to create the above in this format only as corpus_bleu() expects it

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            
            # Going over every image
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist() # img_caps is a list of length cpi where every element is a list of length max_len + 2
                
                # Going over every caption of the image and reving the <start> and <pad>'s
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist() # preds is a list of length = batch_size, every elements is a list of length = max_decode_length
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # predictions were done only done for tokens within the cap_lens, max_len - cap_lens were initialized to 0 (<pad>). Therfore, removing <pad>s
            preds = temp_preds
            hypotheses.extend(preds)

            # During validation, numbers of tokens in predicted sequence is taken to be the same as
            # cap_len. Now, in the true caption,first token is <start> and last token is <end> token. 
            # However, predicted seq starts with the token after <start>. So, we remove <start> from 
            # true caption and <pad>'s in the predicted seq.


            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print('\n * LOSS - {loss.avg:.3f}, BLEU-4 - {bleu}\n'.format(loss=losses, bleu=bleu4))
    
    return bleu4


if __name__ == '__main__':
    main()
