import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates a json file which contains word2id for all the words in all the captions of all the images whose 
    count >= min_word_freq and '<unk>', '<start>', '<end>', '<pad>' tokens

    Creates input files for training, validation, and test data.

    For each training, validation and test, it creates the following files:
        .h5py file: contains all the images for training/validation/test
        .json file: list of caps_per_image*number of images captions, each caption is padded to have max_len
        .json file: list of caption lengths for corresponding captions in the other .json file. Caption lengths incluse <start> and <end> but excludes <pad> if any 

    dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    karpathy_json_path: path of Karpathy JSON file with splits and captions
    image_folder: folder with downloaded images
    captions_per_image: number of captions to sample per image
    min_word_freq: words occuring less frequently than this threshold are binned as <unk>
    output_folder: folder to save files
    max_len: don't sample captions longer than this length 

    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image

    # list of image paths that belong to the train set, where every element is a string
    train_image_paths = []
    # list of all captions for the corresponding images in train_image_path. Every element is again a list 'captions'(list of list) 
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    '''
    >> print(data['images'][0])

    {
    'filepath': 'val2014', 
    'sentids': [770337, 771687, 772707, 776154, 781998], 
    'filename': 'COCO_val2014_000000391895.jpg', 
    'imgid': 0, 
    'split': 'test', 
    'sentences': [{'tokens': ['a', 'man', 'with', 'a', 'red', 'helmet', 'on', 'a', 'small', 'moped', 'on', 'a', 'dirt', 'road'], 'raw': 'A man with a red helmet on a small moped on a dirt road. ', 'imgid': 0, 'sentid': 770337},
     {'tokens': ['man', 'riding', 'a', 'motor', 'bike', 'on', 'a', 'dirt', 'road', 'on', 'the', 'countryside'], 'raw': 'Man riding a motor bike on a dirt road on the countryside.', 'imgid': 0, 'sentid': 771687}, 
     {'tokens': ['a', 'man', 'riding', 'on', 'the', 'back', 'of', 'a', 'motorcycle'], 'raw': 'A man riding on the back of a motorcycle.', 'imgid': 0, 'sentid': 772707}, 
     {'tokens': ['a', 'dirt', 'path', 'with', 'a', 'young', 'person', 'on', 'a', 'motor', 'bike', 'rests', 'to', 'the', 'foreground', 'of', 'a', 'verdant', 'area', 'with', 'a', 'bridge', 'and', 'a', 'background', 'of', 'cloud', 'wreathed', 'mountains'], 'raw': 'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', 'imgid': 0, 'sentid': 776154},
     {'tokens': ['a', 'man', 'in', 'a', 'red', 'shirt', 'and', 'a', 'red', 'hat', 'is', 'on', 'a', 'motorcycle', 'on', 'a', 'hill', 'side'], 'raw': 'A man in a red shirt and a red hat is on a motorcycle on a hill side.', 'imgid': 0, 'sentid': 781998}], 
     'cocoid': 391895}

     NOTE: In Kartpathy json train-val split, some of the images from the val set is labelled as 'TEST'. So, these
     images are taken as test images.

    '''
    # Looping through every image
    for img in data['images']:
        captions = [] # List of all the captions for an image. Every element is a list of tokens of captions of an image

        # Looping over every caption of an image
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:

                # Appending a caption of the image in the captions list. captions is a list of lists where every inside list is one of the captions of an image
                captions.append(c['tokens'])

        # if there are no captions for a particular image, don't consider it in the model
        if len(captions) == 0:
            continue

        # path of the particular image     
        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        # Appending file path and captions of the image based on if the image is from test/train/val set.
        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)# it is of form [ [[cap1 of img1], [cap2 of img1]], [[cap1 of img2], [cap2 of img2]] ]
        
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # List of words whose count >= min_word_freq
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    
    # Create word map: dictionary of word 2 id. This is done only for words whose count >= min_word_freq
    word_map = {k: v + 1 for v, k in enumerate(words)}
    
    # Adding '<unk>', '<start>', '<end>', '<pad>' tokens to the word map dictionary
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        # 'a' mode Read/write if exists
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            # Reading all the image paths for TRAIN/VAL/TEST
            for i, path in enumerate(tqdm(impaths)):

                # Sample captions

                # Case 1: if total no. of captions < captions per image then keep all the captions and 
                # sample captions_per_image - total_captions from total number of captions

                # Case 2: if total no. of captions > captions per image then sample captions_per_image 
                # from total_captions 

                # captions is a list of list, where len(captions) = captions_per_image
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                # Looping over every captions of an image i.e. caps_per_image captions sampled for an image, encoding it and saving it in enc_captions
                for j, c in enumerate(captions):
                    # Encode captions, padded with '<pad>' if number of tokens is less than max len
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2 # +2 is for start and end tokens

                    enc_captions.append(enc_c) #enc_captions is a list of list wherein every element is a list of encoded tokens. It contains total_images*caps_per_image lists 
                    caplens.append(c_len) # caplens is a list of length of corresponding captions in enc_captions

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model

    emb_file: file containing embeddings (stored in Glove format)
    word_map: word map

    Returns:
    embeddings: in the same order as the words in the word map
    emb_dim: embedding size

    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradient computed during backpropagation to avoid exlosion of gradients

    optimizer: optimizer with the gradients to be clipped
    grad_clip: clip_value

     new value = -grad clip     if old val < -grad_clip
                 old value      if -grad_clip < old val < grad_clip
                 grad clip      if old val > grad_clip     

    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint

    data_name: base name of processed dataset
    epoch: epoch number
    epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    encoder: encoder model
    decoder: decoder model
    encoder_optimizer: optimizer to update encoder's weights if fine tune
    decoder_optimizer: optimizer to update decoder's weights 
    bleu4: validaton BLEU-4 score for this epoch
    is_best: is this checkpoint the best so far

    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks the learning rate by a specified factor

    optimizer: optimizer whose learning rate is to be shrunk
    shrink_factor: factor in interval (0,1) to multiply learning rate with

    ######################################################################################
    When you initialize the optimizer using

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    or similar, pytorch creates one param_group. The learning rate is accessible via param_group['lr'] and 
    the list of parameters is accessible via param_group['params']

    If you want different learning rates for different parameters, you can initialise the optimizer like this.

    optim.SGD([
                    {'params': model.base.parameters()},
                    {'params': model.classifier.parameters(), 'lr': 1e-3}
                ], lr=1e-2, momentum=0.9)
    This creates two parameter groups with different learning rates. That is the reason for having param_groups.
    ########################################################################################
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


