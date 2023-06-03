import numpy as np
from tokenizers import Tokenizer

from datumaro.components.annotation import HashKey
from datumaro.components.dataset import Dataset
from datumaro.components.media import MediaElement

templates = [
    "a photo of a {}.",
]

cifar10_templates = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]

cifar100_templates = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]

caltech101_templates = [
    "a photo of a {}.",
    "a painting of a {}.",
    "a plastic {}.",
    "a sculpture of a {}.",
    "a sketch of a {}.",
    "a tattoo of a {}.",
    "a toy {}.",
    "a rendition of a {}.",
    "a embroidered {}.",
    "a cartoon {}.",
    "a {} in a video game.",
    "a plushie {}.",
    "a origami {}.",
    "art of a {}.",
    "graffiti of a {}.",
    "a drawing of a {}.",
    "a doodle of a {}.",
    "a photo of the {}.",
    "a painting of the {}.",
    "the plastic {}.",
    "a sculpture of the {}.",
    "a sketch of the {}.",
    "a tattoo of the {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "the embroidered {}.",
    "the cartoon {}.",
    "the {} in a video game.",
    "the plushie {}.",
    "the origami {}.",
    "art of the {}.",
    "graffiti of the {}.",
    "a drawing of the {}.",
    "a doodle of the {}.",
]

eurosat_templates = [
    "a centered satellite photo of {}.",
    "a centered satellite photo of a {}.",
    "a centered satellite photo of the {}.",
]

flowers101_templates = [
    "a photo of a {}, a type of flower.",
]

food101_templates = [
    "a photo of {}, a type of food.",
]

kitti_templates = [
    "{}",
]

kinetics_templates = [
    "a photo of {}.",
    "a photo of a person {}.",
    "a photo of a person using {}.",
    "a photo of a person doing {}.",
    "a photo of a person during {}.",
    "a photo of a person performing {}.",
    "a photo of a person practicing {}.",
    "a video of {}.",
    "a video of a person {}.",
    "a video of a person using {}.",
    "a video of a person doing {}.",
    "a video of a person during {}.",
    "a video of a person performing {}.",
    "a video of a person practicing {}.",
    "a example of {}.",
    "a example of a person {}.",
    "a example of a person using {}.",
    "a example of a person doing {}.",
    "a example of a person during {}.",
    "a example of a person performing {}.",
    "a example of a person practicing {}.",
    "a demonstration of {}.",
    "a demonstration of a person {}.",
    "a demonstration of a person using {}.",
    "a demonstration of a person doing {}.",
    "a demonstration of a person during {}.",
    "a demonstration of a person performing {}.",
    "a demonstration of a person practicing {}.",
]

mnist_templates = [
    'a photo of the number: "{}".',
]

format_templates = {
    "cifar10": cifar10_templates,
    "cifar100": cifar100_templates,
    "caltech101": caltech101_templates,
    "eurosat": eurosat_templates,
    "flowers101": flowers101_templates,
    "food101": food101_templates,
    "kitti": kitti_templates,
    "kinetics": kinetics_templates,
    "mnist": mnist_templates,
}

def tokenize(tokenizer, texts: str, context_length: int = 77, truncate: bool = True):
    if not tokenizer:
        checkpoint = "openai/clip-vit-base-patch32"
        tokenizer = Tokenizer.from_pretrained(checkpoint)
    tokens = tokenizer.encode(texts).ids
    result = np.zeros((1, context_length))

    if len(tokens) > context_length:
        if truncate:
            eot_token = tokens.ids[-1]
            tokens = tokens[:context_length]
            tokens[-1] = eot_token

    for i, token in enumerate(tokens):
        result[:, i] = token
    return result, tokenizer


def compute_hash(features):
    features = np.sign(features)
    hash_key = np.clip(features, 0, None)
    hash_key = hash_key.astype(np.uint8)
    hash_key = np.packbits(hash_key, axis=-1)
    return hash_key


def select_uninferenced_dataset(dataset):
    uninferenced_dataset = Dataset(media_type=MediaElement)
    for item in dataset:
        if not any(isinstance(annotation, HashKey) for annotation in item.annotations):
            uninferenced_dataset.put(item)
    return uninferenced_dataset

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1]  # max inner product value
    distH = 0.5 * (q - B1 @ B2.transpose())
    return distH
