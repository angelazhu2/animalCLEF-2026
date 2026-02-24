import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
import timm
from transformers import AutoModel
from sklearn.cluster import DBSCAN
from wildlife_datasets.datasets import AnimalCLEF2026
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity


root = '/Users/angelazhu/code/courses/ai535_deeplearning/animalCLEF_project/animalCLEF-2026/'


def load_datasets():
    dataset_full = AnimalCLEF2026(
        root,
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False
    )
    # showing metadata
    #print(dataset_full.metadata.head())
    #print(dataset_full.metadata[['dataset', 'split']].value_counts(sort=False))

    # get test data
    dataset_full = dataset_full.get_subset(dataset_full.df['split'] == 'test')

    datasets = {}
    for name in dataset_full.metadata['dataset'].unique():
        datasets[name] = dataset_full.get_subset(dataset_full.df['dataset'] == name)
    return datasets


def visualize_datasets(datasets):
    for dataset in datasets.values():
        dataset.plot_grid(n_rows=3, n_cols=4, rotate=False)
        plt.show()


def run_similarity(datasets):
    # code for running megadescriptor
    #device = 'cuda'
    device = 'cpu'
    batch_size = 32

    similarities = {}
    for name, dataset in datasets.items():
        # Select the model for feature extraction
        if name in ['SalamanderID2025', 'SeaTurtleID2022']:
            model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval()
            size = 384
        elif name in ['LynxID2025', 'TexasHornedLizards']:
            model = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True)
            size = 512
        else:
            raise ValueError('Name does not exist')

        # Set the extractor and transform for the images
        matcher = CosineSimilarity()
        extractor = DeepFeatures(model=model, device=device, batch_size=batch_size)
        transform = T.Compose([
            T.Resize(size=(size, size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Set the transform for images
        dataset.set_transform(transform)
        # Extract features
        features = extractor(dataset)
        # Compute the similarity matrix
        similarity = matcher(features, features)
        similarities[name] = similarity
    return similarities


if __name__ == '__main__':
    datasets = load_datasets()
    print(datasets)
    #visualize_datasets(datasets)
    run_similarity(datasets)