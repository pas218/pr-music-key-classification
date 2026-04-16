from torch.utils.data import Dataset, Subset, random_split

def get_first_3_channels_lambda(x):
    return x[:3, :, :]

labels_map = {
    0: 'Ab:maj',
    1: 'A:maj',
    2: 'Bb:maj',
    3: 'B:maj',
    4: 'C:maj',
    5: 'Db:maj',
    6: 'D:maj',
    7: 'Eb:maj',    
    8: 'E:maj',
    9: 'F:maj',
    10: 'Gb:maj',
    11: 'G:maj',
    12: 'Ab:min',
    13: 'A:min',
    14: 'Bb:min',
    15: 'B:min',
    16: 'C:min',
    17: 'Db:min',
    18: 'D:min',
    19: 'Eb:min',
    20: 'E:min',
    21: 'F:min',
    22: 'Gb:min',
    23: 'G:min'
}

soundfont_map = {
    0: "arachnosf",
    1: "fzero",
    2: "genuser",
    3: "pokemonredgreen",
    4: "sonic2piano"
}

def scaleup_tuple(value: int, scaleup: int) -> tuple:
    # turns a midi idx to all wav/spect idxs (Ex. 0 --> 0, 1, 2, 3, ..., num_soundfonts)
    export_tuple = tuple(scaleup*value + i for i in range(scaleup))
    return export_tuple

def datasetsplit_coupled(spect_dataset: Dataset, uncoupled_length: int, soundfont_num: int,  proportions: list) -> tuple[Subset, Subset]:
    initial_list = range(uncoupled_length)
    coupled_datasets = random_split(initial_list, proportions)
    full_dataset_subsets = [None] * len(proportions)

    for subset_idx, subset in enumerate(coupled_datasets):
        subset_list = list(subset)
        dataset_split = [idx for item in subset_list for idx in scaleup_tuple(item, soundfont_num)]
        full_dataset_subsets[subset_idx] = Subset(spect_dataset, dataset_split)
    
    return full_dataset_subsets


if __name__ == '__main__':
    trainset, valset = datasetsplit_coupled(range(50), 10, 5, [.8, .2])
    print(list(trainset))
    print(list(valset))
    # scaleup_tuple(0, 5)