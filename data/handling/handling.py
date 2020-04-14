from typing import Iterable, Callable, List, Dict

import torch.utils.data as torchdata
import torch

from __types import Loader


class PreparedDataset(torchdata.Dataset):

    def __init__(self, metadata, metadata_len: int, prepare: Callable):
        self.metadata = metadata
        self.metadata_len: int = metadata_len
        self.prepare: Callable = prepare

    def __getitem__(self, item):
        return self.prepare(self.metadata[item])

    def __len__(self):
        return self.metadata_len


def metadata_to_prepared_dataset(metadata, *args, **kwargs) -> PreparedDataset:
    return PreparedDataset(
        metadata,
        len(metadata),
        *args,
        **kwargs
    )


def dataset_to_loader(dataset: torchdata.Dataset, *args, **kwargs) -> Loader:
    return torchdata.DataLoader(dataset, *args, **kwargs)


# given a function to get a label and a list of class dict columns (the last one being for the label), return a function
# that consumes a metadatum and returns a tensor of the label's class
def get_target(get_label: Callable, class_to_index: List[Dict[str, int]]) -> Callable:

    def _get(metadatum):
        label = get_label(metadatum)
        label = class_to_index[-1][label]
        return torch.tensor(label)
    return _get


# composes a closure to get the image and label for a metadatum given functions for getting each separately
def prepare_example(get_image: Callable, get_label: Callable) -> Callable:
    def _prepare(metadatum):
        return get_image(metadatum), get_label(metadatum)
    return _prepare


# given a transform function that given a metadatum returns a line to write, apply it to each metadatum and write
# the output to the given file
def write_transformed_metadata_to_file(metadata: Iterable, out_path: str, transform: Callable) -> None:
    with open(out_path, 'w+', newline='', encoding="utf8") as out_file:
        for metadatum in metadata:
            out_file.write(transform(metadatum))
