import os
import requests
from PIL import Image
from io import BytesIO
import csv
from typing import Iterable, List, Tuple, Dict, Callable, Union, Collection


# pull the image from the api endpoint and save it if we don't have it, else load it from disk
def get_img_from_file_or_url(img_format: str = 'JPEG') -> Callable[[str, str], Image.Image]:
    def _apply(filepath: str, url: str) -> Image.Image:
        img = from_file(filepath)
        if img is None:
            img = from_url(url)
            img.save(filepath, img_format)
        return img.convert('RGB')  # convert to rgb if not already (eg if grayscale)
    return _apply


def from_url(url: str) -> Image.Image:
    api_response = requests.get(url).content
    response_bytes = BytesIO(api_response)
    return Image.open(response_bytes)


def from_file(path: str) -> Union[Image.Image, None]:
    if os.path.exists(path):
        return Image.open(path)
    else:
        return None


def load_metadata(path: str, cols: Iterable[int], class_cols: Collection[int] = tuple(), valid_only: bool = True, **reader_args)\
        -> Tuple[List, int, List, List[Dict[str, int]], List[Dict[int, str]], int]:
    metadata = []
    # one dict for each class col
    class_to_index: List[Dict[str, int]] = [{}] * len(class_cols)
    index_to_class: List[Dict[int, str]] = [{}] * len(class_cols)
    next_indices = [0] * len(class_cols)  # next index for a new class value
    with open(path, 'r', newline='', encoding="utf8") as metadata_file:
        reader = csv.reader(metadata_file, **reader_args)
        headers = next(reader)
        for row in reader:
            if len(row) != 0:
                metadatum = [row[c] for c in cols]
                # for all class cols, add their vals to the class_to_index and index_to_class dicts if not there already
                for c, class_col in enumerate(class_cols):
                    if not row[class_col] in class_to_index[c]:
                        class_to_index[c][row[class_col]] = next_indices[c]
                        index_to_class[c][next_indices[c]] = row[class_col]
                        next_indices[c] += 1
                if valid_only and '' in metadatum:
                    continue
                metadata.append(metadatum)
    len_metadata = len(metadata)
    num_classes = 0 if len(next_indices) == 0 else next_indices[-1]
    # split off the headers
    return metadata, len_metadata, headers, class_to_index, index_to_class, num_classes
