import torch.utils.data as torchdata
import torch
from nvidia import dali


class PreparedDataset(torchdata.Dataset):

    def __init__(self, metadata, metadata_len, prepare):
        self.metadata = metadata
        self.metadata_len = metadata_len
        self.prepare = prepare

    def __getitem__(self, item):
        return self.prepare(self.metadata[item])

    def __len__(self):
        return self.metadata_len


# TODO: abstract out
# TODO: build in batch size or integrate it w pytorch better
class DALIDataset(dali.pipeline.Pipeline):

    def __init__(self, data_dir, metadata, class_to_index, device='cpu'):
        super(DALIDataset, self).__init__(batch_size=1, num_threads=1, device_id=0)
        self.input = dali.ops.FileReader(file_root=data_dir, file_list='./preprocessed_data.csv')
        self.decode = dali.ops.ImageDecoder() # TODO: make device configurable
        self.cast = dali.ops.Cast(dtype=dali.types.DALIDataType.FLOAT) # TODO: better data type? make settable?
        self.targetcast = dali.ops.Cast(dtype=dali.types.DALIDataType.INT64) # TODO: "
        self.transpose = dali.ops.Transpose(perm=[2, 0, 1], device='gpu')
        self.targetreshape = dali.ops.Reshape(shape=1)
        self.metadata = metadata
        #self.external = dali.ops.ExternalSource()
        self.class_to_index = class_to_index
        self.device = device
        #self.getlabel = dali.ops.PythonFunction(lambda m: m[0][1])

    def define_graph(self):
        img, label = self.input()
        img = self.decode(img)
        img = self.cast(img)
        img = self.transpose(img.gpu())
        return img, self.targetreshape(self.targetcast(label))

    # def iter_setup(self):
    #     self.feed_input(self.external, get_target(lambda x: self.metadata[0][1], self.class_to_index))

    # def __getitem__(self, item):
    #     return self.run()

    def __len__(self):
        return 100 # TODO


def metadata_to_prepared_dataset(metadata, *args, **kwargs):
    return PreparedDataset(
        metadata,
        len(metadata),
        *args,
        **kwargs
    )


def dataset_to_loader(dataset, *args, **kwargs):
    return torchdata.DataLoader(dataset, *args, **kwargs)


# given a function to get a label and a list of class dict columns (the last one being for the label), return a function
# that consumes a metadatum and returns a tensor of the label's class
def get_target(get_label, class_to_index):

    def _get(metadatum):
        label = get_label(metadatum)
        label = class_to_index[-1][label]
        return torch.tensor(label)
    return _get


# composes a closure to get the image and label for a metadatum given functions for getting each separately
def prepare_example(get_image, get_label):
    def _prepare(metadatum):
        return get_image(metadatum), get_label(metadatum)
    return _prepare
