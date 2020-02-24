import torch.utils.data as torchdata
from nvidia import dali
from nvidia.dali.plugin.pytorch import DALIGenericIterator


# an iterable dataset using a DALI pipeline for efficiency
# expects a pipeline closure, ie, a function closure that when called returns the next sample
# NOTE: outputs from the DALI pipeline are still owned by DALI and will be invalidated after the next iterator call,
# so if these outputs need preservation they must be copied before that
class DALIIterableDataset(torchdata.IterableDataset):
    def __init__(self, pipeline_closure, metadata, batch_size, *args, **kwargs):
        super(DALIIterableDataset).__init__()
        self.dali_pipeline = _DALIDataset(pipeline_closure, batch_size, *args, **kwargs)
        self.iterator = None
        self.iter = None
        self.len_metadata = len(metadata)
        self.index = 0
        self.batch_size = batch_size

    def build(self, variable_names=None):
        if variable_names is None:
            variable_names = ['inputs', 'labels']
        self.dali_pipeline.build()
        self.iterator = DALIGenericIterator(self.dali_pipeline, variable_names, self.len_metadata)

    def __iter__(self):
        self.dali_pipeline.reset()
        self.iterator.reset()
        self.iter = iter(self.iterator)
        return self

    # TODO: multiple workers, see https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    def __next__(self):
        # StopIteration bubbles up from iter
        self.index += 1
        batch = next(self.iter)[0]  # default: {'inputs': inputs, 'labels': labels}
        return batch

    def __len__(self):
        return self.len_metadata // self.batch_size


# the implementation of the DALI pipeline for a DALI dataset
# expects a pipeline closure, ie, a function closure that when called returns the next sample
class _DALIDataset(dali.pipeline.Pipeline):

    def __init__(self, pipeline_closure, batch_size, *args, num_threads=1, **kwargs):
        super(_DALIDataset, self).__init__(num_threads=num_threads, device_id=0, batch_size=batch_size, *args, **kwargs)
        self.dali_pipeline = pipeline_closure

    def define_graph(self):
        return self.dali_pipeline()


# a standard DALI pipeline closure for the common case of loading and preformatting images and their respective labels
def dali_standard_image_classification_pipeline(data_dir, metadata_filename):
    input = dali.ops.FileReader(file_root=data_dir, file_list=metadata_filename, random_shuffle=True)
    decode = dali.ops.ImageDecoder()
    cast = dali.ops.Cast(dtype=dali.types.DALIDataType.FLOAT)  # TODO: better data type?
    targetcast = dali.ops.Cast(dtype=dali.types.DALIDataType.INT64)  # TODO: better data type?
    transpose = dali.ops.Transpose(perm=[2, 0, 1], device='gpu')
    targetreshape = dali.ops.Reshape(shape=1)

    def _apply():
        img, label = input()
        img = decode(img)
        img = cast(img)
        img = transpose(img.gpu())
        return img, targetreshape(targetcast(label))
    return _apply
