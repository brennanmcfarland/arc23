import torch.utils.tensorboard as tensorboard
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
from typing import Iterable, Callable, Any, Sequence

from __types import Bindable, OutputBinding, Table, LabelledValue


def tensorboard_writer(*args, **kwargs) -> tensorboard.SummaryWriter:
    return tensorboard.SummaryWriter(*args, **kwargs)


# TODO: make it accept variable args so it can be used in non-step callbacks
def scalar_to_tensorboard(
        scalar_provider: Bindable[OutputBinding, Callable],
        tensorboard_writer: tensorboard.SummaryWriter
) -> Bindable[OutputBinding, Callable]:
    def _bind(steps_per_epoch):
        scalar_func = scalar_provider(steps_per_epoch)

        def _run(loss, step, epoch):
            datapoint = scalar_func(loss, step, epoch)
            return tensorboard_writer.add_scalar(datapoint.label, datapoint.value, epoch * steps_per_epoch + step)
        return _run
    return _bind


def image_to_tensorboard(
        img_func: Callable[[Any], LabelledValue],
        tensorboard_writer: tensorboard.SummaryWriter
) -> Bindable[OutputBinding, Callable]:
    def _bind(steps_per_epoch):
        def _run(*args, **kwargs):
            img = img_func(*args, **kwargs)
            return tensorboard_writer.add_image(img.label, img.value)
        return _run
    return _bind


def images_to_tensorboard(
        img_func: Callable[[Any], LabelledValue],
        tensorboard_writer: tensorboard.SummaryWriter
) -> Bindable[OutputBinding, Callable]:
    def _bind(steps_per_epoch):
        def _run(*args, **kwargs):
            img = img_func(*args, **kwargs)
            return tensorboard_writer.add_images(img.label, img.value)
        return _run
    return _bind


def matrix_to_csv(
        matrix_provider: Bindable[OutputBinding, Callable],
        out_path: str,
        labels: Sequence[str] = None,
) -> Bindable[OutputBinding, Callable]:
    def _bind(steps_per_epoch):
        matrix_func = matrix_provider(steps_per_epoch)

        def _run(*args, **kwargs):
            matrix = matrix_func(*args, **kwargs)
            pd.DataFrame(matrix, columns=labels).to_csv(out_path + '.csv', index=False)
        return _run
    return _bind


# prints the data with epoch/step information; only meant to be called on a step
def print_with_step(step_data_provider: Bindable[OutputBinding, Callable]) -> Bindable[OutputBinding, Callable]:
    def _bind(steps_per_epoch):
        step_data_func = step_data_provider(steps_per_epoch)

        def _run(loss, step, epoch):
            datapoint = step_data_func(loss, step, epoch)
            print(
                'EPOCH ', epoch,
                ' STEP ', step, '/', steps_per_epoch,
                ' ',
                datapoint.label, datapoint.value
            )
            return datapoint.value
        return _run
    return _bind


def print_line(line_provider: Bindable[OutputBinding, Callable]) -> Bindable[OutputBinding, Callable]:
    def _bind(steps_per_epoch):
        line_func = line_provider(steps_per_epoch)

        def _run(*args, **kwargs):
            datapoint = line_func(*args, **kwargs)
            print(datapoint.label + ': ' + ''.join([str(i) for i in datapoint.value]))
            return datapoint.value
        return _run
    return _bind


def print_table(table_provider: Bindable[OutputBinding, Callable], headers: Sequence[str] = None, title: str = None
                ) -> Bindable[OutputBinding, Callable[[Any], Table]]:
    def _bind(steps_per_epoch):
        table_func = table_provider(steps_per_epoch)

        def _run(*args, **kwargs):
            table = table_func(*args, **kwargs)
            _print_table(table, headers, title)
            return table
        return _run
    return _bind


def print_tables(tables_provider: Bindable[OutputBinding, Callable], titles: Iterable[str], headers: Sequence[str]
                 ) -> Bindable[OutputBinding, Callable[[Any], Table]]:
    def _bind(steps_per_epoch):
        tables_func = tables_provider(steps_per_epoch)

        def _run(*args, **kwargs):
            tables = tables_func(*args, **kwargs)
            for table, title in zip(tables, titles):
                _print_table(table, headers, title)
            return tables
        return _run
    return _bind


def _print_table(table: Table, headers: Sequence[str], title: str) -> Table:
    if title is None:
        print(table.label)
    else:
        print(title)
    print(tabulate(table.value, headers))
    return table
