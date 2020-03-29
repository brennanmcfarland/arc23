import torch.utils.tensorboard as tensorboard
from tabulate import tabulate


def tensorboard_writer(*args, **kwargs):
    return tensorboard.SummaryWriter(*args, **kwargs)


# TODO: make it accept variable args so it can be used in non-step callbacks
def record_tensorboard_scalar(scalar_provider, tensorboard_writer):
    def _bind(steps_per_epoch):
        scalar_func = scalar_provider(steps_per_epoch)

        def _run(loss, step, epoch):
            datapoint = scalar_func(loss, step, epoch)
            return tensorboard_writer.add_scalar(datapoint["label"], datapoint["value"], epoch * steps_per_epoch + step)
        return _run
    return _bind


# prints the data with epoch/step information; only meant to be called on a step
def print_with_step(step_data_provider):
    def _bind(steps_per_epoch):
        step_data_func = step_data_provider(steps_per_epoch)

        def _run(loss, step, epoch):
            datapoint = step_data_func(loss, step, epoch)
            print(
                'EPOCH ', epoch,
                ' STEP ', step, '/', steps_per_epoch,
                ' ',
                datapoint["label"], datapoint["value"]
            )
        return _run
    return _bind


def print_line(line_provider):
    def _bind(steps_per_epoch):
        line_func = line_provider(steps_per_epoch)

        def _run(*args, **kwargs):
            datapoint = line_func(*args, **kwargs)
            print(datapoint["label"] + ': ' + ''.join([str(i) for i in datapoint["value"]]))
        return _run
    return _bind


def print_table(table_provider, headers=None, title=None):
    def _bind(steps_per_epoch):
        table_func = table_provider(steps_per_epoch)

        def _run(*args, **kwargs):
            table = table_func(*args, **kwargs)
            _print_table(table, headers, title)
        return _run
    return _bind


def print_tables(tables_provider, titles, headers):
    def _bind(steps_per_epoch):
        tables_func = tables_provider(steps_per_epoch)

        def _run(*args, **kwargs):
            tables = tables_func(*args, **kwargs)
            for table, title in zip(tables, titles):
                _print_table(table, headers, title)
        return _run
    return _bind


def _print_table(table, headers, title):
    if title is None:
        print(table["label"])
    else:
        print(title)
    print(tabulate(table["value"], headers))
