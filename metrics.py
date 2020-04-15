from typing import Dict, TypeVar, Generic, Iterable

import torch

from __types import MetricFuncs


# NOTE: the calling function is expected to handle whether the gradient is tracked for tensors passed in


T = TypeVar("T")


# TODO: if we need to add a bind step, make sure it's a consistent interface across all metrics
def category_accuracy() -> MetricFuncs:
    closure = CategoryAccuracy()
    return MetricFuncs(on_item=closure.on_item, on_end=closure.on_end)


def accuracy_by_category(categories: Iterable[T]) -> MetricFuncs:
    closure = AccuracyByCategory(categories)
    return MetricFuncs(on_item=closure.on_item, on_end=closure.on_end)


# TODO: accuracy matrix, for % of each true category categorized as predicted category?


# helper for category_accuracy, since it's stateful
class CategoryAccuracy:

    def __init__(self):
        self.correct: int = 0
        self.total: int = 0

    def on_item(self, inputs, outputs, gtruth) -> None:
        _, predicted = torch.max(outputs.data, 1)
        self.total += gtruth.size(0)
        self.correct += (predicted == gtruth).sum().item()

    def on_end(self) -> float:
        return self.correct / self.total


# helper for category_accuracy, since it's stateful
class AccuracyByCategory(Generic[T]):

    def __init__(self, categories: Iterable[T]):
        self.correct: Dict[T, int] = {c: 0 for c in categories}
        self.total: Dict[T, int] = {c: 0 for c in categories}

    def on_item(self, inputs, outputs, gtruth) -> None:
        _, predicted = torch.max(outputs.data, 1)
        for e_predicted, e_gtruth in zip(predicted, gtruth):
            i_predicted, i_gtruth = e_predicted.item(), e_gtruth.item()
            self.total[i_gtruth] += 1
            self.correct[i_gtruth] += int((i_predicted == i_gtruth))

    def on_end(self) -> Dict[T, float]:
        return {k: self.correct[k] / (self.total[k] if self.total[k] != 0 else -1) for k in self.total.keys()}
