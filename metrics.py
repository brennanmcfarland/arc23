import torch
from __types import MetricFuncs


# TODO: if we need to add a bind step, make sure it's a consistent interface across all metrics
def category_accuracy() -> MetricFuncs:
    closure = CategoryAccuracy()
    return MetricFuncs(on_item=closure.on_item, on_end=closure.on_end)


# helper for calc_category_accuracy, since it's stateful
class CategoryAccuracy:

    def __init__(self):
        self.correct: int = 0
        self.total: int = 0

    def on_item(self, inputs, outputs, gtruth) -> None:
        _, predicted = torch.max(outputs.data, 1)
        # TODO: I don't think these are right
        self.total += gtruth.size(0)
        self.correct += (predicted == gtruth).sum().item()

    def on_end(self) -> float:
        return self.correct / self.total
