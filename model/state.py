

# get whether this module and each submodule recursively is in train or evaluation mode
def get_train_mode_tree(module):
    return [m.training for m in module.modules()]


# set the train or evaluation state of this module and each submodule recursively
def set_train_mode_tree(module, mode):
    for m, modl in zip(mode, module.modules()):
        modl.train(m)
