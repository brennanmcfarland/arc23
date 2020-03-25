def run_with_hooks(net, run_func, hook_funcs, wrapper_funcs):
    hooks = []

    def get_register_func(direction):
        if direction == "forward":
            return lambda l: l.register_forward_hook
        elif direction == "backward":
            return lambda l: l.register_backward_hook
        else:
            raise ValueError("Invalid pass direction for running with hooks")

    def _add_hook(layer):
        tmp = {}

        for hook_func, wrapper_func in zip(hook_funcs, wrapper_funcs):
            func, direction = hook_func()
            hooks.append(get_register_func(direction)(layer)(lambda l, i, o: wrapper_func(func(l, i, o))))
            # if direction not in tmp.keys():
            #     tmp[direction] = []
        #     tmp[direction].append(lambda l, i, o: wrapper_func(func(l, i, o)))
        # def tmp_rn(t, l, i, o):
        #     for ti in t:
        #         ti(l, i, o)
        # for tmp_name, tmp_funcs in tmp.items():
        #     hooks.append(get_register_func(tmp_name)(layer)(lambda l, i, o: [t(l, i, o) for t in tmp_funcs]))

    def _run():
        net.apply(_add_hook)
        run_func()
        for h in hooks:
            h.remove()
    return _run
