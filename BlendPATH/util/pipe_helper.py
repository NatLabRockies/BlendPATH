import copy


def parent_pipe_helper(pipes):
    parent_pipes = {}
    for pipe in pipes:
        name = pipe._parent_pipe if pipe._parent_pipe is not None else pipe.name
        if name in parent_pipes:
            parent_pipes[name].length_km += pipe.length_km
            if pipe.from_node._report_out:
                parent_pipes[name].from_node = pipe.from_node
            if pipe.to_node._report_out:
                parent_pipes[name].to_node = pipe.to_node
        else:
            parent_pipes[name] = copy.copy(pipe)
    return parent_pipes


def get_attr_try(object, val, default=None):
    try:
        return getattr(object, val, default)
    except:
        return default
