def initialize_class_with_kwargs(class_obj, **kwargs):
    parameters = class_obj.__init__.__code__.co_varnames
    kwargs = dict(filter(lambda e: e[0] in parameters, kwargs.items()))
    return class_obj(**kwargs)
