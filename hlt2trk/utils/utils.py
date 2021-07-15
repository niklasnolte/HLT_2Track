from yaml import safe_load


def load_config(file):
    with open(file, 'r') as f:
        configs = safe_load(f)
    return _AttrDict(configs)


class _AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(_AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
