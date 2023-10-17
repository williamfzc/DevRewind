import typing

from git import Commit


class RuntimeContext(object):
    def __init__(self):
        self.files: typing.Dict[str, typing.List[Commit]] = dict()
