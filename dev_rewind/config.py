from enum import Enum

from pydantic import BaseModel


class FileLevelEnum(str, Enum):
    FILE: str = "FILE"
    DIR: str = "DIR"


class DevRewindConfig(BaseModel):
    repo_root: str = "."

    # set it to -1 will query all the commits
    max_depth_limit: int = 32

    # keyword limit for each file
    keyword_limit: int = 10

    include_regex: str = ""
    file_level: FileLevelEnum = FileLevelEnum.FILE
    cache: str = ".devrewind_cache.json"
