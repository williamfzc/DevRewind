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

    # file scope
    include_regex: str = ""

    # also allow analyzing in dir level
    file_level: FileLevelEnum = FileLevelEnum.FILE

    # cache dir for chromadb / tinydb
    cache_dir: str = ".devrewind_cache"

    # pre-extract keywords before startup if enabled
    # it really helps to search by features
    # also take some token COST if you're using something like OPENAI, of course
    pre_extract: bool = False
