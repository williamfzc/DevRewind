import os

import pytest

from dev_rewind import DevRewind, DevRewindConfig


@pytest.fixture
def meta_fixture():
    config = DevRewindConfig()
    config.repo_root = os.path.dirname(__file__)
    api = DevRewind(config)
    return api.collect_metadata()


def test_agent():
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("no openapi key for test")

    api = DevRewind()
    agent = api.create_agent()
    response = agent.run(input="context.py 的功能是什么？")
    print(response)
