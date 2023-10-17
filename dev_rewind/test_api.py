from dev_rewind import DevRewind, DevRewindConfig


def test_api():
    config = DevRewindConfig()
    config.repo_root = ".."
    api = DevRewind(config)
    api.rewind()
