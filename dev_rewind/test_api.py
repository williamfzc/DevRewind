from dev_rewind import DevRewind, DevRewindConfig


def test_document():
    config = DevRewindConfig()
    config.repo_root = ".."
    api = DevRewind(config)
    docs = api.collect_documents()
    assert docs

