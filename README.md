# DevRewind
 
> This is an experimental Python library designed to establish a relationship between code and actual business by mining development logs, and analyzing with LLM and langchain.

At the current stage, we provide two typical capabilities:

1. Summarizing the functionality of a specific code file.
2. Searching for files related to a specific functionality.

In simple terms, our goal is to establish a connection between code and real-world business scenarios and support bidirectional search, **without touching the real code**.

## Example

With a simple script:

```shell
import click

from dev_rewind import DevRewind, DevRewindConfig

config = DevRewindConfig()

# on your codebase path
config.repo_root = "../requests"

api = DevRewind(config)
agent = api.create_agent()

while True:
    question = click.prompt("Question")
    if question == "exit":
        break
    response = agent.run(input=question)
    click.echo(f"Answer: {response}")
```

and setting a valid `OPENAI_API_KEY` like https://github.com/openai/openai-python#usage:

```shell
export OPENAI_API_KEY=sk-xxxxxxxx
```

And you will get an interactive agent:

```shell
2023-10-31 22:29:17.576 | DEBUG    | dev_rewind.core.agent:create_agent:111 - file docs/user/authentication.rst keywords: ['blacked', 'requires', 'benefit', 'project', 'read', 'openid', 'connect', 'formatted', 'protocol', 'https', 'directly', 'highlighting', 'netrc', 'flow', 'oauth', 'guides', 'malformed', 'kwpolska', 'v1', 'supports', 'meth', 'discoverability', 'images', 'wrap', 'urls', 'required', 'underpinning', 'improve', 'api', 'oauthlib', '1471', 'header', 'twitter', 'folks', 'authentication', 'prefer', 'couple', 'add', 'impression', 'credentials', 'split', 'gmail', 'streaming', 'test_basicauth_with_netrc', 'long', 'test_requests', 'ages', 'server', 'pypi', 'precedence', 'digest', 'spent', '2nd', 'separate', 'ie38844a40ec7a483e6ce5e56077be344242bcd99', 'remove', '2062', 'xauth', 'color', 'basic', 'simple', 'underlying', 'grammar', 'schemes', 'custom', 'explicitly', 'clarifying', 'analytics', 'lines']
2023-10-31 22:29:18.279 | DEBUG    | dev_rewind.core.agent:create_agent:111 - file docs/_static/custom.css keywords: ['compact', 'broken', 'zone', 'footer', 'padding', 'reitz', 'org', 'double', 'css', 'overlap', 'button', 'human', 'request', 'ads', 'framework', 'official', 'expanded', 'carbon', 'moved', 'install', 'correct', 'cpc', 'integrate', 'native', 'space', 'adjust', 'integration', 'vertical', 'width', 'kenneth', 'key', 'direct', 'buysellads', 'placements', 'kennethreitz', 'site', 'design', 'things', 'white', 'custom', 'cta', 'image', 'attempt']
2023-10-31 22:29:18.899 | DEBUG    | dev_rewind.core.agent:create_agent:111 - file docs/community/vulnerabilities.rst keywords: ['red', 'markup', 'early', 'vulnerabilities', 'contact', 'info', 'note', 'clarify', 'cve', 'images', 'urls', 'disclosure', 'longer', 'valid', 'process', '5369', 'fixing', 'jeremycline', 'remove', 'ship', 'vulnerability', 'hat', '5881', 'analytics', 'subdomain']
2023-10-31 22:29:18.970 | DEBUG    | dev_rewind.core.agent:create_agent:114 - keywords ready
Question: 
```

### Ask about a file

```text
Question: Tell me about the feature of tests/test_utils.py
```

Based on the fact we have extracted, it can answer well:

```text
Answer: The file tests/test_utils.py contains keywords related to setting environment variables, bypassing proxies, extracting zipped paths, and comparing uri. It also contains keywords related to renaming, formatting, and parsing files.
```

### Ask about a feature

```text
Question: Which files talk about environment variables?
```

And it should work vice versa.

```text
Answer: The response to your last comment is that the files related to environment variables are tests/utils.py, tests/conftest.py, tests/test_packages.py, .pre-commit-config.yaml, tests/test_hooks.py, tests/__init__.py, tests/test_structures.py, docs/user/advanced.rst, tests/testserver/server.py, and .coveragerc.
```

## Installtion

```bash
pip install DevRewind
```

or just clone this repo. See the "Contribution" part.

## How it works?

<img width="625" alt="image" src="https://github.com/openai/openai-python/assets/13421694/58c90007-845b-4658-be4c-f32db072e718">

The key of this repo is that, it will create a topic list for each file to represent them.

## Contribution

1. clone this repo
2. install [poetry](https://python-poetry.org/)
3. run `poetry install`
4. run `poetry run devr`, which will start an interactive shell

Also, [dev_rewind/test_core.py](dev_rewind/test_core.py) should help.

## License

[Apache 2.0](LICENSE)
