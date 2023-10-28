import click
import time

from dev_rewind import DevRewind


@click.command()
def cli_click():
    click.echo("welcome to DevRewind")

    api = DevRewind()
    agent = api.create_agent()

    # avoid the prompt coming faster than log ...
    time.sleep(0.1)

    while True:
        question = click.prompt("Question")
        if question == "exit":
            break
        response = agent.run(input=question)
        click.echo(f"Answer: {response}")


if __name__ == "__main__":
    cli_click()
