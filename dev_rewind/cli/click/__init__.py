import click
import time

from dev_rewind import DevRewind


@click.command()
def cli_click():
    click.echo("welcome to DevRewind")

    dev_rewind = DevRewind()
    qa_chain = dev_rewind.create_router_chain()

    # avoid the prompt coming faster than log ...
    time.sleep(0.1)

    while True:
        question = click.prompt("Question")
        if question == "exit":
            break
        output = qa_chain.run(question)
        click.echo(f"Answer: {output}")


if __name__ == "__main__":
    cli_click()
