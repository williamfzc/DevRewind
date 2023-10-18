import click

from dev_rewind import DevRewind


@click.command()
def interact():
    click.echo("welcome to DevRewind")

    dev_rewind = DevRewind()
    qa_chain = dev_rewind.create_chain()

    while True:
        question = click.prompt("Question")
        if question == "exit":
            break
        output = qa_chain.run(question)
        click.echo(f"Answer: {output}")


if __name__ == "__main__":
    interact()
