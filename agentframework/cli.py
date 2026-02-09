import asyncio

import click

from main import agent1, fitness_agent


@click.group()
def cli() -> None:
    """AgentFramework CLI."""


@cli.command(name="agent1")
def agent1_command() -> None:
    """Run the main agent workflow demo."""
    asyncio.run(agent1())


@cli.command(name="fitness")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
def fitness_command(image_path: str) -> None:
    """Run the fitness agent (image-based macro estimates)."""
    asyncio.run(fitness_agent(image_path))


if __name__ == "__main__":
    cli()
