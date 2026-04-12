import asyncio

import click

from fitness_agent import fitness_agent
from main import agent1, configure_logging


@click.group()
def cli() -> None:
    """AgentFramework CLI."""
    configure_logging()


@cli.command(name="agent1")
def agent1_command() -> None:
    """Run the main agent workflow demo."""
    asyncio.run(agent1())


@cli.command(name="fitness")
@click.argument("image_path", required=False, type=click.Path(exists=True, dir_okay=False, path_type=str))
def fitness_command(image_path: str | None) -> None:
    """Run the fitness agent in multi-turn mode; optional image path starts with a photo ingestion turn."""
    asyncio.run(fitness_agent(image_path))


if __name__ == "__main__":
    cli()
