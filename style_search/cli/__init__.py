"""Style Search CLI tools."""

import click

from style_search.cli.admin import add_user, get_user, list_users, remove_user
from style_search.cli.embed import embed
from style_search.cli.model_info import model_info
from style_search.cli.scrape import scrape
from style_search.cli.sync import sync
from style_search.cli.train import train_cmd
from style_search.db import init_db


@click.group()
def cli():
    """Style Search CLI tools."""
    init_db()


cli.add_command(embed)
cli.add_command(scrape)
cli.add_command(model_info)
cli.add_command(train_cmd, "train")
cli.add_command(add_user)
cli.add_command(list_users)
cli.add_command(get_user)
cli.add_command(remove_user)
cli.add_command(sync)


def main():
    cli()
