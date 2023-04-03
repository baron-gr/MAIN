import typer
from rich.console import Console
from rich.table import Table

console = Console()

app = typer.Typer()

@app.command(short_help='adds an item')
def add(task: str, category: str):
    typer.echo(f"Adding {task} to {category}")

@app.command()
def delete(position: int):
    typer.echo(f"Deleting {position}")

@app.command()
def update(position: int, task: str = None, category: str = None):
    typer.echo(f"Updating {position}")

if __name__ == "__main__":
    app()