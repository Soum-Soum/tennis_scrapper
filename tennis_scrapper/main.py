import typer
from cli.scrap_tournament import app as scrap_tournament_app

app = typer.Typer()
app.add_typer(scrap_tournament_app)

if __name__ == "__main__":
    app()
