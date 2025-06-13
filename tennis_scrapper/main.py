import typer
from cli.scrap_tournaments import app as scrap_tournaments_app
from cli.scrap_matches import app as scrap_matches_app

app = typer.Typer(pretty_exceptions_enable=False)
app.add_typer(scrap_tournaments_app)
app.add_typer(scrap_matches_app)

if __name__ == "__main__":
    app()
