import typer
from loguru import logger
from tqdm import tqdm

from cli.scrap_matches import app as scrap_matches_app
from cli.scrap_tournaments import app as scrap_tournaments_app
from cli.scrap_ranking import app as scrap_atp_ranking_app

app = typer.Typer(pretty_exceptions_enable=False)
app.add_typer(scrap_tournaments_app)
app.add_typer(scrap_matches_app)
app.add_typer(scrap_atp_ranking_app)

if __name__ == "__main__":
    logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [<level>{level: ^12}</level>] <level>{message}</level>"
    logger.configure(
        handlers=[
            dict(
                sink=lambda msg: tqdm.write(msg, end=""),
                format=logger_format,
                colorize=True,
            )
        ]
    )
    app()
