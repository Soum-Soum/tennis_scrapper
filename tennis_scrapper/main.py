import warnings

import typer
from loguru import logger
from tqdm import tqdm

from cli.scrap_all_matches import app as scarp_all_matches_app
from cli.scrap_players import app as scrap_players_app
from cli.scrap_ranking import app as scrap_atp_ranking_app
from cli.scrap_tournaments import app as scrap_tournaments_app
from cli.fill_missing_data import app as fill_missing_data_app
from cli.generate_stats import app as generate_stats_app

app = typer.Typer(pretty_exceptions_enable=False)
app.add_typer(scrap_tournaments_app)
app.add_typer(scrap_atp_ranking_app)
app.add_typer(scarp_all_matches_app)
app.add_typer(scrap_players_app)
app.add_typer(fill_missing_data_app)
app.add_typer(generate_stats_app)

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
    warnings.simplefilter("error")
    app()
