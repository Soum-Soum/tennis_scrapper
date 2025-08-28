import warnings

import typer
from loguru import logger
from tqdm import tqdm

from tennis_scrapper.cli.generate_stats import app as generate_stats_app
from tennis_scrapper.cli.scrap_all import app as scrap_all_app

app = typer.Typer(pretty_exceptions_enable=False)
app.add_typer(generate_stats_app)
app.add_typer(scrap_all_app)


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
