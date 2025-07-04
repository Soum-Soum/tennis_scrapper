import typer
from loguru import logger
from sqlmodel import Session, text

from data.add_atp import add_atp
from data.add_elo import add_elo
from db.db_utils import engine


def update_match_players_ids():
    with Session(engine) as session:
        logger.info("Updating missing player IDs in matches for player 1")
        session.exec(
            text(
                """
                    UPDATE match
                    SET player_1_id = p1.player_id
                    FROM player p1
                    WHERE match.player_1_url_extension = p1.url_extension
                      AND match.player_1_id IS NULL;
                """
            )
        )

        logger.info("Updating missing player IDs in matches for player 2")
        session.exec(
            text(
                """
                    UPDATE match
                    SET player_2_id = p2.player_id
                    FROM player p2
                    WHERE match.player_2_url_extension = p2.url_extension
                      AND match.player_2_id IS NULL;
                """
            )
        )

        session.commit()


app = typer.Typer()


@app.command()
def fill_missing_data():
    update_match_players_ids()
    add_atp()
    add_elo()

    logger.info("Missing player IDs in matches have been updated.")
