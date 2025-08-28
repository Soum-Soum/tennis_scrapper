import asyncio
from datetime import date, datetime, timedelta
import json
import tempfile
from urllib.request import urlopen
from pathlib import Path
from bs4 import BeautifulSoup
import joblib
import pandas as pd
from sqlmodel import Session
from tqdm import tqdm
import typer

from conf.config import settings
from cli.generate_stats import process_matches
from data.add_elo import K, compute_elo
from db.db_utils import (
    get_engine,
    get_last_ranking,
    get_one_player_matches,
    get_player_by_url_extension,
    get_table,
    get_tournament_by_url,
)
from db.models import Gender, Match, ModelPredictions, Player, Ranking, Surface
from scrap.matches import extract_matches_from_table
from scrap.urls import get_match_list_page_url
from ml.preprocess_data import ColsData, preprocess_dataframe_predict
from ml.models.xgb import XgbClassifierWrapper
from stats.stats_utils import get_elo


def incomming_match_from_html(html: str, date: date, gender: Gender) -> list[Match]:
    soup = BeautifulSoup(html, "lxml")

    tables = soup.find("div", id="center").find_all("table", class_="result")

    matches = set()

    for table in tables:
        matches = matches.union(extract_matches_from_table(table, date, gender))

    matches = list(filter(lambda x: x.score == "", matches))
    matches = list(
        filter(
            lambda x: x.player_1_odds is not None and x.player_2_odds is not None,
            matches,
        )
    )
    matches = list(filter(lambda x: x.surface != Surface.UNKNOWN, matches))

    ban_words = ["utr", "challenger", "futures", "itf", "default"]
    for word in ban_words:
        matches = list(
            filter(lambda x: word not in x.tournament_url_extension.lower(), matches)
        )

    return matches


def add_player_to_match(match: Match, db_session: Session) -> Match:
    player_1 = get_player_by_url_extension(
        match.player_1_url_extension, db_session=db_session
    )
    player_2 = get_player_by_url_extension(
        match.player_2_url_extension, db_session=db_session
    )
    match.player_1_id = player_1.player_id
    match.player_2_id = player_2.player_id
    return match


def get_player_id_to_ranking(db_session: Session) -> dict[str, Ranking]:
    player_id_to_ranking = {}
    for gender in Gender:
        last_rankings = get_last_ranking(db_session, gender)
        for ranking in last_rankings:
            player_id_to_ranking[ranking.player_id] = ranking

    return player_id_to_ranking


def add_tournament_data_to_match(match: Match, db_session: Session) -> Match:
    tournament = get_tournament_by_url(match.tournament_url_extension)
    assert (
        tournament is not None
    ), f"Tournament with url {match.tournament_url_extension} not found"

    match.tournament_id = tournament.tournament_id
    match.surface = tournament.surface
    return match


def add_ranking_data_to_match(
    match: Match, player_id_to_ranking: dict[str, Ranking]
) -> Match:
    ranking_player_1 = player_id_to_ranking.get(match.player_1_id)
    ranking_player_2 = player_id_to_ranking.get(match.player_2_id)

    if ranking_player_1 is not None:
        match.player_1_ranking = ranking_player_1.rank
        match.player_1_ranking_points = ranking_player_1.points

    if ranking_player_2 is not None:
        match.player_2_ranking = ranking_player_2.rank
        match.player_2_ranking_points = ranking_player_2.points

    return match


def add_elo_data_to_match(db_session: Session, match: Match) -> Match:
    last_player_1_match = get_one_player_matches(
        db_session=db_session, player_id=match.player_1_id, limit=1
    )[0]
    last_player_2_match = get_one_player_matches(
        db_session=db_session, player_id=match.player_2_id, limit=1
    )[0]

    last_elo_player_1 = get_elo(last_player_1_match, player_id=match.player_1_id)
    last_elo_player_2 = get_elo(last_player_2_match, player_id=match.player_2_id)

    player_1_new_elo, player_2_new_elo = compute_elo(
        last_elo_player_1, last_elo_player_2, K
    )
    match.player_1_elo = player_1_new_elo
    match.player_2_elo = player_2_new_elo

    last_player_1_match_on_surface = get_one_player_matches(
        db_session=db_session,
        player_id=match.player_1_id,
        limit=1,
        surface=match.surface,
    )[0]
    last_player_2_match_on_surface = get_one_player_matches(
        db_session=db_session,
        player_id=match.player_2_id,
        limit=1,
        surface=match.surface,
    )[0]

    last_elo_player_1_on_surface = get_elo(
        last_player_1_match_on_surface, player_id=match.player_1_id
    )
    last_elo_player_2_on_surface = get_elo(
        last_player_2_match_on_surface, player_id=match.player_2_id
    )

    player_1_new_elo_on_surface, player_2_new_elo_on_surface = compute_elo(
        last_elo_player_1_on_surface, last_elo_player_2_on_surface, K
    )
    match.player_1_elo_on_surface = player_1_new_elo_on_surface
    match.player_2_elo_on_surface = player_2_new_elo_on_surface

    return match


async def generate_stats(matches: list[Match]) -> pd.DataFrame:
    ks = [3, 5, 10, 25, 50, 100, 200]
    async_db_url = settings.async_db_url

    players = get_table(Player)
    player_id_to_player = {p.player_id: p for p in players}

    with tempfile.TemporaryDirectory() as tmpdir:
        X_test = await process_matches(
            matches,
            player_id_to_player,
            ks,
            Path(tmpdir),
            async_db_url,
            False,
        )
    return pd.DataFrame(X_test)


app = typer.Typer(pretty_exceptions_enable=False)


def get_incoming_matches() -> list[Match]:
    dates = [datetime.now().date(), datetime.now().date() + timedelta(days=1)]
    all_incoming_matches = []
    for gender in Gender:
        for current_date in dates:
            url = get_match_list_page_url(gender=gender, date=current_date)
            html = urlopen(url).read()
            matches = incomming_match_from_html(
                html=html, date=current_date, gender=gender
            )
            all_incoming_matches.extend(matches)
    return all_incoming_matches


def add_data_to_matches(all_incoming_matches: list[Match]) -> list[Match]:
    matches = []
    with Session(get_engine()) as db_session:

        player_id_to_ranking = get_player_id_to_ranking(db_session)

        for match in tqdm(all_incoming_matches):
            match = add_player_to_match(match, db_session)
            match = add_tournament_data_to_match(match, db_session)
            match = add_ranking_data_to_match(match, player_id_to_ranking)
            match = add_elo_data_to_match(db_session, match)
            matches.append(match)
    return matches


@app.command()
def predict(base_dir: Path = typer.Option(help="Path to save the base dir")):

    all_incoming_matches = get_incoming_matches()

    matches = add_data_to_matches(all_incoming_matches)

    X_test = asyncio.run(generate_stats(matches))

    with open("/home/pierre/dev/tennis_scrapper/resources/cols_data.json") as f:
        cols_data = ColsData.model_validate(json.load(f))

    scaler = joblib.load(base_dir / "data" / "scaler.pkl")

    X_test_preprocessed = preprocess_dataframe_predict(
        X_df=X_test, cols_data=cols_data, scaler=scaler, min_history_size=10
    )

    model_wrapper = XgbClassifierWrapper.from_model(
        base_dir / "model" / "classifier.json"
    )

    predictions_df = model_wrapper.predict(X_test_preprocessed)

    X_test = pd.concat([X_test, predictions_df], axis=1)

    with Session(get_engine()) as db_session:
        predictions = []
        for match, (predicted_class, predicted_proba) in zip(
            matches, predictions_df.itertuples(index=False)
        ):
            prediction = ModelPredictions(
                player_1_id=match.player_1_id,
                player_2_id=match.player_2_id,
                predicted_class=predicted_class,
                predicted_proba=predicted_proba,
                player_1_odds=match.player_1_odds,
                player_2_odds=match.player_2_odds,
                date=match.date,
            )
            predictions.append(prediction)
        db_session.add_all(predictions)
        db_session.commit()


if __name__ == "__main__":
    app()
