import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta
import json
import tempfile
from urllib.request import urlopen
from pathlib import Path
from bs4 import BeautifulSoup
import joblib
from loguru import logger
import pandas as pd
from sqlmodel import Session, select
from tqdm import tqdm
import typer

from conf.config import settings
from cli.generate_stats import process_matches_async
from data.add_elo import K, compute_elo
from db.db_utils import get_engine, get_one_player_matches, get_player_by_url_extension, get_table, get_tournament_by_url
from db.models import Gender, Match, Player, Ranking, Surface
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
    player_1 = get_player_by_url_extension(match.player_1_url_extension, db_session=db_session)
    player_2 = get_player_by_url_extension(match.player_2_url_extension, db_session=db_session)
    match.player_1_id = player_1.player_id
    match.player_2_id = player_2.player_id
    return match


def get_player_id_to_ranking(db_session: Session) -> dict[str, Ranking]:
    def get_last_ranking(db_session: Session, gender: Gender) -> Ranking:
        logger.info(f"Fetching last ranking for circuit: {gender.circuit}")

        last_ranking_date = db_session.exec(
            select(Ranking.date)
            .where(Ranking.circuit == gender.circuit)
            .order_by(Ranking.date.desc())
            .limit(1)
        ).first()

        statement = select(Ranking).where(
            Ranking.circuit == gender.circuit, Ranking.date == last_ranking_date
        )
        return db_session.exec(statement).all()

    gender_to_last_ranking = {
        gender: get_last_ranking(db_session, gender) for gender in Gender
    }
    player_id_to_ranking = {}
    for gender in Gender:
        for ranking in gender_to_last_ranking[gender]:
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
        match.atp_ranking_player_1 = ranking_player_1.rank
        match.atp_points_player_1 = ranking_player_1.points

    if ranking_player_2 is not None:
        match.atp_ranking_player_2 = ranking_player_2.rank
        match.atp_points_player_2 = ranking_player_2.points

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


async def compute_x_test(matches: list[Match]) -> pd.DataFrame:
    ks = [3, 5, 10, 25, 50, 100, 200]
    async_db_url = settings.async_db_url

    players = get_table(Player)
    player_id_to_player = {p.player_id: p for p in players}

    with tempfile.TemporaryDirectory() as tmpdir:
        X_test = await process_matches_async(
            matches, player_id_to_player, ks, Path(tmpdir), async_db_url, False, min_hist_size=10
        )
    return pd.DataFrame(X_test)


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def predict(
    base_dir: Path = typer.Option(
        help="Path to save the base dir"
    )
):

    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    dates = [today, tomorrow]

    all_incoming_matches = defaultdict(list)
    for gender in Gender:
        for current_date in dates:
            url = get_match_list_page_url(gender=gender, date=current_date)
            html = urlopen(url).read()
            matches = incomming_match_from_html(html=html, date=current_date, gender=gender)
            all_incoming_matches[gender].extend(matches)

    match_with_data = defaultdict(list)
    with Session(get_engine()) as db_session:

        player_id_to_ranking = get_player_id_to_ranking(db_session)

        for gender, matches in tqdm(all_incoming_matches.items()):
            for match in matches:
                match = add_player_to_match(match, db_session)
                match = add_tournament_data_to_match(match, db_session)
                match = add_ranking_data_to_match(match, player_id_to_ranking)
                match = add_elo_data_to_match(db_session, match)
                match_with_data[gender].append(match)

    matches = sum(match_with_data.values(), [])

    X_test = asyncio.run(compute_x_test(matches))
    
    with open("/home/pierre/dev/tennis_scrapper/resources/cols_data.json") as f:
        cols_data = ColsData.model_validate(json.load(f))

    model_wrapper = XgbClassifierWrapper.from_model(base_dir / "model" / "classifier.json")
    scaler = joblib.load(base_dir / "data" / "scaler.pkl")
    
    X_test_preprocessed = preprocess_dataframe_predict(X_df=X_test, cols_data=cols_data, scaler=scaler)

    predictions_df = model_wrapper.predict(X_test_preprocessed)

    X_test["proba"] = predictions_df["predicted_proba"]
    X_test["predicted"] = predictions_df["predicted_class"]

    prediction_path = Path(f"predictions/{datetime.now().strftime('%Y-%m-%d')}")
    prediction_path.mkdir(parents=True, exist_ok=True)
    prediction_file = prediction_path / "predictions.csv"
    matches_save_dir = prediction_path / "matches"
    matches_save_dir.mkdir(parents=True, exist_ok=True)
    for match in matches:
        with open(matches_save_dir / f"{match.match_id}.json", "w") as f:
            f.write(match.model_dump_json(indent=4))

    print("Saving predictions to:", prediction_file.absolute())
    X_test.to_csv(
        prediction_file,
        index=False,
    )

if __name__ == "__main__":
    app()
