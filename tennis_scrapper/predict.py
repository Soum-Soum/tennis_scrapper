import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta
import json
from math import isfinite
import math
import tempfile
from typing import Optional
from urllib.request import urlopen
from pathlib import Path
from bs4 import BeautifulSoup
import joblib
from loguru import logger
import pandas as pd
from sqlmodel import Session, select
from tqdm import tqdm
import typer
from xgboost import XGBClassifier
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from conf.config import settings
from cli.generate_stats import get_elo, process_matches_async
from data.add_elo import K, compute_elo
from db.db_utils import get_engine, get_one_player_matches, get_player_by_id, get_table
from db.models import Gender, Match, Player, Ranking, Surface, Tournament
from scrap.matches import extract_matches_from_table
from scrap.urls import get_match_list_page_url
from train import ColsData, compute_diff_columns, validate_cols


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
    def get_player_by_url_extension(url_extension: str) -> Player:
        statement = select(Player).where(Player.url_extension == url_extension)
        player = db_session.exec(statement).first()
        if player is None:
            raise ValueError(f"Player with url extension {url_extension} not found")
        return player

    player_1 = get_player_by_url_extension(match.player_1_url_extension)
    player_2 = get_player_by_url_extension(match.player_2_url_extension)
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

    def get_tournament_by_url(url: str) -> Optional[Tournament]:
        return db_session.exec(
            select(Tournament).where(Tournament.url_extension == url)
        ).first()

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
            matches, player_id_to_player, ks, Path(tmpdir), async_db_url, False
        )
    return pd.DataFrame(X_test)


def kelly_criterion(p: float, odds: float) -> float:
    if not (0.0 <= p <= 1.0) or odds <= 1.0:
        return 0.0
    b = odds - 1.0
    return max(0.0, (p * b - (1 - p)) / b)


def pretty_print_bets(
    matches: list[Match],
    probas: list[float],
    max_bet_fraction: float,
    bankroll: float,
) -> pd.DataFrame:
    """
    Build and display a betting recommendation table from model probabilities.

    Assumptions:
      - `probas` is the model P(player_2 wins), in [0, 1].
      - If proba < 0.5 -> predict player_1; else predict player_2.
      - Uses capped Kelly criterion with cap = `max_bet_fraction`.

    Returns:
      - A pandas DataFrame with one row per match containing:
        date, players, odds, predicted side, model probability for that side,
        implied probability (from odds), edge, Kelly%, stake%, stake €, EV%.

    Notes:
      - If an odd is missing/invalid (<= 1), the recommendation is PASS.
      - “EV%” is the expected profit per euro staked: (p * odds - 1) * 100.
    """
    if len(matches) != len(probas):
        raise ValueError(
            f"Length mismatch: matches={len(matches)} vs probas={len(probas)}"
        )

    rows = []
    for match, p2 in zip(matches, probas):
        # Players
        p1_player = get_player_by_id(match.player_1_id)
        p2_player = get_player_by_id(match.player_2_id)

        # Prediction side & corresponding probability/odds
        pred_p2 = int(round(p2))  # 0->P1, 1->P2
        if pred_p2 == 0:
            predicted_player = p1_player
            p_win = 1.0 - p2
            odd = match.player_1_odds
            side = "P1"
        else:
            predicted_player = p2_player
            p_win = float(p2)
            odd = match.player_2_odds
            side = "P2"

        # Guard against missing/invalid odds
        valid_odd = isinstance(odd, (int, float)) and odd is not None and odd > 1.0

        implied = (1.0 / odd) if valid_odd else math.nan
        edge = (p_win - implied) if valid_odd else math.nan

        # Kelly (net odds b = odd-1). If invalid odds, kelly=0.
        if valid_odd:
            kelly_frac = kelly_criterion(p_win, odd)  # assumed to max(0, ...)
        else:
            kelly_frac = 0.0

        stake_frac = min(kelly_frac, max_bet_fraction)
        stake_eur = bankroll * stake_frac

        # Expected value (per € staked): profit EV = p*odd - 1
        ev = (p_win * odd - 1.0) if valid_odd else math.nan

        decision = "💰 BET" if kelly_frac > 0 else "⏸️ PASS"

        # A tiny confidence meter: proportion of cap used (0..5 dots)
        conf_steps = 5
        conf_level = (
            0
            if max_bet_fraction <= 0
            else round((stake_frac / max_bet_fraction) * conf_steps)
        )
        conf_level = max(0, min(conf_steps, conf_level))
        confidence = "▪" * conf_level + "·" * (conf_steps - conf_level)  # ▪▪▪··

        rows.append(
            {
                "Players": f"{p1_player.name} vs {p2_player.name}",
                "Predicted Winner": f"{predicted_player.name}({side})",
                "Odds": odd if valid_odd else None,
                "Model p(win)": round(p_win, 3),
                "Implied p": round(implied, 3) if valid_odd else None,
                "Edge %": round(100 * edge, 1) if valid_odd else None,
                "Kelly %": round(100 * kelly_frac, 2),
                "Stake %": round(100 * stake_frac, 2),
                "Stake €": round(stake_eur, 2),
                "EV %": round(100 * ev, 2) if valid_odd else None,
                "Decision": decision,
                "Conf": confidence,  # visual hint, 5 steps
            }
        )

    df = pd.DataFrame(rows)

    # Sort: BETs first, then by highest expected value / stake
    def sort_key(row):
        is_bet = row["Decision"].startswith("💰")
        ev = row["EV %"] if pd.notna(row["EV %"]) else -1e9
        stake = row["Stake €"] if pd.notna(row["Stake €"]) else 0.0
        return (not is_bet, -ev, -stake)

    if not df.empty:
        df = df.sort_values(
            by=["Decision", "EV %", "Stake €"],
            ascending=[True, False, False],
            key=lambda s: s,  # keep default; we’ll rely on custom printing sort below if needed
        )

    # Pretty terminal output with rich (fallback to plain DataFrame)
    console = Console()
    table = Table(
        title="🏆 Betting Recommendations",
        box=box.SIMPLE_HEAVY,
        expand=True,
        show_lines=False,
    )

    cols = [
        "Players",
        "Predicted Winner",
        "Odds",
        "Model p(win)",
        "Implied p",
        "Edge %",
        "Kelly %",
        "Stake %",
        "Stake €",
        "EV %",
        "Decision",
        "Conf",
    ]
    just = {
        "Players": "left",
        "Predicted Winner": "left",
        "Odds": "right",
        "Model p(win)": "right",
        "Implied p": "right",
        "Edge %": "right",
        "Kelly %": "right",
        "Stake %": "right",
        "Stake €": "right",
        "EV %": "right",
        "Decision": "center",
        "Conf": "center",
    }
    for c in cols:
        table.add_column(c, justify=just[c], no_wrap=(c in {"Players", "Side"}))

    # Add rows with light conditional coloring
    for _, r in df.iterrows():
        style = "green" if r["Decision"].startswith("💰") else "bright_black"
        table.add_row(
            str(r["Players"]),
            str(r["Predicted Winner"]),
            f"{r['Odds']:.2f}" if pd.notna(r["Odds"]) else "-",
            f"{r['Model p(win)']:.3f}" if pd.notna(r["Model p(win)"]) else "-",
            f"{r['Implied p']:.3f}" if pd.notna(r["Implied p"]) else "-",
            f"{r['Edge %']:.1f}" if pd.notna(r["Edge %"]) else "-",
            f"{r['Kelly %']:.2f}",
            f"{r['Stake %']:.2f}",
            f"{r['Stake €']:.2f}",
            f"{r['EV %']:.2f}" if pd.notna(r["EV %"]) else "-",
            r["Decision"],
            r["Conf"],
            style=style,
        )

    # Summary panel
    bets = df[df["Decision"].str.startswith("💰")] if not df.empty else df
    total_stake = float(bets["Stake €"].sum()) if not bets.empty else 0.0
    exp_profit = 0.0
    for _, r in bets.iterrows():
        ev_pct = r["EV %"]
        stake_eur = r["Stake €"]
        if pd.notna(ev_pct) and pd.notna(stake_eur):
            exp_profit += stake_eur * (ev_pct / 100.0)
    roi = (exp_profit / total_stake * 100.0) if total_stake > 0 else 0.0

    console.print(table)
    console.print(
        Panel.fit(
            f"💸 Total stake: [bold]{total_stake:.2f} €[/bold]   "
            f"📈 Expected profit: [bold]{exp_profit:.2f} €[/bold]   "
            f"📊 Expected ROI: [bold]{roi:.2f}%[/bold]",
            border_style="cyan",
            title="Summary",
        )
    )

    return df


app = typer.Typer()


@app.command()
def predict():

    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    dates = [today, tomorrow]

    all_incoming_matches = defaultdict(list)
    for gender in Gender:
        for date in dates:
            url = get_match_list_page_url(gender=gender, date=date)
            html = urlopen(url).read()
            matches = incomming_match_from_html(html=html, date=date, gender=gender)
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

    X_test = compute_diff_columns(X_test)
    X_test = validate_cols(X_test, cols_data)

    X_test = X_test.drop(columns=cols_data.categorical)
    X_test = X_test.drop(columns=cols_data.other)
    X_test = X_test.drop(columns=cols_data.date_column)

    model_loaded = XGBClassifier()
    model_loaded.load_model(
        "/home/pierre/dev/tennis_scrapper/output3/model/classifier.json"
    )
    scaler = joblib.load("/home/pierre/dev/tennis_scrapper/output3/data/scaler.pkl")

    X_test_scaled = X_test.copy()
    X_test_scaled = X_test_scaled.reindex(columns=model_loaded.feature_names_in_)
    X_test_scaled[cols_data.numerical] = scaler.transform(X_test[cols_data.numerical])

    probas = model_loaded.predict_proba(X_test_scaled)[:, 1]

    X_test["proba"] = probas
    X_test["predicted"] = X_test["proba"].round().astype(int)

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

    pretty_print_bets(matches, probas, max_bet_fraction=0.05, bankroll=1000.0)


if __name__ == "__main__":
    app()
