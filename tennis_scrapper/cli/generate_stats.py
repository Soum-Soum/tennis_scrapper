#!/usr/bin/env python3
"""
Tennis Match Statistics Generator

This script processes tennis match data to generate comprehensive statistics
for each match including player performance metrics, head-to-head records,
and surface-specific statistics.
"""

import asyncio
import datetime
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import typer
from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel import Session, select, func, or_
from sqlmodel.ext.asyncio.session import AsyncSession
from tqdm.asyncio import tqdm

from conf.config import settings
from db.db_utils import engine
from db.models import Match, Player, Surface

app = typer.Typer(help="Generate comprehensive tennis match statistics")


def get_player_by_name(name: str) -> Optional[Player]:
    """Get a player by name using fuzzy matching."""
    with Session(engine) as session:
        player = session.exec(
            select(Player).where(Player.name.like(f"%{name}%"))
        ).first()
        return player


async def get_player_history_at_dt(
    player_id: str,
    date: datetime.date,
    hit_limit: int,
    surface: Surface,
    db_session: AsyncSession,
) -> Tuple[List[Match], List[Match]]:
    """Get player's match history up to a specific date."""
    result = await db_session.exec(
        select(Match)
        .where(
            or_(Match.player_1_id == player_id, Match.player_2_id == player_id),
            Match.date < date,
        )
        .order_by(Match.date)
    )
    matches = result.all()

    matches_on_surface = [match for match in matches if match.surface == surface]

    matches_on_surface = (
        matches_on_surface[-hit_limit:]
        if len(matches_on_surface) > hit_limit
        else matches_on_surface
    )
    matches = matches[-hit_limit:] if len(matches) > hit_limit else matches

    return matches, matches_on_surface


async def get_h2h_matches(
    player_1_id: str, player_2_id: str, db_session: AsyncSession
) -> List[Match]:
    """Get head-to-head matches between two players."""
    result = await db_session.exec(
        select(Match)
        .where(
            or_(Match.player_1_id == player_1_id, Match.player_2_id == player_1_id),
            or_(Match.player_1_id == player_2_id, Match.player_2_id == player_2_id),
        )
        .order_by(Match.date)
    )
    matches = result.all()
    return sorted(matches, key=lambda x: x.date)


def is_winner(match: Match, player_id: str) -> bool:
    """Check if a player won a match."""
    return match.player_1_id == player_id


def parse_score(score: str) -> List[Tuple[int, int]]:
    """Parse tennis score string into list of set scores."""
    score_list = []
    sets = score.split()
    for current_set in sets:
        games_winner, games_loser = current_set.split("-")
        score_list.append((int(games_winner), int(games_loser)))
    return score_list


def get_games_won(match: Match, player_id: str) -> List[int]:
    """Get games won by player in each set."""
    parsed_score = parse_score(match.score)
    if is_winner(match, player_id):
        return [games[0] for games in parsed_score]
    else:
        return [games[1] for games in parsed_score]


def get_games_conceded(match: Match, player_id: str) -> List[int]:
    """Get games conceded by player in each set."""
    parsed_score = parse_score(match.score)
    if is_winner(match, player_id):
        return [games[1] for games in parsed_score]
    else:
        return [games[0] for games in parsed_score]


def get_elo(match: Match, player_id: str) -> float:
    """Get player's ELO rating for a match."""
    if match.player_1_id == player_id:
        return match.player_1_elo
    elif match.player_2_id == player_id:
        return match.player_2_elo
    else:
        raise ValueError(f"Player {player_id} not found in match {match.id}")


def is_match_sorted(matches: List[Match]) -> bool:
    """Check if matches are sorted by date."""
    return all(matches[i].date <= matches[i + 1].date for i in range(len(matches) - 1))


def compute_player_stats(
    matches: List[Match], player_id: str, k: Optional[List[int]] = None
) -> Dict[str, float]:
    """Compute comprehensive player statistics for different match windows."""
    if k is None:
        k = [3, 5, 10, 25, 50, 100, 200]

    stats = {}
    assert is_match_sorted(matches), "Matches must be sorted by date"

    for k_value in k:
        selected_matches = matches[-k_value:]
        if len(selected_matches) == 0:
            continue

        # Calculate statistics
        all_games_won = sum(
            [get_games_won(match, player_id) for match in selected_matches], start=[]
        )
        all_games_conceded = sum(
            [get_games_conceded(match, player_id) for match in selected_matches],
            start=[],
        )

        games_won_by_set = np.mean(all_games_won).item()
        games_conceded_by_set = np.mean(all_games_conceded).item()
        winning_rate = np.mean(
            [is_winner(match, player_id) for match in selected_matches]
        ).item()

        first_elo = get_elo(selected_matches[0], player_id)
        last_elo = get_elo(selected_matches[-1], player_id)

        stats[f"elo_diff_@k={k_value}"] = last_elo - first_elo
        stats[f"win_rate_@k={k_value}"] = winning_rate
        stats[f"games_won_@k={k_value}"] = games_won_by_set
        stats[f"games_conceded_@k={k_value}"] = games_conceded_by_set

    return stats


def compute_h2h_stats(
    matches: List[Match], player_1_id: str, player_2_id: str
) -> Dict[str, float]:
    """Compute head-to-head statistics between two players."""
    stats = {}
    matches = sorted(matches, key=lambda x: x.date)

    player_1_wins = np.sum([is_winner(match, player_1_id) for match in matches]).item()
    player_2_wins = np.sum([is_winner(match, player_2_id) for match in matches]).item()

    stats["h2h_player_1_wins"] = player_1_wins
    stats["h2h_player_2_wins"] = player_2_wins

    return stats


def compute_player_age(player: Player, date: datetime.date) -> float:
    """Calculate player's age at a specific date."""
    return (date - player.birth_date).days / 365.25


async def compute_one_match_stat(
    match: Match,
    async_engine: AsyncEngine,
    player_id_to_player: Dict[str, Player],
    ks: List[int],
    output_dir: Path,
) -> Dict:

    async with AsyncSession(async_engine) as db_session:

        """Compute comprehensive statistics for a single match."""
        player_1 = player_id_to_player[match.player_1_id]
        player_2 = player_id_to_player[match.player_2_id]

        player_1_age = compute_player_age(player_1, match.date)
        player_2_age = compute_player_age(player_2, match.date)

        # Get player histories
        matches_player_1, matches_player_1_on_surface = await get_player_history_at_dt(
            player_id=match.player_1_id,
            date=match.date,
            hit_limit=max(ks),
            surface=match.surface,
            db_session=db_session,
        )

        matches_player_2, matches_player_2_on_surface = await get_player_history_at_dt(
            player_id=match.player_2_id,
            date=match.date,
            hit_limit=max(ks),
            surface=match.surface,
            db_session=db_session,
        )

        # Compute player statistics
        player_1_stats = compute_player_stats(matches_player_1, match.player_1_id, k=ks)
        player_1_stats = {
            f"player_1_{key}": value for key, value in player_1_stats.items()
        }

        player_1_stats_on_surface = compute_player_stats(
            matches_player_1_on_surface, match.player_1_id, k=ks
        )
        player_1_stats_on_surface = {
            f"player_1_{key}_on_surface": value
            for key, value in player_1_stats_on_surface.items()
        }

        player_2_stats = compute_player_stats(matches_player_2, match.player_2_id, k=ks)
        player_2_stats = {
            f"player_2_{key}": value for key, value in player_2_stats.items()
        }

        player_2_stats_on_surface = compute_player_stats(
            matches_player_2_on_surface, match.player_2_id, k=ks
        )
        player_2_stats_on_surface = {
            f"player_2_{key}_on_surface": value
            for key, value in player_2_stats_on_surface.items()
        }

        # Get head-to-head statistics
        h2h_matches = await get_h2h_matches(
            player_1_id=player_1.player_id,
            player_2_id=player_2.player_id,
            db_session=db_session,
        )
        h2h_stats = compute_h2h_stats(
            h2h_matches, player_1_id=match.player_1_id, player_2_id=match.player_2_id
        )

        # Combine all data
        data = {
            **match.model_dump(),
            "player_1_age": player_1_age,
            "player_2_age": player_2_age,
            **player_1_stats,
            **player_1_stats_on_surface,
            **player_2_stats,
            **player_2_stats_on_surface,
            **h2h_stats,
        }

        # Convert date to string for JSON serialization
        data["date"] = str(data["date"])

        # Save to file
        output_file = output_dir / f"{match.match_id}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

        return data


def load_matches_and_players(
    years_offset: int = 2,
) -> Tuple[List[Match], Dict[str, Player]]:
    """Load matches and players from database."""
    logger.info("Loading matches and players from database...")

    with Session(engine) as session:
        # Get minimum date and add offset
        min_date = session.exec(select(func.min(Match.date).label("date_min"))).first()
        min_date = datetime.date(
            year=min_date.year + years_offset, month=min_date.month, day=min_date.day
        )

        # Load matches
        all_matches = session.exec(
            select(Match)
            .where(
                Match.surface != Surface.UNKNOWN,
                Match.date >= min_date,
            )
            .order_by(Match.date)
        ).all()

        # Load players
        all_players = session.exec(select(Player)).all()
        player_id_to_player = {player.player_id: player for player in all_players}

    logger.info(f"Loaded {len(all_matches)} matches and {len(all_players)} players")
    return all_matches, player_id_to_player


async def process_matches_async(
    matches: List[Match],
    player_id_to_player: Dict[str, Player],
    ks: List[int],
    output_path: Path,
    async_db_url: str,
):
    """Process matches asynchronously."""
    # Create async engine and session
    async_engine = create_async_engine(async_db_url, echo=False)

    # Create tasks for all matches
    tasks = [
        compute_one_match_stat(
            match=match,
            async_engine=async_engine,
            player_id_to_player=player_id_to_player,
            ks=ks,
            output_dir=output_path,
        )
        for match in matches
    ]

    # Execute tasks with progress bar
    await tqdm.gather(*tasks, desc="Processing matches", unit="match")

    await async_engine.dispose()


@app.command()
def generate_stats(
    output_dir: str = typer.Option(
        "Output", "--output", "-o", help="Output directory for JSON files"
    ),
    async_db_url: str = typer.Option(
        settings.async_db_url, "--db-url", help="Async database URL"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Limit number of matches to process"
    ),
    years_offset: int = typer.Option(
        2, "--years-offset", help="Years to offset from minimum date"
    ),
    k_values: str = typer.Option(
        "3,5,10,25,50,100,200",
        "--k-values",
        help="Comma-separated list of k values for statistics",
    ),
):
    """
    Generate comprehensive tennis match statistics.

    This command processes tennis matches from the database and generates
    detailed statistics for each match including player performance metrics,
    head-to-head records, and surface-specific statistics.
    """
    # Parse k values
    ks = [int(k.strip()) for k in k_values.split(",")]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    all_matches, player_id_to_player = load_matches_and_players(years_offset)

    # Apply limit if specified
    if limit:
        all_matches = all_matches[:limit]
        logger.info(f"Processing {len(all_matches)} matches (limited)")
    else:
        logger.info(f"Processing {len(all_matches)} matches")

    # Run async processing
    asyncio.run(
        process_matches_async(
            all_matches, player_id_to_player, ks, output_path, async_db_url
        )
    )

    logger.info(f"✅ Successfully generated statistics for {len(all_matches)} matches")
    logger.info(f"📁 Output saved to: {output_path.absolute()}")


if __name__ == "__main__":
    app()
