import asyncio
import datetime
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import typer
from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel import Session, select, func, or_
from sqlmodel.ext.asyncio.session import AsyncSession
from tqdm.asyncio import tqdm

from conf.config import settings
from db.db_utils import engine
from db.models import Match, Player, Surface
from tennis_scrapper.stats.compute_stats import compute_match_played_stats, compute_h2h_stats, compute_player_match_based_stats
from tennis_scrapper.stats.stats_utils import compute_player_age


def is_match_valid(match: Match) -> bool:
    def validate_score_str(score_str: str) -> bool:
        allowed_chars = set("0123456789- ")
        return all(char in allowed_chars for char in score_str)

    if not match.score or not validate_score_str(match.score):
        return False
    return True


async def get_player_history_at_dt(
    player_id: str,
    date: datetime.date,
    hit_limit: int,
    surface: Surface,
    db_session: AsyncSession,
) -> tuple[list[Match], list[Match]]:
    """Get player's match history up to a specific date."""
    # TODO replace by get_one_player_matches when all sessions will be async
    result = await db_session.exec(
        select(Match)
        .where(
            or_(Match.player_1_id == player_id, Match.player_2_id == player_id),
            Match.date < date,
        )
        .order_by(Match.date)
    )
    matches = result.all()
    matches = list(filter(is_match_valid, matches))

    matches_on_surface = [match for match in matches if match.surface == surface]

    matches_on_surface = (
        matches_on_surface[-hit_limit:]
        if len(matches_on_surface) > hit_limit
        else matches_on_surface
    )
    matches = matches[-hit_limit:] if len(matches) > hit_limit else matches

    return matches, matches_on_surface


async def get_h2h_matches(
    player_1_id: str,
    player_2_id: str,
    match_date: datetime.date,
    db_session: AsyncSession,
) -> list[Match]:
    """Get head-to-head matches between two players."""
    matches = await db_session.exec(
        select(Match)
        .where(
            or_(Match.player_1_id == player_1_id, Match.player_2_id == player_1_id),
            or_(Match.player_1_id == player_2_id, Match.player_2_id == player_2_id),
            Match.date < match_date,
        )
        .order_by(Match.date)
    ).all()
    matches = list(filter(is_match_valid, matches))
    return sorted(matches, key=lambda x: x.date)


def is_match_sorted(matches: list[Match]) -> bool:
    """Check if matches are sorted by date."""
    return all(matches[i].date <= matches[i + 1].date for i in range(len(matches) - 1))


def add_key_prefix_suffix(d: dict, prefix: str = "", suffix: str = "") -> dict:
    if prefix != "" and not prefix.endswith("_"):
        prefix += "_"

    if suffix != "" and not suffix.startswith("_"):
        suffix = "_" + suffix
        
    return {f"{prefix}{key}{suffix}": value for key, value in d.items()}

def compute_one_player_stat(
    matches: list[Match], 
    matches_on_surface: list[Match], 
    match: Match,
    player:Player, 
    ks: list[int]
):  
    player_stats = compute_player_match_based_stats(matches, player.player_id, k=ks)
    player_stats = {
        f"{key}": value for key, value in player_stats.items()
    }

    player_stats_on_surface = compute_player_match_based_stats(
        matches_on_surface, player.player_id, k=ks
    )
    player_stats_on_surface = add_key_prefix_suffix(player_stats_on_surface, suffix="_on_surface")

    match_played_stats = compute_match_played_stats(matches, match.date)

    return {
        "age": compute_player_age(player, match.date),
        **player_stats, 
        **player_stats_on_surface, 
        **match_played_stats
    }


async def compute_one_match_stat(
    match: Match,
    async_engine: AsyncEngine,
    player_id_to_player: Dict[str, Player],
    ks: list[int],
    output_dir: Path,
    override: bool,
) -> Dict:

    def get_save_path(match: Match) -> Path:
        return output_dir / "jsons" / match.match_id[:3] / f"{match.match_id}.json"

    if not override and get_save_path(match).exists():
        # logger.info(f"Skipping match {match.match_id}, already processed.")
        return {}

    async with AsyncSession(async_engine) as db_session:

        """Compute comprehensive statistics for a single match."""
        player_1 = player_id_to_player[match.player_1_id]
        player_2 = player_id_to_player[match.player_2_id]

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
        
        player_1_stats = compute_one_player_stat(
            matches=matches_player_1,
            matches_on_surface=matches_player_1_on_surface,
            player=match.player_1_id,
            match=match,
            ks=ks,
        )
        player_1_stats = add_key_prefix_suffix(player_1_stats, prefix="player_1")

        player_2_stats = compute_one_player_stat(
            matches=matches_player_2,
            matches_on_surface=matches_player_2_on_surface,
            player=match.player_2_id,
            match=match,
            ks=ks,
        )
        player_2_stats = add_key_prefix_suffix(player_2_stats, prefix="player_2")


        # Get head-to-head statistics
        h2h_matches = await get_h2h_matches(
            player_1_id=player_1.player_id,
            player_2_id=player_2.player_id,
            match_date=match.date,
            db_session=db_session,
        )
        h2h_stats = compute_h2h_stats(
            h2h_matches, player_1_id=match.player_1_id, player_2_id=match.player_2_id
        )

        # Combine all data
        data = {
            **match.model_dump(),
            **player_1_stats,
            **player_2_stats,
            **h2h_stats,
        }

        # Convert date to string for JSON serialization
        data["date"] = str(data["date"])

        # Save to file
        output_file = get_save_path(match)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(data, f)
        logger.info(f"Processed match {match.match_id} and saved to {output_file}")

        return data


def load_matches_and_players(
    years_offset: int = 2,
) -> Tuple[list[Match], Dict[str, Player]]:
    """Load matches and players from a database."""
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
    matches: list[Match],
    player_id_to_player: Dict[str, Player],
    ks: list[int],
    output_path: Path,
    async_db_url: str,
    override: bool,
) -> list[dict]:
    """Process matches asynchronously."""
    # Create async engine and session
    async_engine = create_async_engine(async_db_url, echo=False)
    
    stats = []
    for match in tqdm(matches, desc="Processing matches", unit="match"):
        try:
            stats.append(
                await compute_one_match_stat(
                    match=match,
                    async_engine=async_engine,
                    player_id_to_player=player_id_to_player,
                    ks=ks,
                    output_dir=output_path,
                    override=override,
                )
            )
        except Exception as e:
            logger.error(f"Error processing match {match}: {e}")
            raise

    await async_engine.dispose()

    return stats


def pack_all_jsons(output_path: Path):
    chunk_size = 50000

    json_files = list((output_path / "jsons").rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to process.")

    output_path = output_path / "chunks"
    output_path.mkdir(parents=True, exist_ok=True)

    current_chunk = []
    for i, file in tqdm(
        enumerate(json_files),
        desc="Packing JSON files",
        unit="file",
        total=len(json_files),
    ):

        with open(file, "r") as f:
            data = json.load(f)

        if i > 0 and i % chunk_size == 0:
            df = pd.DataFrame(current_chunk)
            df = df.to_parquet(output_path / f"chunk_{i // chunk_size}.parquet")
            logger.info(
                f"Saved chunk {i // chunk_size} with {len(current_chunk)} records."
            )
            current_chunk = [data]
        else:
            current_chunk.append(data)

    if len(current_chunk) > 0:
        df = pd.DataFrame(current_chunk)
        df = df.to_parquet(output_path / "chunk_last.parquet")
        logger.info(f"Saved chunk last with {len(current_chunk)} records.")


app = typer.Typer(help="Generate comprehensive tennis match statistics")


@app.command(help="Generate comprehensive tennis match statistics.")
def generate_stats(
    output_dir: str = typer.Option(
        "output", "--output", "-o", help="Output directory for JSON files"
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
    override: bool = typer.Option(
        False,
        "--override",
        help="Override existing output files if they already exist",
    ),
):
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
            all_matches, player_id_to_player, ks, output_path, async_db_url, override
        )
    )

    logger.info(f"✅ Successfully generated statistics for {len(all_matches)} matches")
    logger.info(f"📁 Output saved to: {output_path.absolute()}")

    pack_all_jsons(output_path)


if __name__ == "__main__":
    app()
