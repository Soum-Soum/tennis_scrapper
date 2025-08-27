import asyncio
import datetime
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import typer
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import Session, func, select
from tqdm.asyncio import tqdm

from conf.config import settings
from db.db_utils import engine
from db.models import Match, Player, Surface
from stats.compute_stats import (
    compute_one_match_stat,
)
from stats.fetch_db import get_data_from_db


def is_match_valid(match: Match) -> bool:
    def validate_score_str(score_str: str) -> bool:
        allowed_chars = set("0123456789- ")
        return all(char in allowed_chars for char in score_str)

    if not match.score or not validate_score_str(match.score):
        return False
    return True


def get_save_path(match: Match, output_dir: Path) -> Path:
    return output_dir / "jsons" / match.match_id[:3] / f"{match.match_id}.json"


async def process_one_match(
    match: Match,
    async_engine: AsyncEngine,
    player_id_to_player: Dict[str, Player],
    ks: list[int],
    output_dir: Path,
    override: bool,
) -> Dict:

    save_path = get_save_path(match, output_dir)
    if not override and save_path.exists():
        return {}

    """Compute comprehensive statistics for a single match."""
    player_1 = player_id_to_player[match.player_1_id]
    player_2 = player_id_to_player[match.player_2_id]

    (
        matches_player_1,
        matches_player_1_on_surface,
        matches_player_2,
        matches_player_2_on_surface,
        h2h_matches,
    ) = await get_data_from_db(
        match=match,
        ks=ks,
        async_engine=async_engine,
    )

    if len(matches_player_1) == 0 or len(matches_player_2) == 0:
        return {}

    data = compute_one_match_stat(
        match=match,
        player_1=player_1,
        player_2=player_2,
        matches_player_1=matches_player_1,
        matches_player_1_on_surface=matches_player_1_on_surface,
        matches_player_2=matches_player_2,
        matches_player_2_on_surface=matches_player_2_on_surface,
        h2h_matches=h2h_matches,
        ks=ks,
    )

    # Save to file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f)
    logger.info(f"Processed match {match.match_id} and saved to {save_path}")

    return data


def load_matches_and_players(
    years_offset: int = 2,
) -> Tuple[list[Match], Dict[str, Player]]:
    """Load matches and players from a database."""
    logger.info("Loading matches and players from database...")

    with Session(engine) as session:
        # Get minimum date and add offset
        min_date = session.exec(select(func.min(Match.date))).first()
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


async def process_matches(
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
        stats.append(
            await process_one_match(
                match=match,
                async_engine=async_engine,
                player_id_to_player=player_id_to_player,
                ks=ks,
                output_dir=output_path,
                override=override,
            )
        )

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
        process_matches(
            all_matches,
            player_id_to_player,
            ks,
            output_path,
            async_db_url,
            override,
        )
    )

    logger.info(f"✅ Successfully generated statistics for {len(all_matches)} matches")
    logger.info(f"📁 Output saved to: {output_path.absolute()}")

    pack_all_jsons(output_path)


if __name__ == "__main__":
    app()
