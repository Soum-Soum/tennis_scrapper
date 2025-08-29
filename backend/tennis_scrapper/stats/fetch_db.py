from datetime import date

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from tennis_scrapper.db.models import Match, Surface
from tennis_scrapper.stats.stats_utils import is_match_valid
from tennis_scrapper.db.db_utils import get_h2h_matches, get_player_history


async def get_player_history_at_dt(
    player_id: str,
    cutting_date: date,
    hit_limit: int,
    surface: Surface,
    db_session: AsyncSession,
) -> tuple[list[Match], list[Match]]:
    """Get player's match history up to a specific date."""
    matches = await get_player_history(
        db_session=db_session,
        player_id=player_id,
        cutting_date=cutting_date,
    )
    matches = list(filter(is_match_valid, matches))

    matches_on_surface = [match for match in matches if match.surface == surface]

    matches_on_surface = (
        matches_on_surface[-hit_limit:]
        if len(matches_on_surface) > hit_limit
        else matches_on_surface
    )
    matches = matches[-hit_limit:] if len(matches) > hit_limit else matches

    return matches, matches_on_surface


async def get_h2h_matches_at_dt(
    player_1_id: str,
    player_2_id: str,
    cutting_date: date,
    db_session: AsyncSession,
) -> list[Match]:
    """Get head-to-head matches between two players."""
    matches = await get_h2h_matches(
        db_session=db_session,
        player_1_id=player_1_id,
        player_2_id=player_2_id,
        cutting_date=cutting_date,
    )
    matches = list(filter(is_match_valid, matches))
    return matches


async def get_data_from_db(
    match: Match,
    hit_limit: int,
    async_engine: AsyncEngine,
) -> tuple[list[Match], list[Match], list[Match], list[Match], list[Match]]:
    async with AsyncSession(async_engine) as db_session:
        matches_player_1, matches_player_1_on_surface = await get_player_history_at_dt(
            player_id=match.player_1_id,
            cutting_date=match.date,
            hit_limit=hit_limit,
            surface=match.surface,
            db_session=db_session,
        )

        matches_player_2, matches_player_2_on_surface = await get_player_history_at_dt(
            player_id=match.player_2_id,
            cutting_date=match.date,
            hit_limit=hit_limit,
            surface=match.surface,
            db_session=db_session,
        )

        h2h_matches = await get_h2h_matches_at_dt(
            player_1_id=match.player_1_id,
            player_2_id=match.player_2_id,
            cutting_date=match.date,
            db_session=db_session,
        )

    return (
        matches_player_1,
        matches_player_1_on_surface,
        matches_player_2,
        matches_player_2_on_surface,
        h2h_matches,
    )
