from datetime import date

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from db.models import Match, Surface
from stats.stats_utils import is_match_valid


async def get_player_history_at_dt(
    player_id: str,
    date: date,
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
    match_date: date,
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
    )
    matches = matches.all()
    matches = list(filter(is_match_valid, matches))
    return sorted(matches, key=lambda x: x.date)


async def get_data_from_db(
    match: Match,
    ks: list[int],
    async_engine: AsyncEngine,
) -> tuple[list[Match], list[Match], list[Match], list[Match], list[Match]]:
    async with AsyncSession(async_engine) as db_session:
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

        h2h_matches = await get_h2h_matches(
            player_1_id=match.player_1_id,
            player_2_id=match.player_2_id,
            match_date=match.date,
            db_session=db_session,
        )

    return (
        matches_player_1,
        matches_player_1_on_surface,
        matches_player_2,
        matches_player_2_on_surface,
        h2h_matches,
    )
