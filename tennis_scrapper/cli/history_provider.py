from datetime import date
from sqlalchemy import or_
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from db.models import Surface, Match
from stats.stats_utils import is_match_valid


class PlayerHistoryProvider:
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.player_id_to_history = {}

    async def get_player_history_at_dt(
        self,
        player_id: str,
        date: date,
        hit_limit: int,
        surface: Surface,
    ) -> tuple[list[Match], list[Match]]:

        if player_id not in self.player_id_to_history:
            result = await self.db_session.exec(
                select(Match)
                .where(
                    or_(Match.player_1_id == player_id, Match.player_2_id == player_id),
                )
            .order_by(Match.date)
            )
            matches = result.all()
            matches = list(filter(is_match_valid, matches))
            self.player_id_to_history[player_id] = matches
            
        matches = self.player_id_to_history[player_id]
        matches = list(filter(lambda m: m.date < date, matches))

        matches_on_surface = [match for match in matches if match.surface == surface]
        matches_on_surface = (
            matches_on_surface[-hit_limit:]
            if len(matches_on_surface) > hit_limit
            else matches_on_surface
        )
        matches = matches[-hit_limit:] if len(matches) > hit_limit else matches

        return matches, matches_on_surface