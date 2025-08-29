from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from tennis_scrapper.db.models import Player
from tennis_scrapper.api.deps import get_async_session


router = APIRouter(prefix="/players", tags=["players"])


@router.get("")
async def list_players(
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_async_session),
):
    statement = select(Player)
    if q:
        statement = statement.where(Player.name.ilike(f"%{q}%"))
    statement = statement.offset(offset).limit(limit)
    result = await session.exec(statement)
    return result.all()


@router.get("/{player_id}")
async def get_player(
    player_id: str, session: AsyncSession = Depends(get_async_session)
):
    player = await session.get(Player, player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player
