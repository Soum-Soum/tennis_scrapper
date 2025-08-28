from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from db.models import Tournament, Gender
from tennis_api.deps import get_async_session


router = APIRouter(prefix="/tournaments", tags=["tournaments"])


@router.get("")
async def list_tournaments(
    gender: Optional[Gender] = None,
    year: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_async_session),
):
    statement = select(Tournament)
    if gender:
        statement = statement.where(Tournament.players_gender == gender)
    if year:
        statement = statement.where(Tournament.year == year)
    statement = statement.offset(offset).limit(limit)
    result = await session.exec(statement)
    return result.all()


@router.get("/{tournament_id}")
async def get_tournament(
    tournament_id: str, session: AsyncSession = Depends(get_async_session)
):
    tournament = await session.get(Tournament, tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")
    return tournament
