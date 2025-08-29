from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from tennis_scrapper.db.models import Match
from tennis_scrapper.api.deps import get_async_session
from tennis_scrapper.db.db_utils import get_h2h_matches, get_player_history


router = APIRouter(prefix="/matches", tags=["matches"])


@router.get("")
async def list_matches(
    player_id: Optional[str] = None,
    before: Optional[date] = None,
    limit: int = Query(default=100, le=500),
    offset: int = 0,
    session: AsyncSession = Depends(get_async_session),
):
    if player_id:
        matches = await get_player_history(
            db_session=session, player_id=player_id, cutting_date=before, limit=limit
        )
        return matches
    statement = select(Match).order_by(Match.date.desc()).offset(offset).limit(limit)
    result = await session.exec(statement)
    return result.all()


@router.get("/{match_id}")
async def get_match(match_id: str, session: AsyncSession = Depends(get_async_session)):
    match = await session.get(Match, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@router.get("/h2h")
async def h2h(
    player_1_id: str = Query(..., alias="p1"),
    player_2_id: str = Query(..., alias="p2"),
    before: Optional[date] = None,
    session: AsyncSession = Depends(get_async_session),
):
    return await get_h2h_matches(
        db_session=session,
        player_1_id=player_1_id,
        player_2_id=player_2_id,
        cutting_date=before,
    )
