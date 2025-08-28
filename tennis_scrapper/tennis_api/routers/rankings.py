from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from db.models import Ranking, Gender
from tennis_api.deps import get_async_session
from db.db_utils import get_last_ranking


router = APIRouter(prefix="/rankings", tags=["rankings"])


@router.get("/latest")
async def latest(gender: Gender, session: AsyncSession = Depends(get_async_session)):
    # Reuse existing sync helper with a temporary sync session if needed
    # But we already have an async session; create a tiny sync shim is non-trivial here.
    # Implement the query inline for async path.
    from sqlmodel import select

    last_date = await session.exec(
        select(Ranking.date)
        .where(Ranking.circuit == gender.circuit)
        .order_by(Ranking.date.desc())
        .limit(1)
    )
    last_date = last_date.first()
    result = await session.exec(
        select(Ranking).where(
            Ranking.circuit == gender.circuit, Ranking.date == last_date
        )
    )
    return result.all()
