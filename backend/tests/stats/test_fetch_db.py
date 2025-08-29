import asyncio
import datetime

from sqlmodel.ext.asyncio.session import AsyncSession

from tennis_scrapper.db.models import Surface
from tennis_scrapper.stats.fetch_db import (
    get_player_history_at_dt,
    get_h2h_matches_at_dt,
    get_data_from_db,
)
from tests.utils.async_db import (
    create_test_async_engine,
    dispose_engine,
    make_player,
    make_match,
    seed_players_and_matches,
)


def test_get_player_history_at_dt_basic():
    async def _run():
        engine = await create_test_async_engine()
        try:
            # players
            p1 = make_player("Player One", "p1")
            p2 = make_player("Player Two", "p2")

            # matches in order
            m1 = make_match(
                date=datetime.date(2020, 1, 10),
                player_1=p1,
                player_2=p2,
                score="6-4 6-4",
                surface=Surface.HARD,
            )
            m2 = make_match(
                date=datetime.date(2020, 2, 10),
                player_1=p2,
                player_2=p1,
                score="6-2 6-2",
                surface=Surface.CLAY,
            )
            m3 = make_match(
                date=datetime.date(2020, 3, 10),
                player_1=p1,
                player_2=p2,
                score="7-6 7-6",
                surface=Surface.HARD,
            )

            async with AsyncSession(engine, expire_on_commit=False) as session:
                await seed_players_and_matches(
                    session=session, players=[p1, p2], matches=[m1, m2, m3]
                )

                all_matches, matches_on_surface = await get_player_history_at_dt(
                    player_id=p1.player_id,
                    cutting_date=datetime.date(2020, 4, 1),
                    hit_limit=10,
                    surface=Surface.HARD,
                    db_session=session,
                )

                assert [m.score for m in all_matches] == [
                    "6-4 6-4",
                    "6-2 6-2",
                    "7-6 7-6",
                ]
                assert [m.score for m in matches_on_surface] == [
                    "6-4 6-4",
                    "7-6 7-6",
                ]

                # With a hit_limit of 2, should keep last 2 chronologically
                all_matches, matches_on_surface = await get_player_history_at_dt(
                    player_id=p1.player_id,
                    cutting_date=datetime.date(2020, 4, 1),
                    hit_limit=2,
                    surface=Surface.HARD,
                    db_session=session,
                )
                assert [m.score for m in all_matches] == ["6-2 6-2", "7-6 7-6"]
                assert [m.score for m in matches_on_surface] == ["6-4 6-4", "7-6 7-6"]
        finally:
            await dispose_engine(engine)

    asyncio.run(_run())


def test_get_h2h_matches_orders_and_filtering():
    async def _run():
        engine = await create_test_async_engine()
        try:
            p1 = make_player("A", "a")
            p2 = make_player("B", "b")
            p3 = make_player("C", "c")

            m_older = make_match(
                date=datetime.date(2019, 5, 1),
                player_1=p1,
                player_2=p2,
                score="6-0 6-0",
                surface=Surface.GRASS,
            )
            m_newer = make_match(
                date=datetime.date(2020, 5, 1),
                player_1=p2,
                player_2=p1,
                score="6-1 6-1",
                surface=Surface.GRASS,
            )
            m_too_late = make_match(
                date=datetime.date(2021, 1, 1),
                player_1=p1,
                player_2=p2,
                score="7-5 7-5",
                surface=Surface.GRASS,
            )
            m_other = make_match(
                date=datetime.date(2020, 5, 1),
                player_1=p1,
                player_2=p3,
                score="6-2 6-2",
                surface=Surface.CLAY,
            )

            async with AsyncSession(engine, expire_on_commit=False) as session:
                await seed_players_and_matches(
                    session=session,
                    players=[p1, p2, p3],
                    matches=[m_older, m_newer, m_too_late, m_other],
                )

                h2h = await get_h2h_matches_at_dt(
                    player_1_id=p1.player_id,
                    player_2_id=p2.player_id,
                    cutting_date=datetime.date(2020, 12, 31),
                    db_session=session,
                )

                assert [m.score for m in h2h] == ["6-0 6-0", "6-1 6-1"]
        finally:
            await dispose_engine(engine)

    asyncio.run(_run())


def test_get_data_from_db_integration():
    async def _run():
        engine = await create_test_async_engine()
        try:
            p1 = make_player("X", "x")
            p2 = make_player("Y", "y")

            # history for p1 (2 hard, 1 clay)
            h1 = make_match(
                date=datetime.date(2022, 1, 1),
                player_1=p1,
                player_2=p2,
                score="6-2 6-3",
                surface=Surface.HARD,
            )
            h2 = make_match(
                date=datetime.date(2022, 2, 1),
                player_1=p2,
                player_2=p1,
                score="6-4 6-4",
                surface=Surface.CLAY,
            )
            h3 = make_match(
                date=datetime.date(2022, 3, 1),
                player_1=p1,
                player_2=p2,
                score="7-6 6-7 7-6",
                surface=Surface.HARD,
            )

            # history for p2 vs p1 already included above; add another vs someone else to ensure filtering
            p3 = make_player("Z", "z")
            other = make_match(
                date=datetime.date(2022, 2, 15),
                player_1=p2,
                player_2=p3,
                score="6-0 6-0",
                surface=Surface.HARD,
            )

            # target match on 2022-04-01 on HARD for p1 vs p2
            target = make_match(
                date=datetime.date(2022, 4, 1),
                player_1=p1,
                player_2=p2,
                score="6-3 6-3",
                surface=Surface.HARD,
            )

            async with AsyncSession(engine, expire_on_commit=False) as session:
                await seed_players_and_matches(
                    session=session,
                    players=[p1, p2, p3],
                    matches=[h1, h2, h3, other, target],
                )

                k_values = [1, 2, 3, 4]
                all_p1, p1_surface, all_p2, p2_surface, h2h = await get_data_from_db(
                    match=target,
                    hit_limit=max(k_values),
                    async_engine=engine,
                )

                # ensure hit limit respected (max k = 3)
                assert [m.date for m in all_p1] == [
                    datetime.date(2022, 1, 1),
                    datetime.date(2022, 2, 1),
                    datetime.date(2022, 3, 1),
                ]
                assert [m.date for m in p1_surface] == [
                    datetime.date(2022, 1, 1),
                    datetime.date(2022, 3, 1),
                ]

                # p2 history contains only matches vs p1 before target date that are counted in all_p2
                assert [m.date for m in all_p2] == [
                    datetime.date(2022, 1, 1),
                    datetime.date(2022, 2, 1),
                    datetime.date(2022, 2, 15),
                    datetime.date(2022, 3, 1),
                ], f"all_p2: {[m.date for m in all_p2]}"

                assert [m.date for m in p2_surface] == [
                    datetime.date(2022, 1, 1),
                    datetime.date(2022, 2, 15),
                    datetime.date(2022, 3, 1),
                ]

                # h2h excludes target and is ordered
                assert [m.date for m in h2h] == [
                    datetime.date(2022, 1, 1),
                    datetime.date(2022, 2, 1),
                    datetime.date(2022, 3, 1),
                ]
        finally:
            await dispose_engine(engine)

    asyncio.run(_run())
