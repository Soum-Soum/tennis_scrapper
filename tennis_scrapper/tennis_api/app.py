from fastapi import FastAPI

from tennis_api.routers.players import router as players_router
from tennis_api.routers.matches import router as matches_router
from tennis_api.routers.tournaments import router as tournaments_router
from tennis_api.routers.rankings import router as rankings_router


def create_app() -> FastAPI:
    app = FastAPI(title="Tennis Scrapper API", version="0.1.0")
    app.include_router(players_router)
    app.include_router(matches_router)
    app.include_router(tournaments_router)
    app.include_router(rankings_router)
    return app
