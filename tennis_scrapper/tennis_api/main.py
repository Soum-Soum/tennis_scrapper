import uvicorn

from tennis_api.app import create_app


app = create_app()


def run():
    uvicorn.run(
        "tennis_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
