import uvicorn

from tennis_scrapper.api.app import create_app


app = create_app()


def run():
    uvicorn.run(
        "tennis_scrapper.api.main:app",
    )
    # app.run(
    #     host="0.0.0.0",
    #     port=8000,
        
    # )


if __name__ == "__main__":
    run()
