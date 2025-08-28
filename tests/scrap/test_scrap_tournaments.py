from urllib.request import urlopen
import pytest

from tennis_scrapper.scrap.tournaments import tournaments_from_html
from tennis_scrapper.db.models import Gender, Surface
from tennis_scrapper.scrap.urls import get_one_year_tournaments_url


@pytest.fixture
def url() -> str:
    return get_one_year_tournaments_url(gender=Gender.MEN, year=2024)


def test_scrap_tournaments_ok(url: str):
    html = urlopen(url).read()
    tournaments = tournaments_from_html(html, Gender.MEN)
    assert (
        len(tournaments) == 298
    ), f"Expected 298 tournaments to be scraped but got {len(tournaments)}"

    assert tournaments[0].name == "Macau Tennis Masters"
    assert tournaments[0].year == 2024
    assert tournaments[0].surface == Surface.INDOOR

    assert tournaments[-1].name == "United Cup"
    assert tournaments[-1].year == 2023
    assert tournaments[-1].surface == Surface.HARD
