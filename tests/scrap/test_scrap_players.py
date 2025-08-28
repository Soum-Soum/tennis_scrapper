from datetime import datetime
from urllib.request import urlopen
import pytest

from tennis_scrapper.db.models import Gender
from tennis_scrapper.scrap.players import player_from_html
from tennis_scrapper.scrap.urls import get_player_detail_url


@pytest.fixture
def url_extension() -> str:
    return "/player/sinner-8b8e8/"


@pytest.fixture
def url(url_extension: str) -> str:
    return get_player_detail_url(url_extension)


def test_scrap_player_ok(url: str, url_extension: str):
    html = urlopen(url).read()
    player = player_from_html(html, url_extension)

    assert player.name == "Sinner Jannik"
    assert player.country == "Italy"
    assert player.birth_date == datetime(2001, 8, 16).date()
    assert player.gender == Gender.MEN
    assert player.url_extension == url_extension
