from datetime import datetime
from bs4 import BeautifulSoup
import pytest
from db.models import Gender, Tournament
from scrap.matches import pair_to_match
from scrap.tournaments import get_default_tournament


@pytest.fixture
def row_one_html():
    ROW_ONE_HTML = """
    <tr id="r10" class="one fRow bott" onmouseover="md_over(this);" onmouseout="md_out(this);" style="">
        <td class="first time" rowspan="2">19:50<br><a href="/match-detail/?id=2988741" title="Click for match highlights"><img src="/res/img/highlights.gif"></a></td>
        <td class="t-name"><a href="/player/sinner-8b8e8/">Sinner J.</a> (1)</td>
        <td class="result">2</td>
        <td class="score">6</td>
        <td class="score">7</td>
        <td class="score">&nbsp;</td>
        <td class="score">&nbsp;</td>
        <td class="score">&nbsp;</td>
        <td class="coursew" rowspan="2">1.03</td>
        <td class="course" rowspan="2">14.33</td>
        <td class="alone-icons" rowspan="2">&nbsp;</td>
        <td rowspan="2"><a href="/match-detail/?id=2988741" title="Click for match detail">info</a></td>
    </tr>
    """
    return BeautifulSoup(ROW_ONE_HTML, "lxml")


@pytest.fixture
def row_two_html():
    ROW_TWO_HTML = """
    <tr id="r10b" class="one" onmouseover="md_over(this);" onmouseout="md_out(this);" style="">
        <td class="t-name"><a href="/player/mannarino-a7108/">Mannarino A.</a></td>
        <td class="result">0</td>
        <td class="score">4</td>
        <td class="score">6<sup>4</sup></td>
        <td class="score">&nbsp;</td>
        <td class="score">&nbsp;</td>
        <td class="score">&nbsp;</td>
    </tr>
    """
    return BeautifulSoup(ROW_TWO_HTML, "lxml")


def test_pair_to_match(row_one_html, row_two_html):
    dummy_tournament = get_default_tournament(2023, Gender.MEN)
    date = datetime(2023, 5, 15).date()
    the_match = pair_to_match(
        tournament=dummy_tournament, row1=row_one_html, row2=row_two_html, date=date
    )

    assert the_match.tournament_id == dummy_tournament.tournament_id
    assert the_match.date == date
    assert the_match.players_gender == dummy_tournament.players_gender
    assert the_match.surface == dummy_tournament.surface
    assert the_match.player_1_url_extension == "/player/sinner-8b8e8/"
    assert the_match.player_2_url_extension == "/player/mannarino-a7108/"
    assert the_match.score == "6-4 7-6"
    assert the_match.player_1_odds == 1.03
    assert the_match.player_2_odds == 14.33
