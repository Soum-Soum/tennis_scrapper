from datetime import date, datetime

from tennis_scrapper.db.models import Gender


BASE_URL = "https://www.tennisexplorer.com"


def get_match_result_list_page_url(
    gender: Gender,
    date: datetime.date,
) -> str:
    extension = gender.player_url_extension
    year = date.year
    month = date.month
    day = date.day
    return f"{BASE_URL}/results/?type={extension}&year={year}&month={month}&day={day}"


def get_match_list_page_url(
    gender: Gender,
    date: datetime.date,
) -> str:
    extension = gender.player_url_extension
    year = date.year
    month = date.month
    day = date.day
    return f"{BASE_URL}/matches/?type={extension}&year={year}&month={month}&day={day}"


def get_player_detail_url(player_url_extension: str) -> str:
    return f"{BASE_URL}/{player_url_extension}/"


def get_one_year_tournaments_url(gender: Gender, year: int) -> str:
    return f"{BASE_URL}/calendar/{gender.tournament_url_extension}/{year}/"


def get_ranking_date_url(gender: Gender, year: int) -> str:
    url = f"{BASE_URL}/ranking/{gender.tournament_url_extension}/{year}/"
    return url


def get_ranking_url(gender: Gender, date: date, page_index: int) -> str:
    extension = gender.tournament_url_extension
    year = date.year
    return f"{BASE_URL}/ranking/{extension}/{year}/?date={date}&page={page_index}"
