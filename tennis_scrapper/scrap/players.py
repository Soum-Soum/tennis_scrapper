from db.models import Gender, Player
from scrap.utils import find_div_by_text


from bs4 import BeautifulSoup


import datetime


def scrap_player_from_html(html: str, player_detail_url_extension: str) -> Player:

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", "plDetail")
    td = table.find_all("td")[1]
    player_name = td.find("h3").get_text(strip=True)
    divs = td.find_all("div", class_="date")

    country_div = find_div_by_text(divs, "Country: ")
    country = (
        country_div.get_text(strip=True).replace("Country: ", "").strip()
        if country_div
        else "Unknown"
    )

    birth_date_div = find_div_by_text(divs, "Age: ")
    birth_date = (
        birth_date_div.get_text(strip=True).split("(")[1].replace(")", "").strip()
        if birth_date_div
        else "01.01.1970"
    )
    day, month, year = map(int, birth_date.split("."))
    birth_date = datetime.date(year=year, month=month, day=day)

    gender_div = find_div_by_text(divs, "Sex: ")
    gender = (
        Gender.from_string(gender_div.get_text(strip=True).replace("Sex: ", "").strip())
        if gender_div
        else "Unknown"
    )

    main_hand_div = find_div_by_text(divs, "Plays: ")
    main_hand = (
        main_hand_div.get_text(strip=True).replace("Plays: ", "").strip().upper()
        if main_hand_div
        else "UNKNOWN"
    )
    return Player(
        name=player_name,
        country=country,
        birth_date=birth_date,
        gender=gender,
        preferred_hand=main_hand,
        url_extension=player_detail_url_extension,
    )
