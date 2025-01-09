from datetime import date
import requests
import os
from typing import List
import time
from tqdm import tqdm


def construct_url(search_date: date) -> str:
    # format in YYYY-MM-DD
    date_str = search_date.strftime("%Y-%m-%d")
    return f"https://ratings.fide.com/a_top_var.php?continent=0&country=&rating=standard&gender=&age1=0&age2=0&period={date_str}&period2=1"


def construct_date_list(num_years: int) -> List[date]:
    today = date.today()
    date_list = []
    for year in range(today.year - num_years, today.year + 1):
        for month in range(1, 13):
            date_list.append(date(year, month, 1))
    date_list.reverse()
    # remove future dates
    date_list = [d for d in date_list if d <= today]
    return date_list


def get_html(out_html_dir: str, num_years: int, sleep_time: float) -> None:
    if not os.path.exists(out_html_dir):
        os.makedirs(out_html_dir)
    date_list = construct_date_list(num_years)
    for search_date in tqdm(date_list):
        url = construct_url(search_date)
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        response = requests.get(url, headers={"User-Agent": user_agent})
        with open(f"fide_html/{search_date}.html", "w") as f:
            f.write(response.text)
        time.sleep(sleep_time)


def main():
    out_html_dir = "fide_html"
    num_years = 10
    sleep_time = 2.0
    get_html(out_html_dir, num_years, sleep_time)


if __name__ == "__main__":
    main()
