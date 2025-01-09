import os
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from tqdm import tqdm


def process_html_file(filepath):
    """
    Process a single HTML file and extract player names and ratings.
    Returns a dictionary mapping player names to their ratings.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    ratings = {}
    # Find all table rows
    rows = soup.find_all("tr")

    for row in rows:
        # Skip header rows
        if row.find("th"):
            continue

        # Extract cells
        cells = row.find_all("td")
        if len(cells) >= 5:  # Ensure we have enough cells
            name = cells[1].text.strip()  # Name is in second column
            rating = cells[4].text.strip()  # Rating is in fifth column

            # Clean up the name (remove the link text if present)
            name = name.replace("\n", " ").strip()

            # Convert rating to integer
            try:
                rating = int(rating)
                ratings[name] = rating
            except ValueError:
                continue

    return ratings


def process_all_files(directory):
    """
    Process all HTML files in the directory and create a DataFrame.
    """
    # Get all HTML files and sort them
    files = [f for f in os.listdir(directory) if f.endswith(".html")]
    files.sort()

    # Initialize data structures
    all_players = set()
    ratings_by_date = {}

    # Process each file
    for filename in tqdm(files):
        # Extract date from filename (assuming format YYYY-MM-DD.html)
        date_str = filename.split(".")[0]  # Remove .html extension

        # Process the file
        filepath = os.path.join(directory, filename)
        ratings = process_html_file(filepath)

        # Store the ratings and update the set of all players
        ratings_by_date[date_str] = ratings
        all_players.update(ratings.keys())

    # Create DataFrame
    df = pd.DataFrame(index=sorted(list(all_players)))

    # Fill in the ratings for each date
    for date_str, ratings in sorted(ratings_by_date.items()):
        df[date_str] = df.index.map(lambda x: ratings.get(x, float("nan")))

    return df


def main():
    # Process all files in the fide_html directory
    df = process_all_files("fide_html")

    # Save to CSV
    df.to_csv("elo.csv")
    print(f"Successfully processed the files and created elo.csv")
    print(f"Shape of the data: {df.shape}")
    print(f"Number of players: {len(df)}")
    print(f"Number of time periods: {len(df.columns)}")


if __name__ == "__main__":
    main()
