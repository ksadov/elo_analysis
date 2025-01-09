import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def plot_top_players_ratings(csv_path, date_str, n_players=10):
    """
    Plot ELO ratings over time for top N players at a specified date.

    Args:
        csv_path (str): Path to the CSV file containing ratings
        date_str (str): Date string in format 'YYYY-MM-DD' to select top players from
        n_players (int): Number of top players to show
    """
    df = pd.read_csv(csv_path, index_col=0)

    df.columns = pd.to_datetime(df.columns)

    target_date = pd.to_datetime(date_str)
    print("date_str", date_str)
    print("target_date", target_date)
    closest_date = min(df.columns, key=lambda x: abs(x - target_date))

    reference_ratings = df[closest_date].dropna()

    top_players = reference_ratings.nlargest(n_players).index

    top_players_df = df.loc[top_players]

    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")

    for player in top_players:
        player_ratings = top_players_df.loc[player]
        plt.plot(
            player_ratings.index,
            player_ratings.values,
            marker="o",
            markersize=3,
            label=player.split(",")[0],
        )

    plt.title(
        f'Top {n_players} Chess Players ELO Ratings Over Time\n(Selected from {closest_date.strftime("%B %Y")})',
        fontsize=14,
        pad=20,
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("ELO Rating", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.ylim(
        min(2600, top_players_df.min().min() - 20),
        max(2900, top_players_df.max().max() + 20),
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    return plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="elo.csv")
    parser.add_argument("--date_str", type=str, default="2025-01-01")
    parser.add_argument("--n_players", type=int, default=10)
    args = parser.parse_args()
    csv_path = "elo.csv"
    plot = plot_top_players_ratings(
        csv_path=args.csv_path, date_str=args.date_str, n_players=args.n_players
    )
    plot.savefig("top_players_ratings.png", bbox_inches="tight", dpi=300)
    plot.show()
