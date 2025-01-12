import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from chess_elo_predictor import ChessEloPredictor
import argparse


def analyze_highest_elo_probability(predictor, players, target_date="2025-12-31"):
    """Calculate probability of each player having the highest rating at target date"""
    simulated_ratings, months = predictor.simulate_rating_changes(
        "2025-01-01", target_date
    )

    # Get final ratings for each simulation
    final_ratings = {
        player: ratings[:, -1] for player, ratings in simulated_ratings.items()
    }

    # Count times each player has highest rating
    n_sims = predictor.n_simulations
    highest_count = {player: 0 for player in players}

    for sim in range(n_sims):
        sim_ratings = {player: final_ratings[player][sim] for player in players}
        highest_player = max(sim_ratings.items(), key=lambda x: x[1])[0]
        highest_count[highest_player] += 1

    probabilities = {player: count / n_sims for player, count in highest_count.items()}
    return probabilities


def plot_projections(
    predictor, players, start_date="2025-01-01", end_date="2025-12-31"
):
    """Plot rating projections for multiple players"""
    simulated_ratings, months = predictor.simulate_rating_changes(start_date, end_date)
    months = pd.DatetimeIndex([pd.to_datetime(start_date)] + list(months))

    plt.figure(figsize=(15, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(players)))

    for player, color in zip(players, colors):
        mean_path = simulated_ratings[player].mean(axis=0)
        std_path = simulated_ratings[player].std(axis=0)

        plt.plot(months, mean_path, color=color, linewidth=2, label=f"{player}")
        plt.fill_between(
            months, mean_path - std_path, mean_path + std_path, color=color, alpha=0.1
        )

        # Plot historical ratings
        historical = predictor.df.loc[player].dropna()
        plt.plot(
            historical.index,
            historical.values,
            color=color,
            linewidth=1,
            linestyle="--",
            alpha=0.5,
        )

    # lower bound should be 100 points below the lowest historical rating
    min_rating = min([predictor.df.loc[p].min() for p in players]) - 100
    plt.ylim(min_rating, 3000)
    plt.title("Rating Projections for Top Players")
    plt.xlabel("Date")
    plt.ylabel("ELO Rating")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def analyze_2800_breakthrough(predictor, players):
    """Analyze which player is most likely to break 2800 first"""
    simulated_ratings, months = predictor.simulate_rating_changes(
        "2025-01-01", "2025-12-31"
    )

    # Only consider players currently below 2800
    eligible_players = [p for p in players if predictor.current_ratings[p] < 2800]
    breakthrough_counts = {player: 0 for player in eligible_players}

    for sim in range(predictor.n_simulations):
        first_breakthrough = None
        earliest_month = len(months)

        for player in eligible_players:
            ratings = simulated_ratings[player][sim]
            breakthrough_month = np.argmax(ratings > 2800)

            if breakthrough_month < len(months) and breakthrough_month < earliest_month:
                earliest_month = breakthrough_month
                first_breakthrough = player

        if first_breakthrough:
            breakthrough_counts[first_breakthrough] += 1

    probabilities = {
        player: count / predictor.n_simulations
        for player, count in breakthrough_counts.items()
    }
    return probabilities


def analyze_top10_probability(predictor, player, target_year=2030):
    """Analyze probability of reaching top 10 before target year"""
    simulated_ratings, months = predictor.simulate_rating_changes(
        "2025-01-01", f"{target_year}-12-31"
    )

    reached_top10 = 0
    for sim in range(predictor.n_simulations):
        for month in range(len(months)):
            # Get all ratings for this simulation and month
            current_ratings = {
                p: simulated_ratings[p][sim, month] for p in simulated_ratings.keys()
            }

            # Sort by rating
            sorted_ratings = sorted(
                current_ratings.items(), key=lambda x: x[1], reverse=True
            )

            # Check if player is in top 10
            if player in [p[0] for p in sorted_ratings[:10]]:
                reached_top10 += 1
                break

    probability = reached_top10 / predictor.n_simulations
    return probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elo_csv",
        type=str,
        default="elo.csv",
        help="Path to CSV file with historical Elo ratings",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=10000,
        help="Number of simulations to run for each scenario",
    )
    parser.add_argument(
        "--matches_per_month",
        type=int,
        default=5,
        help="Number of matches to simulate per month",
    )
    args = parser.parse_args()
    predictor = ChessEloPredictor(
        args.elo_csv, args.n_simulations, args.matches_per_month
    )
    players = [
        "Praggnanandhaa R",
        "Gukesh D",
        "Keymer, Vincent",
        "Erigaisi Arjun",
        "Abdusattorov, Nodirbek",
        "Firouzja, Alireza",
        "Niemann, Hans Moke",
        "Wei, Yi",
    ]
    highest_elo_prob = analyze_highest_elo_probability(predictor, players)
    print("Probability of highest Elo at end of 2025 for selected players:")
    for player, prob in highest_elo_prob.items():
        print(f"{player}: {prob:.2%}")
    plt = plot_projections(predictor, players)
    plt.show()
    plt.savefig("prodigy_elo_projections.png")

    breakthrough_probs = analyze_2800_breakthrough(predictor, players)
    print("\nProbability of breaking 2800 before end of 2025:")
    for player, prob in breakthrough_probs.items():
        print(f"{player}: {prob:.2%}")

    top10_player = "Niemann, Hans Moke"
    top10_prob = analyze_top10_probability(predictor, top10_player)
    print(
        f"\nProbability of {top10_player} reaching top 10 before 2030: {top10_prob:.2%}"
    )


if __name__ == "__main__":
    main()
