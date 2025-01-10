import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def draw_probability(rating_diff, base_draw=0.20, steepness=0.006):
    """
    Returns a draw probability that approaches 'base_draw' when
    rating_diff is large, and increases further when rating_diff is small.
    'steepness' controls how fast draw probability rises as |diff| -> 0.

    This is just one example; you can tweak to taste.
    """
    # The smaller |diff|, the higher the draw probability,
    # but never less than base_draw.
    return base_draw + (0.5 - base_draw) * np.exp(-steepness * (rating_diff**2))


class ChessEloPredictor:
    def __init__(self, csv_path, n_simulations=10000, matches_per_month=5):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.df.columns = pd.to_datetime(self.df.columns)
        self.n_simulations = n_simulations
        self.matches_per_month = matches_per_month

        # Get initial ratings and active players
        self.current_ratings = self.get_latest_ratings()

    def get_latest_ratings(self):
        """Get the most recent valid rating for each player"""
        latest_ratings = {}
        for player in self.df.index:
            ratings = self.df.loc[player].dropna()
            if len(ratings) > 0:
                latest_ratings[player] = ratings.iloc[-1]
        return latest_ratings

    def calculate_expected_score(self, rating_a, rating_b):
        """Calculate expected score (win probability + 0.5 * draw probability) for player A"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def simulate_match(self, rating_a, rating_b):
        """Simulate a match between two players based on Elo probabilities"""
        expected_score = self.calculate_expected_score(rating_a, rating_b)
        result = np.random.random()

        draw_prob = draw_probability(abs(rating_a - rating_b))
        assert expected_score + draw_prob/2 < 1.0, "Draw probability too high"
        if result < draw_prob:  # Draw
            return 0.5
        elif result < expected_score + draw_prob/2:  # Win
            return 1.0
        else:  # Loss
            return 0.0

    def simulate_rating_changes(self, start_date, end_date):
        """Simulate future ratings based on head-to-head matches"""
        months = pd.date_range(start_date, end_date, freq="M")
        simulated_ratings = {
            player: np.zeros((self.n_simulations, len(months) + 1))
            for player in self.current_ratings
        }

        # Set initial ratings for all simulations
        for player in simulated_ratings:
            simulated_ratings[player][:, 0] = self.current_ratings[player]

        # Run simulations
        for sim in tqdm(range(self.n_simulations)):
            # For each month
            for month_idx in range(len(months)):
                # Get current ratings for this simulation
                current_sim_ratings = {
                    player: simulated_ratings[player][sim, month_idx]
                    for player in simulated_ratings
                }

                # Simulate matches for the month
                top_players = sorted(
                    current_sim_ratings.items(), key=lambda x: x[1], reverse=True
                )[
                    :20
                ]  # Focus on top 20 players

                # Each player plays matches_per_month matches against others
                for player_a, rating_a in top_players:
                    # Select opponents weighted by rating proximity
                    rating_diffs = [
                        abs(rating_a - p[1]) for p in top_players if p[0] != player_a
                    ]
                    weights = 1 / (
                        np.array(rating_diffs) + 100
                    )  # Add 100 to avoid division by zero
                    weights = weights / weights.sum()

                    # Sample opponents
                    opponents = np.random.choice(
                        [p[0] for p in top_players if p[0] != player_a],
                        size=self.matches_per_month,
                        p=weights,
                        replace=True,
                    )

                    # Play matches
                    k_factor = 10  # Monthly K-factor
                    monthly_rating_change = 0

                    for opponent in opponents:
                        # Simulate using static "true Elo"
                        result = self.simulate_match(
                            self.current_ratings[player_a], self.current_ratings[opponent]
                        )
                        # Adjust based on current simulated Elo
                        expected_score = self.calculate_expected_score(
                            rating_a, current_sim_ratings[opponent]
                        )
                        rating_change = k_factor * (result - expected_score)
                        monthly_rating_change += rating_change

                    # Update ratings for next month
                    simulated_ratings[player_a][sim, month_idx + 1] = (
                        current_sim_ratings[player_a] + monthly_rating_change
                    )

        return simulated_ratings, months

    def plot_sample_paths(
        self, player_name, start_date="2025-01-01", end_date="2026-12-31", n_paths=50
    ):
        """Plot sample rating paths for a player"""
        simulated_ratings, months = self.simulate_rating_changes(start_date, end_date)
        months = pd.DatetimeIndex([pd.to_datetime(start_date)] + list(months))

        plt.figure(figsize=(12, 6))
        paths = simulated_ratings[player_name][:n_paths]
        for path in paths:
            plt.plot(months, path, alpha=0.1, color="blue")

        # Plot mean path and confidence intervals
        mean_path = simulated_ratings[player_name].mean(axis=0)
        std_path = simulated_ratings[player_name].std(axis=0)
        plt.plot(months, mean_path, color="red", linewidth=2, label="Mean projection")
        plt.fill_between(
            months,
            mean_path - 2 * std_path,
            mean_path + 2 * std_path,
            color="red",
            alpha=0.1,
        )

        # Plot historical ratings
        historical = self.df.loc[player_name].dropna()
        plt.plot(
            historical.index,
            historical.values,
            color="black",
            linewidth=2,
            label="Historical",
        )

        plt.title(f"{player_name} Rating Projections")
        plt.xlabel("Date")
        plt.ylabel("ELO Rating")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt
