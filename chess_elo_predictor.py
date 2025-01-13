from webbrowser import get
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date, datetime


def draw_probability(rating_diff, base_draw=0.20, steepness=0.006):
    return base_draw + (0.5 - base_draw) * np.exp(-steepness * (rating_diff**2))


class ChessEloPredictor:
    def __init__(
        self,
        csv_path,
        n_simulations=10000,
        matches_per_month=5,
        weighting_period=0,
        simulation_start=None,
    ):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.df.columns = pd.to_datetime(self.df.columns)
        self.n_simulations = n_simulations
        self.matches_per_month = matches_per_month
        self.simulation_start = simulation_start

        # Get initial ratings and active players at simulation start
        self.current_ratings = self.get_ratings_at_date(simulation_start)
        if weighting_period > 0:
            self.true_ratings = self.init_weighted_ratings(
                weighting_period, simulation_start
            )
        else:
            self.true_ratings = self.current_ratings

    def get_ratings_at_date(self, date):
        """Get the most recent valid rating for each player up to given date"""
        latest_ratings = {}
        date = pd.to_datetime(date)
        for player in self.df.index:
            # Get ratings up to simulation start
            ratings = self.df.loc[player].loc[:date].dropna()
            if len(ratings) > 0:
                latest_ratings[player] = ratings.iloc[-1]
        return latest_ratings

    def init_weighted_ratings(self, weighting_period, cutoff_date):
        """Take a weighted average of historical ratings to get initial "true" ratings"""
        weighted_ratings = {}
        cutoff_date = pd.to_datetime(cutoff_date)

        for player in self.df.index:
            # Only use ratings up to cutoff date
            ratings = self.df.loc[player].loc[:cutoff_date].dropna()
            if len(ratings) > 0:
                ratings_no_na = ratings.tail(weighting_period)
                weights = np.arange(len(ratings_no_na), 0, -1)
                weighted_rating = np.average(ratings_no_na, weights=weights)
                weighted_ratings[player] = weighted_rating
        return weighted_ratings

    def calculate_expected_score(self, rating_a, rating_b):
        """Calculate expected score for player A"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def draw_probability(self, rating_diff, base_draw=0.20, steepness=0.006):
        """Calculate draw probability based on rating difference"""
        return base_draw + (0.5 - base_draw) * np.exp(-steepness * (rating_diff**2))

    def simulate_match(self, rating_a, rating_b):
        """Simulate a match between two players"""
        expected_score = self.calculate_expected_score(rating_a, rating_b)
        result = np.random.random()

        draw_prob = self.draw_probability(abs(rating_a - rating_b))
        if result < draw_prob:  # Draw
            return 0.5
        elif result < expected_score + draw_prob / 2:  # Win
            return 1.0
        else:  # Loss
            return 0.0

    def simulate_rating_changes(self, start_date, end_date):
        """Simulate ratings based on head-to-head matches"""
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        months = []
        # take date on the first of each month
        for year in range(start_date.year, end_date.year + 1):
            for month in range(1, 13):
                if year == start_date.year and month < start_date.month:
                    continue
                if year == end_date.year and month > end_date.month:
                    continue
                months.append(date(year, month, 1))
        simulated_ratings = {
            player: np.zeros((self.n_simulations, len(months) + 1))
            for player in self.true_ratings
        }

        # Set initial ratings
        for player in simulated_ratings:
            simulated_ratings[player][:, 0] = self.current_ratings[player]

        for sim in tqdm(range(self.n_simulations)):
            for month_idx in range(len(months)):
                current_sim_ratings = {
                    player: simulated_ratings[player][sim, month_idx]
                    for player in simulated_ratings
                }

                top_players = sorted(
                    current_sim_ratings.items(), key=lambda x: x[1], reverse=True
                )[:20]

                for player_a, rating_a in top_players:
                    # Weight opponents by rating proximity
                    rating_diffs = [
                        abs(rating_a - p[1]) for p in top_players if p[0] != player_a
                    ]
                    weights = 1 / (np.array(rating_diffs) + 100)
                    weights = weights / weights.sum()

                    # Sample opponents
                    opponents = np.random.choice(
                        [p[0] for p in top_players if p[0] != player_a],
                        size=self.matches_per_month,
                        p=weights,
                        replace=True,
                    )

                    k_factor = 10
                    monthly_rating_change = 0

                    for opponent in opponents:
                        result = self.simulate_match(
                            self.true_ratings[player_a],
                            self.true_ratings[opponent],
                        )
                        expected_score = self.calculate_expected_score(
                            rating_a, current_sim_ratings[opponent]
                        )
                        rating_change = k_factor * (result - expected_score)
                        monthly_rating_change += rating_change

                    simulated_ratings[player_a][sim, month_idx + 1] = (
                        current_sim_ratings[player_a] + monthly_rating_change
                    )

        return simulated_ratings, months

    def evaluate_predictions(
        self, player_name, simulated_ratings, months, actual_ratings
    ):
        """Calculate prediction accuracy metrics"""
        # Get actual ratings for comparison months
        actual_values = []
        predicted_values = []

        months = pd.DatetimeIndex(
            [pd.to_datetime(self.simulation_start)] + list(months)
        )
        mean_predictions = simulated_ratings[player_name].mean(axis=0)
        for i, month in enumerate(months):
            actual_rating = actual_ratings[month]
            if pd.notna(actual_rating):
                actual_values.append(actual_rating)
                predicted_values.append(mean_predictions[i])

        if len(actual_values) > 0:
            mse = mean_squared_error(actual_values, predicted_values)
            mae = mean_absolute_error(actual_values, predicted_values)
            rmse = np.sqrt(mse)
            return {"RMSE": rmse, "MAE": mae, "n_points": len(actual_values)}
        return None

    def plot_sample_paths(
        self,
        player_name,
        start_date="2025-01-01",
        end_date="2026-12-31",
        n_paths=50,
        save_path=None,
    ):
        """Plot postdicted rating paths against actual historical data"""
        simulated_ratings, months = self.simulate_rating_changes(start_date, end_date)
        months = pd.DatetimeIndex([pd.to_datetime(start_date)] + list(months))

        # Get actual ratings for the period
        actual_ratings = self.df.loc[player_name].loc[start_date:end_date].dropna()

        # Calculate accuracy metrics
        metrics = self.evaluate_predictions(
            player_name, simulated_ratings, months[1:], actual_ratings
        )

        plt.figure(figsize=(12, 6))

        # Plot sample paths
        paths = simulated_ratings[player_name][:n_paths]
        for path in paths:
            plt.plot(months, path, alpha=0.1, color="blue")

        # Plot mean and confidence intervals
        mean_path = simulated_ratings[player_name].mean(axis=0)
        std_path = simulated_ratings[player_name].std(axis=0)
        plt.plot(months, mean_path, color="red", linewidth=2, label="Mean projection")
        plt.fill_between(
            months,
            mean_path - 2 * std_path,
            mean_path + 2 * std_path,
            color="red",
            alpha=0.1,
            label="95% confidence interval",
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
        print("metrics", metrics)
        if metrics:
            plt.suptitle(
                f"RMSE: {metrics['RMSE']:.1f}, MAE: {metrics['MAE']:.1f}, n={metrics['n_points']}",
                fontsize=10,
            )

        plt.xlabel("Date")
        plt.ylabel("ELO Rating")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        return plt


def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = f"Not a valid date: '{s}'. Expected format: YYYY-MM-DD"
        raise argparse.ArgumentTypeError(msg)


def main():
    parser = argparse.ArgumentParser(description="Test Elo rating prediction accuracy")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="elo.csv",
        help="Path to CSV file with historical ratings",
    )
    parser.add_argument(
        "--player",
        type=str,
        default="Carlsen, Magnus",
        help="Player name to simulate (default: Carlsen, Magnus)",
    )
    parser.add_argument(
        "--start_date",
        type=valid_date,
        default=date(2020, 1, 1),
        help="Simulation start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=valid_date,
        default=date.today(),
        help="Simulation end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    parser.add_argument(
        "--matches_per_month",
        type=int,
        default=5,
        help="Average matches per month (default: 5)",
    )
    parser.add_argument(
        "--weighting_period",
        type=int,
        default=0,
        help="Number of months to use for weighted average (default: 0)",
    )
    parser.add_argument("--output", type=str, help="Path to save plot (optional)")

    args = parser.parse_args()

    # Validate dates
    if args.start_date >= args.end_date:
        parser.error("End date must be after start date")

    # Initialize predictor
    predictor = ChessEloPredictor(
        args.csv_path,
        n_simulations=args.n_simulations,
        matches_per_month=args.matches_per_month,
        weighting_period=args.weighting_period,
        simulation_start=args.start_date,
    )

    # Generate and save plot
    predictor.plot_sample_paths(
        args.player, args.start_date, args.end_date, save_path=args.output
    )
    plt.show()


if __name__ == "__main__":
    main()
