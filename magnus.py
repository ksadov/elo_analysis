import numpy as np
from chess_elo_predictor import ChessEloPredictor


def analyze_magnus_scenarios(
    predictor: ChessEloPredictor, start_date="2025-01-01", end_date="2026-12-31"
):
    """Analyze various probability scenarios for Magnus Carlsen"""
    simulated_ratings, months = predictor.simulate_rating_changes(start_date, end_date)

    # Find Magnus's historical peak
    magnus_ratings = predictor.df.loc["Carlsen, Magnus"].dropna()
    magnus_peak = magnus_ratings.max()

    results = {
        "july_2025_highest": 0,
        "top_2025_all_months": 0,
        "below_2800_before_2026": 0,
        "peak_broken_by_2026": 0,
    }

    july_2025_idx = np.where(months.strftime("%Y-%m") == "2025-07")[0][0] + 1
    months_2025 = [i for i, m in enumerate(months) if m.year == 2025]
    months_until_2026 = [i for i, m in enumerate(months) if m.year <= 2025]
    for sim in range(predictor.n_simulations):
        # July 2025 check
        july_ratings = {p: r[sim, july_2025_idx] for p, r in simulated_ratings.items()}
        magnus_july = july_ratings["Carlsen, Magnus"]
        results["july_2025_highest"] += int(magnus_july == max(july_ratings.values()))

        # Top throughout 2025
        top_all_months = True
        for month_idx in months_2025:
            month_ratings = {
                p: r[sim, month_idx + 1] for p, r in simulated_ratings.items()
            }
            if month_ratings["Carlsen, Magnus"] != max(month_ratings.values()):
                top_all_months = False
                break
        results["top_2025_all_months"] += int(top_all_months)

        # Below 2800 check
        magnus_path = simulated_ratings["Carlsen, Magnus"][sim]
        results["below_2800_before_2026"] += int(
            any(magnus_path[months_until_2026] < 2800)
        )

        # Peak broken check
        all_final_ratings = [r[sim, -1] for r in simulated_ratings.values()]
        results["peak_broken_by_2026"] += int(max(all_final_ratings) > magnus_peak)

    # Convert counts to probabilities
    for key in results:
        results[key] = results[key] / predictor.n_simulations

    return results


if __name__ == "__main__":
    # Usage example
    predictor = ChessEloPredictor("elo.csv")

    # Get probability predictions
    results = analyze_magnus_scenarios(predictor)
    print("\nProbability Estimates:")
    print(f"Magnus highest rated in July 2025: {results['july_2025_highest']:.1%}")
    print(f"Magnus stays #1 throughout 2025: {results['top_2025_all_months']:.1%}")
    print(
        f"Magnus drops below 2800 before 2026: {results['below_2800_before_2026']:.1%}"
    )
    print(f"Someone breaks Magnus's peak by 2026: {results['peak_broken_by_2026']:.1%}")

    # Plot sample paths for Magnus
    plot = predictor.plot_sample_paths("Carlsen, Magnus")
    plot.show()
