"""Equilibrium analysis for game theory education."""

from typing import List, Tuple, Dict, Any, Optional
import polars as pl

from ..core.types import GameDefinition


class EquilibriumAnalyzer:
    """Analyzes game equilibria and compares with LLM behavior."""

    def find_nash_equilibria(self, game: GameDefinition) -> List[Tuple[str, ...]]:
        """Find all pure strategy Nash equilibria for a game.

        Supports N-player games. A strategy profile is a Nash equilibrium if
        no player can improve their payoff by unilaterally changing their action.

        Args:
            game: The game definition.

        Returns:
            List of action tuples that are Nash equilibria.
        """
        equilibria = []
        actions = game.actions
        num_players = game.num_players

        for action_combo, payoffs in game.payoff_matrix.items():
            is_equilibrium = True

            # Check if each player's action is a best response
            for player_idx in range(num_players):
                player_payoff = payoffs[player_idx]
                current_action = action_combo[player_idx]

                # Check all possible deviations for this player
                for alternative_action in actions:
                    if alternative_action == current_action:
                        continue

                    # Build the alternative action combo with this player deviating
                    alt_combo = list(action_combo)
                    alt_combo[player_idx] = alternative_action
                    alt_combo = tuple(alt_combo)

                    # Get payoff for this deviation
                    alt_payoffs = game.payoff_matrix.get(alt_combo)
                    if alt_payoffs and alt_payoffs[player_idx] > player_payoff:
                        is_equilibrium = False
                        break

                if not is_equilibrium:
                    break

            if is_equilibrium:
                equilibria.append(action_combo)

        return equilibria

    def find_pareto_optimal(self, game: GameDefinition) -> List[Tuple[str, ...]]:
        """Find Pareto optimal outcomes for a game.

        An outcome is Pareto optimal if no other outcome makes all players
        better off (or at least as well off with at least one strictly better).

        Supports N-player games.

        Args:
            game: The game definition.

        Returns:
            List of action tuples that are Pareto optimal.
        """
        pareto_optimal = []

        for action_combo, payoffs in game.payoff_matrix.items():
            is_dominated = False

            for other_combo, other_payoffs in game.payoff_matrix.items():
                if other_combo == action_combo:
                    continue

                # Check if other_payoffs Pareto dominates payoffs
                # All players at least as well off, at least one strictly better
                all_at_least_good = all(op >= p for op, p in zip(other_payoffs, payoffs))
                at_least_one_better = any(op > p for op, p in zip(other_payoffs, payoffs))

                if all_at_least_good and at_least_one_better:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_optimal.append(action_combo)

        return pareto_optimal

    def find_dominant_strategies(self, game: GameDefinition) -> Dict[int, Optional[str]]:
        """Find dominant strategies for each player.

        A dominant strategy is one that is best regardless of all other players' choices.

        Supports N-player games.

        Args:
            game: The game definition.

        Returns:
            Dictionary mapping player number (1-indexed) to their dominant strategy (or None).
        """
        from itertools import product as itertools_product

        result = {}
        actions = game.actions
        num_players = game.num_players

        # Check for each player's dominant strategy
        for player_idx in range(num_players):
            player_num = player_idx + 1  # 1-indexed for output

            # Generate all possible opponent action combinations
            other_player_indices = [i for i in range(num_players) if i != player_idx]
            opponent_combos = list(itertools_product(actions, repeat=len(other_player_indices)))

            dominant_action = None

            for candidate in actions:
                is_dominant = True

                # Check if candidate is best response to all possible opponent combinations
                for opponent_combo in opponent_combos:
                    # Build the full action combo with this candidate
                    full_combo = [None] * num_players
                    full_combo[player_idx] = candidate
                    for i, other_idx in enumerate(other_player_indices):
                        full_combo[other_idx] = opponent_combo[i]
                    full_combo = tuple(full_combo)

                    candidate_payoff = game.payoff_matrix.get(full_combo, tuple([0] * num_players))[player_idx]

                    # Compare with all other actions
                    for other_action in actions:
                        if other_action == candidate:
                            continue

                        other_full_combo = list(full_combo)
                        other_full_combo[player_idx] = other_action
                        other_full_combo = tuple(other_full_combo)

                        other_payoff = game.payoff_matrix.get(other_full_combo, tuple([0] * num_players))[player_idx]

                        if other_payoff > candidate_payoff:
                            is_dominant = False
                            break

                    if not is_dominant:
                        break

                if is_dominant:
                    dominant_action = candidate
                    break

            result[player_num] = dominant_action

        return result

    def analyze_game_outcomes(
        self,
        game: GameDefinition,
        results_df: pl.DataFrame,
    ) -> Dict[str, Any]:
        """Compare LLM behavior against theoretical equilibria.

        Supports N-player games.

        Args:
            game: The game definition.
            results_df: DataFrame with game results.

        Returns:
            Dictionary with equilibrium analysis.
        """
        num_players = game.num_players

        analysis = {
            "game_name": game.name,
            "num_players": num_players,
        }

        # Find theoretical equilibria
        nash = self.find_nash_equilibria(game)
        pareto = self.find_pareto_optimal(game)
        dominant = self.find_dominant_strategies(game)

        analysis["nash_equilibria"] = [tuple(e) for e in nash]
        analysis["pareto_optimal"] = [tuple(p) for p in pareto]
        analysis["dominant_strategies"] = dominant

        # Calculate outcome distribution from results
        if results_df.is_empty():
            analysis["outcome_distribution"] = {}
            analysis["nash_play_rate"] = 0.0
            analysis["pareto_play_rate"] = 0.0
            return analysis

        # Build dynamic action column list for N players
        action_cols = [f"player{p+1}_action" for p in range(num_players)]

        # Ensure all columns exist
        available_cols = [c for c in action_cols if c in results_df.columns]
        if not available_cols:
            analysis["outcome_distribution"] = {}
            analysis["nash_play_rate"] = 0.0
            analysis["pareto_play_rate"] = 0.0
            analysis["note"] = "No action columns found in results"
            return analysis

        # Count outcomes
        outcome_counts = (
            results_df.group_by(available_cols)
            .agg(pl.len().alias("count"))
            .to_dicts()
        )
        total_rounds = results_df.height

        outcome_dist = {}
        nash_count = 0
        pareto_count = 0

        for row in outcome_counts:
            # Build outcome tuple from available columns
            outcome = tuple(row[col] for col in available_cols)
            count = row["count"]
            rate = count / total_rounds
            outcome_dist[outcome] = {"count": count, "rate": round(rate, 3)}

            if outcome in nash:
                nash_count += count
            if outcome in pareto:
                pareto_count += count

        analysis["outcome_distribution"] = outcome_dist
        analysis["nash_play_rate"] = round(nash_count / total_rounds, 3) if total_rounds > 0 else 0.0
        analysis["pareto_play_rate"] = round(pareto_count / total_rounds, 3) if total_rounds > 0 else 0.0
        analysis["total_rounds"] = total_rounds

        # Generate educational insights
        analysis["insights"] = self._generate_insights(
            nash, pareto, dominant, analysis["nash_play_rate"], analysis["pareto_play_rate"]
        )

        return analysis

    def _generate_insights(
        self,
        nash: List[Tuple[str, ...]],
        pareto: List[Tuple[str, ...]],
        dominant: Dict[int, Optional[str]],
        nash_rate: float,
        pareto_rate: float,
    ) -> List[str]:
        """Generate educational insights about the equilibrium analysis.

        Args:
            nash: List of Nash equilibria.
            pareto: List of Pareto optimal outcomes.
            dominant: Dominant strategies by player.
            nash_rate: Rate of Nash equilibrium play.
            pareto_rate: Rate of Pareto optimal play.

        Returns:
            List of insight strings.
        """
        insights = []

        # Nash equilibrium insights
        if not nash:
            insights.append("This game has no pure strategy Nash equilibrium (may have mixed strategy equilibrium).")
        elif len(nash) == 1:
            insights.append(f"This game has a unique Nash equilibrium at {nash[0]}.")
        else:
            insights.append(f"This game has {len(nash)} Nash equilibria: {nash}.")

        # Dominant strategy insights
        if dominant.get(1) and dominant.get(2):
            insights.append(
                f"Both players have dominant strategies: P1={dominant[1]}, P2={dominant[2]}. "
                "Rational players should always choose these."
            )
        elif dominant.get(1):
            insights.append(f"Player 1 has a dominant strategy: {dominant[1]}.")
        elif dominant.get(2):
            insights.append(f"Player 2 has a dominant strategy: {dominant[2]}.")

        # Compare Nash vs Pareto (social dilemma indicator)
        nash_set = set(nash)
        pareto_set = set(pareto)

        if nash_set and not nash_set.intersection(pareto_set):
            insights.append(
                "The Nash equilibrium is NOT Pareto optimal - this indicates a social dilemma "
                "where rational self-interest leads to a collectively suboptimal outcome."
            )
        elif nash_set and nash_set.issubset(pareto_set):
            insights.append("The Nash equilibrium is Pareto optimal - rational play leads to an efficient outcome.")

        # LLM behavior insights
        if nash_rate > 0.8:
            insights.append(
                f"LLMs played the Nash equilibrium {nash_rate*100:.0f}% of the time, "
                "showing strong alignment with game-theoretic predictions."
            )
        elif nash_rate > 0.5:
            insights.append(
                f"LLMs played the Nash equilibrium {nash_rate*100:.0f}% of the time, "
                "showing moderate alignment with game-theoretic predictions."
            )
        elif nash_rate > 0 and nash:
            insights.append(
                f"LLMs only played the Nash equilibrium {nash_rate*100:.0f}% of the time, "
                "suggesting factors beyond pure rational self-interest influence their decisions."
            )

        if pareto_rate > nash_rate and pareto_rate > 0.5:
            insights.append(
                f"LLMs achieved Pareto optimal outcomes {pareto_rate*100:.0f}% of the time, "
                "suggesting cooperative tendencies."
            )

        return insights

    def get_game_summary(self, game: GameDefinition) -> Dict[str, Any]:
        """Get a summary of game-theoretic properties for a game.

        Supports N-player games.

        Args:
            game: The game definition.

        Returns:
            Dictionary with game properties.
        """
        nash = self.find_nash_equilibria(game)
        pareto = self.find_pareto_optimal(game)
        dominant = self.find_dominant_strategies(game)

        # Classify game type (works for N-player games)
        game_type = "unknown"
        nash_set = set(nash)
        pareto_set = set(pareto)

        # Check for social dilemma (Nash not Pareto optimal)
        if nash_set and not nash_set.intersection(pareto_set):
            game_type = "social_dilemma"
        elif len(nash) == 1 and nash[0] in pareto:
            game_type = "coordination"
        elif len(nash) > 1:
            game_type = "multiple_equilibria"
        elif not nash:
            game_type = "no_pure_equilibrium"
        else:
            game_type = "standard"

        return {
            "game_name": game.name,
            "game_type": game_type,
            "num_actions": len(game.actions),
            "num_players": game.num_players,
            "nash_equilibria_count": len(nash),
            "nash_equilibria": nash,
            "pareto_optimal_count": len(pareto),
            "pareto_optimal": pareto,
            "has_dominant_strategies": any(v is not None for v in dominant.values()),
            "dominant_strategies": dominant,
            "is_social_dilemma": game_type == "social_dilemma",
        }

    def calculate_equilibrium_distance(
        self,
        game: GameDefinition,
        results_df: pl.DataFrame,
    ) -> Dict[str, Any]:
        """Calculate distance from Nash equilibrium play.

        Measures how far observed play deviates from Nash equilibrium,
        using action frequencies and payoff regret.

        Args:
            game: The game definition.
            results_df: DataFrame with game results.

        Returns:
            Dictionary with distance metrics.
        """
        nash = self.find_nash_equilibria(game)
        num_players = game.num_players

        if not nash or results_df.is_empty():
            return {
                "nash_equilibria": nash,
                "has_nash": bool(nash),
                "distance_metrics": None,
                "note": "No Nash equilibrium or no results",
            }

        total_rounds = results_df.height
        action_cols = [f"player{p+1}_action" for p in range(num_players)]

        # Calculate action frequency distance for each Nash equilibrium
        best_nash = None
        min_distance = float("inf")

        for eq in nash:
            distance = 0.0
            for player_idx, action_col in enumerate(action_cols):
                if action_col not in results_df.columns:
                    continue

                eq_action = eq[player_idx]
                # Frequency of playing equilibrium action
                eq_freq = (
                    results_df.filter(pl.col(action_col) == eq_action).height
                    / total_rounds
                )
                # Distance = 1 - frequency (0 = perfect, 1 = never played)
                distance += 1 - eq_freq

            # Average across players
            distance /= num_players

            if distance < min_distance:
                min_distance = distance
                best_nash = eq

        # Calculate regret (how much better players could have done with Nash)
        regret = self._calculate_regret(game, results_df, nash)

        return {
            "nash_equilibria": nash,
            "closest_nash": best_nash,
            "action_distance": round(min_distance, 4),
            "regret": regret,
            "nash_play_rate": round(1 - min_distance, 4),
            "interpretation": self._interpret_distance(min_distance),
        }

    def _calculate_regret(
        self,
        game: GameDefinition,
        results_df: pl.DataFrame,
        nash: List[Tuple[str, ...]],
    ) -> Dict[str, float]:
        """Calculate regret for each player vs Nash play.

        Args:
            game: The game definition.
            results_df: DataFrame with results.
            nash: List of Nash equilibria.

        Returns:
            Dictionary with regret per player.
        """
        num_players = game.num_players
        regret = {}

        for player_idx in range(num_players):
            player_num = player_idx + 1
            payoff_col = f"player{player_num}_payoff"

            if payoff_col not in results_df.columns:
                continue

            actual_payoff = results_df[payoff_col].mean()

            # Calculate expected Nash payoff
            nash_payoffs = []
            for eq in nash:
                if eq in game.payoff_matrix:
                    nash_payoffs.append(game.payoff_matrix[eq][player_idx])

            if nash_payoffs:
                nash_payoff = sum(nash_payoffs) / len(nash_payoffs)
                regret[f"player{player_num}"] = round(nash_payoff - actual_payoff, 4)
            else:
                regret[f"player{player_num}"] = None

        return regret

    def _interpret_distance(self, distance: float) -> str:
        """Interpret equilibrium distance.

        Args:
            distance: Action distance from Nash.

        Returns:
            Human-readable interpretation.
        """
        if distance < 0.1:
            return "Near-perfect Nash equilibrium play"
        elif distance < 0.25:
            return "Strong alignment with Nash equilibrium"
        elif distance < 0.5:
            return "Moderate deviation from Nash equilibrium"
        elif distance < 0.75:
            return "Substantial deviation from Nash equilibrium"
        else:
            return "Significant departure from Nash equilibrium"

    def detect_equilibrium_convergence(
        self,
        game: GameDefinition,
        results_df: pl.DataFrame,
        window_size: int = 5,
    ) -> Dict[str, Any]:
        """Detect if play is converging toward Nash equilibrium.

        Analyzes whether Nash equilibrium action frequency increases over time.

        Args:
            game: The game definition.
            results_df: DataFrame with game results.
            window_size: Window size for trend analysis.

        Returns:
            Dictionary with convergence analysis.
        """
        nash = self.find_nash_equilibria(game)
        num_players = game.num_players

        if not nash or results_df.is_empty() or results_df.height < window_size * 2:
            return {
                "converging": False,
                "note": "Insufficient data or no Nash equilibrium",
            }

        # Track Nash action rate over time windows
        window_nash_rates = []
        total_rounds = results_df.height

        for start in range(0, total_rounds - window_size + 1, window_size // 2):
            window_df = results_df.slice(start, window_size)

            # Count Nash equilibrium plays in this window
            nash_count = 0
            for eq in nash:
                # Build filter for this equilibrium
                filters = []
                for player_idx in range(num_players):
                    action_col = f"player{player_idx + 1}_action"
                    if action_col in window_df.columns:
                        filters.append(pl.col(action_col) == eq[player_idx])

                if filters:
                    combined = filters[0]
                    for f in filters[1:]:
                        combined = combined & f
                    nash_count += window_df.filter(combined).height

            nash_rate = nash_count / (window_size * len(nash)) if nash else 0
            window_nash_rates.append(nash_rate)

        if len(window_nash_rates) < 2:
            return {"converging": False, "note": "Insufficient windows"}

        # Calculate trend
        import numpy as np

        x = np.arange(len(window_nash_rates))
        slope = np.polyfit(x, window_nash_rates, 1)[0]

        # Convergence: positive slope and increasing final rate
        first_rate = window_nash_rates[0]
        last_rate = window_nash_rates[-1]
        converging = slope > 0.02 and last_rate > first_rate

        return {
            "converging": converging,
            "slope": round(float(slope), 4),
            "first_window_rate": round(first_rate, 4),
            "last_window_rate": round(last_rate, 4),
            "improvement": round(last_rate - first_rate, 4),
            "window_rates": [round(r, 4) for r in window_nash_rates],
            "interpretation": (
                f"Play is {'converging toward' if converging else 'not converging to'} Nash equilibrium"
            ),
        }

    def compare_to_mixed_nash(
        self,
        game: GameDefinition,
        results_df: pl.DataFrame,
    ) -> Dict[str, Any]:
        """Compare observed play to theoretical mixed Nash equilibrium.

        For games without pure Nash equilibrium, compares action frequencies
        to what a mixed strategy equilibrium would predict.

        Args:
            game: The game definition.
            results_df: DataFrame with game results.

        Returns:
            Dictionary with mixed Nash comparison.
        """
        pure_nash = self.find_nash_equilibria(game)

        if pure_nash:
            return {
                "has_pure_nash": True,
                "pure_nash": pure_nash,
                "note": "Game has pure strategy Nash equilibrium",
            }

        if results_df.is_empty():
            return {"has_pure_nash": False, "note": "No results data"}

        # For 2-player 2-action games, calculate theoretical mixed Nash
        num_players = game.num_players
        actions = game.actions

        if num_players != 2 or len(actions) != 2:
            return {
                "has_pure_nash": False,
                "note": "Mixed Nash calculation only supported for 2x2 games",
            }

        # Calculate mixed Nash probabilities for 2x2 game
        # Player 1 mixes to make Player 2 indifferent, and vice versa
        a1, a2 = actions

        # Get payoffs for each outcome
        try:
            p2_payoffs = {
                (a1, a1): game.payoff_matrix[(a1, a1)][1],
                (a1, a2): game.payoff_matrix[(a1, a2)][1],
                (a2, a1): game.payoff_matrix[(a2, a1)][1],
                (a2, a2): game.payoff_matrix[(a2, a2)][1],
            }
            p1_payoffs = {
                (a1, a1): game.payoff_matrix[(a1, a1)][0],
                (a1, a2): game.payoff_matrix[(a1, a2)][0],
                (a2, a1): game.payoff_matrix[(a2, a1)][0],
                (a2, a2): game.payoff_matrix[(a2, a2)][0],
            }
        except KeyError:
            return {"has_pure_nash": False, "note": "Incomplete payoff matrix"}

        # P1 plays a1 with prob p to make P2 indifferent
        # P2's EU(a1) = p * p2_payoffs[(a1,a1)] + (1-p) * p2_payoffs[(a2,a1)]
        # P2's EU(a2) = p * p2_payoffs[(a1,a2)] + (1-p) * p2_payoffs[(a2,a2)]
        # Set equal and solve for p
        denom = (
            p2_payoffs[(a1, a1)]
            - p2_payoffs[(a2, a1)]
            - p2_payoffs[(a1, a2)]
            + p2_payoffs[(a2, a2)]
        )

        if abs(denom) < 0.001:
            return {"has_pure_nash": False, "note": "No interior mixed equilibrium"}

        p1_mix = (p2_payoffs[(a2, a2)] - p2_payoffs[(a2, a1)]) / denom

        # Similarly for P2
        denom2 = (
            p1_payoffs[(a1, a1)]
            - p1_payoffs[(a1, a2)]
            - p1_payoffs[(a2, a1)]
            + p1_payoffs[(a2, a2)]
        )

        if abs(denom2) < 0.001:
            return {"has_pure_nash": False, "note": "No interior mixed equilibrium"}

        p2_mix = (p1_payoffs[(a2, a2)] - p1_payoffs[(a1, a2)]) / denom2

        # Check if mix probabilities are valid (between 0 and 1)
        if not (0 < p1_mix < 1 and 0 < p2_mix < 1):
            return {
                "has_pure_nash": False,
                "note": "Mixed equilibrium probabilities outside [0,1]",
            }

        # Compare to observed frequencies
        total_rounds = results_df.height

        p1_observed = (
            results_df.filter(pl.col("player1_action") == a1).height / total_rounds
            if "player1_action" in results_df.columns
            else None
        )
        p2_observed = (
            results_df.filter(pl.col("player2_action") == a1).height / total_rounds
            if "player2_action" in results_df.columns
            else None
        )

        # Calculate deviation from mixed Nash
        p1_deviation = abs(p1_observed - p1_mix) if p1_observed is not None else None
        p2_deviation = abs(p2_observed - p2_mix) if p2_observed is not None else None

        return {
            "has_pure_nash": False,
            "mixed_nash": {
                "player1": {
                    "action": a1,
                    "probability": round(p1_mix, 4),
                },
                "player2": {
                    "action": a1,
                    "probability": round(p2_mix, 4),
                },
            },
            "observed": {
                "player1": round(p1_observed, 4) if p1_observed is not None else None,
                "player2": round(p2_observed, 4) if p2_observed is not None else None,
            },
            "deviation": {
                "player1": round(p1_deviation, 4) if p1_deviation is not None else None,
                "player2": round(p2_deviation, 4) if p2_deviation is not None else None,
            },
            "interpretation": self._interpret_mixed_nash_deviation(
                p1_deviation, p2_deviation
            ),
        }

    def _interpret_mixed_nash_deviation(
        self, p1_dev: Optional[float], p2_dev: Optional[float]
    ) -> str:
        """Interpret deviation from mixed Nash equilibrium.

        Args:
            p1_dev: Player 1's deviation.
            p2_dev: Player 2's deviation.

        Returns:
            Human-readable interpretation.
        """
        if p1_dev is None or p2_dev is None:
            return "Insufficient data for comparison"

        avg_dev = (p1_dev + p2_dev) / 2

        if avg_dev < 0.1:
            return "Play closely matches mixed Nash equilibrium"
        elif avg_dev < 0.2:
            return "Moderate alignment with mixed Nash equilibrium"
        elif avg_dev < 0.35:
            return "Some deviation from mixed Nash equilibrium"
        else:
            return "Substantial deviation from mixed Nash equilibrium"
