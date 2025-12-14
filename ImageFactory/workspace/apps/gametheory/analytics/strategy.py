"""Strategy detection and classification for educational insights."""

from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import polars as pl

from ..core.utils import detect_num_players


class StrategyType(str, Enum):
    """Known game theory strategies that can be detected."""

    ALWAYS_COOPERATE = "always_cooperate"
    ALWAYS_DEFECT = "always_defect"
    TIT_FOR_TAT = "tit_for_tat"
    GRIM_TRIGGER = "grim_trigger"
    RANDOM = "random"
    PAVLOV = "pavlov"  # Win-stay, lose-shift
    # New strategy types
    MIXED = "mixed"  # Probabilistic mixing with stable ratio
    THRESHOLD = "threshold"  # Conditional on payoff threshold
    BEST_RESPONSE = "best_response"  # Myopic best response to opponent
    EQUILIBRIUM_SEEKING = "equilibrium_seeking"  # Converging to Nash
    FORGIVING_TFT = "forgiving_tft"  # TFT with forgiveness
    TWO_TITS_FOR_TAT = "two_tits_for_tat"  # Responds only to 2 defects
    UNKNOWN = "unknown"


STRATEGY_DESCRIPTIONS = {
    StrategyType.ALWAYS_COOPERATE: "Always chooses the cooperative action regardless of opponent behavior.",
    StrategyType.ALWAYS_DEFECT: "Always chooses the defective/selfish action regardless of opponent behavior.",
    StrategyType.TIT_FOR_TAT: "Starts with cooperation, then mirrors the opponent's previous move.",
    StrategyType.GRIM_TRIGGER: "Cooperates until the opponent defects once, then defects forever.",
    StrategyType.RANDOM: "Appears to choose randomly with no clear pattern (near 50/50 split).",
    StrategyType.PAVLOV: "Win-stay, lose-shift: Repeats action if it led to a good outcome, switches otherwise.",
    StrategyType.MIXED: "Probabilistic strategy with stable cooperation/defection ratio over time.",
    StrategyType.THRESHOLD: "Cooperates when cumulative payoff is above threshold, defects otherwise.",
    StrategyType.BEST_RESPONSE: "Myopic best response: plays the action that would have won against opponent's last move.",
    StrategyType.EQUILIBRIUM_SEEKING: "Gradually converges toward Nash equilibrium play over time.",
    StrategyType.FORGIVING_TFT: "Like TFT but occasionally cooperates after opponent defects (forgiveness).",
    StrategyType.TWO_TITS_FOR_TAT: "Only defects after opponent defects twice in a row.",
    StrategyType.UNKNOWN: "No clear strategy pattern detected.",
}


class StrategyDetector:
    """Detects strategy patterns from game history for educational analysis."""

    COOPERATIVE_ACTIONS = ["cooperate", "contribute", "trust", "share", "stag"]
    DEFECTIVE_ACTIONS = ["defect", "free_ride", "betray", "keep", "hare"]

    def __init__(self, min_rounds: int = 5):
        """Initialize the strategy detector.

        Args:
            min_rounds: Minimum number of rounds needed for reliable detection.
        """
        self.min_rounds = min_rounds

    def _classify_action(self, action: str) -> str:
        """Classify an action as cooperative (C) or defective (D).

        Args:
            action: The action string.

        Returns:
            "C" for cooperative, "D" for defective.
        """
        if action.lower() in [a.lower() for a in self.COOPERATIVE_ACTIONS]:
            return "C"
        return "D"

    def detect_strategy(
        self,
        player_actions: List[str],
        opponent_actions: List[str],
        player_payoffs: Optional[List[int]] = None,
    ) -> Tuple[StrategyType, float, str]:
        """Detect the strategy used by a player with confidence score.

        Args:
            player_actions: List of actions taken by the player.
            opponent_actions: List of actions taken by the opponent.
            player_payoffs: Optional list of payoffs received by the player.

        Returns:
            Tuple of (strategy_type, confidence_score, explanation).
        """
        if len(player_actions) < self.min_rounds:
            return (
                StrategyType.UNKNOWN,
                0.0,
                f"Need at least {self.min_rounds} rounds for reliable detection (have {len(player_actions)}).",
            )

        player_cd = [self._classify_action(a) for a in player_actions]
        opponent_cd = [self._classify_action(a) for a in opponent_actions]

        # Check pure strategies first (highest confidence)
        if all(a == "C" for a in player_cd):
            return (
                StrategyType.ALWAYS_COOPERATE,
                1.0,
                f"Player chose cooperative action in all {len(player_cd)} rounds.",
            )

        if all(a == "D" for a in player_cd):
            return (
                StrategyType.ALWAYS_DEFECT,
                1.0,
                f"Player chose defective action in all {len(player_cd)} rounds.",
            )

        # Check Tit-for-Tat (starts C, then mirrors opponent)
        tft_score = self._score_tit_for_tat(player_cd, opponent_cd)
        if tft_score > 0.85:
            matches = sum(1 for i in range(1, len(player_cd)) if player_cd[i] == opponent_cd[i - 1])
            return (
                StrategyType.TIT_FOR_TAT,
                tft_score,
                f"Started with cooperation and mirrored opponent {matches}/{len(player_cd)-1} times.",
            )

        # Check Grim Trigger
        grim_score, grim_details = self._score_grim_trigger(player_cd, opponent_cd)
        if grim_score > 0.85:
            return (
                StrategyType.GRIM_TRIGGER,
                grim_score,
                grim_details,
            )

        # Check Pavlov (if payoffs available)
        if player_payoffs:
            pavlov_score = self._score_pavlov(player_cd, player_payoffs)
            if pavlov_score > 0.8:
                return (
                    StrategyType.PAVLOV,
                    pavlov_score,
                    "Player appears to use win-stay, lose-shift strategy.",
                )

        # Check Two-Tits-for-Tat
        ttft_score, ttft_explanation = self._score_two_tits_for_tat(player_cd, opponent_cd)
        if ttft_score > 0.8:
            return (
                StrategyType.TWO_TITS_FOR_TAT,
                ttft_score,
                ttft_explanation,
            )

        # Check Forgiving TFT
        ftft_score, ftft_explanation = self._score_forgiving_tft(player_cd, opponent_cd)
        if ftft_score > 0.75:
            return (
                StrategyType.FORGIVING_TFT,
                ftft_score,
                ftft_explanation,
            )

        # Check Equilibrium Seeking
        eq_score, eq_explanation = self._score_equilibrium_seeking(player_cd)
        if eq_score > 0.7:
            return (
                StrategyType.EQUILIBRIUM_SEEKING,
                eq_score,
                eq_explanation,
            )

        # Check Threshold strategy (if payoffs available)
        if player_payoffs:
            thresh_score, thresh_explanation = self._score_threshold(player_cd, player_payoffs)
            if thresh_score > 0.8:
                return (
                    StrategyType.THRESHOLD,
                    thresh_score,
                    f"Threshold strategy detected. {thresh_explanation}",
                )

        # Check Mixed strategy
        mixed_score, mix_ratio = self._score_mixed(player_cd)
        if mixed_score > 0.75:
            return (
                StrategyType.MIXED,
                mixed_score,
                f"Stable mixed strategy with ~{mix_ratio*100:.0f}% cooperation rate.",
            )

        # Check random
        random_score = self._score_random(player_cd)
        if random_score > 0.75:
            c_count = sum(1 for a in player_cd if a == "C")
            return (
                StrategyType.RANDOM,
                random_score,
                f"Action split is {c_count}C/{len(player_cd)-c_count}D, suggesting random choice.",
            )

        # Check Best Response (less strict, fallback)
        br_score = self._score_best_response(player_cd, opponent_cd)
        if br_score > 0.85:
            return (
                StrategyType.BEST_RESPONSE,
                br_score,
                "Player appears to play myopic best response.",
            )

        # No clear pattern
        c_count = sum(1 for a in player_cd if a == "C")
        return (
            StrategyType.UNKNOWN,
            0.0,
            f"No clear strategy detected. Cooperation rate: {c_count}/{len(player_cd)} ({100*c_count/len(player_cd):.0f}%).",
        )

    def _score_tit_for_tat(self, player: List[str], opponent: List[str]) -> float:
        """Score how well player actions match Tit-for-Tat strategy.

        Args:
            player: List of player's C/D classifications.
            opponent: List of opponent's C/D classifications.

        Returns:
            Score from 0.0 to 1.0.
        """
        # TFT starts with cooperation
        if player[0] != "C":
            return 0.0

        if len(player) < 2:
            return 0.5  # Can't tell with just one round

        # Check how many times player mirrored opponent's previous move
        matches = sum(1 for i in range(1, len(player)) if player[i] == opponent[i - 1])
        return matches / (len(player) - 1)

    def _score_grim_trigger(self, player: List[str], opponent: List[str]) -> Tuple[float, str]:
        """Score how well player actions match Grim Trigger strategy.

        Args:
            player: List of player's C/D classifications.
            opponent: List of opponent's C/D classifications.

        Returns:
            Tuple of (score, explanation).
        """
        # Find first opponent defection
        first_defect_idx = next((i for i, a in enumerate(opponent) if a == "D"), None)

        if first_defect_idx is None:
            # Opponent never defected - check if player always cooperated
            if all(a == "C" for a in player):
                return (0.9, "Opponent never defected; player cooperated throughout (consistent with Grim Trigger).")
            return (0.3, "Opponent never defected, but player didn't always cooperate.")

        # Check pre-defection phase
        pre_defect_ok = all(a == "C" for a in player[:first_defect_idx])

        # Check post-defection phase (player should defect from next round onward)
        if first_defect_idx + 1 < len(player):
            post_defect = player[first_defect_idx + 1 :]
            post_defect_ok = all(a == "D" for a in post_defect) if post_defect else True
        else:
            post_defect_ok = True  # No rounds after defection

        if pre_defect_ok and post_defect_ok:
            return (1.0, f"Cooperated until round {first_defect_idx+1} when opponent defected, then defected forever.")

        return (0.3, "Pattern doesn't match Grim Trigger.")

    def _score_pavlov(self, player: List[str], payoffs: List[int]) -> float:
        """Score how well player actions match Pavlov (win-stay, lose-shift).

        Args:
            player: List of player's C/D classifications.
            payoffs: List of payoffs received.

        Returns:
            Score from 0.0 to 1.0.
        """
        if len(player) < 2:
            return 0.5

        # Define "good" payoff as above median
        median_payoff = sorted(payoffs)[len(payoffs) // 2]

        matches = 0
        total = len(player) - 1

        for i in range(1, len(player)):
            prev_action = player[i - 1]
            curr_action = player[i]
            prev_payoff = payoffs[i - 1]

            # Good outcome: stay with same action
            # Bad outcome: switch action
            if prev_payoff >= median_payoff:
                expected = prev_action  # Stay
            else:
                expected = "D" if prev_action == "C" else "C"  # Switch

            if curr_action == expected:
                matches += 1

        return matches / total if total > 0 else 0.5

    def _score_random(self, player: List[str]) -> float:
        """Score how random the player's actions appear.

        Args:
            player: List of player's C/D classifications.

        Returns:
            Score from 0.0 to 1.0 (higher = more random).
        """
        if not player:
            return 0.5

        c_ratio = sum(1 for a in player if a == "C") / len(player)

        # Random would be close to 0.5
        # Score is higher when c_ratio is closer to 0.5
        return 1.0 - abs(0.5 - c_ratio) * 2

    def _score_mixed(self, player: List[str], window_size: int = 5) -> Tuple[float, float]:
        """Score how well player matches a stable mixed strategy.

        A mixed strategy maintains a consistent cooperation ratio over time.

        Args:
            player: List of player's C/D classifications.
            window_size: Window size for measuring stability.

        Returns:
            Tuple of (score, estimated_mix_ratio).
        """
        if len(player) < window_size * 2:
            return 0.0, 0.5

        # Calculate cooperation ratio in sliding windows
        window_ratios = []
        for i in range(0, len(player) - window_size + 1, window_size // 2):
            window = player[i : i + window_size]
            ratio = sum(1 for a in window if a == "C") / len(window)
            window_ratios.append(ratio)

        if len(window_ratios) < 2:
            return 0.0, 0.5

        # Mixed strategy should have low variance and ratio not at extremes
        import numpy as np

        std = np.std(window_ratios)
        mean_ratio = np.mean(window_ratios)

        # Score high if: low variance AND not pure strategy (0.15 < ratio < 0.85)
        variance_score = max(0, 1.0 - std * 4)  # Penalize high variance
        not_pure = 1.0 if 0.15 < mean_ratio < 0.85 else 0.0

        score = variance_score * not_pure
        return score, float(mean_ratio)

    def _score_threshold(
        self, player: List[str], payoffs: List[int]
    ) -> Tuple[float, str]:
        """Score how well player matches threshold strategy.

        Threshold strategy: cooperate when cumulative payoff is above threshold.

        Args:
            player: List of player's C/D classifications.
            payoffs: List of payoffs received.

        Returns:
            Tuple of (score, explanation).
        """
        if len(player) < 5 or len(payoffs) < 5:
            return 0.0, "Insufficient data"

        # Calculate cumulative payoff at each round
        cumulative = []
        total = 0
        for p in payoffs:
            total += p
            cumulative.append(total)

        # Try different thresholds to find best fit
        import numpy as np

        avg_payoff = np.mean(payoffs)
        best_score = 0.0
        best_threshold = 0

        # Test thresholds at various points
        for threshold_mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
            threshold = avg_payoff * threshold_mult * len(payoffs) / 2

            matches = 0
            for i in range(1, len(player)):
                # Predict: cooperate if above threshold, defect otherwise
                expected = "C" if cumulative[i - 1] >= threshold else "D"
                if player[i] == expected:
                    matches += 1

            score = matches / (len(player) - 1)
            if score > best_score:
                best_score = score
                best_threshold = threshold

        explanation = f"Best fit threshold: {best_threshold:.1f}"
        return best_score, explanation

    def _score_best_response(
        self,
        player: List[str],
        opponent: List[str],
        payoff_matrix: Optional[Dict] = None,
    ) -> float:
        """Score how well player plays myopic best response.

        Best response: play action that would beat opponent's last action.

        Args:
            player: List of player's C/D classifications.
            opponent: List of opponent's C/D classifications.
            payoff_matrix: Optional payoff matrix for the game.

        Returns:
            Score from 0.0 to 1.0.
        """
        if len(player) < 3:
            return 0.0

        # Default assumption for PD-like games: D beats C
        # In PD: if opponent played C, best response is D; if D, depends on game
        # Simplify: assume best response to C is D, best response to D is D
        matches = 0
        for i in range(1, len(player)):
            opponent_last = opponent[i - 1]
            # In most games, D is best response (dominant if it exists)
            expected_best = "D"
            if player[i] == expected_best:
                matches += 1

        return matches / (len(player) - 1)

    def _score_equilibrium_seeking(
        self,
        player: List[str],
        equilibrium_action: str = "D",
        window_size: int = 5,
    ) -> Tuple[float, str]:
        """Score if player is converging toward Nash equilibrium.

        Looks for increasing rate of equilibrium action over time.

        Args:
            player: List of player's C/D classifications.
            equilibrium_action: The Nash equilibrium action.
            window_size: Window size for trend analysis.

        Returns:
            Tuple of (score, explanation).
        """
        if len(player) < window_size * 2:
            return 0.0, "Insufficient data"

        # Calculate equilibrium action rate in windows
        window_rates = []
        for i in range(0, len(player) - window_size + 1, window_size // 2):
            window = player[i : i + window_size]
            rate = sum(1 for a in window if a == equilibrium_action) / len(window)
            window_rates.append(rate)

        if len(window_rates) < 2:
            return 0.0, "Insufficient windows"

        import numpy as np

        # Check if trend is increasing toward equilibrium
        x = np.arange(len(window_rates))
        slope = np.polyfit(x, window_rates, 1)[0]

        # Score based on positive trend and final rate
        final_rate = window_rates[-1]

        if slope > 0.02 and final_rate > 0.7:
            score = min(1.0, slope * 10 + final_rate * 0.5)
            explanation = f"Converging to {equilibrium_action}: slope={slope:.3f}, final_rate={final_rate:.2f}"
        else:
            score = 0.0
            explanation = "No convergence detected"

        return score, explanation

    def _score_forgiving_tft(self, player: List[str], opponent: List[str]) -> Tuple[float, str]:
        """Score how well player matches Forgiving TFT strategy.

        Forgiving TFT: like TFT but cooperates occasionally after opponent defects.

        Args:
            player: List of player's C/D classifications.
            opponent: List of opponent's C/D classifications.

        Returns:
            Tuple of (score, explanation).
        """
        if len(player) < 5:
            return 0.0, "Insufficient data"

        if player[0] != "C":
            return 0.0, "Did not start with cooperation"

        # Track TFT-like behavior with forgiveness
        tft_matches = 0
        forgiveness_count = 0
        total = len(player) - 1

        for i in range(1, len(player)):
            opponent_last = opponent[i - 1]
            player_action = player[i]

            if player_action == opponent_last:
                tft_matches += 1
            elif opponent_last == "D" and player_action == "C":
                # Forgiveness: cooperated after opponent defected
                forgiveness_count += 1
                tft_matches += 1  # Count as partial match

        tft_rate = tft_matches / total
        forgiveness_rate = forgiveness_count / total if total > 0 else 0

        # Forgiving TFT: high TFT rate with some forgiveness
        if tft_rate > 0.7 and 0.05 < forgiveness_rate < 0.4:
            score = tft_rate * 0.7 + (1 - abs(forgiveness_rate - 0.15) * 3) * 0.3
            explanation = f"TFT rate: {tft_rate:.2f}, forgiveness rate: {forgiveness_rate:.2f}"
            return min(1.0, score), explanation

        return 0.0, "Does not match Forgiving TFT pattern"

    def _score_two_tits_for_tat(self, player: List[str], opponent: List[str]) -> Tuple[float, str]:
        """Score how well player matches Two-Tits-for-Tat strategy.

        2TFT: defect only after opponent defects twice consecutively.

        Args:
            player: List of player's C/D classifications.
            opponent: List of opponent's C/D classifications.

        Returns:
            Tuple of (score, explanation).
        """
        if len(player) < 5:
            return 0.0, "Insufficient data"

        if player[0] != "C":
            return 0.0, "Did not start with cooperation"

        # Check 2TFT pattern
        matches = 0
        total = 0

        for i in range(2, len(player)):
            # Look at opponent's last two actions
            opp_prev1 = opponent[i - 1]
            opp_prev2 = opponent[i - 2]
            player_action = player[i]

            # 2TFT: defect only if opponent defected twice in a row
            if opp_prev1 == "D" and opp_prev2 == "D":
                expected = "D"
            else:
                expected = "C"

            if player_action == expected:
                matches += 1
            total += 1

        if total == 0:
            return 0.0, "Insufficient data"

        score = matches / total
        explanation = f"Matched 2TFT pattern {matches}/{total} times"

        if score > 0.8:
            return score, explanation
        return 0.0, "Does not match 2TFT pattern"

    def analyze_session(self, results_df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze strategies for all players in a session.

        Args:
            results_df: DataFrame with session results.

        Returns:
            Dictionary with strategy analysis per player.
        """
        analysis = {}

        # Detect number of players (cached)
        num_players = detect_num_players(tuple(results_df.columns))

        if num_players < 2:
            return {"error": "Need at least 2 players for strategy analysis"}

        # For 2-player games, analyze each player vs the opponent
        if num_players == 2:
            p1_actions = results_df["player1_action"].to_list()
            p2_actions = results_df["player2_action"].to_list()

            p1_payoffs = results_df["player1_payoff"].to_list() if "player1_payoff" in results_df.columns else None
            p2_payoffs = results_df["player2_payoff"].to_list() if "player2_payoff" in results_df.columns else None

            strategy1, conf1, explanation1 = self.detect_strategy(p1_actions, p2_actions, p1_payoffs)
            strategy2, conf2, explanation2 = self.detect_strategy(p2_actions, p1_actions, p2_payoffs)

            analysis["player1"] = {
                "strategy": strategy1.value,
                "confidence": conf1,
                "explanation": explanation1,
                "description": STRATEGY_DESCRIPTIONS[strategy1],
            }
            analysis["player2"] = {
                "strategy": strategy2.value,
                "confidence": conf2,
                "explanation": explanation2,
                "description": STRATEGY_DESCRIPTIONS[strategy2],
            }

        # For N-player games, analyze each player individually (cooperation patterns)
        else:
            for p in range(1, num_players + 1):
                action_col = f"player{p}_action"
                payoff_col = f"player{p}_payoff"

                if action_col not in results_df.columns:
                    continue

                actions = results_df[action_col].to_list()
                payoffs = results_df[payoff_col].to_list() if payoff_col in results_df.columns else None

                # For N-player, use empty opponent list (pure strategy detection)
                cd_actions = [self._classify_action(a) for a in actions]
                c_count = sum(1 for a in cd_actions if a == "C")
                total = len(cd_actions)

                # Simple classification for N-player
                if total >= self.min_rounds:
                    coop_rate = c_count / total
                    if coop_rate >= 0.95:
                        strategy = StrategyType.ALWAYS_COOPERATE
                        conf = 1.0
                    elif coop_rate <= 0.05:
                        strategy = StrategyType.ALWAYS_DEFECT
                        conf = 1.0
                    elif 0.4 <= coop_rate <= 0.6:
                        strategy = StrategyType.RANDOM
                        conf = 0.7
                    else:
                        strategy = StrategyType.UNKNOWN
                        conf = 0.0
                else:
                    strategy = StrategyType.UNKNOWN
                    conf = 0.0

                analysis[f"player{p}"] = {
                    "strategy": strategy.value,
                    "confidence": conf,
                    "cooperation_rate": c_count / total if total > 0 else 0,
                    "explanation": f"Cooperation rate: {c_count}/{total} ({100*c_count/total:.0f}%)" if total > 0 else "No actions",
                    "description": STRATEGY_DESCRIPTIONS[strategy],
                }

        return analysis
