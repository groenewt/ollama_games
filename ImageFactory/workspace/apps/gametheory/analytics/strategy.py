"""Strategy detection and classification for educational insights."""

from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import polars as pl


class StrategyType(str, Enum):
    """Known game theory strategies that can be detected."""

    ALWAYS_COOPERATE = "always_cooperate"
    ALWAYS_DEFECT = "always_defect"
    TIT_FOR_TAT = "tit_for_tat"
    GRIM_TRIGGER = "grim_trigger"
    RANDOM = "random"
    PAVLOV = "pavlov"  # Win-stay, lose-shift
    UNKNOWN = "unknown"


STRATEGY_DESCRIPTIONS = {
    StrategyType.ALWAYS_COOPERATE: "Always chooses the cooperative action regardless of opponent behavior.",
    StrategyType.ALWAYS_DEFECT: "Always chooses the defective/selfish action regardless of opponent behavior.",
    StrategyType.TIT_FOR_TAT: "Starts with cooperation, then mirrors the opponent's previous move.",
    StrategyType.GRIM_TRIGGER: "Cooperates until the opponent defects once, then defects forever.",
    StrategyType.RANDOM: "Appears to choose randomly with no clear pattern (near 50/50 split).",
    StrategyType.PAVLOV: "Win-stay, lose-shift: Repeats action if it led to a good outcome, switches otherwise.",
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

        # Check random
        random_score = self._score_random(player_cd)
        if random_score > 0.75:
            c_count = sum(1 for a in player_cd if a == "C")
            return (
                StrategyType.RANDOM,
                random_score,
                f"Action split is {c_count}C/{len(player_cd)-c_count}D, suggesting random choice.",
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

    def analyze_session(self, results_df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze strategies for all players in a session.

        Args:
            results_df: DataFrame with session results.

        Returns:
            Dictionary with strategy analysis per player.
        """
        analysis = {}

        # Detect number of players
        num_players = 0
        while f"player{num_players + 1}_action" in results_df.columns:
            num_players += 1

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
