"""Coordination Game definition."""

from .matrix_factory import create_matrix_game

GAME = create_matrix_game(
    id="coordination_game",
    name="Coordination Game",
    description="""Players benefit equally from selecting the same strategy, regardless of which one.

Strategic considerations:
- Both (A,A) and (B,B) are Nash equilibria with equal payoffs (2,2)
- Miscoordination (A,B) or (B,A) yields zero for both players
- Neither action is dominant - the goal is to coordinate
- Without communication, players must use focal points or patterns
- In repeated play, establishing a convention becomes crucial
- If opponent showed a preference before, matching it is wise""",
    actions=["A", "B"],
    matrix={
        ("A", "A"): (2, 2),
        ("A", "B"): (0, 0),
        ("B", "A"): (0, 0),
        ("B", "B"): (2, 2),
    },
)
