import heapq
import math
import numpy as np


def a_star(
    start: tuple[int, int],
    goal: tuple[int, int],
    occupancy_grid: np.ndarray,
) -> list[tuple[int, int]] | None:
    """Find the shortest path from start to goal on a 2D occupancy grid.

    Uses A* with Euclidean heuristic and 8-directional movement.

    Args:
        start: (row, col) of the starting cell.
        goal:  (row, col) of the goal cell.
        occupancy_grid: 2D numpy array where 1 = obstacle, 0 = free.

    Returns:
        Ordered list of (row, col) tuples from start to goal (inclusive),
        or None if no path exists.
    """
    num_rows, num_cols = occupancy_grid.shape

    def in_bounds(row: int, col: int) -> bool:
        return 0 <= row < num_rows and 0 <= col < num_cols

    def is_free(row: int, col: int) -> bool:
        return occupancy_grid[col, row] != 1

    def heuristic(row: int, col: int) -> float:
        goal_row, goal_col = goal
        return math.sqrt((row - goal_row) ** 2 + (col - goal_col) ** 2)

    # 8-directional neighbours: (row_delta, col_delta, step_cost)
    NEIGHBOURS = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    # Validate start and goal
    if not in_bounds(*start) or not is_free(*start):
        return None
    if not in_bounds(*goal) or not is_free(*goal):
        return None

    # cost_from_start[node] = best known cost from start to node
    cost_from_start: dict[tuple[int, int], float] = {start: 0.0}

    # came_from[node] = predecessor on the best known path
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    # open_set entries: (f_score, (row, col))
    open_set: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_set, (heuristic(*start), start))

    # expanded_nodes: nodes already settled (closed set)
    expanded_nodes: set[tuple[int, int]] = set()

    while open_set:
        _, current_node = heapq.heappop(open_set)

        if current_node in expanded_nodes:
            continue
        expanded_nodes.add(current_node)

        if current_node == goal:
            return _reconstruct_path(came_from, current_node)

        current_row, current_col = current_node
        for row_delta, col_delta, step_cost in NEIGHBOURS:
            neighbour_row = current_row + row_delta
            neighbour_col = current_col + col_delta
            neighbour_node = (neighbour_row, neighbour_col)

            if not in_bounds(neighbour_row, neighbour_col) or not is_free(
                neighbour_row, neighbour_col
            ):
                continue
            if neighbour_node in expanded_nodes:
                continue

            tentative_cost = cost_from_start[current_node] + step_cost
            if tentative_cost < cost_from_start.get(neighbour_node, math.inf):
                cost_from_start[neighbour_node] = tentative_cost
                came_from[neighbour_node] = current_node
                f_score = tentative_cost + heuristic(neighbour_row, neighbour_col)
                heapq.heappush(open_set, (f_score, neighbour_node))

    return None  # no path found


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    goal_node: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [goal_node]
    current_node = goal_node
    while current_node in came_from:
        current_node = came_from[current_node]
        path.append(current_node)
    path.reverse()
    return path


def find_retreat_turning_points(
    path: list[tuple[int, int]],
    goal: tuple[int, int],
) -> np.ndarray | None:
    """Find all points on the path that are strict local maxima of distance to goal.

    A turning point is a point where the path peaks in distance before turning
    back toward the goal. Plateaus are not counted.

    Args:
        path: Ordered list of (row, col) tuples as returned by a_star.
        goal: (row, col) of the goal cell.

    Returns:
        np.ndarray of shape (K, 2) containing the turning points in path order,
        or None if no turning points exist.
    """
    if path is None or len(path) < 3:
        return None

    goal_row, goal_col = goal
    distances = [
        math.sqrt((row - goal_row) ** 2 + (col - goal_col) ** 2) for row, col in path
    ]

    turning_points = [
        path[i]
        for i in range(1, len(path) - 1)
        if distances[i - 1] < distances[i] > distances[i + 1]
    ]

    if not turning_points:
        return None

    return np.array(turning_points)
