import numpy as np
import logging
from itertools import combinations
import random

logging.basicConfig(
    level=logging.DEBUG,
)


class SISSO:
    def __init__(self, K: np.ndarray, y: np.ndarray) -> None:
        self.K = K
        self.K_transpose = np.transpose(K)
        self.y = y
        self.data_size = len(y)

    def fit(self, tol: float = 0.01, max_iterations: float = 5) -> np.ndarray:
        sis_subspace_size = max(
            int(np.exp(self.data_size / (3.125 * max_iterations)) / max_iterations), 1
        )
        active_set = np.array([], dtype=int)
        iterate = np.zeros(self.data_size)
        residual = iterate - self.y
        error = np.linalg.norm(residual)
        n = 1
        while error > tol and n <= max_iterations:
            # SIS
            correlations = np.matmul(self.K_transpose, residual)
            sorted_correlations = np.flip(np.argsort(correlations))
            active_set = np.concatenate(
                (active_set, sorted_correlations[:sis_subspace_size])
            )
            active_set = np.unique(active_set)

            # SO
            combinatorial_combinations = combinations(active_set, n)
            min_error = 10 * error
            optimal_combination = None
            optimal_coefficients = None
            combinatorial_counter = 0
            for combination in combinatorial_combinations:
                combinatorial_counter += 1
                submatrix = self.K[:, np.array(combination)]
                least_squares, res, rank, s = np.linalg.lstsq(
                    submatrix, self.y, rcond=None
                )
                if len(res):
                    local_error = np.sqrt(res[0])
                else:
                    local_error = np.linalg.norm(
                        np.matmul(submatrix, least_squares) - self.y
                    )
                if local_error < min_error:
                    min_error = local_error
                    optimal_combination = combination
                    optimal_coefficients = least_squares
            iterate = np.matmul(
                self.K[:, np.array(optimal_combination)], optimal_coefficients
            )
            residual = iterate - self.y
            error = min_error

            logging.info("------------------")
            logging.info(f"Iteration: {n}")
            logging.info(f"Error: {error}")
            logging.info(f"Number of combinations: {combinatorial_counter}")
            logging.info(f"Optimal combination: {optimal_combination}")
            logging.info(f"Optimal coefficients: {optimal_coefficients}")
            n += 1


# if __name__ == "__main__":
#     data_size = 100
#     m = 1000000
#     K = np.array(
#         [
#             np.random.normal(random.uniform(-3, 3), random.uniform(0.5, 3), [m])
#             for _ in range(data_size)
#         ]
#     )
#     y = np.random.normal(random.uniform(-3, 3), random.uniform(0.5, 3), [data_size])
#     sisso = SISSO(K, y)
#     sisso.fit()
