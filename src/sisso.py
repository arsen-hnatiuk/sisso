import numpy as np
import logging
from itertools import combinations
import random

logging.basicConfig(
    level=logging.DEBUG,
)


class SISSO:
    def __init__(self, K: np.ndarray, target: np.ndarray) -> None:
        self.K_norms = np.linalg.norm(K, axis=0)
        self.K_transpose = np.array(
            [row / norm for row, norm in zip(np.transpose(K), self.K_norms)]
        )
        self.K = np.transpose(self.K_transpose)
        self.target = target

    def fit(
        self, tol: float = 0.01, max_iterations: float = 4, sis_subspace_size=25
    ) -> list:
        output = []
        active_set = np.array([], dtype=int)
        iterate = np.zeros(len(self.target))
        residual = iterate - self.target
        error = np.linalg.norm(residual)
        n = 1
        while error > tol and n <= max_iterations:
            # SIS
            correlations = np.absolute(np.matmul(self.K_transpose, residual))
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
                try:
                    least_squares, res, rank, s = np.linalg.lstsq(
                        submatrix, self.target, rcond=None
                    )
                    local_error = np.sqrt(
                        np.mean(
                            np.square(np.matmul(submatrix, least_squares) - self.target)
                        )
                    )  # RMSE
                except np.linalg.LinAlgError:
                    local_error = min_error
                if local_error < min_error:
                    min_error = local_error
                    optimal_combination = combination
                    optimal_coefficients = least_squares
            iterate = np.matmul(
                self.K[:, np.array(optimal_combination)], optimal_coefficients
            )
            residual = iterate - self.target
            error = min_error

            n_solution = np.zeros(self.K.shape[1])
            for i, position in enumerate(optimal_combination):
                n_solution[position] = optimal_coefficients[i] / self.K_norms[position]
            output.append(n_solution)

            logging.info(f"Iteration: {n}")
            logging.info(f"Error: {error}")
            logging.info(f"Number of combinations: {combinatorial_counter}")
            logging.info(f"Optimal combination: {optimal_combination}")
            logging.info(f"Optimal coefficients: {optimal_coefficients}")
            logging.info("------------------")
            n += 1

        return output


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
