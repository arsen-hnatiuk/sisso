import numpy as np
import logging
from itertools import combinations
import random
import time

logging.basicConfig(
    level=logging.DEBUG,
)


class SISSO:
    def __init__(self, K: np.ndarray, target: np.ndarray) -> None:
        self.K_norms = np.linalg.norm(K, axis=0)
        self.K_norms[self.K_norms == 0] = 1  # Avoid division by zero
        self.K_transpose = np.array(
            [row / norm for row, norm in zip(np.transpose(K), self.K_norms)]
        )
        self.K = np.transpose(self.K_transpose)
        self.target = target

    def fit(
        self,
        max_iterations: float = 3,
        sis_subspace_size: int = 25,
        nbr_residuals: int = 10,
    ) -> list:
        output = []
        active_set = np.array([], dtype=int)
        iterate = np.zeros(len(self.target))
        residuals = np.array([iterate - self.target])
        errors = np.array([np.linalg.norm(residual) for residual in residuals])
        n = 1
        while n <= max_iterations:
            start_time = time.time()
            # SIS
            try:
                correlations = np.maximum(
                    *[
                        np.absolute(np.matmul(self.K_transpose, residual))
                        for residual in residuals
                    ]
                )
            except TypeError:
                correlations = np.absolute(np.matmul(self.K_transpose, residuals[0]))
            sorted_correlations = np.flip(np.argsort(correlations))
            active_set = np.concatenate(
                (active_set, sorted_correlations[:sis_subspace_size])
            )
            active_set = np.unique(active_set)

            # SO
            combinatorial_combinations = combinations(active_set, n)
            min_errors = np.array([10e5] * nbr_residuals)
            optimal_combination = None
            optimal_coefficients = None
            best_residuals = np.array([np.zeros(len(self.target))] * nbr_residuals)
            combinatorial_counter = 0
            for combination in combinatorial_combinations:
                combinatorial_counter += 1
                submatrix = np.append(
                    self.K[:, np.array(combination)],
                    np.ones((self.K.shape[0], 1)) / np.sqrt(self.K.shape[0]),
                    axis=1,
                )
                try:
                    least_squares, res, rank, s = np.linalg.lstsq(
                        submatrix, self.target, rcond=None
                    )
                    residual = np.matmul(submatrix, least_squares) - self.target
                    local_error = np.sqrt(np.mean(np.square(residual)))  # RMSE
                except np.linalg.LinAlgError:
                    continue
                if local_error < min_errors[-1]:
                    if local_error < min_errors[0]:
                        optimal_combination = combination
                        optimal_coefficients = least_squares
                    min_errors[-1] = local_error
                    best_residuals[-1] = residual
                    sorted_indices = np.argsort(min_errors)
                    min_errors = min_errors[sorted_indices]
                    best_residuals = best_residuals[sorted_indices]
            errors = min_errors[:combinatorial_counter]
            residuals = best_residuals[:combinatorial_counter]

            n_solution = np.zeros(self.K.shape[1] + 1)
            for i, position in enumerate(optimal_combination):
                # Denormalize
                n_solution[position] = optimal_coefficients[i] / self.K_norms[position]
            n_solution[-1] = optimal_coefficients[-1] / np.sqrt(
                self.K.shape[0]
            )  # The constant term
            output.append(n_solution)
            iteration_time = time.time() - start_time

            logging.info(f"Iteration: {n}")
            logging.info(f"Error: {errors[0]}")
            logging.info(f"Number of combinations: {combinatorial_counter}")
            logging.info(f"Optimal combination: {optimal_combination}")
            logging.info(f"Optimal coefficients: {optimal_coefficients}")
            logging.info(f"Time of iteration: {iteration_time}")
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
