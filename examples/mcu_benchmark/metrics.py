"""Shared calibration metrics for benchmarks."""
import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE).

    ECE measures how well predicted probabilities match empirical frequencies.
    Lower is better. Well-calibrated models have ECE close to 0.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration curve.

    Returns:
        Expected Calibration Error (0 to 1, lower is better).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim > 1:
        # Multi-class: use max probability
        y_pred = np.argmax(y_prob, axis=1)
        y_prob = np.max(y_prob, axis=1)
    else:
        y_pred = (y_prob >= 0.5).astype(int)

    # Bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        bin_size = np.sum(in_bin)
        if bin_size == 0:
            continue

        # Accuracy in bin
        bin_accuracy = np.mean(y_true[in_bin] == y_pred[in_bin])

        # Average confidence in bin
        bin_confidence = np.mean(y_prob[in_bin])

        # Weight by bin size
        bin_weight = bin_size / len(y_true)

        ece += np.abs(bin_accuracy - bin_confidence) * bin_weight

    return float(ece)
