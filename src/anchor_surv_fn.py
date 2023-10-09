import numba
import numpy as np


def process_times_anchors(anchor_times, anchor_values):
    if anchor_times[0] > 0:
        anchor_times = np.concatenate([np.zeros(1), anchor_times])
        anchor_values = np.concatenate([np.ones(1), anchor_values])
    anchor_times = np.concatenate([anchor_times, [anchor_times[-1] * 10]])
    anchor_values = np.concatenate([anchor_values, np.zeros(1)])
    anchor_values[anchor_values < 0] = 0
    anchor_values[anchor_values > 1] = 1
    return anchor_times, anchor_values


def _call_step_bwd(t, anchor_times, anchor_values):
    idx = np.searchsorted(anchor_times, t, side="right")
    return anchor_values[idx]


def _call_step_fwd(t, anchor_times, anchor_values):
    idx = np.searchsorted(anchor_times, t, side="right")
    idx = np.clip(idx, 0, len(anchor_times) + 1)
    return anchor_values[idx - 1]


def _call_linear(t, anchor_times, anchor_values):
    idx = np.searchsorted(anchor_times, t) - 1
    idx = np.clip(idx, 0, len(anchor_times) - 2)
    ret = anchor_values[idx] + (
        anchor_values[idx + 1] - anchor_values[idx]
    ) / (anchor_times[idx + 1] - anchor_times[idx]) * (
        t - anchor_times[idx]
    )
    return np.clip(ret, 0.0, 1.0)


def _call_piecewise_exponential(t, anchor_times, anchor_values):
    idx = np.searchsorted(anchor_times, t) - 1
    idx = np.clip(idx, 0, len(anchor_times) - 2)

    # Get time difference between two anchor points
    delta_t = anchor_times[idx + 1] - anchor_times[idx]

    # Calculate decay rate based on continuous condition
    if anchor_values[idx + 1] == 0:
        k = 0
    else:
        k = (np.log(anchor_values[idx + 1]) - np.log(anchor_values[idx])) / delta_t

    # Now compute the exponential function value at time t
    ret = anchor_values[idx] * np.exp(k * (t - anchor_times[idx]))

    return np.clip(ret, 0.0, 1.0)


@numba.jit(nopython=True)
def _compute_tangents(anchor_times, anchor_values):
    delta_x = np.diff(anchor_times)
    delta_y = np.diff(anchor_values)
    slopes = delta_y / delta_x

    tangents = np.zeros_like(anchor_values)

    # Initial approximation for tangents
    tangents[:-1] = slopes
    tangents[1:] += slopes
    tangents /= 2

    # Fix tangents to ensure monotonicity
    for i in range(len(slopes) - 1):
        if slopes[i] * slopes[i + 1] <= 0:
            tangents[i + 1] = 0
        elif (slopes[i] < 0 and tangents[i + 1] > -3 * slopes[i]) or (
                slopes[i + 1] < 0 and tangents[i + 1] < -3 * slopes[i + 1]):
            tangents[i + 1] = -3 * min(abs(slopes[i]), abs(slopes[i + 1]))

    # Endpoints: set tangents to slopes of adjacent segment
    tangents[0] = slopes[0]
    tangents[-1] = slopes[-1]

    return tangents


@numba.jit(nopython=True)
def _evaluate_spline(x, x_data, y_data, tangents):
    for i in range(len(x_data) - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            t = (x - x_data[i]) / (x_data[i + 1] - x_data[i])
            return ((2 * t ** 3 - 3 * t ** 2 + 1) * y_data[i] +
                    (t ** 3 - 2 * t ** 2 + t) * (x_data[i + 1] - x_data[i]) * tangents[i] +
                    (-2 * t ** 3 + 3 * t ** 2) * y_data[i + 1] +
                    (t ** 3 - t ** 2) * (x_data[i + 1] - x_data[i]) * tangents[i + 1])
    return 0.0


def interpolate_points(t, anchor_times, anchor_values, mode, **kwargs):
    assert np.isscalar(t), t
    if mode == "step_bwd":
        return _call_step_bwd(t, anchor_times, anchor_values)
    if mode == "step_fwd":
        return _call_step_fwd(t, anchor_times, anchor_values)
    if mode == "linear":
        return _call_linear(t, anchor_times, anchor_values)
    if mode == "pce":
        return _call_piecewise_exponential(t, anchor_times, anchor_values)
    if mode == "spline":
        return _evaluate_spline(t, anchor_times, anchor_values, kwargs["tangents"])
    raise ValueError(f"Unknown mode: {mode}")
