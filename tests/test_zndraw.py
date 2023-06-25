import numpy as np

from moldiff.zndraw import calculate_path_length, interpolate_points


def test_calculate_path_length():
    f = lambda x: x**2
    xs = np.array([-1, 0, 1])
    ys = f(xs)
    points = np.stack([xs, ys, np.zeros_like(xs)], axis=1)
    length = calculate_path_length(points)
    expected_length = 2 * np.sqrt(2)
    np.testing.assert_almost_equal(length, expected_length)
    points = interpolate_points(points, num_interpolated_points=500)
    length = calculate_path_length(points)
    expected_length = 0.5 * (np.log(2 + np.sqrt(5)) + 2 * np.sqrt(5))
    np.testing.assert_almost_equal(length, expected_length, decimal=5)
