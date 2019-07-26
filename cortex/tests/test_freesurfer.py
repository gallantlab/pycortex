import numpy as np

from cortex.freesurfer import _remove_disconnected_polys


def test_remove_disconnected_polys_examples():
    polys = np.array([[0, 1, 2],
                      [0, 1, 3],
                      [1, 2, 4],
                      [5, 6, 7]])
    expected_result = np.array([[0, 1, 2],
                                [0, 1, 3],
                                [1, 2, 4]])
    result = _remove_disconnected_polys(polys)
    np.testing.assert_array_equal(result, expected_result)


def test_remove_disconnected_polys_idempotence():
    rng = np.random.RandomState(0)
    for n_polys in [10, 20, 30, 40]:
        polys_0 =rng.randint(0, 100, size=3 * n_polys).reshape(-1, 3)
        
        # make sure this example filters something
        polys_1 = _remove_disconnected_polys(polys_0)
        assert len(polys_0) != len(polys_1)
        
        # make sure calling the function does not change anything
        polys_2 = _remove_disconnected_polys(polys_1)
        np.testing.assert_array_equal(polys_1, polys_2)
