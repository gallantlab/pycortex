import hashlib

import numpy as np

from cortex.dataset.braindata import _hash


def test_hash_uses_tobytes():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = hashlib.sha1(array.tobytes()).hexdigest()
    assert _hash(array) == expected
