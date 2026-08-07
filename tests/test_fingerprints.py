from mavis.fingerprints import hamming


def test_hamming_distance() -> None:
    assert hamming(0b1010, 0b0011) == 2
