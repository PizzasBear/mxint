from collections.abc import MutableSequence
from re import M

import numpy as np

BYTE_BIT_REV_TBL = b"\
\x00\x80\x40\xc0\x20\xa0\x60\xe0\x10\x90\x50\xd0\x30\xb0\x70\xf0\
\x08\x88\x48\xc8\x28\xa8\x68\xe8\x18\x98\x58\xd8\x38\xb8\x78\xf8\
\x04\x84\x44\xc4\x24\xa4\x64\xe4\x14\x94\x54\xd4\x34\xb4\x74\xf4\
\x0c\x8c\x4c\xcc\x2c\xac\x6c\xec\x1c\x9c\x5c\xdc\x3c\xbc\x7c\xfc\
\x02\x82\x42\xc2\x22\xa2\x62\xe2\x12\x92\x52\xd2\x32\xb2\x72\xf2\
\x0a\x8a\x4a\xca\x2a\xaa\x6a\xea\x1a\x9a\x5a\xda\x3a\xba\x7a\xfa\
\x06\x86\x46\xc6\x26\xa6\x66\xe6\x16\x96\x56\xd6\x36\xb6\x76\xf6\
\x0e\x8e\x4e\xce\x2e\xae\x6e\xee\x1e\x9e\x5e\xde\x3e\xbe\x7e\xfe\
\x01\x81\x41\xc1\x21\xa1\x61\xe1\x11\x91\x51\xd1\x31\xb1\x71\xf1\
\x09\x89\x49\xc9\x29\xa9\x69\xe9\x19\x99\x59\xd9\x39\xb9\x79\xf9\
\x05\x85\x45\xc5\x25\xa5\x65\xe5\x15\x95\x55\xd5\x35\xb5\x75\xf5\
\x0d\x8d\x4d\xcd\x2d\xad\x6d\xed\x1d\x9d\x5d\xdd\x3d\xbd\x7d\xfd\
\x03\x83\x43\xc3\x23\xa3\x63\xe3\x13\x93\x53\xd3\x33\xb3\x73\xf3\
\x0b\x8b\x4b\xcb\x2b\xab\x6b\xeb\x1b\x9b\x5b\xdb\x3b\xbb\x7b\xfb\
\x07\x87\x47\xc7\x27\xa7\x67\xe7\x17\x97\x57\xd7\x37\xb7\x77\xf7\
\x0f\x8f\x4f\xcf\x2f\xaf\x6f\xef\x1f\x9f\x5f\xdf\x3f\xbf\x7f\xff\
"


def exgcd(a: int, b: int) -> tuple[int, int, int]:
    ax, ay = 1, 0
    bx, by = 0, 1

    while b != 0:
        q, r = divmod(a, b)

        ax, bx = bx, ax - q * bx
        ay, by = by, ay - q * by
        a, b = b, r

    return ax, ay, a


def fft(a: list[int], root: int, mod: int, first_round: bool = True) -> list[int]:
    if len(a) <= 1:
        assert root == 1
        return a
    assert not len(a) & 1
    assert root != 1

    if first_round:
        x, _y, d = exgcd(root, mod)
        assert d == 1
        root = x % mod

    root2 = (root**2) % mod
    results0 = fft(a[::2], root2, mod, False)
    results1 = fft(a[1::2], root2, mod, False)

    rootp = 1
    results: list[int] = [0] * len(a)
    for i, (r0, r1) in enumerate(zip(results0, results1)):
        results[i] = (r0 + rootp * r1) % mod
        results[i + len(results0)] = (r0 - rootp * r1) % mod
        rootp = (rootp * root) % mod
    assert rootp == mod - 1

    return results


def ifft(a: list[int], root: int, mod: int, first_round: bool = True) -> list[int]:
    if len(a) <= 1:
        assert root == 1
        return a
    assert not len(a) & 1
    assert root != 1

    root2 = (root**2) % mod
    results0 = ifft(a[::2], root2, mod, False)
    results1 = ifft(a[1::2], root2, mod, False)

    rootp = 1
    results: list[int] = [0] * len(a)
    for i, (r0, r1) in enumerate(zip(results0, results1)):
        results[i] = (r0 + rootp * r1) % mod
        results[i + len(results0)] = (r0 - rootp * r1) % mod
        rootp = (rootp * root) % mod
    assert rootp == mod - 1

    if first_round:
        print(len(a))
        for i in range(len(results)):
            q, r = divmod(results[i], len(a))
            assert r == 0
            results[i] = q

    return results


GLOBAL_COUNTER = 0


def mod_fermat(n: int, mod_bits: int) -> int:
    global GLOBAL_COUNTER
    counter = 0
    while mod_bits < n.bit_length():
        n = (n & (1 << mod_bits) - 1) - (n >> mod_bits)
        counter += 1
    if GLOBAL_COUNTER < counter:
        GLOBAL_COUNTER = counter
        print(f"counter = {counter}")
    if n < 0:
        n += (1 << mod_bits) + 1
    return n


def sqrt2(mod_bits: int) -> int:
    assert not mod_bits & 3, "mod_bits must be divisible by 4"
    mod_bits2, mod_bits4 = mod_bits >> 1, mod_bits >> 2
    return (1 << mod_bits2 + mod_bits4) - (1 << mod_bits4)


def bit_rev(n: int, bit_len: int):
    n_bytes = (n << (-bit_len & 7)).to_bytes(bit_len + 7 >> 3, "little")
    return int.from_bytes([BYTE_BIT_REV_TBL[b] for b in n_bytes], "big")


def bit_rev_reord(a: MutableSequence[int]):
    assert not len(a) & len(a) - 1, "len(a) must be a power of 2"
    log_len_a = len(a).bit_length() - 1
    for i in range(len(a)):
        j = bit_rev(i, log_len_a)
        if i < j:
            a[i], a[j] = a[j], a[i]


def schönhage_strassen_fft(
    a: MutableSequence[int],
    el_bit_len: int,
) -> int:
    assert max(map(int.bit_length, a)) <= el_bit_len
    assert not len(a) & len(a) - 1, "len(a) must be a power of 2"

    log_len_a = len(a).bit_length() - 1
    M_tag = (2 * el_bit_len + log_len_a + len(a) - 1) >> log_len_a
    mod_bits = M_tag << log_len_a

    for step in reversed(range(log_len_a)):
        print("fft: step =", step)
        halfway = 1 << step
        for i in range(0, len(a), 2 * halfway):
            for j in range(halfway):
                # t := (a[i+j] - a[i+j+halfway]) * root^(-j * 2^(log_len_a - step))
                t = mod_fermat(a[i + j + halfway] - a[i + j], mod_bits)
                t <<= mod_bits - (j * M_tag << log_len_a - step)

                a[i + j] = mod_fermat(a[i + j] + a[i + j + halfway], mod_bits)
                a[i + j + halfway] = mod_fermat(t, mod_bits)

    # bit_rev_reord(a)

    return mod_bits


def schönhage_strassen_ifft(
    a: MutableSequence[int],
    el_bit_len: int,
) -> int:
    assert not len(a) & len(a) - 1, "len(a) must be a power of 2"

    # bit_rev_reord(a)

    log_len_a = len(a).bit_length() - 1
    M_tag = (2 * el_bit_len + log_len_a + len(a) - 1) >> log_len_a
    mod_bits = M_tag << log_len_a

    for step in range(log_len_a):
        print("ifft: step =", step)
        halfway = 1 << step
        for i in range(0, len(a), 2 * halfway):
            for j in range(halfway):
                # t := -a[i+j+halfway] * root^(j * 2^(log_len_a - step))
                t = a[i + j + halfway] << (j * M_tag << log_len_a - step)
                t = mod_fermat(t, mod_bits)

                a[i + j + halfway] = mod_fermat(a[i + j] - t, mod_bits)
                a[i + j] = mod_fermat(a[i + j] + t, mod_bits)

    for i in range(len(a)):
        if a[i] < 0:
            a[i] += (1 << mod_bits) + 1
        assert a[i] & len(a) - 1 == 0
        a[i] >>= log_len_a

    return mod_bits


def schönhage_strassen_mul(a: int, b: int) -> int:
    assert 0 < a and 0 < b

    n = 1 << max(7, a.bit_length() + b.bit_length() - 1).bit_length() - 3
    a_ds, b_ds = list(a.to_bytes(n, "little")), list(b.to_bytes(n, "little"))

    mod_bits = schönhage_strassen_fft(a_ds, 8)
    schönhage_strassen_fft(b_ds, 8)

    c_ds = [mod_fermat(dg_a * dg_b, mod_bits) for dg_a, dg_b in zip(a_ds, b_ds)]
    schönhage_strassen_ifft(c_ds, 8)

    carry = 0
    for i in range(n):
        c_ds[i] += carry
        carry = c_ds[i] >> 8
        c_ds[i] &= 255
    assert carry == 0

    return int.from_bytes(c_ds, "little")


def main():
    rng = np.random.default_rng()

    a = int.from_bytes(rng.bytes(1 << 10), "little")
    b = int.from_bytes(rng.bytes(1 << 10), "little")
    ss_c = schönhage_strassen_mul(a, b)
    py_c = a * b
    assert py_c == ss_c


if __name__ == "__main__":
    main()
