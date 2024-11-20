from qiskit.quantum_info import SparsePauliOp, Pauli


def getPauliOperator(k_cuts, color_encoding):
    # flip Pauli strings, because of qiskit's little endian encoding
    if k_cuts == 2:
        P = [
            [2 / (2**1), Pauli("ZZ")],
        ]
        Phalf = [
            [2 / (2**1), Pauli("Z")],
        ]
    elif k_cuts == 3:
        if color_encoding == "LessThanK":
            P = [
                [-4 / (2**4), Pauli("IIII"[::-1])],
                [-4 / (2**4), Pauli("IIZI"[::-1])],
                [+4 / (2**4), Pauli("IZIZ"[::-1])],
                [+4 / (2**4), Pauli("IZZZ"[::-1])],
                [-4 / (2**4), Pauli("ZIII"[::-1])],
                [+12 / (2**4), Pauli("ZIZI"[::-1])],
                [+4 / (2**4), Pauli("ZZIZ"[::-1])],
                [+4 / (2**4), Pauli("ZZZZ"[::-1])],
            ]
            Phalf = [
                [-8 / (2**4), Pauli("II"[::-1])],
                [+8 / (2**4), Pauli("ZI"[::-1])],
                [+8 / (2**4), Pauli("IZ"[::-1])],
                [+8 / (2**4), Pauli("ZZ"[::-1])],
            ]
        else:
            raise ValueError("invalid or unspecified color_encoding")
    elif k_cuts == 4:
        P = [
            [-8 / (2**4), Pauli("IIII"[::-1])],
            [+8 / (2**4), Pauli("IZIZ"[::-1])],
            [+8 / (2**4), Pauli("ZIZI"[::-1])],
            [+8 / (2**4), Pauli("ZZZZ"[::-1])],
        ]
        Phalf = [
            [-8 / (2**4), Pauli("II"[::-1])],
            [+8 / (2**4), Pauli("IZ"[::-1])],
            [+8 / (2**4), Pauli("ZI"[::-1])],
            [+8 / (2**4), Pauli("ZZ"[::-1])],
        ]
    elif k_cuts == 5:
        if color_encoding == "max_balanced":
            # ((0, 1), (2,), (3,), (4,5), (6,7,)):
            P = [
                [-36 / (2**6), Pauli("IIIIII"[::-1])],
                [4 / (2**6), Pauli("IIIIZI"[::-1])],
                [-4 / (2**6), Pauli("IIIZII"[::-1])],
                [4 / (2**6), Pauli("IIIZZI"[::-1])],
                [4 / (2**6), Pauli("IIZIIZ"[::-1])],
                [-4 / (2**6), Pauli("IIZIZZ"[::-1])],
                [4 / (2**6), Pauli("IIZZIZ"[::-1])],
                [-4 / (2**6), Pauli("IIZZZZ"[::-1])],
                [4 / (2**6), Pauli("IZIIII"[::-1])],
                [28 / (2**6), Pauli("IZIIZI"[::-1])],
                [4 / (2**6), Pauli("IZIZII"[::-1])],
                [-4 / (2**6), Pauli("IZIZZI"[::-1])],
                [-4 / (2**6), Pauli("IZZIIZ"[::-1])],
                [4 / (2**6), Pauli("IZZIZZ"[::-1])],
                [-4 / (2**6), Pauli("IZZZIZ"[::-1])],
                [4 / (2**6), Pauli("IZZZZZ"[::-1])],
                [-4 / (2**6), Pauli("ZIIIII"[::-1])],
                [4 / (2**6), Pauli("ZIIIZI"[::-1])],
                [28 / (2**6), Pauli("ZIIZII"[::-1])],
                [4 / (2**6), Pauli("ZIIZZI"[::-1])],
                [4 / (2**6), Pauli("ZIZIIZ"[::-1])],
                [-4 / (2**6), Pauli("ZIZIZZ"[::-1])],
                [4 / (2**6), Pauli("ZIZZIZ"[::-1])],
                [-4 / (2**6), Pauli("ZIZZZZ"[::-1])],
                [4 / (2**6), Pauli("ZZIIII"[::-1])],
                [-4 / (2**6), Pauli("ZZIIZI"[::-1])],
                [4 / (2**6), Pauli("ZZIZII"[::-1])],
                [28 / (2**6), Pauli("ZZIZZI"[::-1])],
                [-4 / (2**6), Pauli("ZZZIIZ"[::-1])],
                [4 / (2**6), Pauli("ZZZIZZ"[::-1])],
                [-4 / (2**6), Pauli("ZZZZIZ"[::-1])],
                [4 / (2**6), Pauli("ZZZZZZ"[::-1])],
            ]
            Phalf = [
                [-32 / (2**6), Pauli("III"[::-1])],
                [32 / (2**6), Pauli("IZI"[::-1])],
                [32 / (2**6), Pauli("ZII"[::-1])],
                [32 / (2**6), Pauli("ZZI"[::-1])],
            ]
        elif color_encoding == "LessThanK":
            P = [
                [-24 / (2**6), Pauli("IIIIII"[::-1])],
                [-24 / (2**6), Pauli("IIIZII"[::-1])],
                [+8 / (2**6), Pauli("IIZIIZ"[::-1])],
                [+8 / (2**6), Pauli("IIZZIZ"[::-1])],
                [+8 / (2**6), Pauli("IZIIZI"[::-1])],
                [+8 / (2**6), Pauli("IZIZZI"[::-1])],
                [+8 / (2**6), Pauli("IZZIZZ"[::-1])],
                [+8 / (2**6), Pauli("IZZZZZ"[::-1])],
                [-24 / (2**6), Pauli("ZIIIII"[::-1])],
                [+40 / (2**6), Pauli("ZIIZII"[::-1])],
                [+8 / (2**6), Pauli("ZIZIIZ"[::-1])],
                [+8 / (2**6), Pauli("ZIZZIZ"[::-1])],
                [+8 / (2**6), Pauli("ZZIIZI"[::-1])],
                [+8 / (2**6), Pauli("ZZIZZI"[::-1])],
                [+8 / (2**6), Pauli("ZZZIZZ"[::-1])],
                [+8 / (2**6), Pauli("ZZZZZZ"[::-1])],
            ]
            Phalf = [
                [-48 / (2**6), Pauli("III"[::-1])],
                [+16 / (2**6), Pauli("ZII"[::-1])],
                [+16 / (2**6), Pauli("IIZ"[::-1])],
                [+16 / (2**6), Pauli("ZIZ"[::-1])],
                [+16 / (2**6), Pauli("IZI"[::-1])],
                [+16 / (2**6), Pauli("ZZI"[::-1])],
                [+16 / (2**6), Pauli("IZZ"[::-1])],
                [+16 / (2**6), Pauli("ZZZ"[::-1])],
            ]
        else:
            raise ValueError("invalid or unspecified color_encoding")
    elif k_cuts == 6:
        if color_encoding in ["max_balanced"]:
            # ((0,1), (2), (3), (4,5), (6), (7))
            P = [
                [-40 / (2**6), Pauli("IIIIII"[::-1])],
                [8 / (2**6), Pauli("IIIIZI"[::-1])],
                [8 / (2**6), Pauli("IIZIIZ"[::-1])],
                [-8 / (2**6), Pauli("IIZIZZ"[::-1])],
                [8 / (2**6), Pauli("IZIIII"[::-1])],
                [24 / (2**6), Pauli("IZIIZI"[::-1])],
                [-8 / (2**6), Pauli("IZZIIZ"[::-1])],
                [8 / (2**6), Pauli("IZZIZZ"[::-1])],
                [24 / (2**6), Pauli("ZIIZII"[::-1])],
                [8 / (2**6), Pauli("ZIIZZI"[::-1])],
                [8 / (2**6), Pauli("ZIZZIZ"[::-1])],
                [-8 / (2**6), Pauli("ZIZZZZ"[::-1])],
                [8 / (2**6), Pauli("ZZIZII"[::-1])],
                [24 / (2**6), Pauli("ZZIZZI"[::-1])],
                [-8 / (2**6), Pauli("ZZZZIZ"[::-1])],
                [8 / (2**6), Pauli("ZZZZZZ"[::-1])],
            ]
            Phalf = [
                [-32 / (2**6), Pauli("III"[::-1])],
                [32 / (2**6), Pauli("IZI"[::-1])],
                [32 / (2**6), Pauli("ZII"[::-1])],
                [32 / (2**6), Pauli("ZZI"[::-1])],
            ]
        elif color_encoding == "LessThanK":
            P = [
                [-36 / (2**6), Pauli("IIIIII"[::-1])],
                [-4 / (2**6), Pauli("IIIIIZ"[::-1])],
                [-4 / (2**6), Pauli("IIIIZI"[::-1])],
                [-4 / (2**6), Pauli("IIIIZZ"[::-1])],
                [-12 / (2**6), Pauli("IIIZII"[::-1])],
                [+4 / (2**6), Pauli("IIIZIZ"[::-1])],
                [+4 / (2**6), Pauli("IIIZZI"[::-1])],
                [+4 / (2**6), Pauli("IIIZZZ"[::-1])],
                [-4 / (2**6), Pauli("IIZIII"[::-1])],
                [+12 / (2**6), Pauli("IIZIIZ"[::-1])],
                [+4 / (2**6), Pauli("IIZIZI"[::-1])],
                [+4 / (2**6), Pauli("IIZIZZ"[::-1])],
                [+4 / (2**6), Pauli("IIZZII"[::-1])],
                [+4 / (2**6), Pauli("IIZZIZ"[::-1])],
                [-4 / (2**6), Pauli("IIZZZI"[::-1])],
                [-4 / (2**6), Pauli("IIZZZZ"[::-1])],
                [-4 / (2**6), Pauli("IZIIII"[::-1])],
                [+4 / (2**6), Pauli("IZIIIZ"[::-1])],
                [+12 / (2**6), Pauli("IZIIZI"[::-1])],
                [+4 / (2**6), Pauli("IZIIZZ"[::-1])],
                [+4 / (2**6), Pauli("IZIZII"[::-1])],
                [-4 / (2**6), Pauli("IZIZIZ"[::-1])],
                [+4 / (2**6), Pauli("IZIZZI"[::-1])],
                [-4 / (2**6), Pauli("IZIZZZ"[::-1])],
                [-4 / (2**6), Pauli("IZZIII"[::-1])],
                [+4 / (2**6), Pauli("IZZIIZ"[::-1])],
                [+4 / (2**6), Pauli("IZZIZI"[::-1])],
                [+12 / (2**6), Pauli("IZZIZZ"[::-1])],
                [+4 / (2**6), Pauli("IZZZII"[::-1])],
                [-4 / (2**6), Pauli("IZZZIZ"[::-1])],
                [-4 / (2**6), Pauli("IZZZZI"[::-1])],
                [+4 / (2**6), Pauli("IZZZZZ"[::-1])],
                [-12 / (2**6), Pauli("ZIIIII"[::-1])],
                [+4 / (2**6), Pauli("ZIIIIZ"[::-1])],
                [+4 / (2**6), Pauli("ZIIIZI"[::-1])],
                [+4 / (2**6), Pauli("ZIIIZZ"[::-1])],
                [+28 / (2**6), Pauli("ZIIZII"[::-1])],
                [-4 / (2**6), Pauli("ZIIZIZ"[::-1])],
                [-4 / (2**6), Pauli("ZIIZZI"[::-1])],
                [-4 / (2**6), Pauli("ZIIZZZ"[::-1])],
                [+4 / (2**6), Pauli("ZIZIII"[::-1])],
                [+4 / (2**6), Pauli("ZIZIIZ"[::-1])],
                [-4 / (2**6), Pauli("ZIZIZI"[::-1])],
                [-4 / (2**6), Pauli("ZIZIZZ"[::-1])],
                [-4 / (2**6), Pauli("ZIZZII"[::-1])],
                [+12 / (2**6), Pauli("ZIZZIZ"[::-1])],
                [+4 / (2**6), Pauli("ZIZZZI"[::-1])],
                [+4 / (2**6), Pauli("ZIZZZZ"[::-1])],
                [+4 / (2**6), Pauli("ZZIIII"[::-1])],
                [-4 / (2**6), Pauli("ZZIIIZ"[::-1])],
                [+4 / (2**6), Pauli("ZZIIZI"[::-1])],
                [-4 / (2**6), Pauli("ZZIIZZ"[::-1])],
                [-4 / (2**6), Pauli("ZZIZII"[::-1])],
                [+4 / (2**6), Pauli("ZZIZIZ"[::-1])],
                [+12 / (2**6), Pauli("ZZIZZI"[::-1])],
                [+4 / (2**6), Pauli("ZZIZZZ"[::-1])],
                [+4 / (2**6), Pauli("ZZZIII"[::-1])],
                [-4 / (2**6), Pauli("ZZZIIZ"[::-1])],
                [-4 / (2**6), Pauli("ZZZIZI"[::-1])],
                [+4 / (2**6), Pauli("ZZZIZZ"[::-1])],
                [-4 / (2**6), Pauli("ZZZZII"[::-1])],
                [+4 / (2**6), Pauli("ZZZZIZ"[::-1])],
                [+4 / (2**6), Pauli("ZZZZZI"[::-1])],
                [+12 / (2**6), Pauli("ZZZZZZ"[::-1])],
            ]
            Phalf = [
                [-48 / (2**6), Pauli("III"[::-1])],
                [+16 / (2**6), Pauli("IIZ"[::-1])],
                [+16 / (2**6), Pauli("IZI"[::-1])],
                [+16 / (2**6), Pauli("IZZ"[::-1])],
                [+16 / (2**6), Pauli("ZII"[::-1])],
                [+16 / (2**6), Pauli("ZIZ"[::-1])],
                [+16 / (2**6), Pauli("ZZI"[::-1])],
                [+16 / (2**6), Pauli("ZZZ"[::-1])],
            ]
        else:
            raise ValueError("invalid or unspecified color_encoding")
    elif k_cuts == 7:
        if color_encoding == "LessThanK":
            P = [
                [-44 / (2**6), Pauli("IIIIII"[::-1])],
                [-4 / (2**6), Pauli("IIIIZI"[::-1])],
                [-4 / (2**6), Pauli("IIIZII"[::-1])],
                [+4 / (2**6), Pauli("IIIZZI"[::-1])],
                [+12 / (2**6), Pauli("IIZIIZ"[::-1])],
                [+4 / (2**6), Pauli("IIZIZZ"[::-1])],
                [+4 / (2**6), Pauli("IIZZIZ"[::-1])],
                [-4 / (2**6), Pauli("IIZZZZ"[::-1])],
                [-4 / (2**6), Pauli("IZIIII"[::-1])],
                [+20 / (2**6), Pauli("IZIIZI"[::-1])],
                [+4 / (2**6), Pauli("IZIZII"[::-1])],
                [-4 / (2**6), Pauli("IZIZZI"[::-1])],
                [+4 / (2**6), Pauli("IZZIIZ"[::-1])],
                [+12 / (2**6), Pauli("IZZIZZ"[::-1])],
                [-4 / (2**6), Pauli("IZZZIZ"[::-1])],
                [+4 / (2**6), Pauli("IZZZZZ"[::-1])],
                [-4 / (2**6), Pauli("ZIIIII"[::-1])],
                [+4 / (2**6), Pauli("ZIIIZI"[::-1])],
                [+20 / (2**6), Pauli("ZIIZII"[::-1])],
                [-4 / (2**6), Pauli("ZIIZZI"[::-1])],
                [+4 / (2**6), Pauli("ZIZIIZ"[::-1])],
                [-4 / (2**6), Pauli("ZIZIZZ"[::-1])],
                [+12 / (2**6), Pauli("ZIZZIZ"[::-1])],
                [+4 / (2**6), Pauli("ZIZZZZ"[::-1])],
                [+4 / (2**6), Pauli("ZZIIII"[::-1])],
                [-4 / (2**6), Pauli("ZZIIZI"[::-1])],
                [-4 / (2**6), Pauli("ZZIZII"[::-1])],
                [+20 / (2**6), Pauli("ZZIZZI"[::-1])],
                [-4 / (2**6), Pauli("ZZZIIZ"[::-1])],
                [+4 / (2**6), Pauli("ZZZIZZ"[::-1])],
                [+4 / (2**6), Pauli("ZZZZIZ"[::-1])],
                [+12 / (2**6), Pauli("ZZZZZZ"[::-1])],
            ]
            Phalf = [
                [-48 / (2**6), Pauli("III"[::-1])],
                [+16 / (2**6), Pauli("IZI"[::-1])],
                [+16 / (2**6), Pauli("ZII"[::-1])],
                [+16 / (2**6), Pauli("ZZI"[::-1])],
                [+16 / (2**6), Pauli("IIZ"[::-1])],
                [+16 / (2**6), Pauli("IZZ"[::-1])],
                [+16 / (2**6), Pauli("ZIZ"[::-1])],
                [+16 / (2**6), Pauli("ZZZ"[::-1])],
            ]
        else:
            raise ValueError("invalid or unspecified color_encoding")
    else:
        P = [
            [-48 / (2**6), Pauli("IIIIII"[::-1])],
            [+16 / (2**6), Pauli("IIZIIZ"[::-1])],
            [+16 / (2**6), Pauli("IZIIZI"[::-1])],
            [+16 / (2**6), Pauli("IZZIZZ"[::-1])],
            [+16 / (2**6), Pauli("ZIIZII"[::-1])],
            [+16 / (2**6), Pauli("ZIZZIZ"[::-1])],
            [+16 / (2**6), Pauli("ZZIZZI"[::-1])],
            [+16 / (2**6), Pauli("ZZZZZZ"[::-1])],
        ]
        Phalf = [
            [-48 / (2**6), Pauli("III"[::-1])],
            [+16 / (2**6), Pauli("IIZ"[::-1])],
            [+16 / (2**6), Pauli("IZI"[::-1])],
            [+16 / (2**6), Pauli("IZZ"[::-1])],
            [+16 / (2**6), Pauli("ZII"[::-1])],
            [+16 / (2**6), Pauli("ZIZ"[::-1])],
            [+16 / (2**6), Pauli("ZZI"[::-1])],
            [+16 / (2**6), Pauli("ZZZ"[::-1])],
        ]

    # devide coefficients by 2, since:
    # "The evolution gates are related to the Pauli rotation gates by a factor of 2"
    op = SparsePauliOp([item[1] for item in P], coeffs=[item[0] / 2 for item in P])
    ophalf = SparsePauliOp(
        [item[1] for item in Phalf], coeffs=[item[0] / 2 for item in Phalf]
    )
    return op, ophalf
