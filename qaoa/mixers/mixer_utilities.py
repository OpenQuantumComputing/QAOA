import numpy as np
from sympy.physics.quantum import TensorProduct
from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product
from OLD.binsymbols import binsymbols
from sympy import *
from sympy.core.numbers import One as sympyOne
from openquantumcomputing.utilities import decompose
import itertools
import math

X = Pauli(1)
Y = Pauli(2)
Z = Pauli(3)
I = 1


def hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def get_T(n, mode, d=1, i=None, j=None, shift=1, oddeven="both"):
    if mode == "leftright":
        T = np.zeros((n, n))
        if isinstance(i, list):
            if not len(i) == len(j):
                raise ValueError(
                    'mode "' + mode + '": length of index lists must be equal'
                )
            for ind in range(len(i)):
                T[i[ind], j[ind]] = 1
                T[j[ind], i[ind]] = 1
        else:
            T[i, j] = 1
            T[j, i] = 1
    elif mode == "full":
        T = np.ones((n, n))
        for i in range(n):
            T[i, i] = 0
    elif mode == "nearest_int":
        T = np.zeros((n, n))
        for i in range(n - shift):
            if oddeven == "even" and i % 2 == 0:
                continue
            elif oddeven == "odd" and i % 2 == 1:
                continue
            T[i, i + shift] = 1
            T[i + shift, i] = 1
    elif mode == "nearest_int_cyclic":
        T = np.zeros((n, n))
        for i in range(n - 1):
            if oddeven == "even" and i % 2 == 0:
                continue
            elif oddeven == "odd" and i % 2 == 1:
                continue
            T[i, i + 1] = 1
            T[i + 1, i] = 1
        if not oddeven == "even":
            T[0, -1] = 1
            T[-1, 0] = 1
    elif mode == "random":
        T = np.random.rand(n, n)
        for i in range(0, n):
            for j in range(i, n):
                T[i, j] = T[j, i]
    elif mode == "standard" or mode == "Hamming":
        ispowertwo = (n & (n - 1) == 0) and n != 0
        if not ispowertwo:
            raise ValueError('mode "' + mode + '" needs n to be a power of two')
        T = np.zeros((n, n))
        log2n = int(np.log2(n))
        for i in range(n):
            for j in range(n):
                s1 = "{0:b}".format(i).zfill(log2n)
                s2 = "{0:b}".format(j).zfill(log2n)
                if hamming(s1, s2) == d:
                    T[i, j] = 1
    else:
        raise NotImplementedError('mode "' + mode + '" not implemented')

    return T


class PauliStringTP:
    def __init__(self, excludeI=False):
        self.excludeI = excludeI
        self.items = []

    def get_items_PS(self, tp):
        if isinstance(tp, Pauli):
            if not self.excludeI:
                self.items.append(tp)
            elif not tp == I:
                self.items.append(tp)
        else:
            tpL, tpR = tp.args
            if isinstance(tpL, TensorProduct):
                self.get_items_PS(tpL)
            else:
                if not self.excludeI:
                    self.items.append(tpL)
                elif not tpL == I:
                    self.items.append(tpL)
            if isinstance(tpR, TensorProduct):
                self.get_items_PS(tpR)
            else:
                if not self.excludeI:
                    self.items.append(tpR)
                elif not tpR == I:
                    self.items.append(tpR)


def TPoPtoString(item):
    ret = ""
    tps = PauliStringTP()
    tps.get_items_PS(item)
    for p in tps.items:
        if p == 1:
            ret += "I"
        if p == X:
            ret += "X"
        if p == Y:
            ret += "Y"
        if p == Z:
            ret += "Z"
    return ret


def HtoString(H, symbolic=False):
    ret = ""
    if isinstance(H, TensorProduct) or isinstance(
        H, Pauli
    ):  ### go through Pauli string
        ret += TPoPtoString(H)
    else:
        for item in H.args:  ### go through all items of the sum (Pauli strings)
            if isinstance(item, Mul):  ### remove float
                if symbolic:
                    fval, _, item = item.args
                else:
                    if len(item.args) > 2:
                        fval, tmp, item = item.args
                        if not math.isclose(fval, 0, abs_tol=1e-7):
                            raise AssertionError(
                                "Encountered imaginary part that is not close to zero, aborting!",
                                fval,
                                tmp,
                                item,
                            )
                    else:
                        fval, item = item.args
                        if math.isclose(fval, 0, abs_tol=1e-7):
                            item = None
                            print("depug: close to zero", fval, item)
                ret += f"{fval:+.2f}" + " "
            if isinstance(item, TensorProduct) or isinstance(
                item, Pauli
            ):  ### go through Pauli string
                ret += TPoPtoString(item)
            ret += " "
    return ret


def num_Cnot_TPoP(item, symbolic=False):
    tps = PauliStringTP(excludeI=True)
    tps.get_items_PS(item)
    tmp = len(tps.items)
    if tmp == 1:
        sqg = 1
        cnot = 0
    elif tmp > 1:
        sqg = 0
        cnot = 2 * (tmp - 1)
    return sqg, cnot


def num_Cnot(H, symbolic=False):
    sqg = 0
    cnot = 0
    if isinstance(H, TensorProduct) or isinstance(
        H, Pauli
    ):  ### go through Pauli string
        sqg_, cnot_ = num_Cnot_TPoP(H, symbolic=symbolic)
        sqg += sqg_
        cnot += cnot_
    else:
        for item in H.args:  ### go through all items of the sum (Pauli strings)
            if isinstance(item, Mul):  ### remove float
                if symbolic:
                    fval, _, item = item.args
                else:
                    if len(item.args) > 2:
                        fval, tmp, item = item.args
                        if not math.isclose(fval, 0, abs_tol=1e-7):
                            raise AssertionError(
                                "Encountered imaginary part that is not close to zero, aborting!",
                                fval,
                                tmp,
                                item,
                            )
                    else:
                        fval, item = item.args
                        if math.isclose(fval, 0, abs_tol=1e-7):
                            item = None
                            print("depug: close to zero", fval, item)
            if isinstance(item, TensorProduct) or isinstance(
                item, Pauli
            ):  ### go through Pauli string
                sqg_, cnot_ = num_Cnot_TPoP(item, symbolic=symbolic)
                sqg += sqg_
                cnot += cnot_
    return sqg, cnot


def get_g(binstrings):
    n = len(binstrings[0])
    x = binsymbols("x:" + str(n))
    expr = 1
    for bs in binstrings:
        tmp_expr = 0
        for i in range(n):
            if bs[i] == "0":
                tmp_expr += x[i]
            else:
                tmp_expr += (x[i] - 1) ** 2
        expr *= tmp_expr
    return x, expand(expr)


def convert_to_ps(bs1, bs2):
    n = len(bs1)

    for j in range(n):
        if bs1[j] == "1" and bs2[j] == "0":
            tmp = 1 / 2 * (X - 1j * Y)
        elif bs1[j] == "0" and bs2[j] == "1":
            tmp = 1 / 2 * (X + 1j * Y)
        elif bs1[j] == "1" and bs2[j] == "1":
            tmp = 1 / 2 * (I - Z)
        else:  # bs[j]=="0" and bs[j]=="0":
            tmp = 1 / 2 * (I + Z)
        if j == 0:
            pauli_str = tmp
        else:
            pauli_str = TensorProduct(pauli_str, tmp)
    return pauli_str


def get_overlap(binstringsA, binstringsB):
    overlap = []
    mA = len(binstringsA)
    mB = len(binstringsB)
    for i in range(mA):
        for j in range(mB):
            if binstringsA[i] == binstringsB[j]:
                overlap.append(binstringsA[i])
    return overlap


def get_Pauli_string(binstrings, T, symbolic=False):
    m = len(binstrings)

    pauli_str = 0
    if symbolic:
        for i in range(m):
            for j in range(m):
                tmp_ps = convert_to_ps(binstrings[i], binstrings[j])
                pauli_str += T[i, j] * tmp_ps
    else:
        for i in range(m):
            for j in range(m):
                if not math.isclose(T[i, j], 0, abs_tol=1e-7):
                    tmp_ps = convert_to_ps(binstrings[i], binstrings[j])
                    pauli_str += T[i, j] * tmp_ps

    return pauli_str


def get_A1(w, z):
    if z[0] == "0" and w[0] == "0":
        return I + Z
    if (z[0] == "0" and w[0] == "1") or (z[0] == "1" and w[0] == "0"):
        return X
    return I - Z  # z[0]==1 and w[0]==1


def get_B1(w, z):
    if z[0] == "0" and w[0] == "1":
        return -Y
    if z[0] == "1" and w[0] == "0":
        return Y
    return 0.0  # (z[0]==0 and w[0]==0) or (z[0]==1 and w[0]==1)


def get_A_next(x, y, A_n, B_n):
    if x == "0" and y == "0":
        return 0.5 * TensorProduct(A_n, I + Z)
    if x == "0" and y == "1":
        return 0.5 * (TensorProduct(A_n, X) + TensorProduct(B_n, Y))
    if x == "1" and y == "0":
        return 0.5 * (TensorProduct(A_n, X) - TensorProduct(B_n, Y))
    return 0.5 * TensorProduct(A_n, I - Z)  # x==1 and y==1


def get_B_next(x, y, A_n, B_n):
    if x == "0" and y == "0":
        return 0.5 * (TensorProduct(B_n, I + Z))
    if x == "0" and y == "1":
        return 0.5 * (TensorProduct(B_n, X) - TensorProduct(A_n, Y))
    if x == "1" and y == "0":
        return 0.5 * (TensorProduct(B_n, X) + TensorProduct(A_n, Y))
    return 0.5 * (TensorProduct(B_n, I - Z))  # x==1 and y==1


def get_A_n(w, z):
    A_curr = get_A1(w, z)
    B_curr = get_B1(w, z)
    for n in range(1, len(z)):
        x, y = z[n], w[n]
        A_curr, B_curr = get_A_next(x, y, A_curr, B_curr), get_B_next(
            x, y, A_curr, B_curr
        )
    return A_curr


def get_Pauli_string_with_algorithm3(B, T):
    pauli_string = 0
    J = len(B)
    for j in range(J):
        for k in range(j + 1, J):
            if not math.isclose(T[j, k], 0, abs_tol=1e-7):
                w = B[j]
                z = B[k]
                A_n = get_A_n(w, z)  # recursively compute A_n
                pauli_string += A_n * T[j, k]
    return pauli_string


def simplifyH(H):
    for i in range(10):
        H = H.expand(tensorproduct=True)
    H = evaluate_pauli_product(H)
    return H


def get_H(stringlist, T, simplify=True, symbolic=False, verbose=False):
    H = get_Pauli_string(stringlist, T, symbolic=symbolic)
    if simplify:
        H = simplifyH(H)
    if verbose:
        print("#sqg, #cnots=", num_Cnot(H, symbolic=symbolic))
    return H


def matrix_to_sympy(H):
    """Decompose matrix H to sympy pauli string"""
    array = [i.split("*") for i in decompose(H).replace(" ", "").split("+")[1:]]
    pauli_str_sympy = 0
    pauli_map = {"I": 1, "X": Pauli(1), "Y": Pauli(2), "Z": Pauli(3)}
    for number, pauli_str in array:
        if len(pauli_str) == 1:
            tp = pauli_map[pauli_str]
        else:
            tp = TensorProduct(pauli_map[pauli_str[0]], pauli_map[pauli_str[1]])
            for i in range(2, len(pauli_str)):
                tp = TensorProduct(tp, pauli_map[pauli_str[i]])
        pauli_str_sympy += float(number) * tp
    return pauli_str_sympy


class Circuit_maker:
    """class for creating circuit from sympy expression"""

    def __init__(self, H):
        self.array = self.__create_array(H)

    def add_gates(self, circuit, parameter):
        """iterate over self.array and add gates to circuit,
        parameter is the rotation angle used in the rz gate"""
        for row in self.array:
            if row[0] == "h":
                circuit.h(row[1])
            elif row[0] == "cx":
                circuit.cx(row[1], row[2])
            elif row[0] == "rz":
                circuit.rz(row[1] * parameter, row[2])
            elif row[0] == "s":
                circuit.s(row[1])
            elif row[0] == "sdg":
                circuit.sdg(row[1])

    def __create_array(self, H):
        """create self.array which holds instructions of where gates should be placed,
        corresponding to equation 33 in Fuchs et al. (arXiv:2203.06095).
        H is a sympy expression and could be for example the mixing operator, H_m"""
        array = []
        if H != 0:
            commuting_terms = Add.make_args(H)
            for expr in commuting_terms:
                # remove artificially introduced symbols from H, which was introduced to keep
                # commuting terms separated. Example: expr = c(4*II + XX) -> H_args = [4*II, XX].
                H_args = Add.make_args(
                    list(
                        filter(
                            lambda x: not isinstance(x, Symbol),
                            list(Mul.make_args(expr)),
                        )
                    )[0]
                )
                if len(H_args) == 1:  # if only one term in H
                    H_args = [H_args[0]]

                for pauli_string in H_args:
                    # remove all symbols from expression, and split pauli string to (float, Symbol, TensorProduct)
                    mul_args = list(
                        filter(
                            lambda x: not isinstance(x, Symbol),
                            list(Mul.make_args(pauli_string)),
                        )
                    )
                    angle, tensor_prod = mul_args

                    # split tensor product expression on each tensor product
                    operators = self.__get_operators_from_tensor_product(tensor_prod)

                    # add gates to array
                    self.__add_U(array, operators)
                    self.__add_middle_part(array, operators, angle)
                    self.__add_U_dagger(array, operators)
        return array

    def __get_operators_from_tensor_product(self, tensor_product):
        """split tensor product expr on each tensor product"""
        operators = []
        while not isinstance(tensor_product, (Pauli, sympyOne)):
            tensor_product, operator = tensor_product.args
            operators.insert(0, operator)
        operators.insert(0, tensor_product)
        return operators

    def __add_U(self, array, operators):
        """add gates corresponding to block U in eq. 33"""
        for i, operator in enumerate(operators):
            if operator == Pauli(1):  # X
                array.append(("h", i))
            elif operator == Pauli(2):  # Y
                array.append(("s", i))
                array.append(("h", i))

    def __add_middle_part(self, array, operators, angle):
        """add cnots and rz gate corresponding to the middle part in eq. 33"""
        pairs = self.__get_cx_pairs(operators)

        # if only one operator in operators, add rz gate on correct index (do not add cx gates)
        if len(pairs) == 0:
            for i, operator in enumerate(operators):
                if isinstance(operator, Pauli):
                    array.append(("rz", -2.0 * float(angle), i))
                    break
        else:
            for pair in pairs:
                array.append(("cx", pair[0], pair[1]))
            array.append(("rz", -2.0 * float(angle), pairs[-1][1]))
            for pair in pairs[::-1]:
                array.append(("cx", pair[0], pair[1]))

    def __add_U_dagger(self, array, operators):
        """add gates corresponding to block dagger(U) in eq. 33"""
        for i, operator in enumerate(operators):
            if operator == Pauli(1):  # X
                array.append(("h", i))
            elif operator == Pauli(2):  # Y
                array.append(("h", i))
                array.append(("sdg", i))

    def __get_cx_pairs(self, operators):
        """return list of indices of where cx gates should be placed"""
        pairs = []
        j = None
        for i, operator in enumerate(operators):
            if i != 0 and isinstance(operator, Pauli) and j != None:
                pairs.append((j, i))
            if isinstance(operator, Pauli):
                j = i
        return pairs
