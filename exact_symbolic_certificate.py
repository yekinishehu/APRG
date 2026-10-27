"""Exact-arithmetic certificate for the period-two separator orbit.

Run with: python3 exact_symbolic_certificate.py
Requires only SymPy. No floating-point computation is used in any assertion.
The assertions certify branch selection, both orbit equations, and the
characteristic polynomial printed in Appendix A of the revised manuscript.
"""
import sympy as sp

lam = sp.Rational(1, 20)
eps = 1 / sp.sqrt(99)
sqrt11 = sp.sqrt(11)
a = sp.Matrix([
    -sp.Rational(891, 128140) - 153 * sqrt11 / 64070,
    -sp.Rational(51, 64070) - 27 * sqrt11 / 128140,
])
b = sp.Matrix([
    -sp.Rational(891, 128140) + 153 * sqrt11 / 64070,
    -sp.Rational(51, 64070) + 27 * sqrt11 / 128140,
])
G = sp.Matrix([[1, -3], [3, 1]])
h = sp.Matrix([0, 3])
normals = [sp.Matrix([eps, -1]), sp.Matrix([-eps, -1])]
M = [sp.simplify(sp.eye(2) - v * v.T / (v.T * v)[0]) for v in normals]


def reflected_input(x, previous):
    return sp.simplify(x - lam * (G * (2 * x - previous) + h))


def state_jacobian(projector):
    return sp.Matrix(sp.BlockMatrix([
        [sp.zeros(2), sp.eye(2)],
        [projector * lam * G,
         projector * (sp.eye(2) - 2 * lam * G)],
    ]))


# On T(a), facet 2 (index 1) is the unique most-violated facet; on T(b),
# facet 1 (index 0) is the unique most-violated facet.
z_a = reflected_input(a, b)
z_b = reflected_input(b, a)
residual_a = [sp.simplify(v.dot(z_a)) for v in normals]
residual_b = [sp.simplify(v.dot(z_b)) for v in normals]
assert sp.simplify(residual_a[1] > residual_a[0] > 0) is sp.true
assert sp.simplify(residual_b[0] > residual_b[1] > 0) is sp.true

orbit_residual_a = sp.simplify(M[1] * z_a - b)
orbit_residual_b = sp.simplify(M[0] * z_b - a)
assert orbit_residual_a == sp.zeros(2, 1)
assert orbit_residual_b == sp.zeros(2, 1)

J1 = state_jacobian(M[0])
J2 = state_jacobian(M[1])
J = sp.simplify(J1 * J2)
r = sp.symbols("r")
characteristic = sp.factor((r * sp.eye(4) - J).det())
expected = r**2 * (50000 * r**2 - 43718 * r + 125) / 50000
assert sp.simplify(characteristic - expected) == 0

print("Exact branch-selection residuals at a:", residual_a)
print("Exact branch-selection residuals at b:", residual_b)
print("M2*z_a-b =", orbit_residual_a.T)
print("M1*z_b-a =", orbit_residual_b.T)
print("J1 =", J1)
print("J2 =", J2)
print("J = J1*J2 =", J)
print("det(rI-J) =", characteristic)
print("All exact-arithmetic assertions passed.")
