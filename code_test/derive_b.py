import sympy as sp

# Define variables
a, b, g, y = sp.symbols('alpha beta gamma y')

# Define moments X3 to X6
X3 = a * b + 3 * a * g
X4 = -2 * a**2 + a * b**2 + 6 * a * b * g + 3 * a * g**2
X5 = a * (b**3 - 8 * a * b + 10 * b**2 * g + 15 * g**2 * b - 20 * a * g)
X6 = a * (16 * a**2 - 12 * a * b**2 - 60 * a * b * g + b**4 + 15 * b**3 * g + 45 * b**2 * g**2 + 15 * b * g**3)

# D, E, A, C formulas from the paper
D = 2 * X3 * y**3 + X5 * y**2 - 3 * X3 * X4 * y + 2 * X3**3
E = 2 * y**3 + 3 * X4 * y - 4 * X3**2
A = -E
C = 10 * X3**3 - 7 * X3 * X4 * y - 2 * X3 * y**3

# We know that at y = alpha: D / A = alpha * gamma
print("Testing D / A == a * g at y = a:")
D_sub = D.subs(y, a).expand()
A_sub = A.subs(y, a).expand()
ag = (a * g).expand()
print("D - A*(a*g) at y=a: ", sp.simplify(D_sub - A_sub * ag))

# Now we want B such that B / C = alpha * gamma^2 or something?
# The paper says p6(y) = A(y)B(y) - C(y)D(y). The root y=a occurs when A B = C D.
# So B(a) = C(a) * D(a) / A(a) = C(a) * (a * g).
B_target = sp.simplify(C.subs(y, a) * a * g)

# The paper's B is:
B_paper = 4 * X4**3 - 4 * X3**2 * X4 * y - 8 * X3**2 * y**3 - X4**2 * y**2 + 8 * X4 * y**4 + X6 * y**3 + 4 * y**6
B_paper_sub = B_paper.subs(y, a).expand()

print("\nDifference between Paper B and Target B:")
diff = sp.simplify(B_paper_sub - B_target)
print(diff)

# Let's see what diff actually is in terms of X's:
print("\nCan we express B_target using X3..X6 ?")
# It's known that B corresponds to an equation for alpha * gamma^2 (or similar).
