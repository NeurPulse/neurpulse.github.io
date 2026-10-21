import math


def close(a, b, tol=1e-8):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def production_feasible(x, y):
    return (x >= 0 and y >= 0
            and 8*x + 5*y <= 450 + 1e-9
            and 4*x + 9*y <= 150 + 1e-9
            and 13*x + 6*y <= 300 + 1e-9)


# Example 1.1: continuous solution and complete integer enumeration.
x, y = 600/31, 250/31
assert production_feasible(x, y)
assert close(0.9*x + 1.5*y, 915/31)
plans = [(0.9*i + 1.5*j, i, j)
         for i in range(24) for j in range(17)
         if production_feasible(i, j)]
profit, ix, iy = max(plans)
assert (ix, iy) == (19, 8) and close(profit, 29.1)
print("Production continuous:", x, y, "profit =", 915/31)
print("Production integer:", ix, iy, "profit =", profit)

# Example 1.2: the small teaching investment problem.
projects = [(6, 12), (5, 9), (5, 9)]
choices = []
for mask in range(2**len(projects)):
    bits = [(mask >> i) & 1 for i in range(len(projects))]
    cost = sum(a*z for (a, b), z in zip(projects, bits))
    gain = sum(b*z for (a, b), z in zip(projects, bits))
    if cost <= 10:
        choices.append((gain, bits))
assert max(choices) == (18, [0, 1, 1])
print("Investment teaching example:", max(choices))

# Example 1.3: trough with sheet width a = 6.
a = 6.0
bottom, side, angle = a/3, a/3, math.pi/3
upper = bottom + 2*side*math.cos(angle)
height = side*math.sin(angle)
area = (bottom + upper)*height/2
assert close(area, math.sqrt(3)*a*a/12)
print("Trough:", "bottom =", bottom, "area =", area)

# Example 1.4: the two-by-two teaching transport problem.
shipments = [[10, 10], [0, 30]]
assert [sum(row) for row in shipments] == [20, 30]
assert [sum(row[j] for row in shipments) for j in range(2)] == [10, 40]
costs = [[2, 4], [3, 1]]
transport_cost = sum(costs[i][j]*shipments[i][j]
                     for i in range(2) for j in range(2))
assert transport_cost == 90
print("Transport teaching example:", shipments, transport_cost)

# Exercise 1.2: cone, using original sheet radius r = 1.
r = 1.0
base = r*math.sqrt(2/3)
cone_height = r/math.sqrt(3)
alpha = 2*math.pi*(1 - math.sqrt(2/3))
cone_volume = math.pi*base*base*cone_height/3
assert close(2*math.pi*base, r*(2*math.pi-alpha))
assert close(base*base + cone_height*cone_height, r*r)
assert close(cone_volume, 2*math.pi*r**3/(9*math.sqrt(3)))
print("Cone cut angle in degrees:", math.degrees(alpha))

# Exercise 1.3: triangle, checking Heron's formula.
side_length = 50/3
s = 25
triangle_area = math.sqrt(s*(s-side_length)**3)
assert close(triangle_area, 625*math.sqrt(3)/9)
print("Triangle side and area:", side_length, triangle_area)

# Exercise 1.4: bisection for the common marginal return.
def spending(lam):
    return [(4/lam)**2, (0.6/lam)**(10/7),
            (1.8/lam)**2.5, (4.4/lam)**(5/3)]


lo, hi = 0.001, 10.0
assert sum(spending(lo)) > 1000 > sum(spending(hi))
for _ in range(100):
    mid = (lo + hi)/2
    if sum(spending(mid)) > 1000:
        lo = mid
    else:
        hi = mid
lam = (lo + hi)/2
budget = spending(lam)
gain = (8*budget[0]**0.5 + 2*budget[1]**0.3
        + 3*budget[2]**0.6 + 11*budget[3]**0.4)
assert close(sum(budget), 1000)
assert all(z > 0 for z in budget)
assert close(gain, 364.7519235947512)
print("Advertising allocation:", budget, "gain =", gain)

# Exercise 1.5: distance-weighted iteration for these three cities.
cities = [(60.0, 30.0), (0.0, 0.0), (0.0, 50.0)]
point = [20.0, 80/3]
for _ in range(10000):
    distances = [math.dist(point, q) for q in cities]
    if min(distances) < 1e-12:
        raise ValueError("City point reached; zero-distance handling is needed.")
    weight = sum(1/d for d in distances)
    updated = [sum(q[k]/d for q, d in zip(cities, distances))/weight
               for k in range(2)]
    if math.dist(updated, point) < 1e-12:
        point = updated
        break
    point = updated
else:
    raise RuntimeError("Iteration did not meet the stopping tolerance.")
distances = [math.dist(point, q) for q in cities]
gradient = [sum((point[k]-q[k])/d for q, d in zip(cities, distances))
            for k in range(2)]
assert math.hypot(*gradient) < 1e-10
assert close(sum(distances), 103.42220468887054)
print("Airport:", point, "distance sum =", sum(distances))

# Exercise 1.6: cylinder and all original constraints.
radius = math.sqrt(187.5/math.pi)
height = 2.4*radius
lateral = 2*math.pi*radius*height
volume = math.pi*radius*radius*height
assert 5 <= radius <= 20 and 0 < height <= 20
assert 2.4 - 1e-12 <= height/radius <= 3.4 + 1e-12
assert close(lateral, 900)
assert close(volume, 3476.4678182087064)
print("Cylinder:", radius, height, "volume =", volume)
print("All numerical checks passed.")
