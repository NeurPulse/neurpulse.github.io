# -*- coding: utf-8 -*-
# 数学建模：学习地图与完整流程
# Source: /notes/mathematical-modeling-roadmap/
# Generated from the Markdown examples; edit the note and export again.

def solve_plan(material):
    feasible = [
        (3*x + 2*y, x, y)
        for x in range(6)
        for y in range(6)
        if x + y <= 5 and 2*x + y <= material
    ]
    return max(feasible)

profit, x, y = solve_plan(8)
print("利润、A批数、B批数:", profit, x, y)
print("原料用量:", 2*x + y, "总批数:", x + y)
print("原料减少后:", solve_plan(7))
assert (profit, x, y) == (13, 3, 2)
assert 2*x + y <= 8 and x + y <= 5
