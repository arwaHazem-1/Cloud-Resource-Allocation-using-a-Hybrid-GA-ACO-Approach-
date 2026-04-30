from operators import *

p1 = [0, 1, 2, 1]
p2 = [2, 0, 1, 0]

print("Parents:", p1, p2)

print("\n--- Crossover ---")
print("Single:", single_point_crossover(p1, p2))
print("Uniform:", uniform_crossover(p1, p2))

print("\n--- Mutation ---")
print("Random Reset:", random_reset_mutation(p1, 3))
print("Swap:", swap_mutation(p1))
print("\n--- Validation ---")
print("Valid:", validate_solution(p1, 3))



child1, child2 = single_point_crossover(p1, p2)
assert len(child1) == len(p1)
assert len(child2) == len(p1)

assert validate_solution(child1, 3)
assert validate_solution(child2, 3)

mut = random_reset_mutation(p1, 3, mutation_rate=1)
assert validate_solution(mut, 3)

print("\nAll tests passed ✅")