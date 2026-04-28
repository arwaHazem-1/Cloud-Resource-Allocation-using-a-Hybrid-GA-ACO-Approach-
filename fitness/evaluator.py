def evaluate(individual, env):
    tasks = env.tasks
    vms = env.vms

    vm_cpu_usage = [0] * len(vms)
    vm_ram_usage = [0] * len(vms)
    vm_time = [0] * len(vms)

    total_cost = 0
    penalty = 0

    for i, vm_id in enumerate(individual):
        task = tasks[i]
        vm = vms[vm_id]

        vm_cpu_usage[vm_id] += task.cpu
        vm_ram_usage[vm_id] += task.ram

        exec_time = task.length / vm.speed
        vm_time[vm_id] += exec_time

    for i, vm in enumerate(vms):
        if vm_cpu_usage[i] > vm.cpu_capacity:
            penalty += (vm_cpu_usage[i] - vm.cpu_capacity)

        if vm_ram_usage[i] > vm.ram_capacity:
            penalty += (vm_ram_usage[i] - vm.ram_capacity)

        total_cost += vm_time[i] * vm.cost_per_time

    makespan = max(vm_time)

    fitness = total_cost + makespan + 10 * penalty

    return fitness