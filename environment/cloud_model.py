class Task:
    def __init__(self, task_id, cpu, ram, length):
        self.id = task_id
        self.cpu = cpu
        self.ram = ram
        self.length = length


class VM:
    def __init__(self, vm_id, cpu_capacity, ram_capacity, cost_per_time, speed):
        self.id = vm_id
        self.cpu_capacity = cpu_capacity
        self.ram_capacity = ram_capacity
        self.cost_per_time = cost_per_time
        self.speed = speed


class CloudEnvironment:
    def __init__(self, tasks, vms):
        self.tasks = tasks
        self.vms = vms