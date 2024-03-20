import numpy as np
import random


class CriticalSection:
    def __init__(self, duration, nested_level):
        self.duration = duration
        self.nested_level = nested_level

    def __repr__(self):
        return f"CriticalSection(duration={self.duration}, nested_level={self.nested_level})"


class Resource:
    def __init__(self, resource_id, nestable):
        self.resource_id = resource_id
        self.nestable = nestable
        self.nested_access = None

    def __repr__(self):
        if self.nestable and self.nested_access:
            return f"Resource {self.resource_id} (Nestable, can access: Resource {self.nested_access.resource_id})"
        elif self.nestable:
            return f"Resource {self.resource_id} (Nestable, no nested access defined)"
        else:
            return f"Resource {self.resource_id} (Non-Nestable)"


def create_resources(num_resources):
    resources = [Resource(i + 1, random.choice([True, False])) for i in range(num_resources)]
    for resource in resources:
        if resource.nestable:
            potential_resources = [res for res in resources if res != resource and res.nested_access]
            if potential_resources:
                accessed_resource = random.choice(potential_resources)
                resource.nested_access = accessed_resource
    return resources


class Task:
    def __init__(self, unique_id, arrival, execution, deadline, period=None):
        self.id = unique_id
        self.arrival = arrival
        self.execution = execution
        self.deadline = deadline
        self.start = 0
        self.finish = 0
        self.period = period
        self.execution_timeline = []

    def generate_critical_sections(self, f):
        execution_as_int = int(self.execution)
        critical_sections_count = random.randint(0, min(8, execution_as_int))
        print(critical_sections_count)
        critical_sections_positions = random.sample(range(execution_as_int), critical_sections_count)
        for i in range(execution_as_int):
            if i in critical_sections_positions:
                if random.random() < 2 * f * (1 - f):
                    nesting_level = 1
                elif random.random() < f ** 2:
                    nesting_level = 2
                else:
                    nesting_level = 0
                self.execution_timeline.append(('Critical', nesting_level))
            else:
                self.execution_timeline.append(('Non-Critical', 0))

    def __repr__(self):
        return (f"Task(id={self.id}, critical_sections={self.execution_timeline},\n"
                f"and Task(id={self.id}, arrival={self.arrival}, execution={self.execution}, deadline={self.deadline})")


def UUniFast(n, u_bar):
    sum_u = u_bar
    vect_u = np.zeros(n)
    for i in range(n - 1):
        next_sum_u = sum_u * (1 - (np.random.rand() ** (1.0 / (n - i))))
        vect_u[i] = sum_u - next_sum_u
        sum_u = next_sum_u
    vect_u[n - 1] = sum_u
    return vect_u


def generate_tasks(num_tasks, u_bar):
    utilizations = UUniFast(num_tasks, u_bar)
    tasks = []
    for i in range(num_tasks):
        arrival = np.random.randint(0, 100)
        max_execution_time = 100
        execution = np.ceil(utilizations[i] * max_execution_time)
        buffer = np.random.randint(1, 50)
        deadline = arrival + execution + buffer
        tasks.append(Task(f"Task_{i + 1}", arrival, execution, deadline))
    return tasks


generated_tasks = generate_tasks(100, 0.5)
print(generated_tasks[:5])
resources = create_resources(10)
for res in resources:
    print(res)
f = 0.05

for task in generated_tasks[:5]:
    task.generate_critical_sections(f)
    print(task)
