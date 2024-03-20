import numpy as np
import random


class Processor:
    def __init__(self, processor_id):
        self.processor_id = processor_id
        self.is_idle = True
        self.current_task = None

    def assign_task(self, assigned_task):
        self.current_task = assigned_task
        self.is_idle = False

    def complete_task(self):
        self.current_task = None
        self.is_idle = True

    def __repr__(self):
        if self.is_idle:
            return f"Processor {self.processor_id} is idle"
        else:
            return f"Processor {self.processor_id} is executing {self.current_task.id}"


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
    created_resources = [Resource(i + 1, random.choice([True, False])) for i in range(num_resources)]
    for resource in created_resources:
        if resource.nestable:
            potential_resources = [choose_res for choose_res in created_resources
                                   if choose_res != resource and choose_res.nestable]
            if potential_resources:
                accessed_resource = random.choice(potential_resources)
                resource.nested_access = accessed_resource
    return created_resources


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
        self.linked = False
        self.scheduled = False
        self.cpu_name = None
        self.state = 'preemptable'

    def generate_critical_sections(self, nesting_factor, all_resources):
        critical_sections_count = random.randint(0, min(8, int(self.execution)))
        self.execution_timeline = [('Non-Critical', 0, None)] * int(self.execution)
        critical_sections_positions = random.sample(range(int(self.execution)), critical_sections_count)

        for pos in critical_sections_positions:
            nested_probability = random.random()
            resource = random.choice(all_resources)
            if (nested_probability < nesting_factor ** 2 and (pos + 2 < int(self.execution))
                    and resource.nestable and resource.nested_access):
                for q in all_resources:
                    if q.resource_id == resource.nested_access and q.nested_access:
                        self.execution_timeline[pos] = ('Critical', 2, resource.resource_id)
                        self.execution_timeline[pos + 1] = ('Nested', 2, resource.nested_access)
                        self.execution_timeline[pos + 2] = ('Nested', 2, q.nested_access)
            elif (nested_probability < 2 * nesting_factor * (1 - nesting_factor) and (pos + 1 < int(self.execution))
                  and resource.nestable and resource.nested_access):
                self.execution_timeline[pos] = ('Critical', 1, resource.resource_id)
                self.execution_timeline[pos + 1] = ('Nested', 1, resource.nested_access)
            else:
                self.execution_timeline[pos] = ('Critical', 0, resource.resource_id)

    def is_runnable(self):
        return self.state in ['preemptable', 'non-preemptable']

    def resume(self):
        self.state = 'preemptable'

    @staticmethod
    def assign_priorities(tasks):
        sorted_tasks = sorted(tasks, key=lambda x: (x.deadline, x.id))
        for priority, task in enumerate(sorted_tasks):
            task.priority = priority + 1
        return sorted_tasks

    def __repr__(self):
        critical_sections_str = ', '.join([f"{status}({level}, res={chosen_resource})"
                                           for status, level, chosen_resource in self.execution_timeline])
        return (f"Task(id={self.id}, execution_timeline=[{critical_sections_str}])\n and"
                f"Task(id={self.id}, arrival={self.arrival}, execution={self.execution}, deadline={self.deadline})")


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


def GSN_EDF_scheduler(all_tasks, all_processors, all_resources):
    # Implement the algorithm by iterating over the jobs and processors
    # Use the states and conditions from your pseudocode
    pass


generated_tasks = generate_tasks(100, 0.5)
print(generated_tasks[:5])
resources = create_resources(10)
for res in resources:
    print(res)
f = 0.05
for task in generated_tasks[:5]:
    task.generate_critical_sections(f, resources)
    print(task)
processors = [Processor(i) for i in range(1, 5)]
print(processors)
generated_tasks = Task.assign_priorities(generated_tasks)
GSN_EDF_scheduler(generated_tasks, processors, resources)
