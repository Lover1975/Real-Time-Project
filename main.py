import numpy as np
import random
import queue


class Processor:
    def __init__(self, processor_id):
        self.processor_id = processor_id
        self.is_idle = True
        self.current_task = None
        self.linked_task = None

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


class Lock:
    def __init__(self, lock_id):
        self.id = lock_id
        self.queued_tasks = queue.Queue()
        self.current_task = None
        self.is_acquired = False

    def free_lock(self):
        self.is_acquired = False
        self.current_task = None

    def __repr__(self):
        return f"Lock ID is {self.id} and its queued tasks are:\n{self.queued_tasks}"


class Resource:
    def __init__(self, resource_id, nestable):
        self.resource_id = resource_id
        self.nestable = nestable
        self.nested_access = None

    def __repr__(self):
        if self.nestable and self.nested_access:
            return f"Resource {self.resource_id} (Nestable, can access: Resource {self.nested_access})"
        elif self.nestable:
            return f"Resource {self.resource_id} (Nestable, no nested access defined)"
        else:
            return f"Resource {self.resource_id} (Non-Nestable)"


def create_resources(num_resources):
    created_resources = [Resource(i + 1, random.choice([True, False])) for i in range(num_resources)]
    created_resources[0].nested_access = [created_resources[1], created_resources[2]]
    created_resources[3].nested_access = [created_resources[4]]
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
        self.is_finished = False
        self.priority = None
        self.current_execution = 0
        self.state = 'preemptable'

    def check_valid_critical_assignment(self, position_index, nesting_degree_space):
        for i in range(nesting_degree_space):
            if self.execution_timeline[position_index + i][0] != 'Non-Critical':
                return False
        return True

    def generate_critical_sections(self, nesting_factor):
        critical_sections_count = random.randint(0, 8)
        self.execution_timeline = [('Non-Critical', 0, "None")] * int(self.execution)
        if critical_sections_count > self.execution:
            critical_sections_count = int(self.execution)
        critical_sections_positions = random.sample(range(int(self.execution)), critical_sections_count)
        # print(critical_sections_positions)
        for i in range(critical_sections_count):
            for pos in critical_sections_positions:
                nested_probability = random.random()
                if (nested_probability < nesting_factor ** 2 and self.execution >= pos + 4
                        and self.check_valid_critical_assignment(pos, 4)):
                    self.execution_timeline[pos] = ('Critical', 2, "group1")
                    self.execution_timeline[pos + 1] = ('Critical', 2, "group1")
                    self.execution_timeline[pos + 2] = ('Critical', 2, "group1")
                    self.execution_timeline[pos + 3] = ('Critical', 2, "group1")
                elif nested_probability < 2 * nesting_factor * (1 - nesting_factor) and self.execution >= pos + 3 \
                        and self.check_valid_critical_assignment(pos, 3):
                    self.execution_timeline[pos] = ('Critical', 1, "group2")
                    self.execution_timeline[pos + 1] = ('Critical', 1, "group2")
                    self.execution_timeline[pos + 2] = ('Critical', 1, "group2")
                elif self.execution >= pos + 2 \
                        and self.check_valid_critical_assignment(pos, 2):
                    chosen_group = random.randint(5, 9)
                    self.execution_timeline[pos] = ('Critical', 0, f"group{chosen_group}")
                    self.execution_timeline[pos + 1] = ('Critical', 0, f"group{chosen_group}")

    def is_runnable(self):
        return self.state in ['preemptable', 'non-preemptable']

    def resume(self):
        self.state = 'preemptable'

    @staticmethod
    def assign_priorities(tasks):
        sorted_tasks = sorted(tasks, key=lambda x: (x.deadline, x.id))
        for priority, c_task in enumerate(sorted_tasks):
            c_task.priority = priority + 1
        # for q in sorted_tasks:
        #    print(q.priority)
        return sorted_tasks

    def __repr__(self):
        critical_sections_str = ', '.join([f"{status}({level}, res={chosen_resource})"
                                           for status, level, chosen_resource in self.execution_timeline])
        return (f"Task(id={self.id}, execution_timeline=[{critical_sections_str}])\n and "
                f"arrival={self.arrival}, execution={self.execution}, deadline={self.deadline}), "
                f"start={self.start}, finish={self.finish}")


def UUniFast(n, u_bar):
    sum_u = u_bar
    vect_u = np.zeros(n)
    for i in range(n - 1):
        next_sum_u = sum_u * (np.random.rand() ** (1.0 / (n - i)))
        vect_u[i] = sum_u - next_sum_u
        sum_u = next_sum_u
    vect_u[n - 1] = sum_u
    # print(vect_u)
    return vect_u


def generate_tasks(num_tasks, u_bar):
    utilizations = UUniFast(num_tasks, u_bar)
    tasks = []
    for i in range(num_tasks):
        arrival = np.random.randint(0, 300)
        max_execution_time = 500
        execution = np.ceil(utilizations[i] * max_execution_time)
        buffer = np.random.randint(5, 250)
        deadline = arrival + execution + buffer
        tasks.append(Task(f"Task_{i + 1}", arrival, execution, deadline))
    return tasks


def find_ready_executing_tasks(all_tasks, timer, all_processors):
    # print("hi")
    ready_tasks = []
    processors_tasks = []
    for i in all_processors:
        if i is not None:
            processors_tasks.append(i.current_task)
    for each_task in all_tasks:
        if each_task.arrival <= timer and (not each_task.is_finished):
            if each_task not in processors_tasks:
                ready_tasks.append(each_task)
    return ready_tasks, processors_tasks


def update_tasks_processors_resources(all_processors, all_locks, timer):
    for i in all_processors:
        if (i.current_task is not None) and i.current_task.is_finished:
            # print(f"Inside utpr: {i.current_task}")
            i.current_task.linked = False
            i.current_task.scheduled = False
            for each_lock in all_locks:
                if (each_lock.current_task is not None) and each_lock.current_task.id == i.current_task.id:
                    each_lock.free_lock()
            i.current_task.state = "preemptable"
            i.current_task.finish = timer + 1
            print(f"Inside utpr: {i.current_task}")
            i.complete_task()


def handle_critical_sections(all_processors, all_locks):
    for i in all_processors:
        if not i.is_idle:
            print("Enter to handle critical sections:")
            print(f"task et before anything: {i.current_task.current_execution}")
            if i.current_task.execution_timeline[i.current_task.current_execution][0] == 'Non-Critical':
                print(f"{i.current_task.id} is not critical now.")
                i.current_task.current_execution += 1
                i.current_task.state = "preemptable"
                for j in all_locks:
                    if (j.current_task is not None) and j.current_task.id == i.current_task.id:
                        j.free_lock()
            elif i.current_task.execution_timeline[i.current_task.current_execution][0] == "Critical":
                print(f"{i.current_task.id} is critical now.")
                cs, nes_level, group_lock = i.current_task.execution_timeline[i.current_task.current_execution]
                for j in all_locks:
                    if j.id == group_lock and (not j.is_acquired):
                        print(f"{i.current_task.id} can take the lock {j.id}.")
                        j.is_acquired = True
                        j.current_task = i.current_task
                        i.current_task.current_execution += 1
                        break
                    elif j.id == group_lock and j.is_acquired and j.current_task.id != i.current_task.id:
                        print(f"{i.current_task.id} can not take the lock {j.id}")
                        j.queued_tasks.put(i.current_task)
                        i.current_task.state = "non-preemptable"
                        break
                    else:
                        i.current_task.current_execution += 1
                        break
                print(f"task et after everything: {i.current_task.current_execution}")


def assign_and_execute(all_processors, four_high_priority_tasks, all_locks, timer):
    for i in all_processors:
        if len(four_high_priority_tasks) >= 1:
            if i.is_idle:
                i.assign_task(four_high_priority_tasks[0])
                four_high_priority_tasks[0].linked = True
                four_high_priority_tasks[0].scheduled = True
                four_high_priority_tasks[0].start = timer
                four_high_priority_tasks = four_high_priority_tasks[1:]
                # i.current_task.current_execution += 1
    for j in all_processors:
        if len(four_high_priority_tasks) >= 1:
            if not j.is_idle and j.current_task.state == "preemptable":
                j.current_task.linked = False
                j.current_task.scheduled = False
                j.assign_task(four_high_priority_tasks[0])
                four_high_priority_tasks[0].linked = True
                four_high_priority_tasks[0].scheduled = True
                four_high_priority_tasks[0].start = timer
                four_high_priority_tasks = four_high_priority_tasks[1:]
                # i.current_task.current_execution += 1
    for q in all_processors:
        if len(four_high_priority_tasks) >= 1:
            if not q.is_idle and q.current_task.state == "non-preemptable":
                q.current_task.linked = False
                four_high_priority_tasks[0].linked = True
                # i.current_task.current_execution += 1

        # elif i.current_task is not None:
        #     i.current_task.current_execution += 1
    handle_critical_sections(all_processors, all_locks)
    for i in all_processors:
        if (i.current_task is not None) and int(i.current_task.current_execution) == int(i.current_task.execution):
            i.current_task.is_finished = True
            print(f"Task got finished: {i.current_task.id}")
    update_tasks_processors_resources(all_processors, all_locks, timer)


def create_lock():
    all_locks = [Lock("group1"), (Lock("group2")), Lock("group5"), (Lock("group6")), Lock("group7"), (Lock("group8")),
                 Lock("group9")]
    return all_locks


def all_are_finished(all_tasks):
    for i in all_tasks:
        if not i.is_finished:
            return True
    return False


def GSN_EDF_scheduler(all_tasks, all_processors, all_locks):
    clock = 0
    while all_are_finished(all_tasks):
        print()
        print(f"Inside while and clock is {clock}")
        ready_tasks, processors_tasks = find_ready_executing_tasks(all_tasks, clock, all_processors)
        print(f"ready tasks are: {ready_tasks}\nand processors tasks are: {processors_tasks}")
        four_high_priority_tasks = sorted(ready_tasks, key=lambda x: x.priority)[0:4]
        print(f"4 HPTs at the moment are: {four_high_priority_tasks}")
        # update_tasks_processors_resources(all_processors, all_locks, clock)
        assign_and_execute(all_processors, four_high_priority_tasks, all_locks, clock)
        clock += 1
        #if clock == 50:
        #    break


generated_tasks = generate_tasks(100, 0.5)
resources = create_resources(10)
f = 0.05
for task in generated_tasks[:100]:
    task.generate_critical_sections(f)
    # print(task)
processors = [Processor(i) for i in range(1, 5)]
# print(processors)
generated_tasks = Task.assign_priorities(generated_tasks)
locks = create_lock()
print()
print()
print("Entering processing:")
GSN_EDF_scheduler(generated_tasks, processors, locks)
print()
print("End of processing:\n")
for w in generated_tasks:
    print(w)
