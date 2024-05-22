import numpy as np
import random
import queue
import matplotlib.pyplot as plt
import sys
import json


class Processor:
    def __init__(self, processor_id):
        self.processor_id = processor_id
        self.is_idle = True
        self.current_task = None
        self.linked_task = None
        self.time_line_utilization = []

    def assign_task(self, assigned_task):
        self.current_task = assigned_task
        self.is_idle = False

    def complete_task(self):
        self.current_task = None
        self.is_idle = True

    def add_usage(self, usage_tuple):
        self.time_line_utilization.append(usage_tuple)

    def to_dict(self):
        return {'processor_id': self.processor_id, 'is_idle': self.is_idle,
                'current_task': self.current_task.id if self.current_task else None,
                'linked_task': self.linked_task.id if self.linked_task else None,
                'time_line_utilization': self.time_line_utilization}

    def __repr__(self):
        if self.is_idle:
            return f"Processor {self.processor_id} is idle"
        else:
            return f"Processor {self.processor_id} is executing {self.current_task.id}"


class Lock:
    def __init__(self, lock_id):
        self.id = lock_id
        self.queued_tasks = queue.Queue()
        self.main_queued_tasks = queue.Queue()
        self.current_task = None
        self.is_acquired = False

    def free_lock(self):
        self.is_acquired = False
        self.current_task = None

    def to_dict(self):
        return {
            'id': self.id,
            'current_task': self.current_task.id if self.current_task else None,
            'is_acquired': self.is_acquired,
            'queued_tasks': [task_list.id for task_list in list(self.queued_tasks.queue)]
        }

    def __repr__(self):
        return f"Lock ID is {self.id} and its queued tasks are:\n{self.queued_tasks}"


class Resource:
    def __init__(self, resource_id, nestable):
        self.resource_id = resource_id
        self.nestable = nestable
        self.nested_access = None

    def to_dict(self):
        return {'resource_id': self.resource_id, 'nestable': self.nestable,
                'nested_access': [res.resource_id for res in self.nested_access] if self.nested_access else None}

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
        self.start = None
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
        self.current_lock_group = None

    def check_valid_critical_assignment(self, position_index, nesting_degree_space):
        for i in range(nesting_degree_space):
            if self.execution_timeline[position_index + i][0] != 'Non-Critical':
                return False
        return True

    def generate_critical_sections(self, nesting_factor):
        if int(self.execution) <= 2:
            self.execution_timeline = [('Non-Critical', 0, "None")] * int(self.execution)
        else:
            critical_sections_count = random.randint(0, 8)
            self.execution_timeline = [('Non-Critical', 0, "None")] * int(self.execution)
            if critical_sections_count > self.execution - 2:
                critical_sections_count = int(self.execution) - 2
            critical_sections_positions = random.sample(range(1, int(self.execution) - 1), critical_sections_count)
            for i in range(critical_sections_count):
                for pos in critical_sections_positions:
                    nested_probability = random.random()
                    if (nested_probability < nesting_factor ** 2 and self.execution - 1 >= pos + 4
                            and self.check_valid_critical_assignment(pos, 4)):
                        self.execution_timeline[pos] = ('Critical', 2, "group1")
                        self.execution_timeline[pos + 1] = ('Critical', 2, "group1")
                        self.execution_timeline[pos + 2] = ('Critical', 2, "group1")
                        self.execution_timeline[pos + 3] = ('Critical', 2, "group1")
                    elif (nested_probability < 2 * nesting_factor * (1 - nesting_factor) and
                          self.execution - 1 >= pos + 3 and self.check_valid_critical_assignment(pos, 3)):
                        self.execution_timeline[pos] = ('Critical', 1, "group2")
                        self.execution_timeline[pos + 1] = ('Critical', 1, "group2")
                        self.execution_timeline[pos + 2] = ('Critical', 1, "group2")
                    elif self.execution - 1 >= pos + 2 \
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
        return sorted_tasks

    def to_dict(self):
        return {
            'id': self.id,
            'arrival': self.arrival,
            'execution': self.execution,
            'deadline': self.deadline,
            'period': self.period,
            'start': self.start,
            'finish': self.finish,
            'execution_timeline': self.execution_timeline,
            'linked': self.linked,
            'scheduled': self.scheduled,
            'cpu_name': self.cpu_name,
            'is_finished': self.is_finished,
            'priority': self.priority,
            'current_execution': self.current_execution,
            'state': self.state,
            'current_lock_group': self.current_lock_group
            if self.current_lock_group is None else self.current_lock_group.id
        }

    def __repr__(self):
        critical_sections_str = ', '.join([f"{status}({level}, res={chosen_resource})"
                                           for status, level, chosen_resource in self.execution_timeline])
        return (f"Task(id={self.id}, priority={self.priority}, execution_timeline=[{critical_sections_str}])\n and "
                f"arrival={self.arrival}, execution={self.execution}, deadline={self.deadline}), "
                f"start={self.start}, finish={self.finish}\n")


def UUniFast(n, u_bar):
    sum_u = u_bar
    vect_u = np.zeros(n)
    for i in range(n - 1):
        next_sum_u = sum_u * (np.random.rand() ** (1.0 / (n - i)))
        vect_u[i] = sum_u - next_sum_u
        sum_u = next_sum_u
    vect_u[n - 1] = sum_u
    return vect_u


def generate_tasks(num_tasks, u_bar):
    utilization = UUniFast(num_tasks, u_bar)
    tasks = []
    for i in range(num_tasks):
        arrival = np.random.randint(0, 100)
        max_execution_time = 1300
        execution = np.ceil(utilization[i] * max_execution_time)
        buffer = np.random.randint(5, 150)
        deadline = arrival + execution + buffer
        tasks.append(Task(f"Task_{i + 1}", arrival, execution, deadline))
    return tasks


def find_ready_executing_tasks(all_tasks, timer, all_processors):
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
            i.current_task.linked = False
            i.current_task.scheduled = False
            for each_lock in all_locks:
                if (each_lock.current_task is not None) and each_lock.current_task.id == i.current_task.id:
                    each_lock.free_lock()
                    i.current_task.current_lock_group = None
            i.current_task.state = "preemptable"
            i.current_task.finish = timer + 1
            print(f"Inside UTPR: {i.current_task}")
            i.complete_task()


def peek(queue_p):
    print(queue_p.qsize())
    if not queue_p.empty():
        return queue_p.queue[0]
    return None


def handle_critical_sections(all_processors, all_locks, timer):
    for i in all_processors:
        if not i.is_idle:
            print("Enter to handle critical sections:")
            print(f"task et before anything: {i.current_task.current_execution}")
            if i.current_task.execution_timeline[i.current_task.current_execution][0] == 'Non-Critical':
                print(f"{i.current_task.id} is not critical now.")
                i.current_task.current_execution += 1
                i.current_task.state = "preemptable"
                i.add_usage((f"{i.current_task.id[5:]}", timer, timer + 1))
                for j in all_locks:
                    if (j.current_task is not None) and j.current_task.id == i.current_task.id:
                        j.free_lock()
                        i.current_task.current_lock_group = None
            elif i.current_task.execution_timeline[i.current_task.current_execution][0] == "Critical":
                print(f"{i.current_task.id} is critical now.")
                cs, nes_level, group_lock = i.current_task.execution_timeline[i.current_task.current_execution]
                for j in all_locks:
                    if j.id == group_lock and (not j.is_acquired):
                        if i.current_task.current_lock_group is None:
                            print(f"{i.current_task.id} can take the lock {j.id}.")
                            if j.main_queued_tasks.qsize() == 0:
                                j.is_acquired = True
                                j.current_task = i.current_task
                                i.current_task.state = "non-preemptable"
                                i.current_task.current_execution += 1
                                i.current_task.current_lock_group = j.id
                                i.add_usage((f"{i.current_task.id[5:]}", timer, timer + 1))
                            elif peek(j.main_queued_tasks).id == i.current_task.id:
                                j.is_acquired = True
                                j.current_task = i.current_task
                                i.current_task.state = "non-preemptable"
                                i.current_task.current_execution += 1
                                i.current_task.current_lock_group = j.id
                                i.add_usage((f"{i.current_task.id[5:]}", timer, timer + 1))
                                print(f"Inside checking locks {j.main_queued_tasks.qsize()} and {j.id}")
                                j.main_queued_tasks.get()
                                print(f"Inside Checking locks after dequeue {j.main_queued_tasks.qsize()} and {j.id}")
                            else:
                                j.queued_tasks.put(i.current_task)
                                if i.current_task not in list(j.main_queued_tasks.queue):
                                    j.main_queued_tasks.put(i.current_task)
                                i.current_task.state = "non-preemptable"
                                i.add_usage((f"B{i.current_task.id[5:]}", timer, timer + 1))
                            break
                        else:
                            for w in all_locks:
                                if w.id == i.current_task.current_lock_group:
                                    i.current_task.current_lock_group = None
                                    w.free_lock()
                                    if j.main_queued_tasks.qsize() == 0:
                                        j.is_acquired = True
                                        j.current_task = i.current_task
                                        i.current_task.state = "non-preemptable"
                                        i.current_task.current_execution += 1
                                        i.current_task.current_lock_group = j.id
                                        i.add_usage((f"{i.current_task.id[5:]}", timer, timer + 1))
                                    elif peek(j.main_queued_tasks).id == i.current_task.id:
                                        j.is_acquired = True
                                        j.current_task = i.current_task
                                        i.current_task.state = "non-preemptable"
                                        i.current_task.current_execution += 1
                                        i.current_task.current_lock_group = j.id
                                        i.add_usage((f"{i.current_task.id[5:]}", timer, timer + 1))
                                        print(f"Inside checking locks second condition"
                                              f" {j.main_queued_tasks.qsize()} and {j.id}")
                                        j.main_queued_tasks.get()
                                        print(f"Inside Checking locks after dequeue"
                                              f" {j.main_queued_tasks.qsize()} and {j.id}")
                                    else:
                                        j.queued_tasks.put(i.current_task)
                                        if i.current_task not in list(j.main_queued_tasks.queue):
                                            j.main_queued_tasks.put(i.current_task)
                                        i.current_task.state = "non-preemptable"
                                        i.add_usage((f"B{i.current_task.id[5:]}", timer, timer + 1))
                                    break
                            break

                    elif j.id == group_lock and j.is_acquired and j.current_task.id != i.current_task.id:
                        if i.current_task.current_lock_group is None:
                            print(f"{i.current_task.id} can not take the lock {j.id}")
                            print(f"the task that has the lock is: {j.current_task.id}")
                            j.queued_tasks.put(i.current_task)
                            if i.current_task not in list(j.main_queued_tasks.queue):
                                j.main_queued_tasks.put(i.current_task)
                            i.current_task.state = "non-preemptable"
                            i.add_usage((f"B{i.current_task.id[5:]}", timer, timer + 1))
                            break
                        else:
                            for w in all_locks:
                                if w.id == i.current_task.current_lock_group:
                                    i.current_task.current_lock_group = None
                                    w.free_lock()
                                    print(f"{i.current_task.id} can not take the lock {j.id}")
                                    print(f"the task that has the lock is: {j.current_task.id}")
                                    j.queued_tasks.put(i.current_task)
                                    if i.current_task not in list(j.main_queued_tasks.queue):
                                        j.main_queued_tasks.put(i.current_task)
                                    i.current_task.state = "non-preemptable"
                                    i.add_usage((f"B{i.current_task.id[5:]}", timer, timer + 1))
                                    break
                            break
                    elif j.id == group_lock and j.is_acquired and j.current_task.id == i.current_task.id:
                        i.current_task.current_execution += 1
                        i.add_usage((f"{i.current_task.id[5:]}", timer, timer + 1))
                        break
                print(f"task et after everything: {i.current_task.current_execution}")


def assign_and_execute(all_processors, four_high_priority_tasks, all_locks, timer):
    for i in all_processors:
        if len(four_high_priority_tasks) >= 1:
            if i.is_idle:
                i.assign_task(four_high_priority_tasks[0])
                four_high_priority_tasks[0].linked = True
                four_high_priority_tasks[0].scheduled = True
                if four_high_priority_tasks[0].start is None:
                    four_high_priority_tasks[0].start = timer
                four_high_priority_tasks = four_high_priority_tasks[1:]

    for j in all_processors:
        if len(four_high_priority_tasks) >= 1:
            if (not j.is_idle and j.current_task.state == "preemptable"
                    and four_high_priority_tasks[0].priority < j.current_task.priority):
                j.current_task.linked = False
                j.current_task.scheduled = False
                j.assign_task(four_high_priority_tasks[0])
                four_high_priority_tasks[0].linked = True
                four_high_priority_tasks[0].scheduled = True
                if four_high_priority_tasks[0].start is None:
                    four_high_priority_tasks[0].start = timer
                four_high_priority_tasks = four_high_priority_tasks[1:]

    for q in all_processors:
        if len(four_high_priority_tasks) >= 1:
            if (not q.is_idle and q.current_task.state == "non-preemptable"
                    and four_high_priority_tasks[0].priority < q.current_task.priority):
                q.current_task.linked = False
                four_high_priority_tasks[0].linked = True

    handle_critical_sections(all_processors, all_locks, timer)
    for i in all_processors:
        if (i.current_task is not None) and int(i.current_task.current_execution) == int(i.current_task.execution):
            i.current_task.is_finished = True
            print(f"Task got finished: {i.current_task.id}")
            for j in all_locks:
                if (j.current_task is not None) and j.current_task.id == i.current_task.id:
                    j.free_lock()
                    i.current_task.current_lock_group = None
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


def depict_the_scheduling(all_processors):
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    for u_list in all_processors:
        if u_list.processor_id == 1:
            list_1 = u_list.time_line_utilization
        if u_list.processor_id == 2:
            list_2 = u_list.time_line_utilization
        if u_list.processor_id == 3:
            list_3 = u_list.time_line_utilization
        if u_list.processor_id == 4:
            list_4 = u_list.time_line_utilization

    processor_tasks = {
        'Processor_1': list_1,
        'Processor_2': list_2,
        'Processor_3': list_3,
        'Processor_4': list_4
    }

    fig, gnt = plt.subplots()

    gnt.set_xlabel('Time')
    gnt.set_ylabel('Processor')

    gnt.grid(True)

    y_labels = []
    y_ticks = []
    for i, (processor, tasks) in enumerate(processor_tasks.items(), 1):
        y_labels.append(processor)
        y_ticks.append(i)
    gnt.set_yticks(y_ticks)
    gnt.set_yticklabels(y_labels)
    gnt.invert_yaxis()

    for i, (processor, tasks) in enumerate(processor_tasks.items(), 1):
        print(tasks)
        for each_task, start, end in tasks:
            if each_task != 'idle':
                gnt.broken_barh([(start, end - start)], (i - 0.4, 0.8), facecolors='skyblue', edgecolors='black')
                mid_point = start + (end - start) / 2
                gnt.text(mid_point, i, each_task, ha='center', va='center', color='black')
            else:
                gnt.broken_barh([(start, end - start)], (i - 0.4, 0.8), facecolors='white', edgecolors='black',
                                linewidth=0.5)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', edgecolor='black', label='Task'),
                       Patch(facecolor='white', edgecolor='black', label='Idle')]
    gnt.legend(handles=legend_elements, loc='upper right')

    max_time = max(end for tasks in processor_tasks.values() for _, _, end in tasks)
    gnt.set_xlim(0, max_time)

    plt.show()


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


def save_to_json(processors_p, locks_p, tasks_p, resources_p, filename):
    data = {
        'processors': [p.to_dict() for p in processors_p],
        'resources': [r.to_dict() for r in resources_p],
        'locks': [q.to_dict() for q in locks_p],
        'tasks': [t.to_dict() for t in tasks_p]
    }
    try:
        with open(filename, 'w') as fq:
            json.dump(data, fq, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving data to {filename}: {e}")


def calculate_qos(tasks):
    qos_data = []
    for tasks_list in tasks:
        lateness = tasks_list.finish - tasks_list.deadline
        if lateness <= 0:
            qos = 100
        else:
            qos = max(0, 100 - lateness * 5)
        qos_data.append((tasks_list.id, qos))
    return qos_data


def plot_qos(qos_data, filename):
    task_ids = [data[0] for data in qos_data]
    qos_values = [data[1] for data in qos_data]
    plt.figure(figsize=(10, 5))
    plt.bar(task_ids, qos_values, color='green')
    plt.xlabel('Task IDs')
    plt.ylabel('Quality of Service (%)')
    plt.title('Quality of Service by Task')
    plt.ylim(0, 100)  # QoS ranges from 0 to 100
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def run_scheduler_and_save_results(tasks, processors_list, locks_list,
                                   resources_list, output_file_name, qos_output_image, json_output_file):
    with open(output_file_name, 'w') as output_file:
        print("\nEntering processing:", file=output_file)
        original_stdout = sys.stdout
        sys.stdout = output_file
        GSN_EDF_scheduler(tasks, processors_list, locks_list)
        sys.stdout = original_stdout
        print("\nEnd of processing:\n", file=output_file)

        sorted_tasks_by_arrival = sorted(tasks, key=lambda task_list: task_list.arrival)

        print(len(sorted_tasks_by_arrival), file=output_file)
        print(len(sorted_tasks_by_arrival))
        for task_lists in sorted_tasks_by_arrival:
            print(task_lists, file=output_file)
            print(task_lists)
        output_file.flush()

        for processor in processors_list:
            print(processor.time_line_utilization)

        save_to_json(processors_list, locks_list, tasks, resources_list, json_output_file)
        qos_data = calculate_qos(tasks)
        plot_qos(qos_data, qos_output_image)
        sys.stdout = output_file
        depict_the_scheduling(processors_list)
        sys.stdout = original_stdout


generated_tasks_l1 = generate_tasks(100, 0.5)
generated_tasks_l2 = generate_tasks(100, 0.75)
generated_tasks_l3 = generate_tasks(100, 0.5)
generated_tasks_l4 = generate_tasks(100, 0.75)
resources_l1 = create_resources(10)
resources_l2 = create_resources(10)
resources_l3 = create_resources(10)
resources_l4 = create_resources(10)
f_r1 = 0.05
f_r2 = 0.05
f_r3 = 0.08
f_r4 = 0.08
for task_l1 in generated_tasks_l1[:100]:
    task_l1.generate_critical_sections(f_r1)
for task_l2 in generated_tasks_l2[:100]:
    task_l2.generate_critical_sections(f_r2)
for task_l3 in generated_tasks_l3[:100]:
    task_l3.generate_critical_sections(f_r3)
for task_l4 in generated_tasks_l4[:100]:
    task_l4.generate_critical_sections(f_r4)
processors_l1 = [Processor(i) for i in range(1, 5)]
processors_l2 = [Processor(i) for i in range(1, 5)]
processors_l3 = [Processor(i) for i in range(1, 5)]
processors_l4 = [Processor(i) for i in range(1, 5)]
generated_tasks_l1 = Task.assign_priorities(generated_tasks_l1)
generated_tasks_l2 = Task.assign_priorities(generated_tasks_l2)
generated_tasks_l3 = Task.assign_priorities(generated_tasks_l3)
generated_tasks_l4 = Task.assign_priorities(generated_tasks_l4)
locks_l1 = create_lock()
locks_l2 = create_lock()
locks_l3 = create_lock()
locks_l4 = create_lock()

run_scheduler_and_save_results(generated_tasks_l1, processors_l1, locks_l1, resources_l1,
                               'first_scheduling.txt', 'first_qos.png', 'first_data.json')
run_scheduler_and_save_results(generated_tasks_l2, processors_l2, locks_l2, resources_l2,
                               'second_scheduling.txt', 'second_qos.png', 'second_data.json')
run_scheduler_and_save_results(generated_tasks_l3, processors_l3, locks_l3, resources_l3,
                               'third_scheduling.txt', 'third_qos.png', 'third_data.json')
run_scheduler_and_save_results(generated_tasks_l4, processors_l4, locks_l4, resources_l4,
                               'fourth_scheduling.txt', 'fourth_qos.png', 'fourth_data.json')
