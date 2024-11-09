import random
from schedule import Schedule
import copy

RESTARTS = 50
NUMBER_OF_ITERATIONS = 200000
def solve(schedule):
    """
    Your solution of the problem
    Local search algorithm with restarts
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    start_state = None
    for _ in range(RESTARTS):
        state = local_search(schedule)
        if start_state is None or get_solution_cost(state) < get_solution_cost(start_state):
            start_state = state
        print(f"Current best solution: {get_solution_cost(start_state)}")
    return start_state

def local_search(schedule: Schedule) -> dict:
    """
    Local search algorithm
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot.
    """
    random_course_list = random.sample(list(schedule.course_list), len(schedule.course_list))
    best_schedule = {c: i for i, c in enumerate(random_course_list)}
    
    for _ in range(NUMBER_OF_ITERATIONS):
        neighbors = generate_neighbors(best_schedule, schedule)
        if not neighbors:
            break

        best_neighbor = select_neighbor(neighbors)
        if get_solution_cost(best_neighbor) < get_solution_cost(best_schedule):
            best_schedule = best_neighbor

    return best_schedule

def get_solution_cost(proposed_schedule: dict) -> int:
    """
    Return the number of time slots used by a proposed_schedule
    :param proposed_schedule: a dictionnary where the keys are the list of the courses and the values are the time periods associated
    :return: an int containing the number of used time slots
    """
    return len(set(proposed_schedule.values()))

def generate_neighbors(proposed_schedule: dict, schedule: Schedule) -> list[dict]: 
    """
    Generate neighbors of a solution
    :param proposed_schedule: a dictionnary where the keys are the list of the courses and the values are the time periods associated
    :param schedule: object describing the input
    :return: a list of dictionaries representing the neighbors
    """
    available_time_slots = list(proposed_schedule.values())
    least_used_time_slot = min(available_time_slots, key=available_time_slots.count)
    time_slots = set(available_time_slots)
    time_slots.remove(least_used_time_slot)
    neighbors = []
    reassigned_courses = [course for course in proposed_schedule if proposed_schedule[course] == least_used_time_slot]

    for time_slot in time_slots:
        can_reassign = True
        neighbor = copy.copy(proposed_schedule)
        for course in reassigned_courses:
            if move_has_conflicts(schedule=schedule, course=course, proposed_schedule=neighbor, time_slot=time_slot):
                can_reassign = False
                break

            neighbor[course] = time_slot

        if can_reassign:
            neighbors.append(neighbor)
            
    return neighbors

def move_has_conflicts(schedule: Schedule, course: str, proposed_schedule: dict, time_slot: int) -> bool:
    """
    Return True if the move has conflicts, False otherwise
    :param course: a course
    :param time_slot: the new time slot
    :param proposed_schedule: a dictionnary where the keys are the list of the courses and the values are the time periods associated
    :param schedule: object describing the input
    :return: True if the move has conflicts, False otherwise
    """
    for conflicting_course in schedule.get_node_conflicts(course):
        if proposed_schedule[conflicting_course] == time_slot:
            return True
        
    return False 

def select_neighbor(neighbors: list[dict]) -> dict:
    """
    Return the neighbor with the minimum number of time slots
    Args:
        neighbors: a list of dictionaries representing the neighbors
    Returns:
        the neighbor with the minimum number of time slots
    """
    return min(neighbors, key=lambda solution: get_solution_cost(solution))