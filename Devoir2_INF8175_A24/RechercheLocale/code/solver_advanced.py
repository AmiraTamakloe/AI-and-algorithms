import random
from schedule import Schedule
import copy
from collections import Counter

RESTARTS = 45
NUMBER_OF_ITERATIONS = 30000
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
        if start_state is None or get_schedule_cost(state) < get_schedule_cost(start_state):
            start_state = state
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
        neighbors = define_neighbors(best_schedule, schedule)
        if not neighbors:
            break

        best_neighbor = select_neighbor(neighbors)
        if get_schedule_cost(best_neighbor) < get_schedule_cost(best_schedule):
            best_schedule = best_neighbor

    return best_schedule

def get_schedule_cost(proposed_schedule: dict) -> int:
    """
    Return the number of time slots used by a proposed_schedule
    :param proposed_schedule: a dictionnary where the keys are the list of the courses and the values are the time periods associated
    :return: an int containing the number of used time slots
    """
    return len(set(proposed_schedule.values()))

def define_neighbors(proposed_schedule: dict, schedule: Schedule) -> list[dict]: 
    """
    Generate neighbors of a schedule
    :param proposed_schedule: a dictionnary where the keys are the list of the courses and the values are the time periods associated
    :param schedule: object describing the input
    :return: a list of dictionaries representing the neighbors
    """
    # Find the least-used time slot
    time_slot_counts = Counter(proposed_schedule.values())
    least_used_time_slot = min(time_slot_counts, key=time_slot_counts.get)
    
    # Identify courses in the least-used time slot
    courses_to_reassign = [
        course for course, slot in proposed_schedule.items() if slot == least_used_time_slot
    ]
    
    # Get other time slots to consider
    other_time_slots = set(time_slot_counts.keys()) - {least_used_time_slot}
    neighbors = []

    # Generate neighbors by reassigning each course to other time slots if no conflicts
    for course in courses_to_reassign:
        for time_slot in other_time_slots:
            # Check for conflicts only once per time slot
            if not schedule_has_conflicts(schedule, course, proposed_schedule, time_slot):
                # Create a neighbor by reassigning the course
                neighbor_schedule = copy.copy(proposed_schedule)
                neighbor_schedule[course] = time_slot
                neighbors.append(neighbor_schedule)
                break  # Only add the first valid reassignment for each course
    
    return neighbors

def schedule_has_conflicts(schedule: Schedule, course: str, proposed_schedule: dict, time_slot: int) -> bool:
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
    return min(neighbors, key=lambda schedule: get_schedule_cost(schedule))