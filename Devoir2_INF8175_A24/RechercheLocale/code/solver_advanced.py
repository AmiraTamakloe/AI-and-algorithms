

def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """

    solution = {} 
    courses = {}

    for c in schedule.course_list:
        courses[c] = (len(schedule.get_node_conflicts(c)), schedule.get_node_conflicts(c))
    
    courses_sorted_by_num_of_conflict = dict(sorted(courses.items(), key=lambda x:x[1][0], reverse=True))

    courses_in_conflict_by_slot = []
    for current_course, current_course_info in courses_sorted_by_num_of_conflict.items() :
        not_assigned = True
        start_arr = 0
        while not_assigned :
            if start_arr >= len(courses_in_conflict_by_slot):
                courses_in_conflict_by_slot.append([])
            if current_course in courses_in_conflict_by_slot[start_arr] :
                start_arr += 1
            else :
                solution[current_course] = start_arr + 1
                courses_in_conflict_by_slot[start_arr] += current_course_info[1] # adding conflicting class so they can't be assigned to this period
                not_assigned = False
    return solution