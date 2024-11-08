def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """

    solution = dict()
    courses_dict = dict()

    for course1, course2 in schedule.conflict_list:
        if course1 not in courses_dict:
            courses_dict[course1] = []
        if course2 not in courses_dict:
            courses_dict[course2] = []
        
        courses_dict[course1].append(course2)
        courses_dict[course2].append(course1)
    
    sorted_courses = sorted(courses_dict.keys(), key=lambda c: len(courses_dict[c]), reverse=True)

    for course in sorted_courses:  
        time_slots_taken = set()
        for neighbors in courses_dict[course]:
            if neighbors in solution:
                time_slots_taken.add(solution[neighbors])
        
        time_slot = 1
        while time_slot in time_slots_taken:
            time_slot += 1
        solution[course] = time_slot
    
    return solution
