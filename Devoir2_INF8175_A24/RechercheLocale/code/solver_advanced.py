def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """

    solution = dict()
    courses_dict = dict() #Associe chaque cours a une liste d'autres cours en conflit avec lui

    for course1, course2 in schedule.conflict_list: #Parcours chaque paire de cours en conflit
        if course1 not in courses_dict: #Si le cours a gauche n'est pas dans courses_dict, il est rajouté
            courses_dict[course1] = []
        if course2 not in courses_dict: #Si le cours a droite n'est pas dans courses_dict, il est rajouté
            courses_dict[course2] = []
        
        courses_dict[course1].append(course2) #Ajoute course2 dans la liste des cours en conflit de course1
        courses_dict[course2].append(course1) #Ajoute course1 dans la liste des cours en conflit de course2

    
    for course in courses_dict:
        time_slots_taken = set() #Contient les créneaux horaires deja pris par des cours en conflit avec le cours actuel
        for neighbors in courses_dict[course]: #Parcours chaque cours dans le dictionnaire
        #Si le cours en conflit à déja un créneau attribué, alors il est rajouté à time_slots_taken
            if neighbors in solution: 
                time_slots_taken.add(solution[neighbors])
        
        time_slot = 1
        while time_slot in time_slots_taken: #Tant que le créneau horaire est deja pris par un des cours en conflit
            time_slot += 1 #on incrémente pour trouver le prochain céneau libre
        solution[course] = time_slot #Attribue le premier créneau horaire disponible au cours actuel dans solution
    
    return solution
