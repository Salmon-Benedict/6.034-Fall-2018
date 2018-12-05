# MIT 6.034 Lab 4: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem
from itertools import combinations


#### Part 1: Warmup ############################################################

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    
    for var in csp.get_all_variables():
        if len(csp.get_domain(var)) == 0:
            return True
    return False
        
def check_all_constraints(csp):
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    
    assignments = csp.assignments
    constraints = csp.get_all_constraints()
    
    for constraint in constraints:
        val1 = assignments.get(constraint.var1, None)
        val2 = assignments.get(constraint.var2, None)
        if val1 != None and val2 != None and constraint.check(val1,val2) == False:
            return False
    return True


#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(given_problem) :
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    #Initialize agenda and the extension count.
    agenda = [given_problem]
    ext_count = 0
    

    while agenda:
        #Until the agenda is empty, pop the first problem off the list and 
        #increment the extension count
        ext_count += 1
        problem = agenda.pop(0)
        
        #If any variable's domain is empty or if any constraints are violated, 
        #the problem is unsolvable with the current assignments.
        if not check_all_constraints(problem) or has_empty_domains(problem):
            continue 

        #If none of the constraints have been violated, check whether the 
        #problem has any unassigned variables. 
        #If not, you've found a complete solution!
        if check_all_constraints(problem):
            if len(problem.unassigned_vars) == 0:
                return (problem.assignments, ext_count)
     
            else: 
                #However, if the problem has some unassigned variables:
                #Take the first unassigned variable off the list 
                var = problem.pop_next_unassigned_var()
                new_problems = []
                
                #For each value in the variable's domain, create a new problem 
                #with that value assigned to the variable, and add it to a list 
                #of new problems. 
                for val in problem.get_domain(var):
                    new_problem = problem.copy()
                    new_problem.set_assignment(var, val)
                    new_problems.append(new_problem) 
  
                #Then, add the new problems to the appropriate end of the agenda.
                agenda = new_problems + agenda
                
    return (None, ext_count)


# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?
#pokemon_problem = get_pokemon_problem()
#print (solve_constraint_dfs(pokemon_problem))
#print ("here")

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

ANSWER_1 = 20


#### Part 3: Forward Checking ##################################################

#One problem with the solve_constraint_dfs algorithm is that it explores all 
#possible branches of the tree. We can use a trick called forward checking to 
#avoid exploring branches that cannot possibly lead to a solution: each time 
#we assign a value to a variable, we'll eliminate incompatible or inconsistent 
#values from that variable's neighbors.

def violations(csp,V,W,v,w):
    """
    a helper function that takes in two variables V and W, and two values v 
    and w in their respective domains, and checks if there are any constraint 
    violations between V=v and W=w.
    """

    return [constraint.check(v,w) 
            for constraint in csp.constraints_between(V,W)]
    
    
def eliminate_from_neighbors(csp, var) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
#Suppose V is a variable with neighbor W. If W's domain contains a value w 
#which violates a constraint with every value in V's domain, then the 
#assignment W=w can't be part of the solution we're constructing — we can safely 
#eliminate w from W's domain.   
#don't eliminate values from a variable while iterating over its domain
    
    neighbors = csp.get_neighbors(var)
    reduced_neighbors = set()
    
    for neighbor in neighbors:
        violators = []
        for neighbor_val in csp.get_domain(neighbor):
            counter = 0
            for var_val in csp.get_domain(var):
                if all(violations(csp,var,neighbor,var_val,neighbor_val)):
                    counter += 1

            if counter ==0:
                violators.append(neighbor_val)

        if len(violators) > 0: 
            for violator in violators:
                csp.eliminate(neighbor,violator)
                reduced_neighbors.add(neighbor)
                
            if not(csp.get_domain(neighbor)):
                return None

#    return an alphabetically sorted list of the neighbors whose domains were 
#    reduced (i.e. which had values eliminated from their domain), with each 
#    neighbor appearing at most once in the list. If no domains were reduced, 
#    return an empty list

    return sorted(list(reduced_neighbors))
                                

# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def solve_constraint_forward_checking(given_problem) :
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
#The implementation for this function will be very similar to that of 
#solve_constraint_dfs, except now the solver must apply forward checking 
#(eliminate_from_neighbors) after each assignment, to eliminate incompatible 
#values from the assigned variable's neighbors.


#lack of comments because it is the same as the other solve constraint but 
    #only impliments a line of code to do some forward checking
    agenda = [given_problem]
    ext_count = 0

    while agenda:
        ext_count += 1
        problem = agenda.pop(0)
        

        if not check_all_constraints(problem) or has_empty_domains(problem):
            continue 

        if check_all_constraints(problem):
            if len(problem.unassigned_vars) == 0:
                return (problem.assignments, ext_count)
     
            else: 
              
                var = problem.pop_next_unassigned_var() 
                new_problems = []
               
                for val in problem.get_domain(var):
                    new_problem = problem.copy()
                    new_problem.set_assignment(var, val)
                    #after each assignment, to eliminate incompatible values 
                    #from the assigned variable's neighbors.
                    forward_check(new_problem, var)
                    new_problems.append(new_problem) 
  
                agenda = new_problems + agenda
                
    return (None, ext_count)



# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?
    
#pokemon_problem = get_pokemon_problem()
#print (solve_constraint_forward_checking(pokemon_problem))

ANSWER_2 = 9


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
#    Establish a queue. If using domain reduction during search, this queue 
#   should initially contain only the variable that was just assigned. If before 
#   search (or if no queue is specified), the queue can contain all variables in 
#   the problem. (Hint: csp.get_all_variables() will make a copy of the variables
#   list.)
    dequeued = []
    if queue is None:
        queue = csp.get_all_variables()
   
    while queue:
        #Until the queue is empty, pop the first variable var off the queue.
        var = queue.pop(0)
        dequeued.append(var)

        #Iterate over that var's neighbors: if some neighbor n has values that
        #are incompatible with the constraints between var and n, remove the
        #incompatible values from n's domain. If you reduce a neighbor's domain,
        #add that neighbor to the queue (unless it's already in the queue).
        to_eliminate = forward_check(csp, var)
        if to_eliminate:
            for eliminated_neighbor in to_eliminate:
                if eliminated_neighbor not in queue:
                    queue.append(eliminated_neighbor)
                
        #If any variable has an empty domain, quit immediately and return None.
        elif to_eliminate is None:
            return None        
    
    return dequeued
        
        

# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?

#pokemon_problem = get_pokemon_problem()
#domain_reduction(pokemon_problem, queue=None)
#print (solve_constraint_forward_checking(pokemon_problem))


ANSWER_3 = 6


def solve_constraint_propagate_reduced_domains(given_problem) :
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """

#lack of comments because it is the same as the previous forward chaining 
    #constraint function but instead has 1extra line of code that impliments 
    #the reduced domains


    agenda = [given_problem]
    ext_count = 0
    

    while agenda:
        ext_count += 1
        problem = agenda.pop(0)

        if not check_all_constraints(problem) or has_empty_domains(problem):
            continue 

        if check_all_constraints(problem):
            if len(problem.unassigned_vars) == 0:
                return (problem.assignments, ext_count)
     
            else: 
                
                var = problem.pop_next_unassigned_var()
                new_problems = []
                
                for val in problem.get_domain(var):
                    new_problem = problem.copy()
                    new_problem.set_assignment(var, val)
                    #propogation for reduced domains
                    domain_reduction(new_problem, [var])
                    forward_check(new_problem, var)
                    new_problems.append(new_problem) 
  
                agenda = new_problems + agenda
                
    return (None, ext_count)



# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?

ANSWER_4 = 7


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
#The domain_reduction procedure is comprehensive, but expensive: it eliminates
# as many values as possible, but it continually adds more variables to the 
# queue. As a result, it is an effective algorithm to use before solving a 
# constraint satisfaction problem, but is often too expensive to call
# repeatedly during search.

#Instead of comprehensively reducing all the domains in a problem, as
# domain_reduction does, you can instead reduce only some of the domains.
# This idea underlies propagation through singleton domains — a reduction
# algorithm which does not detect as many dead ends, but which is significantly 
# faster.

#Propagation through singletons is like propagation through reduced domains, 
#except that variables must pass a test in order to be added to the queue:

    dequeued = []
    if queue is None:
        queue = csp.get_all_variables()
                
    while queue:
        var = queue.pop(0)
        dequeued.append(var)

        to_eliminate = forward_check(csp, var)
        if to_eliminate:
            for eliminated_neighbor in to_eliminate:
                #adding it if it is new to the queue and if it passes the given 
                #queue condition (as specified below)
                if eliminated_neighbor not in queue and enqueue_condition_fn(csp, eliminated_neighbor):
                    queue.append(eliminated_neighbor)
                
        elif to_eliminate is None:
            return None        
    
    return dequeued


def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    
#    domain reduction / propagation through reduced domains: adds a neighboring 
#    variable to the queue if its domain has been reduced in size
    
    return True

    
def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
#    propagation through singleton domains: adds a neighboring variable to the 
#   queue if its domain has exactly one value in it
    
    if len(csp.get_domain(var)) == 1:
        return True
    return False
        
def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    
#    forward checking: never adds other variables to the queue
    
    return False


#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(given_problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
#lack of comments because it impliments DFS like the other constraint
    #problems but instead it has the checkpoint of the propogation function

    agenda = [given_problem]
    ext_count = 0
    
    while agenda:
        ext_count += 1
        problem = agenda.pop(0)

        if not check_all_constraints(problem) or has_empty_domains(problem):
            continue 

        if check_all_constraints(problem):
            if len(problem.unassigned_vars) == 0:
                return (problem.assignments, ext_count)
     
            else: 
                
                var = problem.pop_next_unassigned_var()
                new_problems = []
                
                for val in problem.get_domain(var):
                    new_problem = problem.copy()
                    new_problem.set_assignment(var, val)
                    #checking if there are any constraints we need to follow
                    if enqueue_condition is not None:
                        propagate(enqueue_condition, new_problem, [var])
                    new_problems.append(new_problem) 
  
                agenda = new_problems + agenda
       
    return (None, ext_count)


# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)

ANSWER_5 = 8


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    if abs(m-n) == 1:
        return True
    return False


def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    if abs(m-n) > 1 or m==n:
        return True
    return False

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    
#Instead, write a function that takes a list of variables and returns a list 
#containing, for each pair of variables, a constraint object requiring the 
#variables to be different from each other.
    
    combos = combinations(variables, 2)
    constraint_list = []
    for var1,var2 in combos:
        constraint_list.append(Constraint(var1,var2, constraint_different))
    return constraint_list


#### SURVEY ####################################################################

NAME = "Taylor"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = "14"
WHAT_I_FOUND_INTERESTING = "finally getting the hang of DFS"
WHAT_I_FOUND_BORING = "learning is never boring SMH"
SUGGESTIONS = "Keep it up with all the pokemon (this pset) and LoTR (quiz1) could add some DC and marvel too"
