# MIT 6.034 Lab 8: Bayesian Inference
# Written by 6.034 staff

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    
    agenda = [net.get_parents(var)]
    ancestors = set()
    
    while agenda:
        parents = agenda.pop()
        
        for parent in parents:
            ancestors.add(parent)
            agenda.append(net.get_parents(parent))
        
    return ancestors

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    
    agenda = [net.get_children(var)]
    descendants = set()
    
    while agenda:
        children = agenda.pop()
        
        for child in children:
            descendants.add(child)
            agenda.append(net.get_children(child))
            
    return descendants

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"

    return set(net.get_variables()) - get_descendants(net, var) - set(var)


#### Part 2: Computing Probability #############################################

#The Bayes net assumption
#Every variable in a Bayes net is conditionally independent of its 
#non-descendants, given its parents.

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    given_set = set(givens.keys())
    parents = net.get_parents(var)
    descendants = get_descendants(net,var)
    nondescendants = get_nondescendants(net, var)

    
    if parents.issubset(given_set) and given_set.isdisjoint(descendants):
        modified_givens = givens.copy()
        
        for key in given_set:
            if key in nondescendants and key not in parents:
                del modified_givens[key]
        
        return modified_givens        

    else:
        return givens
    

def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    
    #probability is not conditioned on anything, quick look up
    if givens == None:
        return net.get_probability(hypothesis)
    
    else: #probability is possibly conditioned
        var = list(hypothesis.keys())[0]
        simplified_givens = simplify_givens(net, var, givens)
        
        try:
            return net.get_probability(hypothesis, simplified_givens)
        
        except:
            raise LookupError

                        
def probability_joint(net, hypothesis):
    """
    Uses the chain rule to compute a joint probability
    assume that the hypothesis represents a valid joint probability
    """
    givens = hypothesis.copy()
    probabilities = []

    for var in net.topological_sort()[::-1]:
        if var in hypothesis:
            val = hypothesis[var]
            assert_var = {var: val}
            del givens[var]
            probabilities.append(probability_lookup(net, assert_var, givens))
    
    return product(probabilities)


def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"

    probabilities = []
    
    for combo in net.combinations(net.get_variables(), hypothesis):
        probabilities.append(probability_joint(net, combo))
        
    return sum(probabilities)

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    
    #base marginal probability
    if givens == None:
        return probability_marginal(net, hypothesis)
    
    #edge case, contradictory
    for var,val in hypothesis.items():
        if var in givens:
            if givens[var] != val:
                return 0.0
            
    #edge case, hypothesis is already given     
    if hypothesis == givens:
        return 1.0
        
    numerator_hypothesis = dict(hypothesis, **givens)
    denominator_hypothesis = numerator_hypothesis.copy()
    
    for key in hypothesis:
        del denominator_hypothesis[key]
    
    numerator = probability_marginal(net, numerator_hypothesis)
    denominator = probability_marginal(net, denominator_hypothesis)
    
    return numerator/ denominator
            
def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    
    return probability_conditional(net, hypothesis, givens)


#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    parameters = []
    
    for var in net.get_variables(): 
        dimensions = [] 
        
        for parent in net.get_parents(var):
            dimensions.append(len(net.get_domain(parent)))
            
        parameters.append((len(net.get_domain(var)) - 1) * product(dimensions))
        
    return sum(parameters)

#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
   
    for var1_domain_option in net.get_domain(var1):
        for var2_domain_option in net.get_domain(var2):
            assert_var_1 = {var1: var1_domain_option}
            assert_var_2 = {var2: var2_domain_option}
            probability_1 = probability(net, assert_var_1 , givens)
            
            #marginally independent
            if givens == None:
                probability_2 = probability(net, assert_var_1, assert_var_2 )

            #conditionally independent
            else:
                conditional_givens = dict(assert_var_2 , **givens)
                probability_2 = probability(net, assert_var_1, conditional_givens)
            
            #early return
            if not approx_equal(probability_1, probability_2):
                return False    
            
    #all parameters are independent
    return True
    
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    
    #1. Draw the ancestral graph.
    #includes all the variables mentioned in the probability expression
    ancestral_graph = set()
    
    ancestral_graph.add(var1)
    ancestral_graph.add(var2)
    
    var1_ancestors = get_ancestors(net, var1)
    var2_ancestors = get_ancestors(net, var2)

    ancestral_graph.update(var1_ancestors)
    ancestral_graph.update(var2_ancestors)    

    
    if givens != None:
        for given in givens.keys():
            ancestral_graph.add(given)
        
    connection_variables = set()
    
    for ancestor_1 in ancestral_graph:
        for ancestor_2 in ancestral_graph:
            path = net.find_path(ancestor_1, ancestor_2)
            
            if path:
                connection_variables.update(set(path))
     
    ancestral_graph.update(connection_variables)
            
    modified_net = net.subnet(ancestral_graph)
    
    
    #2. “Moralize” the ancestral graph by “marrying” the parents
    visited = set()
    
    for variable_1 in modified_net.get_variables():
        for variable_2 in modified_net.get_variables():
            
            if variable_1 == variable_2 or variable_1 in visited and variable_2 in visited:
                continue
                        
            variable_1_children = modified_net.get_children(variable_1)
            variable_2_children = modified_net.get_children(variable_2)
            
            #drawing an edge between pair of variables with a common child
            if not variable_1_children.isdisjoint(variable_2_children):
                modified_net.link(variable_1, variable_2)
            
            visited.add(variable_1)
            visited.add(variable_2)
                
    
    #3. "Disorient" the graph by replacing the directed edges (arrows) with 
    #undirected edges (lines).
    modified_net.make_bidirectional()
    
    #4. Delete the givens and their edges.
    if givens != None:
        for given in givens.keys():
            modified_net.remove_variable(given)
        
    #5. Read the answer off the graph.
    if modified_net.find_path(var1, var2) == None:
        return True
    
    return False


#### SURVEY ####################################################################

NAME = "Taylor Burke"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = "10"
WHAT_I_FOUND_INTERESTING = "d separation"
WHAT_I_FOUND_BORING = "Learning is never boring!"
SUGGESTIONS = "Nope! liked this lab, good hints and diagrams"
