# MIT 6.034 Lab 1: Rule-Based Systems
# Written by 6.034 staff

from production import IF, AND, OR, NOT, THEN, DELETE, forward_chain, pretty_goal_tree
from data import *
import pprint

pp = pprint.PrettyPrinter(indent=1)
pprint = pp.pprint

#### Part 1: Multiple Choice #########################################

ANSWER_1 = '2'

ANSWER_2 = '4'

ANSWER_3 = '2'

ANSWER_4 = '0'

ANSWER_5 = '3'

ANSWER_6 = '1'

ANSWER_7 = '0'

#### Part 2: Transitive Rule #########################################

# Fill this in with your rule 
transitive_rule = IF( AND('(?x) beats (?y)','(?y) beats (?z)'), 
                     THEN('(?x) beats (?z)') )

# You can test your rule by uncommenting these pretty print statements
#  and observing the results printed to your screen after executing lab1.py
#pprint(forward_chain([transitive_rule], abc_data))
#pprint(forward_chain([transitive_rule], poker_data))
#pprint(forward_chain([transitive_rule], minecraft_data))


#### Part 3: Family Relations #########################################

# Define your rules here. We've given you an example rule whose lead you can follow:
#friend_rule = IF( AND("person (?x)", "person (?y)"), THEN ("friend (?x) (?y)", "friend (?y) (?x)") )


# Add your rules to this list:

#in order to avoid the sibling x x problem, we first need to populate the
#data set with duplicates so we can not fire a rule if there is a duplicate
duplicate = IF('person (?x)', 
               THEN('person (?x) (?x)'))

#basic rule for defining when you are a child of a parent
child_rule = IF( AND('parent (?x) (?y)'), 
                THEN ('child (?y) (?x)'))


#sibling rule based on one shared parent and the fact that it cannot be the 
#parent of the same person (thus, avoiding the duplicate sibling problem)
sibling_rule = IF( AND('parent (?p) (?x)','parent (?p) (?y)', 
                       NOT('person (?x) (?y)')), 
                THEN('sibling (?x) (?y)', 'sibling (?y) (?x)'))

#cousin rule where two are cousins if their parents are siblings but the
#children themselves are not siblings 
cousin_rule = IF ( AND('parent (?p) (?x)', 'parent (?q) (?y)',
                       'sibling (?p) (?q)', NOT('sibling (?x) (?y)')), 
                THEN ('cousin (?x) (?y)','cousin (?y) (?x)'))

#grandparent and grandchild rule combined 
grandparent_rule =  IF ( AND('parent (?x) (?p)','parent (?p) (?y)'),
                        THEN('grandparent (?x) (?y)', 'grandchild (?y) (?x)'))

#forward chaining the rules such that you start with basic order and build upon
#previous statements     
family_rules =  [duplicate, 
                 child_rule, 
                 sibling_rule,
                 cousin_rule,
                 grandparent_rule]


# Uncomment this to test your data on the Simpsons family:
#pprint(forward_chain(family_rules, simpsons_data, verbose=True))

# These smaller datasets might be helpful for debugging:
#pprint(forward_chain(family_rules, sibling_test_data, verbose=True))
# pprint(forward_chain(family_rules, grandparent_test_data, verbose=True))

# The following should generate 14 cousin relationships, representing 7 pairs
# of people who are cousins:

black_family_cousins = [
    relation for relation in
    forward_chain(family_rules, black_data, verbose=False)
    if "cousin" in relation ]

# To see if you found them all, uncomment this line:
#pprint(black_family_cousins)


#### Part 4: Backward Chaining #########################################

# Import additional methods for backchaining
from production import PASS, FAIL, match, populate, simplify, variables

def backchain_to_goal_tree(rules, hypothesis):
    """
    Takes a hypothesis (string) and a list of rules (list
    of IF objects), returning an AND/OR tree representing the
    backchain of possible statements we may need to test
    to determine if this hypothesis is reachable or not.

    This method should return an AND/OR tree, that is, an
    AND or OR object, whose constituents are the subgoals that
    need to be tested. The leaves of this tree should be strings
    (possibly with unbound variables), *not* AND or OR objects.
    Make sure to use simplify(...) to flatten trees where appropriate.
    """

    #creating the results of our hypothesis
    results = [hypothesis]
    
    #looping through all the rules (necessary for each new hypothesis under recursion)
    for rule in rules:
        
            #taking the binder variable
            binder = match(rule.consequent(), hypothesis)
            
            #seeing if there is actually a matching statement
            if binder or rule.consequent() == hypothesis:
                
                #collecting all the antecedents to a rule
                antecedents = rule.antecedent()
                
                #special case if there is only 1 antecedent
                if isinstance(antecedents, str):
                    #resetting our hypothesis so we can recursively go 
                    #through the one hypothesis
                    hypothesis = populate(antecedents, binder)
                    #have to append the hypothesis itself
                    results.append(hypothesis)
                    #recursively go through rules again with new hypothesis
                    results.append(backchain_to_goal_tree(rules, hypothesis))

                else: #if the antecedent has more AND/OR statements (more depth)
                    hypotheses = [populate(antecedent, binder) 
                                for antecedent in antecedents]
                    
                    #recursive results for each antecent(which became a new
                    #hypothesis)
                    sub_results = [backchain_to_goal_tree(rules, hypothesis) 
                                for hypothesis in hypotheses]
                    
                    #binding the sub_results correctly based on how the 
                    #antecedents were stored 
                    if isinstance(antecedents, AND):
                        results.append(AND(sub_results))
                    elif isinstance(antecedents, OR):
                        results.append(OR(sub_results))
                    
    #returning simplified version as we go                
    return simplify(OR(results))
       
            
# Uncomment this to test out your backward chainer:
#pretty_goal_tree(backchain_to_goal_tree(zookeeper_rules, 'opus is a penguin'))


#### Survey #########################################

NAME = 'Taylor Burke'
COLLABORATORS = 'TA:Kifle'
HOW_MANY_HOURS_THIS_LAB_TOOK = '9'
WHAT_I_FOUND_INTERESTING = 'the back chaining problem! took many hours and help'
WHAT_I_FOUND_BORING = 'Challenges are never boring'
SUGGESTIONS = "offer more insight into how to avoid the sibling X X problem" 
                


###########################################################
### Ignore everything below this line; for testing only ###
###########################################################

# The following lines are used in the tester. DO NOT CHANGE!
print("(Doing forward chaining. This may take a minute.)")
transitive_rule_poker = forward_chain([transitive_rule], poker_data)
transitive_rule_abc = forward_chain([transitive_rule], abc_data)
transitive_rule_minecraft = forward_chain([transitive_rule], minecraft_data)
family_rules_simpsons = forward_chain(family_rules, simpsons_data)
family_rules_black = forward_chain(family_rules, black_data)
family_rules_sibling = forward_chain(family_rules, sibling_test_data)
family_rules_grandparent = forward_chain(family_rules, grandparent_test_data)
family_rules_anonymous_family = forward_chain(family_rules, anonymous_family_test_data)
