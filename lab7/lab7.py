# MIT 6.034 Lab 7: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce
from math import *
from svm_api import *


#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented 
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""
    
    return sum([u_i * v_i for u_i, v_i in zip(list(u), list(v))])

def norm(v):
    """Computes the norm (length) of a vector v, represented 
    as a tuple or list of coords."""
    
    return sqrt(dot_product(v, v))


#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
        
    return dot_product(svm.w, point.coords) + svm.b

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    
    pos = positiveness(svm, point)
    
    if pos > 0:
        return 1
    
    elif pos < 0:
        return -1
    
    return 0
    

def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    
    return 2/norm(svm.w)

def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    
    violators = set()
    
    for training_point in svm.training_points:
        
        if training_point in svm.support_vectors and training_point.classification != positiveness(svm, training_point):
            violators.add(training_point)

        if positiveness(svm, training_point) < 1 and positiveness(svm, training_point) > -1:
            violators.add(training_point)

    return violators

#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    
    violators = set()
    
    for training_point in svm.training_points:
        
        if training_point not in svm.support_vectors and training_point.alpha != 0:
            violators.add(training_point)
            
        if training_point in svm.support_vectors and training_point.alpha <= 0:
            violators.add(training_point)
            
    return violators 

def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    
    equation_4 = [] 
    equation_5 = []

    for training_point in svm.training_points:
        y_i = training_point.classification
        alpha_i = training_point.alpha
        
        equation_4.append(y_i * alpha_i)
        equation_5.append(scalar_mult(y_i * alpha_i, training_point.coords))
    
    return sum(equation_4) == 0 and svm.w == reduce(vector_add, equation_5)


#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    
    violators = set()
    
    for training_point in svm.training_points:
        
        if training_point.classification != classify(svm, training_point):
            violators.add(training_point)
            
    return violators 

#### Part 5: Training an SVM ###################################################


def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""
    
    #step 1: find all support vectors
    support_vectors= []
    equation_5 = []
    
    for training_point in svm.training_points:
        y_i = training_point.classification
        alpha_i = training_point.alpha

        if alpha_i > 0: 
            support_vectors.append(training_point)
            equation_5.append(scalar_mult(y_i * alpha_i, training_point.coords))

            
    #step 2: calculate new w, using equation_5  
    new_w = reduce(vector_add, equation_5)
    
    #step_3 calculate the new b
    positive_support_vector_bs = []
    negative_support_vector_bs = []
    
    for support_vector in support_vectors:
        
        if support_vector.classification == 1:
            new_b = 1 - dot_product(new_w, support_vector)
            positive_support_vector_bs.append(new_b)
            
        elif support_vector.classification == -1:
            new_b = -1 - dot_product(new_w, support_vector)
            negative_support_vector_bs.append(new_b)
            
    
    new_b = (max(positive_support_vector_bs) + min(negative_support_vector_bs))/2
    
    #step 4: update the boundries of the svm and support vectors, mutating the old svm
    svm.support_vectors = support_vectors
    
    
    #step 5: return the updated svm
    return svm.set_boundary(new_w, new_b)


#### Part 6: Multiple Choice ###################################################

ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = ["A", "D"]
ANSWER_6 = ["A", "B", "D"]
ANSWER_7 = ["A", "B", "D"]
ANSWER_8 = []
ANSWER_9 = ["A", "B", "D"]
ANSWER_10 = ["A", "B", "D"]

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1,3,6,8]
ANSWER_18 = [1,2,4,5,6,7,8]
ANSWER_19 = [1,2,4,5,6,7,8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = "Taylor Burke"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = "4"
WHAT_I_FOUND_INTERESTING = "nothing super particular this pset, but the approach was new and I liked that" 
WHAT_I_FOUND_BORING = "learning is never boring!"
SUGGESTIONS = "really well structured pset, keep as is"
