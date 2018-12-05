# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    
    point_to_weight = {}
    weight = make_fraction(1, len(training_points))
    
    for training_point in training_points:
        point_to_weight[training_point] = weight
        
    return point_to_weight

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    
    classifier_to_error_rate = {}
    
    for classifier, points in classifier_to_misclassified.items():
        weight_sum = 0
        
        for training_point in points:
            weight_sum += point_to_weight[training_point]
        
        classifier_to_error_rate[classifier] = weight_sum
        
    return classifier_to_error_rate
    

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""


    threshold = make_fraction(1,2)
    sorted_classifier_to_error_rate = sorted(classifier_to_error_rate.items(), 
                                             key = lambda entry: entry[0])
    
    max_classifier, max_error_rate = max(sorted_classifier_to_error_rate, 
                                         key=lambda entry: entry[1])

    min_classifier, min_error_rate = min(sorted_classifier_to_error_rate, 
                                         key=lambda entry: entry[1])
    
    if use_smallest_error:
        best_classifier, error_rate = min_classifier, min_error_rate
    
    else:
        max_diff = make_fraction(max_error_rate - threshold)
        min_diff = make_fraction(threshold - min_error_rate)
        
        if max_diff > min_diff:
            best_classifier, error_rate = max_classifier, max_error_rate
        
        elif min_diff > max_diff:
            best_classifier, error_rate = min_classifier, min_error_rate
            
        else: #max_diff == min_diff: #alphabetical sort of classifier name
            if max_classifier < min_classifier:
                best_classifier, error_rate = max_classifier, max_error_rate
            else:
                best_classifier, error_rate = min_classifier, min_error_rate
                
    if error_rate == threshold:
        raise NoGoodClassifiersError
        
    return best_classifier
        
    
def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""

    if error_rate == 1.0:
        return -INF
    
    elif error_rate == 0.0:
        return INF
    
    return make_fraction(1,2) * ln(make_fraction(1-error_rate, error_rate))
    
def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""

    misclassified_points = set()
    
    for training_point in training_points:
        score_summation = 0
        
        for classifier, voting_power in H:
            
            if training_point not in classifier_to_misclassified[classifier]:
                score_summation += voting_power
            
            else:
                score_summation -= voting_power

        if score_summation <= 0:
            misclassified_points.add(training_point)
    
    return misclassified_points
    

def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
        
    misclassified_points = get_overall_misclassifications(H, training_points, classifier_to_misclassified)
    
    if len(misclassified_points) > mistake_tolerance:
        return False
    
    return True

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""

    for point, old_weight in point_to_weight.items():
        
        if point not in misclassified_points:
            new_weight = make_fraction(1,2) * make_fraction(1, 1 - error_rate) * old_weight
            
        else:
            new_weight = make_fraction(1,2) * make_fraction(1, error_rate) * old_weight
        
        point_to_weight[point] = new_weight
            
    return point_to_weight


#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""


    point_to_weight = initialize_weights(training_points)
    H = []
    rounds = 0

    while rounds < max_rounds:
        
        classifier_to_error_rate = calculate_error_rates(point_to_weight, classifier_to_misclassified)
        
        try:
            best_classifier = pick_best_classifier(classifier_to_error_rate, use_smallest_error)

        except NoGoodClassifiersError:
            break
        
        error_rate = make_fraction(classifier_to_error_rate[best_classifier])
        voting_power = calculate_voting_power(error_rate)
        
        H.append((best_classifier, voting_power))
        
        misclassified_points = classifier_to_misclassified[best_classifier]
        update_weights(point_to_weight, misclassified_points, error_rate)
        
        if is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance):
            break

        rounds += 1

    return H  

#### SURVEY ####################################################################

NAME = "Taylor Burke"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = "3"
WHAT_I_FOUND_INTERESTING = "implementing adaboost"
WHAT_I_FOUND_BORING = "learning is never boring!"
SUGGESTIONS = "None"
