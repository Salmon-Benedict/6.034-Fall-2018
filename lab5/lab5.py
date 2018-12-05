# MIT 6.034 Lab 5: k-Nearest Neighbors and Identification Trees
# Written by 6.034 Staff

from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')


################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################


#### Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    
    if id_tree.is_leaf():
        return id_tree.get_node_classification()
    
    return id_tree_classify_point(point, id_tree.apply_classifier(point))


#### Part 1B: Splitting data with a classifier #################################

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""

    result ={}
    for point in data:
        classified = classifier.classify(point)
        if classified not in result:
            result[classified] = [point]
        else:
            result[classified].append(point)
    return result

#### Part 1C: Calculating disorder #############################################


def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""
    num_branch = len(data)
    classified_data = split_on_classifier(data, target_classifier)
    
    disorder = 0
    for classified_branch in classified_data.values():
        num_classfied_branch = len(classified_branch)
        disorder += (-(num_classfied_branch/num_branch))*(log2(num_classfied_branch/num_branch))
    
    return disorder   
    

def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    
    classified_data = split_on_classifier(data, test_classifier)
    
    disorder = 0 
    for branch in classified_data.values():
        a_branch_disorder = branch_disorder(branch, target_classifier)
        weight = len(branch) / len(data)
        disorder += a_branch_disorder * weight
        
    return disorder 


## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab5.py:

#for classifier in tree_classifiers:
#    print(classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type")))


#### Part 1D: Constructing an ID tree ##########################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""
    
    disorder_list = []
    
    for test_classifier in possible_classifiers:
        
        test_classifier_disorder = average_test_disorder(data, test_classifier, target_classifier)
        disorder_list.append((test_classifier, test_classifier_disorder))
    
    min_disorder =  min(disorder_list, key = lambda x: x[1])
    
    if min_disorder[1] == 1.0:
        raise NoGoodClassifiersError
        
    else:
        return min_disorder[0]
        

## To find the best classifier from 2014 Q2, Part A, uncomment:
#print(find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")))

def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    
#    1. Once the current id_tree_node is defined, perform one of three actions, 
#    depending on the input node's data and available classifiers:
    if not id_tree_node:
        id_tree_node = IdentificationTreeNode(target_classifier)
    
#    1.1 If the node is homogeneous, then it should be a leaf node, so add the 
#    classification to the node
    if branch_disorder(data, target_classifier) == 0:
        id_tree_node.set_node_classification(target_classifier.classify(data[0]))


    else:
        try:
#            1.2 If the node is not homogeneous and the data can be divided further, 
#            add the best classifier to the node.
            
            best_classifier = find_best_classifier(data, possible_classifiers, \
                                                   target_classifier)
            
            possible_classifiers.remove(best_classifier)
            
            dict_branches = split_on_classifier(data, best_classifier)
            
            id_tree_node = id_tree_node.set_classifier_and_expand(best_classifier, dict_branches)
            
            new_branches = id_tree_node.get_branches()
            

#            2. If you added a classifier to the node, use recursion to 
#            complete each subtree.
            for branch in new_branches:
                construct_greedy_id_tree(dict_branches[branch], \
                possible_classifiers, target_classifier, new_branches[branch])
                
#        1.3 If the node is not homogeneous but there are no good classifiers left 
#        (i.e. no classifiers with more than one branch), leave the node's 
#        classification unassigned (which defaults to None).        
        except NoGoodClassifiersError:
            pass
        
#    3. Return the original input node.       
    return id_tree_node



## To construct an ID tree for 2014 Q2, Part A:
#print(construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type")))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
#tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
#print(id_tree_classify_point(tree_test_point, tree_tree))

# To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
#print(construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification")))
#print(construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class")))


#### Part 1E: Multiple choice ##################################################

ANSWER_1 = 'bark_texture'
ANSWER_2 = 'leaf_shape'
ANSWER_3 = 'orange_foliage'

ANSWER_4 = [2,3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = 'No'
ANSWER_9 = 'No'


#### OPTIONAL: Construct an ID tree with medical data ##########################

## Set this to True if you'd like to do this part of the lab
DO_OPTIONAL_SECTION = False

if DO_OPTIONAL_SECTION:
    from parse import *
    medical_id_tree = construct_greedy_id_tree(heart_training_data, heart_classifiers, heart_target_classifier_discrete)


################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### Part 2A: Drawing Boundaries ###############################################

BOUNDARY_ANS_1 = 3
BOUNDARY_ANS_2 = 4

BOUNDARY_ANS_3 = 1
BOUNDARY_ANS_4 = 2

BOUNDARY_ANS_5 = 2
BOUNDARY_ANS_6 = 4
BOUNDARY_ANS_7 = 1
BOUNDARY_ANS_8 = 4
BOUNDARY_ANS_9 = 4

BOUNDARY_ANS_10 = 4
BOUNDARY_ANS_11 = 2
BOUNDARY_ANS_12 = 1
BOUNDARY_ANS_13 = 4
BOUNDARY_ANS_14 = 4


#### Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    
    return sum([u_i * v_i for u_i, v_i in zip(list(u), list(v))])


def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    
    return math.sqrt(dot_product(v, v))


def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    
    return math.sqrt(sum([math.pow((coord_1 - coord_2),2) 
                for coord_1, coord_2 in zip(point1.coords, point2.coords)]))


def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    
    return sum([abs(coord_1 - coord_2)
            for coord_1, coord_2 in zip(point1.coords, point2.coords)])

def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    
    return len([True for coord_1, coord_2 in zip(point1.coords, point2.coords)
                if coord_1 != coord_2])

def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    
    coords_1 = point1.coords
    coords_2 = point2.coords
    
    return 1 - (dot_product(coords_1,coords_2)/((norm(coords_1))*(norm(coords_2))))
    

#### Part 2C: Classifying points ###############################################

def get_k_closest_points(given_point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    
    
    tup_point_distance = [(point, distance_metric(point, given_point)) for point in data]

    sorted_tup_point_distance = sorted(tup_point_distance, key =lambda tup: (tup[1], tup[0].coords))
        
    return [sorted_tup_point_distance[i][0] for i in range(k)]

def knn_classify_point(given_point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    
    closest_points = get_k_closest_points(given_point, data, k, distance_metric)
    
    classifications_list = list((map(lambda x: x.classification, closest_points)))
    
    classifications_set = set(classifications_list)
    
    classification_occurance = [(classification,classifications_list.count(classification))
                                for classification in classifications_set]

    return sorted(classification_occurance,key=lambda tup: tup[1], reverse=True)[0][0]


## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
# print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


#### Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    
    binary_scores = []
    
    for i in range(len(data)):
        
        training_data = data.copy()
        test_point = training_data.pop(i)
        expected_classification = knn_classify_point(test_point, training_data, k, distance_metric)
        
        if expected_classification == test_point.classification:
           binary_scores.append(1)
        else:
            binary_scores.append(0)

    return float((sum(binary_scores)/len(data)))
    
                

def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""

    largest_cross_score = 0
    
    for i in range(1,len(data)):
        
        k = i
        
        euc_score =  cross_validate(data, k, euclidean_distance)
        man_score = cross_validate(data, k, manhattan_distance)
        ham_score = cross_validate(data, k, hamming_distance)
        cos_score = cross_validate(data, k, cosine_distance)
        
        
        if largest_cross_score < euc_score:
            largest_cross_score = euc_score
            result = (k, euclidean_distance)
            
        if largest_cross_score <  man_score:
            largest_cross_score =  man_score
            result = (k, manhattan_distance) 
            
        if largest_cross_score < ham_score:
            largest_cross_score = ham_score
            result = (k, hamming_distance)
            
        if largest_cross_score < cos_score:
            largest_cross_score = cos_score
            result = (k, cosine_distance)  
            
            
    return result
    
    
## To find the best k and distance metric for 2014 Q2, part B, uncomment:
# print(find_best_k_and_metric(knn_tree_data))


#### Part 2E: More multiple choice #############################################

kNN_ANSWER_1 = "Overfitting"
kNN_ANSWER_2 = "Underfitting"
kNN_ANSWER_3 = 4

kNN_ANSWER_4 = 4
kNN_ANSWER_5 = 1
kNN_ANSWER_6 = 3
kNN_ANSWER_7 = 3


#### SURVEY ####################################################################

NAME = "Taylor Burke"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = '15'
WHAT_I_FOUND_INTERESTING = "learning and implementing completely new material"
WHAT_I_FOUND_BORING = "lol learning is never boring when it comes to CS"
SUGGESTIONS = "the hardest part of lab every time is learning the API (/getting comfortable enough to actual use it properly). since I didn't write it, it's confusing to understand even with all the explanations. It would be cool to have a SMALL part due earlier in the week that allows you to 'write' some of the API ourselves"
