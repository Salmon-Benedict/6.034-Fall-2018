# MIT 6.034 Lab 6: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2, 1]

nn_cross = [2, 2, 1]

nn_stripe = [3, 1]

nn_hexagon = [6, 1]

nn_grid = [4, 2, 1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    
    if x >= threshold:
        return 1
    return 0

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    
    S = steepness
    M = midpoint 

    return 1/ (1+ e**(-S*(x-M)))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    
    return max(0,x)

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    
    return (-1/2)*((desired_output - actual_output)**2)


#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given 
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))
    
    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node
    
    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    
    #initializing resultant dictionary
    results = {}
    
    #going through each neuron layer by layer
    for neuron in net.topological_sort():
        
        #each neuron gets a new summation
        summation = 0
        
        #going through the neuron's neighbors
        for neighbor in net.get_incoming_neighbors(neuron):
            
            #storing wire variables
            wire = net.get_wire(neighbor,neuron)
            wire_weight = wire.weight
            
            if isinstance(neighbor, int):
                weighted_input = neighbor * wire_weight
            
            elif isinstance(neighbor, str):
                if neighbor in input_values:
                    weighted_input = input_values[neighbor] * wire_weight
                    
                else:
                    weighted_input = results[neighbor] * wire_weight
            #continually summing weights
            summation += weighted_input
            
            
        #passing each summation through the threshold function and 
        #mapping the neuron to its immediate output in results
        results[neuron] = threshold_fn(summation)

    for a_input in input_values:
        
        if isinstance(neighbor, int):
            results[a_input] = a_input
            
        elif isinstance(a_input, str):   
            results[a_input] = input_values[a_input]
        
    final_output = results[net.get_output_neuron()]
    return (final_output, results)
    

#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    
    data = []
    
    for step_a in [-step_size, 0, step_size]:
        
        for step_b in[-step_size, 0, step_size]:
            
            for step_c in  [-step_size, 0, step_size]:

                inputs_copy = inputs.copy()
                [input_a, input_b, input_c] = inputs_copy
                inputs_copy[0] = float(input_a) + step_a
                inputs_copy[1] = float(input_b) + step_b
                inputs_copy[2] = float(input_c) + step_c
                
                result = func(inputs_copy[0], inputs_copy[1], inputs_copy[2])
                data.append((result, inputs_copy))
            
    sorted_data = sorted(data, key = lambda tup: tup[0], reverse = True)
    result = sorted_data.pop(0)
    
    return result

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""

# In particular, updating the weight between nodes A and B requires the 
# output from node A, the current weight on the wire from A to B, the
# output of node B, and all neurons and weights downstream to the final layer
    
    start_node = wire.startNode
    end_node = wire.endNode
    
    #maintain a set of already-recorded dependencies
    result = set()
    result.add(wire)
    result.add(start_node)
    result.add(end_node)
    
    if not net.is_output_neuron(end_node):
        agenda = net.get_outgoing_neighbors(end_node)
        agenda.append(end_node)
        
        while agenda:
            neuron = agenda.pop(0)
            result.add(neuron)
            
            for neighbor in net.get_outgoing_neighbors(neuron):
                agenda.append(neighbor)
                result.add(net.get_wire(neuron, neighbor))
                
    return result


def final_layer_b(outStar, outB):
    """
    algorthim for calculating final delta if neuron B is the final neuron
    Args:
        1) outStar: desired output
        2) outB = neuronb
    """
    return outB*(1-outB)*(outStar-outB)

def non_final_layer_b(net, neuron, outB, results):
    """
    algorithm for caculating the summation delta since neuron B is not
    the final neuron
    """
    deltas = []

    for wire in net.get_wires(neuron):
        deltas.append(wire.weight * results[wire.endNode])
        
    return outB*(1-outB)*sum(deltas)

def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    
    results = {}
    outStar = desired_output
    # Note: The reverse of a topologically sorted list is equivalent to the
    # topological sort of a reversed graph.  (This has been proved.)
    reversed_order_neurons = net.topological_sort()
    reversed_order_neurons.reverse()
    

    for neuron in reversed_order_neurons: 

        outB = neuron_outputs[neuron]
        
        if net.is_output_neuron(neuron):
            results[neuron] = final_layer_b(outStar, outB)
            
        else:
            results[neuron] = non_final_layer_b(net, neuron, outB, results)

    return results

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    
    deltas = calculate_deltas(net, desired_output, neuron_outputs)
    
    for wire in net.get_wires():
        
        start_node = wire.startNode
        end_node = wire.endNode
        initial_wire_weight = wire.get_weight()
        delta = deltas[end_node]
        
        if start_node in net.inputs and start_node in input_values:
            #Sets the weight of the wire and returns the new weight.
            wire.set_weight(initial_wire_weight + r * input_values[start_node] * delta)
        
        elif start_node not in net.inputs and start_node in neuron_outputs:
            #Sets the weight of the wire and returns the new weight.
            wire.set_weight(initial_wire_weight + r * neuron_outputs[start_node] * delta)
            
        else:
            #Sets the weight of the wire and returns the new weight.
            wire.set_weight(initial_wire_weight + r * start_node * delta)
            
    return net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    
    iterations = 0
    #first forward prop
    (final_output, results) = forward_prop(net, input_values, sigmoid)
    
    while accuracy(desired_output, final_output) <= minimum_accuracy:

        update_weights(net, input_values, desired_output, results, r)
        
        (final_output, results) = forward_prop(net, input_values, sigmoid)
        
        iterations += 1
        
    return (net, iterations)


#### Part 5: Training a Neural Net #############################################

ANSWER_1 = 15
ANSWER_2 = 11
ANSWER_3 = 7
ANSWER_4 = 96
ANSWER_5 = 67

ANSWER_6 = 1
ANSWER_7 = 'checkerboard'
ANSWER_8 = ['small', 'medium', 'large']
ANSWER_9 = 'B'

ANSWER_10 = 'D'
ANSWER_11 = ['A', 'C']
ANSWER_12 =  ['A', 'E']


#### SURVEY ####################################################################

NAME = "Taylor Burke"
COLLABORATORS = "Daniel (graduated MIT already)"
HOW_MANY_HOURS_THIS_LAB_TOOK = "10"
WHAT_I_FOUND_INTERESTING = "neural nets!"
WHAT_I_FOUND_BORING = "learning is never boring"
SUGGESTIONS = "None"
