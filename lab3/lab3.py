# MIT 6.034 Lab 3: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
#    The game is over when the board contains a chain of length 4 (or longer), 
#    or all columns are full.
    
    lengths = [True for item in board.get_all_chains() if len(item) >= 4]
    full_columns = [board.is_column_full(col) for col in range(board.num_cols)]

    return any(lengths) or all(full_columns)

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    
    return [board.add_piece(col) for col in range(board.num_cols) 
                        if not board.is_column_full(col) and 
                        not is_game_over_connectfour(board)]
    

def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
#    We'll return a score of 1000 if the maximizer has won, or -1000 if the
#    minimizer has won. In case of a tie, return 0 instead. A tie occurs when 
#    the entire board fills up but has no chains of 4 or more same-colored 
#    tokens.
    
    #smimilar for testing if a game is over but reversed and altered lengths
    #so that we
    lengths = [True for item in board.get_all_chains() if len(item) < 4]
    full_columns = [board.is_column_full(col) for col in range(board.num_cols)]
    
    #if the game is a tie (but still a winning board)
    #because the board if over and all players have less than 4 in a row
    if all(lengths) and all(full_columns):
        return 0
    
    #if the game is over by a player having 4 or more in a row
    #if the minimizer made the playing move
    elif is_current_player_maximizer:
        return -1000
        
    else: #if the maximizer made the playing move
        return 1000
            
def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    
    #calculting how many pieces are left
    pieces_left = (board.num_rows * board.num_cols) - board.count_pieces()
    #calculating how much incentive to give
    incentive = (2*pieces_left)
    #instead of repeating code, just build on the previous function
    result = endgame_score_connectfour(board, is_current_player_maximizer)
    
    #going through each possible result and applying the incentive
    if result == 0:
        return 0
    elif result == -1000:
        return -1000 - incentive
    else: 
        return 1000 + incentive

def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer.""" 
        
    #creating a score variable to be returned
    score = 0
    
    #current player will have swapped in the next board
    for a_board in next_boards_connectfour(board):
        #the highest heuristic should be 500 and should indicate a win for the
        #next move
        if is_game_over_connectfour(a_board):
                return 500
            
        for a_chain in a_board.get_all_chains(current_player=True):
            if len(a_chain) == 3:
                score += 10
            elif len(a_chain) == 2:
                score += 5
        
        #the other player is always a move behind and should not be considered
        #the same weight (since the board is the "next" board the players 
        #already switched)
        for a_chain in a_board.get_all_chains(current_player=False):
            if len(a_chain) == 3:
                score -= 8
            elif len(a_chain) == 2:
                score -= 4
    
    #higher scores indicate that the game is favorable for maximizer
    if is_current_player_maximizer:
        score = -(score)
    
    return score
    

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def extensions(game_state, path, all_paths):
        """Returns a list of paths. Each path in the list should be a one-node
        extension of the input path, where an extension is defined as a path formed
        by adding a "next move" node (until and including the final node in the
        path) to the path/goal node/movethat wins the game."""
         #basic idea taken from lab 2 but needed to modify since we need to check 
         #for ending and continue expanding in a DFS manner
        #getting copy of the path
        result = list(path)
        result.append(game_state)
        
        if game_state.is_game_over():
            all_paths.append(result)
            return True
        
        else: #state.is_game_over() 
            for extension in game_state.generate_next_states():
                extensions(extension, result, all_paths)
                
        return all_paths

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    
    #expaning upon to given state recursively
    #the starting path is an empty list since we have yet to start the path
    all_paths = extensions(state, [], [])
    

    #tuple and list comprehension storing each path with path end game score
    ending_paths_and_scores = [ 
            (path, path[-1].get_endgame_score(is_current_player_maximizer=True))
                            for path in all_paths]
        
    
    #since we are evaluating every path
    num_static_evaluations = len(all_paths)
    #finding a returning the path with max score
    best_path_list,score =  max(ending_paths_and_scores, key=lambda item:item[1])
    #collecting our results
    results = (best_path_list, score, num_static_evaluations)
    
    #returning our results
    return results 

             

# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

#pretty_print_dfs_type(dfs_maximizing(GAME1))


def minimax_endgame_search(state, maximize) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    final = []
    (node, path, count) = (state, [state], 0)
    if node.is_game_over():
        count += 1
        final.append((path, node.get_endgame_score(), count))
    else:
        if maximize == True:
            max = -INF
            for nextnode in node.generate_next_states():
                (nextpath, value, count1) = minimax_endgame_search(nextnode, False)
                count += count1
                if value > max:
                    max = value
                    (nextpath1, value1, count2) = (path+nextpath, value, count)
            final.append((nextpath1,value1, count))

        else:
            min = INF
            for nextnode in node.generate_next_states():
                (nextpath, value, count1) = minimax_endgame_search(nextnode, True)
                count += count1
                if value < min:
                    min = value
                    (nextpath1, value1, count2) = (path+nextpath, value, count)

            final.append((nextpath1,value1, count))

    variable = final[0]
    for state1 in final:
        if state1[1] > variable[1]:
            variable = state1
        elif state1[1] == variable[1]:
            if state1[2] < variable[2]:
                variable = state1

    return variable

#    paths = extensions(state, [], [])
    


#    def minimax_score(state, maximize):
#        if state.is_game_over():
#            return [state.get_endgame_score(maximize), state]
#        else:
#            if maximize:
#                max_score = max([minimax_score(child, False)[0] for child in state.generate_next_states()])
#                for child in state.generate_next_states():
#                    if minimax_score(child, False)[1].get_endgame_score(maximize) == max_score:
#                        max_state = child
#
#                return [max_score, max_state]
#            else:
#                min_score = min([minimax_score(child, True)[0] for child in state.generate_next_states()])
#                for child in state.generate_next_states():
#                    if minimax_score(child, True)[1].get_endgame_score(maximize) == min_score:
#                        min_state = child
#
#                return [min_score, min_state]
#
#    # Get min/max scores at from leaves, as well as corresponding parent state
#    score, end_node = minimax_score(state, maximize)[0], minimax_score(state, maximize)[1]
#
#    num_evaluations = 0
#    relevant_paths = []
#    for p in paths:
#        if p[-1].get_endgame_score(maximize) == score and p[-2] == end_node:
#            relevant_paths.append(p)
#        num_evaluations += 1
#
#    return (relevant_paths[0], score, num_evaluations)
    #overarching path that we keep wanting to add to
#    paths = extensions(state, [], [])
#    
#    
#    def minimax_score(state, maximize):
#        if state.is_game_over():
#            return [state.get_endgame_score(True), state]
#        else:
#            if maximize:
#                scores = []
#                for child in state.generate_next_states():
#                    scores.append(minimax_score(child, False)[0])
#                    if minimax_score(child, False)[1].get_endgame_score(True) == max(score):
#                        max_state = child
#                        
#                return [max(score), max_state]
#            else:
#                min_score = min([minimax_score(child, True)[0] for child in state.generate_next_states()])
#                for child in state.generate_next_states():
#                    if minimax_score(child, True)[1].get_endgame_score(True) == min_score:
#                        min_state = child
#
#                return [min_score, min_state]
#
#    # Get min/max scores at from leaves, as well as corresponding parent state
#    score, end_node = minimax_score(state, maximize)[0], minimax_score(state, maximize)[1]
#
#    num_evaluations = 0
#    relevant_paths = []
#    for p in paths:
#        if p[-1].get_endgame_score(maximize) == score and p[-2] == end_node:
#            relevant_paths.append(p)
#        num_evaluations += 1
#
#    return (relevant_paths[0], score, num_evaluations) 
#    
    
#    #since we are evaluating every path
#    num_static_evaluations = len(all_paths)
#    #finding a returning the path with max score
#    best_path_list,score =  max(ending_paths_and_scores, key=lambda item:item[1])
#    #collecting our results
#    results = (best_path_list, score, num_static_evaluations)
#    
#    #returning our results
#    return results 

# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

#pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    raise NotImplementedError


# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    raise NotImplementedError


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    raise NotImplementedError


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = ''

ANSWER_2 = ''

ANSWER_3 = ''

ANSWER_4 = ''


#### SURVEY ###################################################

NAME = 'Taylor Burke'
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
