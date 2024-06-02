from copy import deepcopy
from collections import deque
from functools import singledispatchmethod
from itertools import chain
from time import perf_counter

import numpy as np

from section import Row, Column, Block, InvalidState
from debug_msg import msg, clear, msgpause


def read_puzzle(puz):

    """
    Good webpage dedicated to sudoko.
    https://www.thonky.com/sudoku/
    """

    # Create a dict of puzzle dicts
    
    puzzle_data = {
        "test_puzzle": [(1,5,8)],
        "multiple_solutions": [
            (1,1,4), (1,5,1), (1,9,9),
            (2,4,2), (2,6,7),
            (3,3,7), (3,5,5), (3,7,3),
            (4,2,6), (4,8,5),
            (5,1,1), (5,3,4), (5,7,9), (5,9,6),
            (6,2,7), (6,8,3),
            (7,3,1), (7,5,3), (7,7,4),
            (8,4,7), (8,6,4),
            (9,1,5), (9,5,9)
        ],
        "fiendish": [
            (1,1,4), (1,5,1), (1,9,9),
            (2,4,2), (2,6,7),
            (3,3,7), (3,5,5), (3,7,3),
            (4,2,6), (4,8,5),
            (5,1,1), (5,3,4), (5,7,9), (5,9,6),
            (6,2,7), (6,8,3),
            (7,3,1), (7,5,3), (7,7,4),
            (8,4,7), (8,6,4),
            (9,1,5), (9,5,9), (9,9,8)
        ],
        "inkala": [
            (1,1,8),
            (2,3,3), (2,4,6),
            (3,2,7), (3,5,9), (3,7,2),
            (4,2,5), (4,6,7),
            (5,5,4), (5,6,5), (5,7,7),
            (6,4,1), (6,8,3),
            (7,3,1), (7,8,6), (7,9,8),
            (8,3,8), (8,4,5), (8,8,1),
            (9,2,9), (9,7,4)
        ],
        "partial_puzzle": [
        ]
    }
    
    # Return the specified puzzle as a set
    #
    # TODO - updated from set to list. Not tested
    #
    #puzzle = set()
    #for p in puzzle_data[puz]: puzzle.add(p)

    return [p for p in puzzle_data[puz]]


def print_puzzle(puz: list | np.ndarray, size: int=None, reindex: bool=False):
    """Produces a formatted printout of a sudko or puzzle. 

    Parameters
    ----------
    puz : list, ndarray
        List of 3-tuples comprising the puzzle (row, column, number)
    size
        Size of the puzzle being printed.
    reindex
        Flag indicating puzzle must be amended to from one-indexing
        to zero-indexing.
    
    """
    
    #
    # DEBUG
    #
    #msg("Printing type:", type(puz))
    #msg("Size:", np.shape(puz)[0])

    # If puz is an ndarray, convert it to a list of tuples.
    if type(puz) == np.ndarray:
        size = np.shape(puz)[0]
        puz = [(r, c, puz[r,c]) for r in range(size) for c in range(size)]         

    #
    # TODO - Untested
    #

    # Infer the size from the length of the list.
    # This calculation is NOT guaranteed. It assumes puz is either a
    # sudoko or a constrained puzzle. It will underestimate the size
    # of an underconstrained puzzle.           
    elif type(puz) == list and size is None:

        print("[print_puzzle]WARNING size was not given. It will be infered from the length of the puzzle.")

        if len(puz) > 256:
            size = 25
        elif len(puz) > 81:
            size = 16
        elif len(puz) > 16:
            size = 9
        elif len(puz) > 1:
            size = 4
        else:
            size = 1

    # 
    # TODO - Make this work for any sized puzzle
    # 1,4,9,16,25, ...100
    #

    box_size = int(size ** .5)
    if size <=9:
        row_sep_maj = "+===" * size + "+"
        row_sep_min = "+---" * size + "+"
        w = 1
    else:
        row_sep_maj = "+====" * size + "+"
        row_sep_min = "+----" * size + "+"
        w = 2

    # Create a blank grid.
    blank_row = ["-" for n in range(size)]
    puzzle_grid = [blank_row.copy() for n in range(size)]


    # If reindex=True then subtract 
    #
    # DEBUG
    #
    #print("[print_puzzle]first Cell:", puz[0])

    # Populate puzzle_grid with available clues.
    for r, c, n in puz: 
        if reindex:
            puzzle_grid[r-1][c-1] = n
        else:
            puzzle_grid[r][c] = n
    
    # Print the puzzle as a formatted table.
    for r in range(size):
        print("\n", row_sep_maj, sep="") if r % box_size == 0 else print("\n", row_sep_min, sep="")

        for c in range(size):
            print("|", sep="", end="") if c % box_size == 0 else print(":", sep="", end="")
            #print(" ", puzzle_grid[r][c], " ",sep="", end="")
            print(" {0:{width}} ".format(puzzle_grid[r][c], width=w), sep="", end="")
        print("|", sep="", end="")

    print("\n", row_sep_maj, sep="")


def print_puzzle2(puz):
    """
    """
    row_sep = "+–––+–––+–––+–––+–––+–––+–––+–––+–––+"

    # Create a blank grid.
    puzzle_grid = [
        ["-","-","-","-","-","-","-","-","-"],["-","-","-","-","-","-","-","-","-"],["-","-","-","-","-","-","-","-","-"],
        ["-","-","-","-","-","-","-","-","-"],["-","-","-","-","-","-","-","-","-"],["-","-","-","-","-","-","-","-","-"],
        ["-","-","-","-","-","-","-","-","-"],["-","-","-","-","-","-","-","-","-"],["-","-","-","-","-","-","-","-","-"]
    ]
    for r, c, n in puz:
        puzzle_grid[r][c] = n
    
    for r in puzzle_grid:
        print(row_sep)

        for c in r: 
            print("| ",c, " ",end="", sep="")

        print("|")
    print(row_sep)
    
    #[]===+===+===[]===+===+===[]
    #[] - | - | - [] - | - | - [] 
    #[] - | - | - [] - | - | - []
    #[] - | - | - [] - | - | - []
    #[]---+---+---[]---+---+---[]
    #[] -   -   - [] -   -   - |
    #[] -   7   - [] -   -   - |
    #[] -   8   - [] -   -   - |
    #[]-----------+------------+
        


class InvalidPuzzle(Exception):
    pass

class SudokoSolver():

    """
    Main class describing a sudoko, with method allowing it to be solved.
    
    """
    
    # Initialise the board
    #rows, cols, blks = [], [], []
    #new_sols = set()

    ## TODO
    # Move this logic to the Block class
    
    #
    # TODO - Enumerate these values
    #
    INVALID = 0
    SOLVED = 1
    UNSOLVED = 2

    #
    # TODO replace with actual maps of all sizes. Not just initialised lists.
    # test with tuple indexing into a dict is faster than indexing into 2-d list.
    #

    #block_map = [
    #    [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]
    #]
    #block_map_rev = [
    #    [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]
    #]
    guess_depth = 0

    
    def __init__(self, size=9, zero_index=False):

        # The class works for any sudoko from 1 to 25
        # [1, 4, 9, 16, 25]. Max stack depth 200 will not accomodate >25
        if size not in [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]:
            raise InvalidPuzzle("Size must be one of [1, 4, 9, 16, 25]")
        
        self.size = size
        self.box_size = int(size ** .5)
        self.zero_index = zero_index

        # To support many_sols, set the number of solutions required.
        # Set to 1 if a solution is required.
        # Set to 2 if many_sols has been called. This indicates that a
        # solution if not required, just whether more than 1 solution exists.

        #
        # TODO - Delete this. this is passed as argument to solve
        #
        self.max_solutions = 1

        # Create a mapping between row-column indexing and block indexing
        self.create_block_maps()
        self.metrics = self.init_metrics()
        self.solutions_found = 0

    def init_metrics(self):
        """
        Initialise dictionary to hold performance metrics.

        Metrics are updated by solve, and accessible as an attribute.
        """

        # Create a list of zeroes, and list of empty lists to copy
        # to initialise the members of the dict.
        empty = [0 for idx in range(self.size**2 + 1)]
        emptyb = [False for idx in range(self.size**2 + 1)]
        empty2d = [[] for idx in range(self.size**2 + 1)]

        metrics = {
            "lvl": -1,
            "deduced_lvl": emptyb.copy(),
            "guess_last_lvl": 0,
            "n_deduced_lvl": empty.copy(),
            "n_guessed_lvl": empty.copy(),
            "n_visited_lvl": empty.copy(),
            "n_deduced": 0,
            "n_guessed": 0,
            "n_visited": 0,
            "n_deduced_sol": 0,
            "n_guessed_sol": 0,
            "n_visited_sol": 0,
            "n_branches_lvl": deepcopy(empty2d),
            "avg_branches_lvl": empty.copy(),
            "max_branches_lvl": empty.copy(),
            "avg_branches": 0,
            "max_branches": 0,
            "guess_branches_lvl": deepcopy(empty2d),
            "avg_guess_branches_lvl": empty.copy(),
            "max_guess_branches_lvl": empty.copy(),
            "avg_guess_branches": 0,
            "max_guess_branches": 0,
            "n_path_lvl": empty.copy(),
            "refresh_lvl": deepcopy(empty2d),
            "refresh_path_lvl": empty.copy(),
            "sum_refresh_lvl": empty.copy(),
            "avg_refresh_lvl": empty.copy(),
            "avg_refresh": 0,
            "max_refresh_lvl": empty.copy(),
            "max_refresh": 0,
            "n_clues": 0,
            "max_sols": 0,
            "n_sols": 0,
        }
        return metrics
        
        
    def update_metrics(self, metrics: dict):    
        """
        Update the metrics which describe the object behaviour and 
        performance.

        Level 0 (root noe):     Empty board. Currently not refreshed.
        Level 1:                1 solution on board
        Level n:                n solutions on board. Clues and/or guesses
        Level size^2:           Solved board. Branches=0. Currently not processed.
        """

        #  There will be one more level than there are cells in the
        # sudoko because of the extra root node.
        levels = self.size**2 + 1

        # Correct n_guessed_lvl. A node is assumed to be a guess when
        # the first sol is popped from the sols list. This is true
        # except for level one, when the first clue is added.
        for idx in [0, 1]:
            metrics["n_guessed_lvl"][idx] = 0
            metrics["n_visited_lvl"][idx] = 1
            metrics["deduced_lvl"][idx] = True
            metrics["n_deduced_lvl"][idx] = 1
            metrics["n_branches_lvl"][idx] = [1]

        metrics["refresh_lvl"][0].append(0)
        metrics["refresh_path_lvl"][0] = metrics["refresh_start"]

        # Calculate average and max values
        msg("metrics['lvl']:", metrics["lvl"])
        msg("size**2 + 1:", levels)

        # guess_last_lvl: last level which was guessed.
        # Search backwards through the list to find the last node
        # which was not deduced.
        for idx in range(levels-1, -1, -1):
            if not metrics["deduced_lvl"][idx]:
                metrics["guess_last_lvl"] = idx
                break

        # Aggregate values by level.
        for idx in range(levels - 1):
            #
            #
            #
            msg("Aggregating level:", idx)

            metrics["avg_branches_lvl"][idx] = (
                np.mean(metrics["n_branches_lvl"][idx]))
            metrics["max_branches_lvl"][idx] = (
                max(metrics["n_branches_lvl"][idx]))
            
            # Branching for guessed nodes (branches > 1).
            metrics["guess_branches_lvl"][idx] = (
                [b for b in metrics["n_branches_lvl"][idx] if b > 1])
            if metrics["guess_branches_lvl"][idx] != []:
                metrics["avg_guess_branches_lvl"][idx] = np.mean(
                    metrics["guess_branches_lvl"][idx])
                metrics["max_guess_branches_lvl"][idx] = np.max(
                    metrics["guess_branches_lvl"][idx])
            else:
                metrics["avg_guess_branches_lvl"][idx] = 0
                metrics["max_guess_branches_lvl"][idx] = 0
            
            #continue
            # Refresh time
            metrics["refresh_path_lvl"][idx] += -metrics["refresh_start"]
            metrics["sum_refresh_lvl"][idx] = (
                np.sum(metrics["refresh_lvl"][idx]))
            metrics["avg_refresh_lvl"][idx] = (
                np.mean(metrics["refresh_lvl"][idx]))
            metrics["max_refresh_lvl"][idx] = (
                np.max(metrics["refresh_lvl"][idx]))

        # Aggregate values across all levels.
        metrics["n_deduced"] = sum(metrics["n_deduced_lvl"])
        metrics["n_guessed"] = sum(metrics["n_guessed_lvl"])
        metrics["n_visited"] = (
            metrics["n_deduced"] + metrics["n_guessed"])
        metrics["avg_branches"] = np.mean(metrics["avg_branches_lvl"])
        metrics["max_branches"] = np.max(metrics["max_branches_lvl"])

        # Branching for guesses
        guess_branches = list(chain(*metrics["guess_branches_lvl"]))
        #
        # DEBUG
        #
        msg("guess_branches:", guess_branches)

        metrics["avg_guess_branches"] = np.mean(guess_branches)
        metrics["max_guess_branches"] = np.max(guess_branches)

        # Deduced v guessed in the solution.
        metrics["n_deduced_sol"] = metrics["deduced_lvl"].count(True)
        metrics["n_guessed_sol"] = metrics["deduced_lvl"].count(False)
        metrics["n_visited_sol"] = levels

        # Refresh time
        metrics["avg_refresh"] = np.mean(metrics["avg_refresh_lvl"])
        metrics["max_refresh"] = np.max(metrics["max_refresh_lvl"])
        
        self.metrics=metrics

        #
        # DEBUG
        #
        #msg("n_deduced:", metrics["n_deduced_lvl"])
        #msg("branches_lvl[36]:", metrics["branches_lvl"][36])
        #msg("avg_branches_lvl:", metrics["avg_branches"])


    def set_puzzle(self, puz=None):
        """Initialise the Board with a list of solutions to process
        
        puz : str, list, or set
            Name of known puzzle. Or collections of 3-tuples
        """
        if puz is None:
            puz = []

        self.rows, self.cols, self.blks = [], [], []
        
        # Create lists of each type of section.
        for i in range(self.size):
            self.rows.append(Row(self.size))
            self.cols.append(Column(self.size))
            self.blks.append(Block(self.size))

        self.board = []

        # Create a deque of sols based on the puz argument
        self.new_sols = self.init_new_sols(puz)

        #
        # DEBUG
        #
        #print("[set_puzzle]self.new_sols", self.new_sols)

        # 
        # TODO - Error checking. Symbols in puz from 0 to size.
        #

        self.solutions_found = 0    # Reset the solution counter.
    
    @singledispatchmethod
    def init_new_sols(self, puz):
        """If a string was passed, read the puzzle from the puzzle library.
        """
        #
        # TODO - Error checking to ensure a string was passed
        #

        # Read the puzzle then encode it
        return self.encode(deque(read_puzzle(puz)))

    @init_new_sols.register
    def _(self, puz : list | set | deque):
        """If a collection was passed, add it directly to the solutions set.
        """
        # Ensure collection is a set then encode
        return self.encode(deque(puz))

    @init_new_sols.register
    def _(self, puz : np.ndarray):
        """If numpy.ndarray was passed, extract the clues into a list.
        """

        # Flatten ndarray and filter on cells with values.
        # ndarrys are assumed to be zero indexed. Therefore do not encode.
        clues = [(r, c, puz[r, c]) 
                 for (r, c) in np.ndindex(self.size, self.size)
                 if puz[r, c] is not None]
        return deque(clues)

    def encode(self, puz):
        """Convenience function to decrement all clues by on and
        reindex to zero.
        """

        # Reindex the rows and columns, and decrement the numbers
        # to match the SudokoSolver internal representation.
        # Skip this step if zero_index==True
        if not self.zero_index:
            return deque((r-1, c-1, n-1) for (r, c, n) in puz)
        else:
            return puz
    

    def copy(self):

        """Create a deep copy of the board
        """
        
        #
        # TODO - create a deep copy using Python deep copy function
        #

        b_copy = self.__new__(SudokoSolver)
        
        b_copy.rows, b_copy.cols, b_copy.blks = [], [], []
        
        for i in range(self.size):
            b_copy.rows.append(self.rows[i].copy())
            b_copy.cols.append(self.cols[i].copy())
            b_copy.blks.append(self.blks[i].copy())
        
        # The new board should only copied if there are no processed solutions
        if len(self.new_sols) > 0:
            print("[Board.copy ERROR] attempted to copy board with len(new_sols) > 0")
            return None
        else:
            b_copy.size = self.size
            b_copy.box_size = self.box_size
            b_copy.zero_index = self.zero_index
            b_copy.max_solutions = self.max_solutions
            b_copy.solutions_found =self.solutions_found
            b_copy.new_sols = self.new_sols.copy()
            b_copy.board = self.board.copy()

        return b_copy

    
    def refresh(self, lvl, metrics: dict):
        """Iterate through each solution in the new solution set. Ordering is unspecified.
        New items may be added to the set during refresh.
        Maintain a list of rows, columns and blocks which were updated. These need to be rechecked 
        to see if they contain additional solutions. If so they are added to the new_sols set

        Parameters
        ----------
        lvl : int
            The depth of the current node in the search.
        metrics : dict
            Performance measures updated while executing the 
            algorithm.
        """
        #
        # TODO - Remove max_updates. Not required for puzzle making.
        # Also, don't care about the order of processing the queue.
        #

        #
        # TODO Delete. n_sols only referenced by _solve.
        #
        #n_sols = 0

        #
        # DEBUG
        #
        #msg("Iterating through new_sols...")
        #metrics["n_guessed_lvl"][lvl] += 1

        # Iterate through solutions in new_sols
        first_sol = True
        refresh_time = perf_counter()
        while len(self.new_sols) > 0:

            #
            # DEBUG
            #
            #msg("First sol being added...")
            #msg("len(new_sols):", len(self.new_sols))
            #msg("depth:", depth)

            # pop a solution from new_sols set and append to board list
            sr, sc, sn = self.new_sols.pop()
            self.board.append((sr, sc, sn))

            #
            # DEBUG
            #
            #msg("[refresh]popped solution (r, c, n):", sr, sc, sn)
            #msg("popped solution (r, c, n):", sr, sc, sn )
            #msg("new board:", self.board)

            #
            # DEBUG
            #
            #msg("[refresh]Current board list:", self.board)
            #msgpause()


            # Calculate the block indices and initialise sets to 
            # store sections which were updated.
            sb, sbp = self.block_map[sr][sc]
            updated_rows = set()
            updated_cols = set()
            updated_blks = set()

            #
            # DEBUG
            #
            # Print the board after each new solution added - for testing purposes
            #print("Added:", sr, sc, sn)
            #print_puzzle(self.board)
            #print("Solution Row, Col, Blk, Blk_pos, Num:", sr, sc, sb, sbp, sn)

            # First - remove the solution from the associated row, column, and block

            #
            # DEBUG
            #
            #msg("removing sol from rows, cols, and blks...")

            try:
                self.rows[sr].remove_sol(sc, sn)
                self.cols[sc].remove_sol(sr, sn)
                self.blks[sb].remove_sol(sbp, sn)
            except InvalidState as error:
                #
                # DEBUG
                #
                #msg("Failed to remove sol from row, col, or blk")
                
                return SudokoSolver.INVALID

            # Add updated sections to updated lists to be checked
            updated_rows.add(sr)
            updated_cols.add(sc)
            updated_blks.add(sb)

            #print("Row w removal")
            #print(rows[sr].sol_space)
            #print("column w removal")
            #print(rows[sr].sol_space)
            #print("block w removal")
            #print(blks[sb].sol_space)

            # Second - update impacted cells in the solution space for every row, column, and block
            # The will be no overlap in cells, because the solution which was just removed is the 
            # intersection of the rows and columns

            # Update columns and blocks based on the unsolved positions in the row which was just updated
            # Also add updated sections to the updated list, but only if the update returns True

            # Update columns and blocks for each element in the updated row
            for refresh_col in self.rows[sr].sol_space.index:

                ##TODO
                # Move this mapping logic to the Block class

                #
                # DEBUG
                #
                #print("[refresh]sr & refresh_col", sr, refresh_col)
                #print("[refresh]self.block_map_rev:", self.block_map_rev[sr][refresh_col])
                
                refresh_blk, refresh_blk_pos = self.block_map_rev[sr][refresh_col]

                if self.cols[refresh_col].update_sol(sr, sn): 
                    updated_cols.add(refresh_col)

                if self.blks[refresh_blk].update_sol(refresh_blk_pos, sn):
                    updated_blks.add(refresh_blk)

                #print("Column:",refresh_col)
                #print(cols[refresh_col].sol_space)
                #print("Block:", refresh_blk)
                #print(blks[refresh_blk].sol_space)

            # Update rows and blocks based on the unsolved positions in the column which was just updated
            # Also add updated sections to the updated list, but only if the update retrurns True

            for refresh_row in self.cols[sc].sol_space.index:

                ##TODO
                # Move this logic to the Block class
                refresh_blk, refresh_blk_pos = self.block_map_rev[refresh_row][sc]

                if self.rows[refresh_row].update_sol(sc, sn):
                    updated_rows.add(refresh_row)

                if self.blks[refresh_blk].update_sol(refresh_blk_pos, sn):
                    updated_blks.add(refresh_blk)

                #print("Row:",refresh_row)
                #print(rows[refresh_row].sol_space)
                #print("Block:", refresh_blk)
                #print(blks[refresh_blk].sol_space)

            ##TODO
            # Create a loop like the ones above for row-wise and column-wise updates, to do block-wise updates
            # Is this really neccessary? Not sure. Theoretically it will and additional 4 locations in 
            # the row and column solution spaces.
            #
            # Make the updates parameterisable. Then measure the performance with different options enabled.
            

            ### TODO ###
            # UNTESTED

            # Finally, check the solution space for the updated rows, columns, and block for any new solutions
            # new_sol returns a position and number. So iterate through them and add the full
            # row-columm-number to the new_sols global deque.
            
            #print("Rows updated:", updated_rows)
            #print("Cols updated:", updated_cols)
            #print("Blks updated:", updated_blks)

            # For each section collect partial solutions into sols_to_add
            # then combine the full solutions for all sections into 
            # rcb_sols_to_add.
            sols_to_add = set()
            rcb_sols_to_add = set()

            #
            # DEBUG
            #
            #msg("finding new sols in rows...")

            for sr in updated_rows:
                #print("Updated row:", sr)
                try:
                    sols_to_add = self.rows[sr].new_sols()
                except InvalidState as error:
                    return SudokoSolver.INVALID


                for sol_pos, sol_num in sols_to_add:
                    #print("New solution:", sr, sol_pos, sol_num)
                    
                    rcb_sols_to_add.add((sr, sol_pos, sol_num))
                    #self.new_sols.add((sr, sol_pos, sol_num))

            #
            # DEBUG
            #
            #msg("finding new sols in cols...")

            for sc in updated_cols:
                #print("Updated col:", sc)
                try:
                    sols_to_add = self.cols[sc].new_sols()
                except InvalidState as error:
                    return SudokoSolver.INVALID

                for sol_pos, sol_num in sols_to_add:
                    #print("New solution:", sol_pos, sc, sol_num)

                    #self.new_sols.append((sol_pos, sc, sol_num))
                    rcb_sols_to_add.add((sol_pos, sc, sol_num))
                    #self.new_sols.add((sol_pos, sc, sol_num))

            #
            # DEBUG
            #
            #msg("finding new sols in blocks...")

            for sb in updated_blks:
                #print("Updated blk:", sb)
                try:
                    sols_to_add = self.blks[sb].new_sols()
                except InvalidState as error:
                    return SudokoSolver.INVALID

                for sol_pos, sol_num in sols_to_add:
                    sr, sc = self.block_map_rev[sb][sol_pos]
                    #print("New solution:", sr, sc, sol_num)

                    #self.new_sols.append((sr, sc, sol_num))
                    rcb_sols_to_add.add((sr, sc, sol_num))
                    #self.new_sols.add((sr, sc, sol_num))

            #
            # DEBUG
            #
            #msg("found more solutions to add:", rcb_sols_to_add)

            # Add the new solutions from rows, columns, and blocks 
            # to the new_sols deque. To ensure uniqueness, cast
            # new_sols as a set before combining, then cast back
            # to a deque.
            self.new_sols = deque(set.union(set(self.new_sols), rcb_sols_to_add))

            # Metric Calculations
            #====================
            # The level is now one greater than it was before the
            # solution was added.
            lvl += 1

            #
            # DEBUG
            #
            #msg("Level being refreshed:", lvl)

            metrics["lvl"] = lvl
            metrics["n_visited"] += 1
            metrics["n_visited_lvl"][lvl] += 1
            metrics["n_path_lvl"][lvl] = metrics["n_visited"]

            # Branching
            if first_sol == True:
                metrics["n_guessed_lvl"][lvl] += 1
                metrics["deduced_lvl"][lvl]=False
                first_sol = False
            else:
                #msg("before - metrics['branches_lvl'][lvl-1]:", metrics["branches_lvl"][lvl-1])
                metrics["n_branches_lvl"][lvl-1].append(1)
                metrics["n_deduced_lvl"][lvl] += 1
                metrics["deduced_lvl"][lvl] = True

            # Refresh times
            refresh_start = refresh_time
            refresh_time = perf_counter()

            metrics["refresh_lvl"][lvl].append(refresh_time - refresh_start)
            metrics["refresh_path_lvl"][lvl] = refresh_time

        # END OF MAIN LOOP
        # Finally - Return the status of the board
        
        #
        # TODO - move the block to immediately after the board update
        # This will avoid a small amount of admin when the board is
        # solved.
        #
        
        ## TODO - TESTING
        ## Print the final state of the board after refreshing
        #print_puzzle(self.board)
            
        #
        # DEBUG
        #
        #msg("returning. Length of board:", len(self.board), "size**2:", self.size**2)
        
        if len(self.board) == self.size**2:
            return SudokoSolver.SOLVED
        else:
            return SudokoSolver.UNSOLVED

    def add_sol(self, solution: tuple, lifo: bool=False):
        """
        Adds a single solution to the solution space. This is used to make a guess.
        solution is a set of a 3-tuples of row, column, number.

        lifo=True changes the behaviour of element being added if it
        is already in the deque. In this case, it removes the element
        then appends it to the top of the deque to ensure it is popped
        first. lifo is only set when generating puzzles. For normal
        refreshes, if the solution is already in add_sol appending is
        skipped.

        Parameters
        ----------
        solution : 3-tuple
            (row, column, number)
        """

        #
        # TODO - change new_sols back to a set. faster than checking
        # if solution is in new_sols
        #

        ##TODO
        # Untested
        # Do I  need try-except block around add()? Maybe not

        #
        # DEBUG
        #
        #print("[add_sol]Appending solution (r, c, n):", solution)
        #msg("adding sol... (r, c, n):", solution)


        if (solution) in self.board:
            print("New solution is already on the board. Cannot add an existing solution again:", solution)

        elif lifo:
            # if lifo=True ensure the solution is on top of new_sols
            # by removing before reappending it
            if solution in self.new_sols:
                self.new_sols.remove(solution)
            self.new_sols.append(solution)
        else:
            # If lifo=False and the solution is already in new_sols
            # then do not reappend it.
            if solution not in self.new_sols:
                self.new_sols.append(solution)

        #
        # DEBUG
        #
        #msg("sol added. Returning from add_sol")
               

    
    def create_block_maps(self):
        
        #
        # TODO
        # Replace this with a hard coded map, and remove the function
        # The Board should not be aware of the mapping. Perform this in the Block class.
        
        # This function is only called by init, and not by copy

        # Initialise two lists of lists
        self.block_map = [0 for i in range(self.size)]
        self.block_map = [self.block_map.copy() for i in range(self.size)]
        self.block_map_rev = self.block_map.copy()

        # Create mapping between row-col and block-position indices.
        for r in range(self.size):
            for c in range(self.size):
                blk = int((r//self.box_size)*self.box_size + c//self.box_size)
                blk_pos = int((r%self.box_size)*self.box_size + c%self.box_size)

                self.block_map[r][c] = (blk, blk_pos)
                self.block_map_rev[blk][blk_pos] = (r, c)


    def get_guesses(self):
        """
        Derive a list of guesses to try. 
        Guesses can be either:
        * All possible numbers for a given position in a given row, column, or block
        * All possible positions for a given number in a given row, column, or block
        """

        # Iterate through the solution spaces for every row, column, and block
        # In each, find the position or number with the fewest possible solutions
        # Using the solution space with the fewest solutions will minimise branching in the search for a
        # correct solution.
        #
        # Examples:
        #   Position 1 can only have the numbers 3 or 4.
        #   Number 5 can only be in position 7 or 8.

        # Check whether the board has been fully solved already.
        # This can happen when manually adding clues and refreshing.
        # If it has, return an empty list of quesses.
        if len(self.board)==self.size**2:
            return []
        
        # section.get_guesses returns a 2-tuple (number of solutions, list of solution tuples)
        # Each solution is a 2-tuple (position, number)
        
        #
        # DEBUG
        #
        guess_section = ("fubar", "fubar", "fubar")

        min_guesses = 16
        for section in [self.rows, self.cols, self.blks]:

            # Check for guesses only if the section has not been solved
            for idx, s in enumerate(section):

                if not s.solved:

                    n_g, guesses = s.get_guesses()
                    #print(s.section_type, "guesses:", n_g, guesses)

                    # Exit the loop early if a 2 element solution space
                    # is found as this is the minimum possible size.
                    if n_g < min_guesses: 
                        min_guesses = n_g
                        #
                        # TODO - This line updated to be consistent with zero indexing.
                        # Untested.

                        #guess_section = (s.section_type, idx+1, guesses) # Add 1 to idx because it starts from 0
                        guess_section = (s.section_type, idx, guesses)
                        if min_guesses == 2: break

            if min_guesses == 2: break
        
        # Map the solutions found for a section to the board
        # Board coordinates are (row, column)
        # Sections have just a position which needs to be combined with the number of the 
        # row, column or block itself to determine the row and column.
        #   guess_section = (section type, section number, guess list)
            
        #
        # DEBUG
        #
        #msg("guess_section.type:", guess_section[0], "guess_section.idx:", guess_section[1])
        #msg("guess_section.guesses:", guess_section[2])
        #msgpause()

        board_guesses = []

        if guess_section[0] == "Row":
            row = guess_section[1]
            board_guesses = [(row, g[0], g[1]) for g in guess_section[2]]
                
        if guess_section[0] == "Column":
            col = guess_section[1]
            board_guesses = [(g[0], col, g[1]) for g in guess_section[2]]

        # Storing block-based guesses requires mapping from the board and position to a row and column
        if guess_section[0] == "Block":
            blk = guess_section[1]
            board_guesses = [ self.block_map_rev[blk][g[0]] + (g[1], ) for g in guess_section[2]]
            
        ##TODO
        ## Refer to block_map_rev above. This should be done in the Block class.
        ## The board should not be aware of the mapping

        return board_guesses
    
    def many_sols(self):
        """Test whether more than one solution exists for the puzzle.

        This function is a wrapper for solve. It sets max_solutions=2
        which forces solve to search for one more solutions after it 
        finds the first.
        """

        status, solution = self.solve(max_sols=2)
        
        return status==SudokoSolver.SOLVED

    def solve(self, max_sols=1):
        """
        Wrapper function for __solve to derive arguments.

        Parameters
        ==========
        max_sols: int
            Maximum number of solutions to find.
        """

        # Initialise the metrics dictionary. It will not be an
        # argument in the initial call to solve.
        metrics = self.init_metrics()

        metrics["n_clues"] = len(self.new_sols)
        metrics["max_sols"] = max_sols
        metrics["refresh_start"] = perf_counter()
    
        status, solution = self.__solve(lvl=0, metrics=metrics)

        self.board = solution
        self.update_metrics(metrics)
        self.metrics = metrics
        
        return status, solution


    def __solve(self, lvl, metrics: dict):
        """
        This function attempts to solve a board recursively. 
        
        Parameters
        ----------
        lvl : int
            Depth of the search. Root instance of solve will have
            depth=0, and will be an empty board.
        metrics : dict
            Metrics describing the performance of the algorithm.

        #
        # TODO Parameters deleted
        #

        solutions_found : int
            Number of solutions found. Required when searching for
            multiple solutions.
        max_solutions : int
            Number of solutions required. 1 for regular solve, 2 when
            called from many_sols.
        
        Return values
        -------------
        status : int
            INVALID=0, SOLVED=1, UNSOLVED=2. SOLVED(1) if successful else INVALID(0)
        board : SudokoSolver | None
            Fully solved board if solved, else None
        """
        #
        # TODO Add flag to enable metrics
        #

        # If refresh solved the puzzle then return SOLVED unless more
        # than one solution is being searched for. This is the case 
        # when solve is called by many_sols. No need to search for 
        # more solutions in that case, because the board was solved 
        # without making any guesses. Therefore no other solutions 
        # can exist.

        #
        # DEBUG
        #
        #msg("Calling refresh... Level:", lvl)

        status = self.refresh(lvl, metrics)

        #msg("Returned from refresh. status:", status)
        #msg("Sols to be added (should be empty):", self.new_sols)
        #msg("Refresehed board:", self.board)
        #msgpause()

        if status == SudokoSolver.SOLVED:
            metrics["n_sols"] += 1
            if metrics["n_sols"] >= metrics["max_sols"]:
                #return SudokoSolver.SOLVED, self
                #status = SudokoSolver.SOLVED
                solution = self.board
            else:
                #return SudokoSolver.INVALID, None
                status = SudokoSolver.INVALID

        elif status == SudokoSolver.UNSOLVED:

            guesses = self.get_guesses()
            #print("G:", len(guesses), "Depth:", depth)

            # Update metrics with the branching factor
            #
            # TODO Implement lvl (level) in refresh
            #
            lvl = len(self.board)
            metrics["lvl"] = lvl
            metrics["n_branches_lvl"][lvl].append(len(guesses))

            #
            # DEBUG
            #
            #msg("Guessing. guesses:", guesses, "depth:", depth)
            #msg("branches (len(guesses)):", len(guesses))
            #msg("branches:", metrics["branches_lvl"][depth])
            #msg("Iterate guesses...")
            #msgpause()

            g = 0
            while g < len(guesses) and status != SudokoSolver.SOLVED:
            #for g in guesses:

                #
                # DEBUG
                #
                #msg("making a guess:", g)
                
                # Make a copy of the original board. If backtracking is required another copy
                # will be made from the original for the next guess.
                brd = deepcopy(self)

                #
                # DEBUG
                #
                #msg("Adding guess... Depth:", depth, "guess:", guesses[g])
                brd.add_sol(guesses[g])

                # Recurrsively try to solve the board. 
                # solve returns either SOLVED or INVALID
                status, solution = brd.__solve(lvl, metrics)
                
                g += 1

        # If all guesses tried and invalid, return INVALID then 
        # backtrack.
        
        #return SudokoSolver.INVALID, None
        if status == SudokoSolver.INVALID:
            return SudokoSolver.INVALID, None
        else:
            return SudokoSolver.SOLVED, solution


    def __solve2(self, depth=0, solutions_found=0, max_solutions=1, metrics: dict=None):
        """
        This function attempts to solve a board recursively. 
        
        Parameters
        ----------
        depth : int
            Depth of the search. Root instance of solve will have
            depth=0.
        solution_found : int
            Number of solutions found. Required when searching for
            multiple solutions.
        max_solutions : int
            Number of solutions required. 1 for regular solve, 2 when
            called from many_sols.
        metrics : dict
            Metrics describing the performance of algorithm.
        
        Return values
        -------------
        status : int
            INVALID=0, SOLVED=1, UNSOLVED=2. SOLVED(1) if successful else INVALID(0)
        board : SudokoSolver | None
            Fully solved board if solved, else None
        """
        #
        # TODO Add flag to enable metrics
        #

        #
        # TODO - Test this extra run of refresh.
        # I added it after solve was already tested and working.
        #

        #
        # DEBUG
        #
        #msg("calling refres...")

        # If refresh solved the puzzle then return SOLVED unless more
        # than one solution is being searched for.
        # This is the case when solved is called by many_sols. No need to
        # search for more solutions, because the board was solved without
        # making any guesses. Therefore no other solutions can exist.
        if self.refresh(metrics) == SudokoSolver.SOLVED:
            if max_solutions==1:
                return SudokoSolver.SOLVED, self
            else:
                return SudokoSolver.INVALID, None

        guesses = self.get_guesses()
        #print("G:", len(guesses), "Depth:", depth)

        # Update metrics with the branching factor
        #
        # TODO Implement lvl (level) in refresh
        #
        metrics["lvl"] = len(self.board)
        metrics["branches_lvl"][metrics["lvl"]].append(len(guesses))
        
        for g in guesses:

            #
            # DEBUG
            #
            #msg("making a guess:", g)
            
            # Make a copy of the original board. If backtracking is required another copy
            # will be made from the original for the next guess.
            #brd = self.copy()
            brd = deepcopy(self)

            #
            # DEBUG
            #
            #msg("Adding sol and calling refresh...")

            brd.add_sol(g)
            status = brd.refresh(metrics)

            if status == SudokoSolver.SOLVED: 

                # Return SOLVED. Unless more than one solution is being searched for.
                # This is the case when solved is called by many_sols.
                solutions_found += 1
                if solutions_found >= max_solutions:
                    #
                    # TODO metrics only need to be updated once. Not
                    # at every level.
                    #
                    return SudokoSolver.SOLVED, brd
            
            elif status == SudokoSolver.INVALID:
                # Do nothing. Just keep looping through guesses.
                # status==INVALID. 
                pass            
            
            else:
                # Recurrsively try to solve the board. 
                # solve returns either SOLVED or INVALID
                status, solved_brd = brd.__solve2(
                    depth+1, max_solutions, solutions_found, metrics)
                
                # No need to check max_solutions because this was
                # done during the call to solve following refresh.
                if status == SudokoSolver.SOLVED:
                    return SudokoSolver.SOLVED, solved_brd
                else:
                    # status==INVALID. Just keep looping through guesses
                    # Do nothing

                    # Solve will keep trying until it either succeeds
                    # or fails. It will not give up. So it will never 
                    # return UNSOLVED. Only refresh returns UNSOLVED
                    pass
        else:
            # If all guesses tried and invalid, return INVALID then 
            # backtrack.
            return SudokoSolver.INVALID, None


import unittest

class TestBoard(unittest.TestCase):
    def test_initialise(self):
        brd = Board("test_puzzle")
        brd.refresh()
        self.assertEqual(brd.board, [(1,5,8)])
        
    def test_add_sol(self):
        brd = Board("test_puzzle")
        brd.add_sol((1,1,1))
        brd.refresh()
        self.assertEqual(brd.board, [(1,5,8),(1,1,1)])
        
    def test_get_guesses(self):
        brd = Board("test_puzzle")
        self.assertIsNotNone(brd.get_guesses)
        
        guesses = brd.get_guesses()
        self.assertIsNotNone(guesses)

#unittest.main(defaultTest=["TestBoard"], argv=['ingored', '-v'], exit=False)


class TestInitialise(unittest.TestCase):

    """
    To Do - I don't think this test is relevant. Delete it.
    No 
    """
    def test_read_puzzle(self):
        puzzle = read_puzzle("test_puzzle")
        self.assertEqual(puzzle, {(1,5,8)})
        
    def test_section_sizes(self):
        initialise_board("test_puzzle")
        self.assertEqual(len(rows), 10)      # lists are 10 long, but zeroth element is ignored
        
    def test_refresh_board(self):
        #Soution row 1, col 5, num 8
        # Mapped to block 2 position 2
        initialise_board("test_puzzle")
        refresh_board()
        r_shape = rows[1].sol_space.shape
        c_shape = cols[5].sol_space.shape
        b_shape = blks[2].sol_space.shape
        r_indices = list(rows[1].sol_space.index.values)
        c_indices = list(cols[5].sol_space.index.values)
        b_indices = list(blks[2].sol_space.index.values)
        
        self.assertEqual(r_shape, (8,8))
        self.assertEqual(c_shape, (8,8))
        self.assertEqual(b_shape, (8,8))
        self.assertEqual(r_indices, [1,2,3,4,6,7,8,9])
        self.assertEqual(c_indices, [2,3,4,5,6,7,8,9])
        self.assertEqual(b_indices, [1,3,4,5,6,7,8,9])    

#unittest.main(defaultTest=["TestInitialise"], argv=['ingored', '-v'], exit=False)