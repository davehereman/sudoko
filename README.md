# sudoko
Suite of programs for creating, solving, and analysing sudokos of different sizes

## Overview
This project includes programs, written in Python, which implement functionality related to the popular number placement puzzle named Sudoko.  Classes are available to:
* Create Sudokos i.e. completed board with valid placement of numbers
* Create puzzle i.e. partially complete board, with only one possible solution.
* Solve puzzles

The size of sudok can be set to any of 1 x 1, 4 x 4, 9 x 9, 16 x 16, or 25 x 25. The solver currently uses a recursive back tracking algorithm and cannot support larger puzzles.

The solver can be run with user-defined functions for refreshing the board. It can provide detailed metrics which allow different functions to be evaluated.
