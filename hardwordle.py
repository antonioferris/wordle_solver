"""
    This file contains HardWordle, an implementation of a hard mode Wordle
    solver that tries to use numpy matrices to allow for greater efficiency
    when searching the wordle problem space.

    Most of the benefit of HardWordle comes from storing precomputed data in a specific form.

    The matrix M is a large array for each pair of possible guess and possible word in the dictionary.
    Each element in the array is the index of the result that guess gives when the true word is that word.

                word_1  word_2  word_3 ...
    guess_1       4       2        2
    guess_2       1       2        2
    guess_3       0       10       11
    ...

    Suppose that poss_words is a list of the indexes of possible words, and guess is the index of a possible guess.
    Then, M[guess, poss_words] is an array of the results each possible word gives for our guess!!
    Also, np.where(M[guess, poss_words] == res) is an array of the indices of the words that could still be the word after our guess!
"""
from collections import defaultdict
import numpy as np
import time

class HardWordle:
    """
        This class contains a solver for Hard mode wordle.
        This means that every new guess must use the results
        of your previous guesses. i.e. It must be a valid word
        given your previous guesses.
    """
    def __init__(self, M, word_dict, guess_dict, results_dict, global_guess_list=None):
        """
        Initialize the HardWordle, pretty self explanatory.
        Given a global_guess_list, this class will try to solve only those guesses.

        Recommended usage is either have global_guest_list as default and call solve with verbose=False
        and come back a couple days later,

        or give global_guest_list as your personal favorite guesses for hard mode and call solve
        with verbose=True to see which one is best.

        Args:
            M (numpy array): A large, precomputed array of the result of any possible word as a guess with any other word.
            word_dict (list[str]): the dictionary, also the de-indexing key for all of the word indices being used in this class
            guess_dict (list[str]): the allowed guesses, also the de-indexing key for all of the guest indices used.
            results_dict (list[str]): the possible results, also the de-indexing key for all possible results
            global_guess_list (list[int], optional): the guesses to try as an initial guess. Defaults to None.
        """
        self.n = M.shape[0]
        self.m = 2315
        self.M = M
        self.word_dict = np.array(word_dict)
        self.guess_dict = guess_dict
        self.results_dict = results_dict
        self.global_guess_list = global_guess_list # the guesses this Wordle will try as first guesses
        self.hist = [] # history of guesses and results.
        self.tree = {} # decision tree - actual solution to the game!!

        # get letter frequency for our heuristic
        self.letter_freqs = defaultdict(int)
        for word in word_dict:
            for letter in word:
                self.letter_freqs[letter] += 1

        self.guess_order = sorted(range(self.n), key = lambda g : self.h(g), reverse=True)

        if not self.global_guess_list:
            self.global_guess_list = self.guess_order

        self.cap = 10000
        self.cache = {}

    def solve(self, words, guess_words, verbose=False):
        """
        Recursively solves the Wordle problem given a dictionary
        and an allowed guess list.

        Args:
            words (list[int]): list of word indexes of dictionary (index into self.word_dict)
            guess_words (list[int]): list of word indexes of allowed guesses (index into self.guess_dict)
            verbose (bool, optional): Give more verbose output to command line. Defaults to False.

        Returns:
            solution_tree (dict): The solution tree gives the correct first guess and second guess
            for every possible result. This isn't the full solution, but is enough such that the full solution
            can be computed very quickly.
            The solution_tree is structured as a dictionary from a history tuple to a (guess, cost) tuple giving
            the optimal guess and the optimal cost obtained from that state.
            solution_tree[()]  = (initial_guess, total_cost) gives the best initial guess and the best cost obtained from the solve function.
            solution_tree[(initial_guess, r)] = (second_guess, seond_cost) gives the optimal second guess after obtaining result r.
        """
        min_cost = 4 * len(words) # we know that we can do better than 4 guesses on average.
        guess = None

        # for tracking progress
        w = len(words)

        for g in self.global_guess_list: # guess words in our guess list
            self.hist.append(g) # add guess to history
            cost = len(words) # need 1 guess for every current word.
            results = self.M[g, words]
            guess_results = self.M[g, guess_words]

            r = 1
            t = len(np.unique(results))
            # search the easier results first.
            unique_results, counts = np.unique(results, return_counts=True)
            unique_results = unique_results[np.argsort(counts)]

            for result in unique_results: # for each unique result
                if result == 0:
                    continue # result of 0 means we guessed it! No additional cost.
                self.hist.append(result) # add result to history
                s = time.time()

                new_words = words[np.where(results == result)[0]]
                new_guess_words = guess_words[np.where(guess_results == result)[0]]
                alpha = min_cost - cost # if the cost of this call is greater than min_cost - cost, this whole guess isn't going to be used.

                w -= len(new_words)

                next_guess, inc = self.solve_cost(new_words, new_guess_words, alpha, 1)
                self.tree[tuple(self.hist)] = next_guess, inc
                if verbose and inc >= alpha: # if we failed, let's print what we failed on.
                    print(f"Failed to solve ({self.guess_dict[g]}, {self.results_dict[result]}) giving {len(new_words)} words in {5} guesses.")
                cost += inc

                self.hist.pop() # remove result from history after consideration
                d = time.time() - s
                if verbose and d > 3:
                    print(f"Result {r} / {t} ({len(new_words)} words) has cost {inc} (cumulative {cost}) took {d:.2f}s {w} words left to explore.")
                    r += 1

                if cost >= min_cost: # break early if this guess is not good enough.
                    break

            if cost < min_cost: # update min_cost and guess if we are minimal
                if verbose and g in words:
                    print(self.guess_dict[g], cost)
                min_cost = cost
                guess = g

            self.hist.pop() # remove guess from hstory after consideration

        guess_word = (self.guess_dict[guess], min_cost)
        if verbose and guess:
            print(f"FINAL GUESS {guess_word} cost {min_cost}")

        # we return the tree of the solved game, made from the tree as we went.
        # the only part of the tree we need is the branches that start with the "correct" guess.
        solution_tree = {hist : sol for hist, sol in self.tree.items() if hist[0] == guess}
        solution_tree[tuple([])] = (guess, min_cost)
        return solution_tree

    def solve_cost(self, words, guess_words, alpha, d):
        """
        The recursive workhorse of the solve function, solve_cost
        recursively computes the best guess to give in any situation
        and returns that best guess along with the minimum cost obtained.

        Args:
            words (list[int]): list of word indices
            guess_words (list[int]): list of guess indices
            alpha (int): the maximal "useful" cost, used for alpha pruning
            d (int): depth of the search

        Returns:
            tuple(guess, min_cost)): tuple of the optimal guess and the cost it achieves on the words given.
        """
        n = len(words)

        if d > 5: # if we exceeded maximal recursive depth
            return None, 10000

        if d == 5 and len(words) > 1: # this is an impossible case to succeed in
            return None, 10000

        if n <= 2:
            return words[0], 2 * n - 1 # we are always optimal in the 1 or 2 words case.

        min_cost = alpha # any cost great than alpha is useless.
        min_guess = None
        guess_words = np.sort(guess_words)

        for g in guess_words: # for each guess index,
            cost = n
            results = self.M[g, words]
            guess_results = self.M[g, guess_words]
            unique_results, counts = np.unique(results, return_counts=True)

            # we guess once for each word for a cost of n
            # we then guess at minimum (2 * count - 1) for each possible result that occurs count times
            # except for the result 0, which is only in results if our guess is one of the words
            min_possible_cost = n + 2 * sum(counts) - len(counts) - (g in words)

            if min_possible_cost >= min_cost: # no use exploring this guess
                continue

            # if we are passed guesses that are words, (we ALWAYS guess our actual words before other words)
            if g not in words:
                # if none of the words work as a guess themselves, any guess that with score 2 * n is optimal
                if min_cost == 2 * n:
                    return min_guess, min_cost

                # skip truly useless guesses
                if max(counts) == n:
                    continue

            # we now consider actually recurring on this guess.

            # search the easier results first.
            unique_results = unique_results[np.argsort(counts)]

            for result in unique_results:
                if cost >= min_cost: # break early if we are already too expensive.
                    break
                if result == 0:
                    continue # result of 0 means we guessed it! No additional cost.
                # recusively compute on the new words, guess_words available to us.

                new_alpha = min_cost - cost # any cost greater than or equal to this is useless to us.
                new_words = words[np.where(results == result)[0]] # get the new set of possible words and guesses
                new_guess_words = guess_words[np.where(guess_results == result)[0]]
                alpha = min_cost - cost # if the cost of this call is greater than min_cost - cost, this whole guess isn't going to be used.
                cost += self.solve_cost(new_words, new_guess_words, new_alpha, d + 1)[1]

            if cost < min_cost: # update min_cost if a better guess was found.
                min_guess = g
                min_cost = cost

            # check for pure optimality.
            if min_cost == 2 * len(words) - 1:
                # the optimal guess perfectly discriminates words while being in the words
                return min_guess, min_cost

        return min_guess, min_cost

    def h(self, g):
        """
            A heuristic funciton to optimize the order we choose to evaluate guesses.
        """
        r =  sum(self.letter_freqs[letter] for letter in set(self.guess_dict[g]))

        return r


