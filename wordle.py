"""
    This file contains FastWordle, an implementation of a Wordle
    solver that tries to use numpy matrices to allow for greater efficiency
    when searching the wordle problem space.

    Most of the benefit of FastWordle comes from storing precomputed data in a specific form.

    The matrix M is a large array for each pair of possible guess and possible word in the dictionary.
    Each element in the array is the index of the result that guess gives when the true word is that word.

                word_1  word_2  word_3 ...
    guess_1       4       2        2
    guess_2       1       2        2
    guess_3       0       10       11
    ...

    Suppose that poss_words is a list of the indexes of possible words, and guess is the index of a possible guess.
    Then, M[guess, poss_words] is an array of the results each possible word gives for our guess!!
    Also, M[guess, poss_words] == res is an array of the indices of the words that could still be the word after our guess!
"""
from collections import defaultdict
import numpy as np
import time


class FastWordle:
    def __init__(self, M, word_dict, guess_dict, results_dict, global_guess_list=None):
        self.n = M.shape[0]
        self.m = 2315
        self.M = M
        self.len_cnts = defaultdict(int)
        self.len_csts = defaultdict(int)
        self.word_dict = np.array(word_dict)
        self.guess_dict = guess_dict
        self.results_dict = results_dict
        self.global_guess_list = global_guess_list
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

    def _solve(self, words, guess_words, verbose=False, timeout=600):
        wt = tuple(words)
        if wt not in self.cache: # possible bug different guesses for words with different guess arrays.
            self.cache[wt] = self._solve(words, guess_words, verbose, timeout)
        return self.cache[wt]


    def solve(self, words, guess_words, verbose=False, timeout=600):
        """
            Given a list of words, calls solve_cost to find the
            guess with the minimum cost.
        """
        min_cost = 4 * len(words)
        guess = None
        self.start = time.time()
        if verbose:
            print(f"initial min cost {min_cost}, {len(words)}")

        # for tracking progress
        w = len(words)

        for g in self.global_guess_list: # guess words in our guess list
            self.hist.append(g) # add guess to history
            if time.time() - self.start > timeout:
                print(f"Time limit exceeded. {time.time() - self.start}s taken.")
                break
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

                next_guess, inc = self.cache_cost(new_words, new_guess_words, alpha, 1)
                self.tree[tuple(self.hist)] = next_guess, inc
                if inc == alpha:
                    print(f"Failed to solve ({self.guess_dict[g]}, {self.results_dict[result]}) giving {self.word_dict[new_words]} in {5} guesses.")
                cost += inc

                self.hist.pop() # remove result from history after consideration
                if verbose:
                    d = time.time() - s
                    print(f"Result {r} / {t} ({len(new_words)} words) has cost {inc} (cumulative {cost}) took {d:.2f}s {w} words left to explore.")
                    r += 1
                if cost >= min_cost:
                    break

            if cost < min_cost: # update min_cost and guess if we are minimal
                if verbose and g in words:
                    print(self.guess_dict[g], cost)
                min_cost = cost
                guess = g

            self.hist.pop() # remove guess from hstory after consideration

        guess_word = (self.guess_dict[guess], min_cost)
        if verbose:
            print(f"FINAL GUESS {guess_word} cost {min_cost}")

        # we return the tree of the solved game, made from the tree as we went.
        # the only part of the tree we need is the branches that start with the "correct" guess.
        solution_tree = {hist : sol for hist, sol in self.tree.items() if hist[0] == guess}
        solution_tree[tuple([])] = guess
        print("WORD LEN COSTS:")
        print({j : self.len_csts[j] / self.len_cnts[j] for j in self.len_cnts.keys()})
        return solution_tree

    def cache_cost(self, words, guess_words, alpha, d):
        # hist = tuple(self.hist)
        # if len(hist) % 2 != 0:
        #     raise ValueError(f"Uneven History! {hist}")
        # if hist not in self.cache: # possible bug different guesses for words with different guess arrays.
        s = time.time()
        TEMP_R = self.solve_cost(words, guess_words, alpha, d)

        dt = time.time() - s
        if dt > 20:
            print(f"Cost call with {len(words)} words,  {len(guess_words)} guess words, {alpha} alpha took {dt:.2f} seconds")

        # self.len_cnts[len(words)] += 1
        # self.len_csts[len(words)] += self.cache[hist][1]
        return TEMP_R # return min cost

    def solve_cost(self, words, guess_words, alpha, d):
        """
            Given the list of words as a list of indices and our depth,
            solve_cost recursively computes the cost function.
            alpha : the minimum cost that isn't useful for this function
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

            # if we are passed guesses that are words, (we ALWAYS guess our actual words before other words.)
            if g not in words:
                # if none of the words work as a guess themselves, any guess that with score 2 * n is optimal
                if min_cost == 2 * n:
                    return min_guess, min_cost


                # skip truly useless guesses
                if max(counts) == n:
                    continue

            # we now consider actually recurring on this guess.

            # self.hist.append(g) # add guess to history

            # search the easier results first.
            unique_results = unique_results[np.argsort(counts)]

            for result in unique_results:
                if cost >= min_cost: # break early if we are already too expensive.
                    break
                if result == 0:
                    continue # result of 0 means we guessed it! No additional cost.
                # self.hist.append(result) # add result to history
                # recusively compute on the new words, guess_words available to us.

                new_alpha = min_cost - cost # any cost greater than or equal to this is useless to us.
                new_words = words[np.where(results == result)[0]] # get the new set of possible words and guesses
                new_guess_words = guess_words[np.where(guess_results == result)[0]]
                alpha = min_cost - cost # if the cost of this call is greater than min_cost - cost, this whole guess isn't going to be used.
                cost += self.cache_cost(new_words, new_guess_words, new_alpha, d + 1)[1]

                # self.hist.pop() # remove result after consideration

            # self.hist.pop() # remove guess from history

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


