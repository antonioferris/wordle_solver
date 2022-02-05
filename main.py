"""
    This file contains code that attempts to find the optimal
    guessing path for the online game Wordle.
"""
from collections import defaultdict
import pickle
from re import I
from socket import timeout
from tabnanny import verbose
import time
import random
import numpy as np
import itertools
from wordle import HardWordle


# ONE TIME FILE CREATION

### ------------------------------------------------------ ###
### ------------------------------------------------------ ###
### ------------------------------------------------------ ###

def make_M():
    start = time.time()
    print("Creating the matrix M. This may take a while.")
    wordle_dict = load_dict('wordle-dictionary.txt')
    wordle_guess_dict  = wordle_dict + load_dict('wordle-allowed-guesses.txt')
    results_dict = list(itertools.product('GY_', repeat=5))

    n = len(wordle_guess_dict)
    # m = len(wordle_dict)
    M = np.zeros((n, n))

    # define this function for getting results in terms of results_dict indices
    def get_results_idx(guess, word):
        res = []
        for i in range(len(guess)):
            if guess[i] == word[i]:
                res.append("G")
            elif guess[i] in word:
                res.append("Y")
            else:
                res.append("_")
        return results_dict.index(tuple(res))

    for i in range(n):
        for j in range(n):
            v = get_results_idx(wordle_guess_dict[i], wordle_guess_dict[j])
            M[i, j] = v

    # save M along with the lists used to dereference the indices.
    data = (M, wordle_dict, wordle_guess_dict, results_dict)

    with open('M.pckl', 'wb') as f:
        pickle.dump(data, f)

    print(f"Creating M took {time.time() - start} seconds.")


def load_M():
    print("Loading M ...")
    M, wordle_dict, wordle_guess_dict, results_dict = pickle.load(open('M.pckl', 'rb'))
    print("Finished loading M.")
    return M.astype(int), wordle_dict, wordle_guess_dict, results_dict

# UTIL FUNCTIONS

### ------------------------------------------------------ ###
### ------------------------------------------------------ ###
### ------------------------------------------------------ ###

def load_dict(filename):
    """
        Loads the dictionary
        from a txt file of words
        each captialized on its own line
    """
    with open(filename) as f:
        lines = f.readlines()

    return [line.strip().upper() for line in lines]

def get_results(guess, word):
    res = []
    #
    for i in range(len(guess)):
        if guess[i] == word[i]:
            res.append("G")
        elif guess[i] in word:
            # only write "Y" if the actual placement of this letter isn't green.
            y = False
            for j in range(len(guess)):
                if i == j:
                    continue
                if word[j] == guess[i] and guess[j] != word[j]: # a column where there is this letter not green
                    y = True
                    break
            if y:
                res.append("Y")
            else:
                res.append("_")
        else:
            res.append("_")
    return tuple(res)

def get_buckets(guess, poss_words):
    """
        given a guess, return a dictionary
    """
    buckets = defaultdict(list)
    for word in poss_words:
        if guess == word:
            continue
        res = get_results(guess, word)
        buckets[res].append(word)

    return buckets

def solve(guesser, poss_words, true_word, verbose=True):
    true_word = true_word.upper()
    if verbose:
        print(f'GUESSING FOR {true_word}')
    for i in range(6):
        g = guesser(poss_words)
        results = get_results(g, true_word)
        buckets = get_buckets(g, poss_words)
        if verbose:
            print(f"{g} -> {results}")
        if results in buckets:
            poss_words = buckets[results]
            if verbose:
                if len(poss_words) > 20:
                    print(f"{len(poss_words)} words remain.")
                else:
                    print(f"words that remain {poss_words}")
        else:
            # did we already get it?
            if g == true_word:
                r = i + 1
            else:
                r = i + 2
            if verbose:
                print(f"It took me {r} guesses!")
            break
        if i == 5:
            print(f'We failed still have {poss_words} left after guess {g}.')
            r = None
    if verbose:
        print()
    return r

### ------------------------------------------------------ ###
### ------------------------------------------------------ ###
### ------------------------------------------------------ ###

def get_input(msg, preloaded_input):
    if preloaded_input:
        print(msg, end="")
        i = preloaded_input.pop(0)
        print(i)
        return i
    return input(msg)


def interactive(preloaded_input):
    M, word_dict, guess_dict, results_dict = load_M()
    word_dict = np.array(word_dict)
    guess_dict = np.array(guess_dict)
    words = np.array(range(len(word_dict)))
    w = HardWordle(M, word_dict, guess_dict, results_dict)
    guess_words = np.array(w.guess_order)

    while len(words) > 1:
        guess = get_input("Enter your guess.\n$ ", preloaded_input).upper()

        if len(guess) > 5 and ("SOLVE" in guess or "TAKE" in guess):
            guess = w.solve(np.array(words), np.array(guess_words), verbose=True, timeout=120)
            print("GUESS SUPPLIED:", guess)
        g = np.where(guess_dict == guess)[0][0]

        cr = get_input("Enter your result.\n$ ", preloaded_input)
        try:
            result = int(cr)
        except:
            result = results_dict.index(tuple(cr.replace('-', '_').upper()))

        results = M[g, words]
        words = words[np.where(results == result)[0]]
        guess_words = guess_words[np.where(M[g, guess_words] == result)[0]]
        print("Remaining words", word_dict[words], "Guess words", guess_dict[guess_words])

def eval(solution_tree, true_words=None, verbose=False):
    """
        Given a solution tree for a given set of words, evaluates it on the word_dict. This will be a dictionary
        from observation -> guess.
        For example,
        {
            () : "MATCH",
            (3,) : "GUEST",
            (1, ) : "FREED",
            (3, 3) : "HELLO",
            (1, 3) : "FRANK"
        }
        Would mean a strategy of guessing MATCH first. If the observated response is 3, guess "GUEST" next.
        Otherwise, guess "FREED". and so on.
    """
    M, word_dict, guess_dict, results_dict = load_M()
    n = len(word_dict)
    m = len(guess_dict)
    tree = {}
    guess_score = 0

    w = HardWordle(M, word_dict, guess_dict, results_dict)

    if true_words == None:
        true_words = range(n)
    else:
        true_words = [word_dict.index(w) for w in true_words]


    solution_tree[tuple([])] = solution_tree[()], 0

    for true_word in true_words:
        tw_str = word_dict[true_word] # get the string repr of the truw word
        words = np.array(range(n)) # initialize the remaining words and guesses to start at all of them
        guess_words = np.array(range(m))
        hist = []

        while len(words) > 0: # while we still have words left to choose from
            guess_score += 1
            h = tuple(hist)

            if h in solution_tree: # input precomputed guesses from the tree.
                guess, _ = solution_tree[h]
            else:
                s = time.time()
                guess = w.solve_cost(words, guess_words, alpha=9999, d=len(h) // 2)[0]
                dur = time.time() - s
                if dur > 1:
                    print(f"Needed 1 second for non-precomputed {len(words)} words")

            hist.append(guess)
            guess_str = guess_dict[guess]

            if isinstance(guess, str):
                guess = guess_dict.index(guess)

            result = M[guess, true_word]
            hist.append(result)
            results = M[guess, words]
            guess_results = M[guess, guess_words]
            words = words[np.where(results == result)[0]]
            guess_words = guess_words[np.where(guess_results == result)[0]]
            if guess_str == tw_str:
                if verbose:
                    print(f"{word_dict[guess]}\n")
                break
            else:
                if verbose:
                    print(guess_dict[guess] + " -> " + "".join(results_dict[result]) + ". ")

    print(f"Average number of guesses {guess_score / len(true_words)}")

def solve(guesses, save=True):
    M, word_dict, guess_dict, results_dict = load_M()
    guesses = [guess_dict.index(guess) for guess in guesses]
    word_dict = np.array(word_dict)
    guess_dict = np.array(guess_dict)
    words = np.array(range(len(word_dict)))
    w = HardWordle(M, word_dict, guess_dict, results_dict, global_guess_list=guesses)
    guess_words = np.array(w.guess_order)

    solution_tree = w.solve(words, guess_words, verbose=True, timeout=3600 * 5)
    print()
    print("Solution Tree:", solution_tree)

    if save and solution_tree:
        with open("solutions.pckl", "wb") as f:
            pickle.dump(solution_tree, f)

def main():
    # make_M()
    # interactive(preloaded_input=[])
    # solve(["RAISE"], save=False)
    solution_tree = pickle.load(open("solutions.pckl", "rb"))
    eval(solution_tree, true_words=['ALOFT'], verbose=True)



if __name__ == '__main__':
    main()