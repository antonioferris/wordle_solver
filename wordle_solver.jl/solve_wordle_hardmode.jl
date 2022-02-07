using JLD2

f = jldopen("wordle_hardmode.jld2")
W = f["W"]
N_sol = f["N_sol"]
all_words = f["words"]
entropy_order = f["guess_order"]

function solve_cost(words, guesses, alpha, d)
    n = size(words)[1]

    if d > 5
        return nothing, 100000
    end

    if d == 5 && n > 1
        return nothing, 100000
    end

    if n <= 2
        return words[1], 2*n-1
    end

    min_cost = alpha
    min_guess = nothing


    entropies = entropy(words, guesses)
    split = findfirst(guesses .> N_sol)
    if isnothing(split)
        guess_order = sortperm(entropies)
        tail = 0
        while isapprox(entropies[guess_order[end-tail]], 0)
            tail += 1
        end
        guess_order = guess_order[1:end-tail]
    else
        sol_guess_order = sortperm(entropies[1:split-1])
        non_sol_guess_order = sortperm(entropies[split:end]) .+ (split-1)
        tail = 0
        while isapprox(entropies[sol_guess_order[end-tail]], 0)
            tail += 1
        end
        sol_guess_order = sol_guess_order[1:end-tail]
        tail = 0
        while isapprox(entropies[non_sol_guess_order[end-tail]], 0)
            tail += 1
        end
        non_sol_guess_order = non_sol_guess_order[1:end-tail]
        guess_order = vcat(sol_guess_order, non_sol_guess_order)
    end

    for g in guesses[guess_order]
        cost = n
        global W
        results = W[g, words]
        guess_results = W[g, guesses]
        unique_results = unique(results)
        counts = [count(==(r), results) for r in unique_results]

        min_possible_cost = 3 * n - size(unique_results)[1] - (g <= N_sol)
        if min_possible_cost >= min_cost
            continue
        end

        if g > N_sol
            if min_cost == 2*n
                return min_guess, min_cost
            end

            if size(unique_results)[1] == 1
                continue
            end
        end

        unique_results = unique_results[sortperm(counts)]

        for result in unique_results
            if cost >= min_cost
                break
            end

            if result == 0
                continue
            end

            new_alpha = min_cost - cost
            new_words = words[results .== result]
            new_guesses = guesses[guess_results .== result]
            cost += solve_cost(new_words, new_guesses, new_alpha, d+1)[2]
        end

        if cost < min_cost
            min_cost = cost
            min_guess = g

            if min_cost == 2*n-1
                return min_guess, min_cost
            end
        end
    end

    return min_guess, min_cost
end

function solve(g; min_cost=6*N_sol)
    global W, guess_order, N_sol
    words = collect(1:N_sol)
    guesses = collect(1:size(W)[1])

    n = N_sol

    cost = n
    results = W[g, words]
    guess_results = W[g, guesses]
    unique_results = unique(results)
    counts = [count(==(r), results) for r in unique_results]
    unique_results = unique_results[sortperm(counts)]

    for result in unique_results
        if cost >= min_cost
            break
        end

        if result == 0
            continue
        end

        new_alpha = min_cost - cost
        new_words = words[results .== result]
        new_guesses = guesses[guess_results .== result]
        cost += solve_cost(new_words, new_guesses, new_alpha, 1)[2]
    end

    if cost < min_cost
        min_cost = cost
        min_guess = g
    end

    return g, min_cost
end

function eval(str)
    global all_words
    g = findall(all_words .== str)[1]
    _, cost = solve(g)
    return cost / N_sol
end

function entropy(words, guesses)
    global W
    n = size(guesses)[1]
    entropies = zeros(n)
    for i in 1:n
        results = W[guesses[i], words]
        unique_results = unique(results)
        counts = [count(==(r), results) for r in unique_results]
        entropies[i] = sum(counts/n .* log.(counts / n))
    end
    return entropies
end

# scores = zeros(size(all_words))
# min_guess = entropy_order[1]
# min_score = 9000
# for j in 1:size(entropy_order)[1]
#     global min_score, min_guess
#     i = entropy_order[j]
#     w = all_words[i]
#     print("$j ")
#     print(w)
#     print(": ")
#     _, s = solve(i; min_cost=min_score)
#     scores[i] = s
#     if s >= min_score
#         print("worse than ")
#         println(all_words[min_guess])
#     else
#         min_score = s
#         min_guess = i
#         println(s/N_sol)
#     end
# end


    

