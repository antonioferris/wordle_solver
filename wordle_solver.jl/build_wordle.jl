using JLD2, LinearAlgebra

words = Vector{String}()

open("wordle-dictionary.txt") do f
    while ! eof(f)
        w = readline(f)
        push!(words, w)
    end
end

N_sol = size(words)[1]

open("wordle-allowed-guesses.txt") do f
    while ! eof(f)
        w = readline(f)
        if  ! (w in words)
            push!(words, w)
        end
    end
end

function query(word, guess)
    result::UInt8 = 0
    for i in 1:5
        result *= 3
        if guess[i] != word[i]
            if guess[i] in word
                x = findall(collect(word) .== guess[i])
                if any(collect(word[x]) .== collect(guess[x]))
                    result += 2
                else
                    result += 1
                end
            else
                result +=  2
            end
        end
    end
    return result
end

N_total = size(words)[1]

W = zeros(UInt8, N_total, N_sol)

for i in 1:N_total
    for j in 1:N_sol
        W[i, j] = query(words[j], words[i])
    end
end

entropy = zeros(N_total)
for i in 1:N_total
    results = W[i, :]
    unique_results = unique(results)
    counts = [count(==(r), results) for r in unique_results]
    global entropy
    entropy[i] = sum(counts/N_total .* log.(counts / N_total))
end
guess_order = sortperm(entropy)

jldsave("wordle.jld2"; W, words, guess_order, N_sol)