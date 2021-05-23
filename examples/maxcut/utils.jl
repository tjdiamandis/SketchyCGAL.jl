using SparseArrays

# Read GSET files
function graph_from_file(filename)
    file_vec = readlines(filename)
    n, m = parse.(Int, split(file_vec[1], ' '))

    G = spzeros(Float64, n, n)
    for line in 2:length(file_vec)
        i, j, v = parse.(Int, split(file_vec[line], ' '))
        G[i, j] = G[j, i] = v
    end

    return G
end


# MAXCUT cost function (true -- not used in optimization)
function cost(G, y)
    return 0.25*(sum(G) - dot(y, G*y))
end
