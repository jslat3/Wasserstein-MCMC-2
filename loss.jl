using Distances
using Ripserer, PersistenceDiagrams
using PythonOT
using LinearAlgebra

"""Compute the distance matrix between x and y with metric D"""
distance_matrix(x::T, y::T, D) where {T} = pairwise(D, x', y')

"""Calculate the Euclidean distance (squared) to the diagonal"""
diagonal_cost(x::T) where {T} = ((@view(x[:, 2]) .- @view(x[:, 1])) ./ sqrt(2.0)) .^ 2;

"""Get birth/death values from PersistenceDiagram object"""
extract(dgm::PersistenceDiagram) = [birth.(dgm) death.(dgm)];
   

function Wasserstein_Distance(x::PersistenceDiagram, y::PersistenceDiagram)::Float64

    dgmx, dgmy = extract(x), extract(y)
    n, m = size(dgmx, 1), size(dgmy, 1)

    if n > 1
        if m > 1
            finite_x, finite_y = @view(dgmx[1:n-1, :]), @view(dgmy[1:m-1, :])
    
            # Calculate the cost of matching each point to the diagonal
            dcx, dcy = diagonal_cost(finite_x), diagonal_cost(finite_y) 
            dm = distance_matrix(finite_x, finite_y, SqEuclidean(1e-12))
    
            # Cost matrix
            C = zeros(n,m)
            C[axes(dm)...] .= dm

            C[1:n-1, m] .= dcx
            C[n, 1:m-1] .= dcy

            # Weight vectors
            a = ones(n)
            a[n] = float(m)
            b = ones(m)
            b[m] = float(n)

            ot_cost = PythonOT.emd2(a, b, C) # cost from the matching
            inf_cost = (dgmx[n] - dgmy[m]) ^ 2 # cost from infinite points

            return sqrt(ot_cost + inf_cost)
        else
            inf_cost = (dgmx[n] - dgmy[m]) ^ 2
            diag_cost = diagonal_cost(@view(dgmx[1:n-1, :]))
            return sqrt(sum(diag_cost) + inf_cost)
        end
    else 
        if m > 1 
            inf_cost = (dgmx[n] - dgmy[m]) ^ 2
            diag_cost = diagonal_cost(@view(dgmy[1:m-1, :]))
            return sqrt(sum(diag_cost) + inf_cost)
        else 
            return abs.(dgmx[n] - dgmy[m])
        end
    end

end

