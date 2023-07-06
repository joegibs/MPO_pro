#not relaly working, dont see an increase in bip-ent
#need to change the sampling
using ITensors
using Random
using Plots
using Statistics
using LaTeXStrings
using LinearAlgebra

function thermalize_rho(rho,s,)
function rec_ent(rho,b,s)
  n = length(rho)
  # orthogonalize!(rho,b)
  rho_temp = deepcopy(rho)
  # s = siteinds("Qubit",n) 

  #contract half   x x x x | | | |
  L = ITensor(1.0)
    for i = 1:b
      L *= tr(rho_temp[i])
    end
    # absorb
    rho_temp[b+1] *= L
    # no longer a proper mpo
    M =MPO(n-b)
    for i in 1:(n-b)
        M[i]=rho_temp[b+i]
    end
    #turn this mpo into a single tensor
    T = prod(M)
 
    # @show T
    _,S,_ = svd(T,s)#[inds(T)[i] for i = 1:2:length(inds(T))])
    SvN = 0.0
    for n in 1:dim(S, 1)
      p = S[n,n]
      if p != 0
        SvN -= p * log2(p)
      end
    end
    return SvN
  end
# function rec_ent(psi,b)
#   s = siteinds(psi)  
#   orthogonalize!(psi, b)
#   _,S = svd(psi[b], (linkind(psi, b-1), s[b]))
#   SvN = 0.0
#   for n in 1:dim(S, 1)
#     p = S[n,n]^2
#     if p != 0
#       SvN -= p * log(p)
#     end
#   end
#   return SvN
# end

function ren(rho)
  return -log(tr(apply(rho,rho)))
end
function split_ren(rho,b)
  n = length(rho)
  rho_temp = deepcopy(rho)
  s = siteinds("Qubit",n) 

  #contract half   x x x x | | | |
  L = ITensor(1.0)
  for i = 1:b
    L *= tr(rho_temp[i])
  end
  # absorb
  rho_temp[b+1] *= L
  # no longer a proper mpo
  M =MPO(n-b)
  for i in 1:(n-b)
      M[i]=rho_temp[b+i]
  end
  M=M/tr(M)
  ren = -log2(tr(apply(M,M)))
  return ren
end

function RandomUnitaryMatrix(N::Int)
  x = (rand(N,N) + rand(N,N)*im) / sqrt(2)
  f = qr(x)
  diagR = sign.(real(diag(f.R)))
  diagR[diagR.==0] .= 1
  diagRm = diagm(diagR)
  u = f.Q * diagRm
  
  return u
end
function IXgate(N::Int)
  theta = rand()*2*pi
  eps=0.25#rand()
  u =exp(1im*theta).*(sqrt((1-eps)).*Matrix(I,4,4) + 1im.*sqrt(eps).*[[0,0,0,1] [0,0,1,0] [0,1,0,0] [1,0,0,0]])
  return u
end

function make_row(N,eoo,pc)
  if eoo
      lst =[[i,i+1] for i in 1:2:N-1]
  else
      lst = [[i,i+1] for i in 2:2:N-1]
  end
  if pc
      if !eoo && !Bool(N%2)
          append!(lst,[[N,1]])
      end
  end
  return lst
end


ITensors.op(::OpName"Pup",::SiteType"Qubit") =
 [1 0
  0 0]
ITensors.op(::OpName"Pdn",::SiteType"Qubit") =
 [0 0
  0 1]
ITensors.op(::OpName"Rand",::SiteType"Qubit") = 
    RandomUnitaryMatrix(4)
ITensors.op(::OpName"IX",::SiteType"Qubit") = 
    IXgate(4)

ITensors.op(::OpName"Iden",::SiteType"Qubit") = 
  [1 0
  0 1]
ITensors.op(::OpName"Iden2",::SiteType"Qubit") = 
  [1 0 0 0
  0 1 0 0
  0 0 1 0
  0 0 0 1]

function gen_samp_row(N,meas_p)
  return [rand()<meas_p ? 1 : 0 for i in 1:N]
end
function samp_mps(rho,s,samp_row)
  cutoff = 1E-8
  N = length(rho)
  samp =deepcopy(rho)
  samp = samp/tr(samp)
  samples= sample(samp)
  magz = [x == 1 ? "Pup" : "Pdn" for x in samples]

  gates = ITensor[]

  for i in 1:N
    if Bool(samp_row[i])
        hj = op(magz[i],s[i])
        push!(gates, hj)
    end
  end
  rho = apply(gates, rho;apply_dag=true)
 
  normalize!(rho)
  return rho
end

function gen_step(N,rho,s,step_num,meas_p)
    #apply gates
  row = make_row(N,Bool(step_num%2),false)
  gates = ITensor[]
  for j in row
      s1 = s[j[1]]
      s2 = s[j[2]]
      hj = op("Rand",[s1,s2])
      Gj=hj
      push!(gates, Gj)
  end
  cutoff = 1E-8

  rho = apply(gates, rho;apply_dag=true,cutoff=1E-8)

  #calculate obs
  measured_vals = rec_ent(rho,Int(round(N/2)),s)#,Int(round(N/2)))

  normalize!(rho)
  #sample as needed
  samp_row=gen_samp_row(N,meas_p)
  rho = samp_mps(rho,s,samp_row)

  return rho,measured_vals
end

function do_exp(N,steps,meas_p)
  s = siteinds("Qubit", N) #+1 for ancilla
  psi = productMPS(s, "Up" )
  rho=outer(psi',psi)

  svn =[]
  for i in 1:steps
      rho ,meas_svn= gen_step(N,rho,s,i,meas_p)
      append!(svn,meas_svn)
  end
  #tri_mut = tri_part_MI(psi,[1,2],[3,4],[5,6])
  tri_mut = []
  # for i in 3:length(psi)-1
  #     arr = two_point_MI(psi,2,i)
  #     append!(tri_mut,arr)
  # end
  return svn,tri_mut
end

function do_trials(N,steps,meas_p,trials)
  svn_trials,tri_trials = do_exp(N,steps,meas_p)
  for i in 2:trials
      print(i)
      nSvn,ntm = do_exp(N,steps,meas_p)
      svn_trials = 2*mean([(i-1)/i*svn_trials,1/i*nSvn])
      tri_trials = 2*mean([(i-1)/i*tri_trials,1/i*ntm])

      
  end
  return svn_trials,tri_trials
end

decays=[]
# for n in [8,10]
N = 6
# cutoff = 1E-8
steps = 4*N
meas_p=0.
svns=[]
mut = []
for i in [0.0:0.2:1...]
    print("\n meas_p $i \n")
    svn,tri_mut =do_trials(N,steps,i,100)
    avgsvn = [(svn[x]+svn[x+1])/2 for x in 1:2:(size(svn)[1]-1)]
    append!(svns,[avgsvn])
    append!(mut,tri_mut)
end
decay = [svns[i][end] for i in 1:size(svns)[1]]
append!(decays,[decay])
# end
p = plot(svns,title=string("MPO Gate Rand", ", ", N, " qubit sites, varying meas_p"), label=string.(transpose([0.0:0.2:1...])), linewidth=3,xlabel = "Steps", ylabel = L"$\textbf{S_{vn}}(L/2)$")
# p = plot([0.1:0.2:1...],decays,title=string("Bip_ent Gat: IX0.9", ", ", N, " qubit sites, varying meas_p"), label=string.(transpose([6:2:14...])), linewidth=3,xlabel = "Meas_P", ylabel = L"$\textbf{S_{vn}}(L/2)$")
# m = plot(real(mut))
display(p)





# N=4
# s= siteinds("Qubit",N)
# samp_row = [1,1,1,1,1]
# magz = ["Pup","Pup","Pup","Pup","Pup"]
# ampo = OpSum()
# for i in 1:N
#   print(N)
#   if Bool(samp_row[i])
#       ampo += magz[i],i
#   end
# end
# H = MPO(ampo,s)


# n=5
# s = siteinds("S=1/2", n)
# psi = randomMPS(s)

# rho = outer(psi',psi)
# os = OpSum()
# for j in 1:(n- 1)
#   os += "Sz", j, "Sz", j + 1
#   os += 0.5, "S+", j, "S-", j + 1
#   os += 0.5, "S-", j, "S+", j + 1
# end
# # Convert these terms to an MPO tensor network
# H = MPO(os, s)
# rho = apply(H,rho; apply_dag=true)
# normalize!(rho)
# M=deepcopy(rho)
# N = length(M)
# s = siteinds(M)
# R = Vector{ITensor}(undef, N)
# R[N] = M[N] * δ(dag(s[N]))
# for n in reverse(1:(N - 1))
#   R[n] = M[n] * δ(dag(s[n])) * R[n + 1]
# end


L = Vector{ITensor}(undef, N)
L[N] = tr(M[N])
for n in reverse(1:(N - 1))
  L[n] = tr(M[n]) *R[n+1]
end

function sample_mpo(M::MPO)
  N = length(M)
  s = siteinds(M)
  R = Vector{ITensor}(undef, N)
  R[N] = M[N] * δ(dag(s[N]))
  for n in reverse(1:(N - 1))
    R[n] = M[n] * δ(dag(s[n])) * R[n + 1]
  end

  if abs(1.0 - norm(M)) > 1E-8
    error("sample: MPO is not normalized, norm=$(norm(M[1]))")
  end

  result = zeros(Int, N)
  ρj = M[1] * R[2]
  Lj = ITensor()

  for j in 1:N
    s = siteind(M, j)
    d = dim(s)
    # Compute the probability of each state
    # one-by-one and stop when the random
    # number r is below the total prob so far
    pdisc = 0.0
    r = rand()
    # Will need n, An, and pn below
    n = 1
    projn = ITensor()
    pn = 0.0
    while n <= d
      projn = ITensor(s)
      projn[s => n] = 1.0
      pnc = (ρj * projn * prime(projn))[]
      if imag(pnc) > 1e-8
        @warn "In sample, probability $pnc is complex."
      end
      pn = real(pnc)
      pdisc += pn
      (r < pdisc) && break
      n += 1
    end
    result[j] = n
    if j < N
      if j == 1
        Lj = M[j] * projn * prime(projn)
      elseif j > 1
        Lj = Lj * M[j] * projn * prime(projn)
      end
      if j == N - 1
        ρj = Lj * M[j + 1]
      else
        ρj = Lj * M[j + 1] * R[j + 2]
      end
      s = siteind(M, j + 1)
      normj = (ρj * δ(s', s))[]
      ρj ./= normj
    end
  end
  return result
end

