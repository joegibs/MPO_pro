using ITensors
using Printf

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
 
    @show T
    @show s
    _,S,_ = svd(T,s)
    SvN = 0.0
    @show S
    for n in 1:dim(S, 1)
      p = S[n,n]
      if p != 0
        SvN -= p * log2(p)
      end
    end
    return SvN
end
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
ITensors.op(::OpName"Rand",::SiteType"Qubit") = 
    RandomUnitaryMatrix(4)
#test

N=4
s = siteinds("Qubit", N)
psi1 = productMPS(s, "Up" )
rho = outer(psi1',psi1)
gates = ITensor[]

s1 = s[1]
s2 = s[2]
s3 = s[3]
s4 = s[4]
hj = op("H",s1)
push!(gates, hj)
hj = op("CNOT",[s1,s2])
push!(gates, hj)
hj = op("H",s3)
push!(gates, hj)
hj = op("CNOT",[s3,s4])
push!(gates, hj)
# rho = apply(gates, rho; apply_dag=true, cutoff = 1E-8)
# gates = ITensor[]
hj = op("H",s2)
push!(gates, hj)
hj = op("CNOT",[s2,s3])
push!(gates, hj)
# hj = op("H",s3)
# push!(gates, hj)
# hj = op("CNOT",[s3,s4])
# push!(gates, hj)

# hj = op("Rand",[s1,s2])
# push!(gates,hj)
# hj = op("Rand",[s3,s4])
# push!(gates,hj)
# hj = op("Rand",[s2,s3])
# push!(gates,hj)
# hj = op("Rand",[s1,s2])
# push!(gates,hj)
# hj = op("Rand",[s3,s4])
# push!(gates,hj)
# hj = op("Rand",[s2,s3])
# push!(gates,hj)


cutoff = 1E-8

rho = apply(gates, rho; apply_dag=true, cutoff = 1E-8)
psi = apply(gates,psi1)
# rho = apply(rho, rho; cutoff)

rec_ent(rho,2,s)
split_ren(rho,2)
rec_ent_mps(psi,2)
###
b=2
n = length(rho)
rho_temp = deepcopy(rho)
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
T = prod(M)


_,S,_ = svd(T,s)
SvN = 0.0
for n in 1:dim(S, 1)
  p = S[n,n]
  if p != 0
    SvN -= p * log2(p)
  end
end
SvN


# Make an array of integers of the element we
# want to obtain
T2 = apply(M,M)
# T2 = T2/tr(T2)
el = [1,1,1,1]

V = ITensor(1.)
for j=1:2
  V *= prime(state(s[j+b],el[j]))*M[j]*state(s[j+b],el[j])
end
v = scalar(V)



V = ITensor(1.)
for j=1:2
  V *= prime(state(s[j+b],el[j]))*T2[j]*state(s[j+b],el[j])
end
v = scalar(V)
# v is the element we wanted to obtain:
@show v

###
function rec_ent_mps(psi,b)
  s = siteinds(psi)  
  orthogonalize!(psi, b)
  _,S = svd(psi[b], (linkind(psi, b-1), s[b]))
  SvN = 0.0
  for n in 1:dim(S, 1)
    p = S[n,n]^2
    if p != 0
      SvN -= p * log2(p)
    end
  end
  return SvN
end



# psi=deepcopy(rho)
# s= siteinds(psi)  
# orthogonalize!(psi, b)


# C = combiner(inds(psi[2])[1],inds(psi[2])[3])
# C2 = combiner(inds(psi[2])[2],inds(psi[2])[4])
# test_ten=(C2*(C*psi[2]))

# _,S = svd(psi[b], (linkind(psi, b-1), s[b]))
# SvN = 0.0
# for n in 1:dim(S, 1)
#   p = S[n,n]^2
#   if p != 0
#     SvN -= p * log2(p)
#   end
# end
# return SvN

###################################
t=0
for i in 9:64
  t+=(1/i)
end

t-=7/16

t/2
########################################
ITensors.op(::OpName"Pup",::SiteType"Qubit") =
 [1 0
  0 0]
ITensors.op(::OpName"Pdn",::SiteType"Qubit") =
 [0 0
  0 1]

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

N=4
s = siteinds("Qubit", N)
psi1 = productMPS(s, "Up" )
rho = outer(psi1',psi1)
gates = ITensor[]

s1 = s[2]
s2 = s[3]
hj = op("H",s1)
push!(gates, hj)
hj = op("CNOT",[s1,s2])
push!(gates, hj)

cutoff = 1E-8

rho = apply(gates, rho; apply_dag=true)
psi = apply(gates,psi1)
# rho = apply(rho, rho; cutoff)

rec_ent(rho,2,s)
rho2 = samp_mps(rho,s,[1,0,1,1])
rec_ent(rho2,2,s)


function ITensors.op(::OpName"Rz2", ::SiteType"Qubit"; θ=nothing, ϕ=nothing)
  isone(count(isnothing, (θ, ϕ))) || error(
    "Must specify the keyword argument `θ` (or the deprecated `ϕ`) when creating an Rz gate, but not both.",
  )
  isnothing(θ) && (θ = ϕ)
  return [
    exp(-im * θ / 2) 0
    0 exp(im * θ / 2)
  ]
end
function kraus_dephase(rho,s,p)
  #define the two operators
  #(1-p)ρ + pZρZ
  N=length(rho)
  gates = ITensor[]
  for i in 1:N
    hj = op("Z", s[i])
    push!(gates, hj)
  end
  #apply the operators
  rho = (1-p)*rho + p*apply(gates,rho;apply_dag=true)
  return rho
end
N = 4
sites = siteinds("Qubit",N)
psi1 = productMPS(sites, "Up" )
rho = outer(psi1',psi1)

gates = ITensor[]
  for i in 1:length(s)
    hj = op("Z",s[i])
    push!(gates, hj)
    end

rho2 = apply(gates,rho;apply_dag=true)
  #sum the results

rho2= kraus_dephase(rho,sites,0.5)



s1 = sites[2]

p1 = op("Rz2",s1,θ=2)
gates = ITensor[]
push!(gates, p1)

################## partial transpose ##################33

N = 2
sites = siteinds("Qubit",N)
psi1 = productMPS(sites, "Up" )
rho = outer(psi1',psi1)

gates = ITensor[]

s1 = sites[1]
s2 = sites[2]
hj = op("H",s1)
push!(gates, hj)
hj = op("CNOT",[s1,s2])
push!(gates, hj)

cutoff = 1E-8

rho = apply(gates, rho; apply_dag=true)
# rho = kraus_dephase(rho,sites,0.3)

function partial_transpose(A::MPO, sites)
  A = copy(A)
  for n in sites
    A[n] = swapinds(A[n], siteinds(A, n)...)
  end
  return A
end

rhot = partial_transpose(rho,1)
norm(rhot)

function log_negativity(A::MPO,sites)
  Apt=partial_transpose(A,sites)
  ϵ = sqrt()
  return ϵ
end


log_negativity(rho,1:2,1)
rec_ent(rho,2,sites)

function rho_to_dense(rho,s)
  Hitensor = ITensor(1.)
  N = length(s)
  for i = 1:N
      Hitensor *= rho[i]
  end

  A=Array(Hitensor,prime(s),s)
  return reshape(A,2^N,2^N)
end
function kraus_dephase(rho,s,p)
  #define the two operators
  #(1-p)ρ + pZρZ
  N=length(rho)
  gates = ITensor[]
  for i in 1:N
    hj = op("Z", s[i])
    push!(gates, hj)
  end
  #apply the operators
  rho = (1-p)*rho + p*apply(gates,rho;apply_dag=true)
  return rho/tr(rho)
end

N = 2
sites = siteinds("Qubit",N)
psi1 = productMPS(sites, "Up" )
rho = outer(psi1',psi1)

gates = ITensor[]

s1 = sites[1]
s2 = sites[2]
hj = op("H",s1)
push!(gates, hj)
hj = op("Rand",[s1,s2])
push!(gates, hj)

cutoff = 1E-8

rho = apply(gates, rho; apply_dag=true)
rho_to_dense(rho,sites)
rhot=kraus_dephase(rho,sites,0.5)
rho_to_dense(rhot,sites)
A=partial_transpose(rho,2)


log_negativity(rho,1,2)
log_negativity(rhot,1,2)