function bond_dim_array(rho)
    arr=[]
    for i in rho
        for j in inds(i)
            if occursin("Link",string(tags(j)))
                push!(arr,dim(j))
            end
        end
    end
    return arr
end

N=8
s = siteinds("Qubit", N)
psi1 = productMPS(s, "Up" )
rho = outer(psi1',psi1)
gates = ITensor[]

h = op("H",[s[1]])
h1 = op("H",[s[3]])
h2 = op("H",[s[5]])
h3 = op("H",[s[7]])
push!(gates, h)
push!(gates, h1)
push!(gates, h2)
push!(gates, h3)
h = op("CNOT",[s[1],s[2]])
h1 = op("CNOT",[s[3],s[4]])
h2 = op("CNOT",[s[5],s[6]])
h3 = op("CNOT",[s[7],s[8]])
push!(gates, h)
push!(gates, h1)
push!(gates, h2)
push!(gates, h3)

cutoff = 1E-8

rho = apply(gates, rho; apply_dag=true)
psi = apply(gates,psi1)

Hitensor = ITensor(1.)
for i = 1:N
    Hitensor *= rho[i]
end

A=Array(Hitensor,s[1]',s[2]',s[3]',s[4]',s[5]',s[6]',s[7]',s[8]',s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8])
display(reshape(A,256,256))






using ITensors
using Random
using Plots
using Statistics
using LaTeXStrings
using LinearAlgebra
using PyCall

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
    return rho
  end

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

function gen_step(N,rho,s,step_num,meas_p,noise)
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
  
  #kraus_dephase
  rho = kraus_dephase(rho,s,noise)
  return rho,measured_vals
end

function do_exp(N,steps,meas_p,noise)
  s = siteinds("Qubit", N) #+1 for ancilla
  psi = productMPS(s, "Up" )
  rho=outer(psi',psi)

  svn =[]
  for i in 1:steps
      rho ,meas_svn= gen_step(N,rho,s,i,meas_p,noise)
      append!(svn,meas_svn)
  end
  #tri_mut = tri_part_MI(psi,[1,2],[3,4],[5,6])
  tri_mut = []
  # for i in 3:length(psi)-1
  #     arr = two_point_MI(psi,2,i)
  #     append!(tri_mut,arr)
  # end
  return svn,tri_mut,rho,s
end


decays=[]
svns=[]
n=6
# N = 6
# cutoff = 1E-8
steps = 50


svn,tri_mut,rho,sites =do_exp(n,steps,0.0,0.1);
print(bond_dim_array(rho))

#checked 0 meas 0 noise, say cannonizaation scaling d^2^n as kinda expected and happy to see ☑
#checked 1 meas 0 noise , all dim 1 links as expected ☑
#checked 0.5 meas 0 noise some growth in bond dim but not much ☑
#check 0 meas variable noise see a critical value around 0.23 ☑

Hitensor = ITensor(1.)
for i = 1:n
    Hitensor *= rho[i]
end

A=Array(Hitensor,sites[1]',sites[2]',sites[3]',sites[4]',sites[5]',sites[6]',sites[1],sites[2],sites[3],sites[4],sites[5],sites[6]);
display(reshape(A,64,64))
#check svd
F=svd(reshape(A,64,64))
reshape(A,64,64) ≈ F.U * diagm(F.S) * F.Vt
for projansk_iter in 1:64
colm1 = F.U[:,projansk_iter]
cutoff = 1E-8
colmtens=ITensor(ComplexF64,reshape(colm1,2,2,2,2,2,2),sites)
colmmps = MPS(colmtens,sites;cutoff=cutoff)
print(bond_dim_array(colmmps),"\n")
end
#for high meas bond dim for the mps is low bond dim as expected pure state mpo which is expected because
# i equivilant to each case, pure remains pure

#for no meas but noise, high noise give mpo with low bond and diagonalizing it yeilds low bond dim mps (steven white paper) Pca
#for no meas but low gives a high dim mpo and diagonalizing it yeilds high dim mps
#this kinda says a low entangled density operator can be decompose into low entangled states
# and high entangled density operators can be decomposed into high entangled states