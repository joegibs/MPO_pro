using ITensors
using Statistics
using LinearAlgebra


decays=[]
svns=[]
n=4
steps = 50
trials=5
noise_interval=0.0:0.1:.4
meas_interval=0:0.25:1

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

let
N=6
for tria in 1:50
print(tria)
for interj in 1:50
step_num=6
s = siteinds("Qubit", N) #+1 for ancilla
psi = productMPS(s, "Up" )
rho=outer(psi',psi)

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
step_num=7
s = siteinds("Qubit", N) #+1 for ancilla
psi = productMPS(s, "Up" )
rho=outer(psi',psi)

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
meas_p=0.15
samp_row=gen_samp_row(N,meas_p)
# rho = samp_mps(rho,s,samp_row)

for interi in 1:10
N = length(rho)
samp =deepcopy(rho)
samp = samp/tr(samp)
if norm(samp)>1.5
    print("ahh")
    print(norm(rho))
    print(norm(samp))
end
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
end
end
end
end