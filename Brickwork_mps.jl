using ITensors
using Random
using Plots
using Statistics
using LaTeXStrings
using LinearAlgebra
using PyCall


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
    eps=0.5#rand()
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


function rec_ent(psi,b,s)  
    orthogonalize!(psi, b)
    _,S = svd(psi[b], (linkind(psi, b-1), s[b]))
    SvN = 0.0
    for n in 1:dim(S, 1)
      p = S[n,n]^2
      if p != 0
        SvN -= p * log2(p)
      end
    end
    @show(diag(S).^2)
    return SvN
end

function split_ren(psi,b)
    rho=outer(psi',psi)
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

function entropy_subsys(psi,inds)
    i = inds[1];
    j = inds[end];
    #'gauge' the MPS to site i
    #any 'position' between i and j, inclusive, would work here
    orthogonalize!(psi,i)

    psidag = dag(psi);
    prime(psidag,"Link");

    #//index linking i to i-1:
    li_1 = linkind(psi,i);

    rho = prime(psi[i],li_1)*prime(psidag[i],"Site");

    for k in inds[2:end-1]#(int k = i+1; k < j; ++k)
        # print(k)
        lk = linkind(psi,k)
        rho *= prime(psi[k],lk);
        rho *= prime(psidag[k],"Site");
    end
    for k in setdiff([i:j...],[inds])
        rho *= psi[k];
        rho *= psidag[k];
    end
    #index linking j to j+1:
    lj = linkind(psi,inds[end]+1);
    rho *= prime(psi[j],lj);
    rho *= prime(psidag[j],"Site");

    D,U = eigen(rho)
    SvN = 0.0
    for n in 1:dim(D, 1)
        # print(n,'\n')
        p = D[n,n]^2
        if p != 0
        SvN -= p * log(p)
        end
    end
    return SvN
end

function two_point_entropy(psi,i,j)
    #'gauge' the MPS to site i
    #any 'position' between i and j, inclusive, would work here
    orthogonalize!(psi,i)

    psidag = dag(psi);
    prime(psidag,"Link");

    #//index linking i to i-1:
    li_1 = linkind(psi,i);

    rho = prime(psi[i],li_1)*prime(psidag[i],"Site");

    for k in i+1:j-1
        rho *= psi[k];
        rho *= psidag[k];
    end
    #index linking j to j+1:
    lj = linkind(psi,j+1);
    rho *= prime(psi[j],lj);
    rho *= prime(psidag[j],"Site");
    # print(rho)
    D,U = eigen(rho)
    SvN = 0.0
    for n in 1:dim(D, 1)
        # print(n,'\n')
        p = D[n,n]*conj(D[n,n])
        if p != 0
        SvN -= p * log2(p)
        end
    end
    return real(SvN)
end

function tri_part_MI(psi_abc,sysa,sysb,sysc)
    hab = entropy_subsys(psi_abc, vcat(sysa,sysb))
    hac = entropy_subsys(psi_abc, vcat(sysa,sysc))
    hbc = entropy_subsys(psi_abc, vcat(sysc,sysb))
    
    habc = entropy_subsys(psi_abc, vcat(sysa,sysb,sysc))

    ha = entropy_subsys(psi_abc, sysa)
    hb = entropy_subsys(psi_abc, sysb)
    hc = entropy_subsys(psi_abc, sysc)
    return hb + ha + hc - hab - hac - hbc + habc
end

function two_point_MI(psi, index_a, index_b)
    ha = rec_ent(psi, index_a)
    hb = rec_ent(psi, index_b)
    hab = two_point_entropy(psi, index_a,index_b)
    print("entropies" , ha, " " , hb , " ",hab,"\n")
    return ha + hb - hab
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
ITensors.op(::OpName"Ihaar",::SiteType"Qubit") = 
    Ihaargate(4)
ITensors.op(::OpName"Iden",::SiteType"Qubit") = 
[1 0
0 1]
function gen_samp_row(N,meas_p)
    return [rand()<meas_p ? 1 : 0 for i in 1:N]
end
function samp_mps(psi,s,site)
    cutoff = 1E-8

    magz = expect(psi,"Pup")
    proj="Pup"
    if magz[site]<rand()
        proj ="Pdn"
    end
    G = ITensor[op(proj,s[site])]
    psi = apply(G, psi; cutoff)
    normalize!(psi)
    return psi
end

function gen_step(N,psi,s,step_num,meas_p)
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

    psi = apply(gates, psi; cutoff)
    #calculate obs
    measured_vals = rec_ent(psi,Int(round(N/2)),s)
    
    #metts shenanagins
    # psi = metts(psi,s)
    #sample as needed
    samp_row=gen_samp_row(N,meas_p)
    for (i,x) in enumerate(samp_row)
        if Bool(x)
            psi = samp_mps(psi,s,i)
        end
    end
    return psi,measured_vals
end

function do_exp(N,steps,meas_p)
    s = siteinds("Qubit", N)
    psi = productMPS(s, "Up" )

    svn =[]
    for i in 1:steps
        psi ,meas_svn= gen_step(N,psi,s,i,meas_p)
        append!(svn,meas_svn)
    end
    #tri_mut = tri_part_MI(psi,[1,2],[3,4],[5,6])
    tri_mut = []
    # for i in 3:length(psi)-1
    #     arr = two_point_MI(psi,2,i)
    #     append!(tri_mut,arr)
    # end
    return psi,s,svn,tri_mut
end
function do_trials(N,steps,meas_p,trials)
    psi,s,svn_trials,tri_trials = do_exp(N,steps,meas_p)
    for i in 2:trials
        print(i)
        psi,s,nSvn,ntm = do_exp(N,steps,meas_p)
        svn_trials = 2*mean([(i-1)/i*svn_trials,1/i*nSvn])
        tri_trials = 2*mean([(i-1)/i*tri_trials,1/i*ntm])

        
    end
    return psi,s,svn_trials,tri_trials
end

function rec_ent_mpo(rho,b,s)
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
      @show(diag(S))
      SvN = 0.0
      for n in 1:dim(S, 1)
        p = S[n,n]
        if p != 0
          SvN -= p * log2(p)
        end
      end
      return SvN
    end

N=6
psi,s,not,anything = do_trials(N,20,0,1)
rec_ent(psi,Int(N/2),s)
# A=reshape(array(contract(psi)),(2^N,))
rho = outer(psi',psi)
rec_ent_mpo(rho,Int(N/2),s)







decays=[]
sits = [6]
interval = 0.0:0.1:1
for n in sits#[6,8,10]
N = n
# cutoff = 1E-8
steps = 4*N
meas_p=0.
svns=[]
mut = []
    for i in interval
        print("\n meas_p $i \n")
        svn,tri_mut =do_trials(N,steps,i,50)
        avgsvn = [(svn[x]+svn[x+1])/2 for x in 1:2:(size(svn)[1]-1)]
        append!(svns,[avgsvn])
        append!(mut,tri_mut)
    end
decay = [svns[i][end] for i in 1:size(svns)[1]]
append!(decays,[decay])
end


p = plot(real(svns),title=string("Gate Rand", ", ", N, " qubit sites, varying meas_p"), label=string.(transpose([interval...])), linewidth=3,xlabel = "Steps", ylabel = L"$\textbf{S_{vn}}(L/2)$")
p = plot([0.0:0.1:1...],decays,title=string("Bip_ent Gat: 2haar, varying meas_p"), label=string.(transpose([6:2:14...])), linewidth=3,xlabel = "Meas_P", ylabel = L"$\textbf{S_{vn}}(L/2)$")
# # m = plot(real(mut))
# display(p)


py"""
import numpy as np
import scipy
L=$sits
interval = $interval#[x/10 for x in range(9)]
tot_vonq = $decays
def xfunc(p,l,pc,v):
    return (p-pc)*l**(1/v)

def Spc(pc,l):
    spot, = np.where(np.array(L)==l)
    return np.interp(pc,interval,tot_vonq[spot[0]])

def yfunc(p,l,pc):
    spot, = np.where(np.array(L)==l)

    a=np.interp(p,interval,tot_vonq[spot[0]])
    b = Spc(pc,l)
    return  a-b 

def mean_yfunc(p,pc):
    return np.mean([yfunc(p,l,pc) for l in L])
    from scipy.optimize import minimize

def R(params):
    pc,v = params
    #sum over all the square differences
    x_vals = [[xfunc(p,l,pc,v) for p in interval] for l in L]
    y_vals = [[yfunc(p,l,pc) for p in interval] for l in L]

    min_x = np.max([x[0] for x in x_vals]) #max for smallest value st all overlap
    max_x = np.min([x[-1] for x in x_vals]) # min again to take overlap
    xi = np.linspace(min_x,max_x)
    mean_x_vals = np.mean(x_vals,axis=0)
    mean_y_vals = [mean_yfunc(p,pc) for p in interval]
    
    def mean_y(x):
        return np.interp(x,mean_x_vals,mean_y_vals)
    
    return np.sum([[(np.interp(x,x_vals[i],y_vals[i]) - mean_y(x))**2 for x in xi] for i in range(len(L))]) 
initial_guess = [0.0,0.1]
res = scipy.optimize.minimize(R, initial_guess)
    
"""


ppc,vv=py"res.x"

py"""
ppc,vv=res.x
x_vals = [[xfunc(p,l,ppc,vv) for p in interval] for l in L]
y_vals = [[yfunc(p,l,ppc) for p in interval] for l in L]
# mean_y_vals = [mean_yfunc(p,0.26) for p in interval]
"""
plot(transpose(py"x_vals"),transpose(py"y_vals"),linewidth=3)