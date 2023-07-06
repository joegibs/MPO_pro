using ITensors


#code to check mps
d = 2
N = 5
A = randn(d^N)
sites = siteinds(d,N)
cutoff = 1E-8
maxdim = 10
M = MPS(A,sites;cutoff=cutoff,maxdim=maxdim)
check=reshape(array(contract(M)),(d^N,))
print(isapprox(A,check)) #approx due to numerical error
print(maximum(A-check))

### check mpos

d = 2
N = 5
# A = randn((d^N,d^N))
A = zeros(d^N)
A[1] = 1
checkA=zeros((d^N,d^N))
checkA[1,1]=1
sites = siteinds(d,N)
cutoff = 1E-8
maxdim = 10
M = MPS(A,sites;cutoff=cutoff,maxdim=maxdim)
mpo = outer(M',M)
check=reshape(array(contract(mpo)),(d^N,d^N))
print(isapprox(checkA,check)) #approx due to numerical error
print(maximum(checkA-check))