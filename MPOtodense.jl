using ITensors


N = 2
sites = siteinds("S=1/2",N;conserve_qns = true)

ampo = OpSum()
for j=1:N-1
    ampo += "Sz",j,"Sz",j+1
    ampo += 0.5,"S+",j,"S-",j+1
    ampo += 0.5,"S-",j,"S+",j+1
end

H = MPO(ampo,sites)

Hitensor = ITensor(1.)
for i = 1:N
    Hitensor *= H[i]
end

A=Array(Hitensor,sites[1]',sites[2]',sites[1],sites[2])
println(reshape(A,4,4))