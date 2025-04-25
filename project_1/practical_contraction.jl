using TensorKit

χ = U₁Space([i=>(4-abs(i))*10 for i in -3:3]...)
D = U₁Space([i=>3-abs(i) for i in -2:2]...)
d = U₁Space(0=>2,1=>1,-1=>1)
@show χ D d
# χ = ℂ^dim(χ)
# D = ℂ^dim(D)
# d = ℂ^dim(d)
# @show χ D d

# double layout
# A = rand(ComplexF64, D*D*D'*D' ← d);
# ALu = rand(ComplexF64, χ*D*D' ← χ);
# ALd = rand(ComplexF64, χ*D*D' ← χ);
# FL = rand(ComplexF64, χ*D'*D ← χ);
# @tensoropt FL[-1 -2 -3; -4] := FL[6 5 4; 1] * ALu[1 2 3; -4] * A[5 8 -2 2; 9] * 
# conj(A[4 7 -3 3; 9]) * adjoint(ALd)[-1; 6 8 7];

# single layout
DD = fuse(D, D');
It = isomorphism(DD, D*D');
A = rand(ComplexF64, D*D*D'*D' ← d);
@tensoropt M[10 11; 12 13] := A[1 2 3 4; 5] * conj(A[6 7 8 9; 5]) * It[10; 1 6] * It[11; 2 7] * conj(It[12; 3 8]) * conj(It[13; 4 9]);
ALu = rand(ComplexF64, χ*DD ← χ);
ALd = rand(ComplexF64, χ*DD ← χ);
FL = rand(ComplexF64, χ*DD' ← χ);
println("======================")
@tensoropt FL[-1 -2; -3] := FL[4 3; 1] * ALu[1 2; -3] * M[3 5; -2 2] * conj(ALd[4 5; -1]);