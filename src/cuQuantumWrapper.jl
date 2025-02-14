module cuQuantumWrapper

using CUDA
using cuQuantum_jll

include("cuStateVec/cuStateVec.jl")

export custatevecHandle
export custatevecCreate
export custatevecCreate!
export custatevecDestroy!

export custatevecGetDefaultWorkspaceSize
export custatevecSetWorkspace

export custatevecGetProperty

export custatevecStateVector
export custatevecInitializeStateVector
export custatevecInitializeStateVector!
export custatevecZero
export custatevecUniform
export custatevecGHZ
export custatevecW
export custatevecZero!
export custatevecUniform!
export custatevecGHZ!
export custatevecW!

export custatevecGate
export PauliI
export PauliX
export PauliY
export PauliZ
export Hadamard
export CNOT
export custatevecApplyMatrix!

export custatevecMeasureOnZBasis!

end
