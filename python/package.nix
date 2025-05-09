{
  buildPythonPackage,
  cmake,
  nanobind,
  ninja,
  scikit-build-core,
  cudaPackages,
  numpy,
  pytest,
}:
buildPythonPackage {
  pname = "ising-mcmc";
  version = "0.1.0";

  pyproject = true;

  src = builtins.path {
    path = ./.;
    name = "ising-mcmc-src";
  };

  build-system = [
    nanobind
    ninja
    scikit-build-core
  ];

  nativeBuildInputs = [
    cmake
    cudaPackages.cuda_nvcc
  ];

  buildInputs = [
    cudaPackages.libcurand
    cudaPackages.cuda_cudart
  ];

  dependencies = [
    numpy
  ];

  dontUseCmakeConfigure = true;

  pythonImportsCheck = [
    "ising_mcmc"
    "ising_mcmc.cuda"
  ];

  disabledTests = [
    "test_wrapper" # requires GPU
  ];

  passthru.optional-dependencies.dev = [ pytest ];
}
