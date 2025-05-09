{
  description = "A simple Nix flake for a CUDA development shell";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  inputs.git-hooks.url = "github:cachix/git-hooks.nix";
  inputs.nixgl.url = "github:nix-community/nixGL";

  outputs =
    {
      self,
      nixpkgs,
      git-hooks,
      nixgl,
    }:

    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
        overlays = [
          nixgl.overlays.default
          (import ./nix/overlays/nsight-compute-symlinks.nix)
        ];
      };
    in
    {
      checks.${system} = {
        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;

          package = pkgs.prek;

          hooks = {
            clang-format.enable = true;
            clang-tidy = {
              enable = true;
              types_or = [
                "c"
                "c++"
                "cuda"
              ];
            };
            cmake-format.enable = true;
            nixfmt-rfc-style.enable = true;
            ruff.enable = true;
            ruff-format.enable = true;
          };
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        inherit (self.checks.${system}.pre-commit-check) shellHook;
        packages =
          let
            python = pkgs.python3.withPackages (
              ps: with ps; [
                ipywidgets
                jupytext
                notebook
                pandas
                scipy
                seaborn
                tqdm
              ]
            );
          in
          [
            python

            pkgs.cudaPackages.cuda_nvcc
            pkgs.cudaPackages.cuda_cudart
            pkgs.cudaPackages.libcurand
            pkgs.cudaPackages.nsight_compute

            pkgs.gsl

            pkgs.bear
            pkgs.clang-tools
            pkgs.cudaPackages.cuda_sanitizer_api
            pkgs.gdb

            pkgs.prek
          ]
          ++ nixpkgs.lib.optional (builtins ? currentTime) pkgs.nixgl.auto.nixGLDefault
          ++ self.checks.${system}.pre-commit-check.enabledPackages;
      };
    };
}
