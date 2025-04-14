{
  description = "A simple Nix flake for a CUDA development shell";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  inputs.git-hooks.url = "github:cachix/git-hooks.nix";

  outputs =
    {
      self,
      nixpkgs,
      git-hooks,
    }:

    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      checks.${system} = {
        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
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
            pkgs.stdenv.cc
            pkgs.gsl

            pkgs.bear
            pkgs.clang-tools
            pkgs.gdb

            python
          ]
          ++ self.checks.${system}.pre-commit-check.enabledPackages;
      };
    };
}
