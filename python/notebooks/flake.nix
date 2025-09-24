{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  inputs.nixgl.url = "github:nix-community/nixGL";

  outputs =
    {
      self,
      nixpkgs,
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
        overlays =
          let
            overlay = self: super: {
              pythonPackagesExtensions = (super.pythonPackagesExtensions or [ ]) ++ [
                (pySelf: pySuper: {
                  ising-mcmc = pySelf.callPackage ../package.nix { };
                })
              ];
            };
          in
          [
            nixgl.overlays.default
            overlay
          ];
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages =
          let
            python = pkgs.python3.withPackages (
              ps: with ps; [
                ipywidgets
                ising-mcmc
                jax
                jaxlib
                jupytext
                notebook
                seaborn
                tqdm
              ]
            );
          in
          [
            python
            pkgs.basedpyright
          ]
          ++ nixpkgs.lib.optional (builtins ? currentTime) pkgs.nixgl.auto.nixGLDefault;

        shellHook = ''
          export PYTHONPATH=build/:$PYTHONPATH
        '';
      };
    };
}
