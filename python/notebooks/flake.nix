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
        config.allowUnfree = true;
        overlays =
          let
            overlay = self: super: {
              pythonPackagesExtensions = (super.pythonPackagesExtensions or [ ]) ++ [
                (pySelf: pySuper: {

                  ising-mcmc = pySelf.callPackage ../package.nix { };

                  nanobind = pySuper.nanobind.overrideAttrs (old: {
                    postInstall =
                      (old.postInstall or "")
                      + ''
                        ln -sf $out/lib/${pySelf.python.libPrefix}/site-packages/nanobind/include $out
                      '';
                  });
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
      devShells.${system}.default =
        let
          impure = builtins ? currentSystem;
          useNixgl = impure && pkgs.stdenv.isLinux;
        in
        pkgs.mkShell {
          packages =
            let
              python = pkgs.python3.withPackages (
                ps: with ps; [
                  diskcache
                  ipywidgets
                  ising-mcmc
                  jupytext
                  notebook
                  seaborn
                  tqdm
                ]
              );
            in
            [
              pkgs.clang-tools
              pkgs.gdb

              pkgs.cudaPackages.cuda_sanitizer_api

              pkgs.basedpyright
              python
            ];

          shellHook =
            ''
              export PYTHONPATH=build/:$PYTHONPATH
            ''
            + nixpkgs.lib.optionalString useNixgl ''
              export LD_LIBRARY_PATH=$(${pkgs.nixgl.auto.nixGLDefault}/bin/nixGL printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH
            '';
        };
    };
}
