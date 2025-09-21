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
                  nanobind = pySuper.nanobind.overrideAttrs (old: {
                    postInstall = (old.postInstall or "") + ''
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
      packages.${system}.default = pkgs.python3Packages.callPackage ./package.nix { };

      devShells.${system}.default =
        let
          impure = builtins ? currentSystem;
          useNixgl = impure && pkgs.stdenv.isLinux;
        in
        pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.default ];

          packages =
            let
              python = pkgs.python3.withPackages (
                ps: with ps; [
                  ipywidgets
                  jupytext
                  notebook
                  pip
                  seaborn
                ]
              );
            in
            [
              pkgs.bear
              pkgs.clang-tools
              pkgs.gdb

              pkgs.cudaPackages.cuda_sanitizer_api

              pkgs.basedpyright
              python
            ]
            ++ self.packages.${system}.default.optional-dependencies.dev;

          shellHook = ''
            export PYTHONPATH=build/:$PYTHONPATH
          ''
          + nixpkgs.lib.optionalString useNixgl ''
            export LD_LIBRARY_PATH=$(${pkgs.nixgl.auto.nixGLDefault}/bin/nixGL printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH
          '';
        };
    };
}
