# https://github.com/NixOS/nixpkgs/issues/423609#issuecomment-3053328236
final: prev: {
  cudaPackages = prev.cudaPackages // {
    nsight_compute = prev.cudaPackages.nsight_compute.overrideAttrs (old: {
      postInstall = old.postInstall + ''
        ln -s $out/bin/target/linux-desktop-glibc_2_11_3-x64 \
          $out/bin/target/linux-desktop-glibc_2_11_3-x86
        ln -s $out/sections $out/bin/sections
      '';
    });
  };
}
