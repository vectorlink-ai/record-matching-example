{pyproject-nix, pyproject-build-systems, pyproject-overrides, cudaPackages, lib, workspace, python3, callPackage}:
let overlay = workspace.mkPyprojectOverlay {
      sourcePreference = "wheel";
    };
    fix-vectorlink-gpu-build = final: prev: {
      vectorlink-gpu = prev.vectorlink-gpu.overrideAttrs (old: {
        buildInputs = old.buildInputs or [] ++ [final.hatchling final.pathspec final.pluggy final.packaging final.trove-classifiers];
      });
    };
in
(callPackage pyproject-nix.build.packages {
  python = python3;
}).overrideScope (
  lib.composeManyExtensions [
    pyproject-build-systems.overlays.default
    overlay
    pyproject-overrides.cuda
    pyproject-overrides.default
    fix-vectorlink-gpu-build
  ]
)
