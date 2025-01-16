{
  inputs = {
    nixpkgs.url = "github:nixOS/nixpkgs?ref=nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-nix-overrides.url = "github:vectorlink-ai/pyproject-nix-overrides";
  };


  outputs = {
      nixpkgs,
      flake-utils,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      pyproject-nix-overrides,
      ...
  }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        workspace = uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ./.;
        };
        pythonSet = pkgs.callPackage ./nix/uv-python.nix {
          inherit pyproject-nix pyproject-build-systems workspace;
          pyproject-overrides = pyproject-nix-overrides.overrides pkgs;
        };
      in
      {
        devShells = {
          default = pkgs.callPackage ./nix/uv-shell.nix {
            inherit workspace pythonSet;
          };
        };
      }
  );
}
