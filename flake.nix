{
  description = "Style search - doomp scraper and embedding explorer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    bun2nix.url = "github:nix-community/bun2nix";
    bun2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = inputs @ {
    flake-parts,
    nixpkgs,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];

      flake.nixosModules = rec {
        style-search = import ./nix/modules/style-search.nix { self = inputs.self; };
        default = style-search;
      };

      perSystem = {
        pkgs,
        lib,
        system,
        ...
      }: let
        libs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          libGL
          glib
          xorg.libxcb
          cudaPackages_13_0.cudatoolkit
          cudaPackages_13_0.cudnn
        ];
      in {
        _module.args.pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        packages = rec {
          style-search = pkgs.callPackage ./nix/packages/style-search.nix {
            bun2nix = inputs.bun2nix.packages.${system}.default;
          };
          default = style-search;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs;
            [
              python313
              uv
              bun
              process-compose
              sqlite
            ]
            ++ libs;

          shellHook = ''
            export LD_LIBRARY_PATH="${lib.makeLibraryPath libs}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            export CUDA_HOME="${pkgs.cudaPackages_13_0.cudatoolkit}"
            export CUDA_PATH="${pkgs.cudaPackages_13_0.cudatoolkit}"
            export CUDNN_HOME="${pkgs.cudaPackages_13_0.cudnn}"
            export UV_PYTHON_PREFERENCE=system
          '';
        };
      };
    };
}
