{
  description = "Moonlab quantum simulator -- distributed cloud quantum platform";

  inputs = {
    nixpkgs.url    = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";

    # jetpack-nixos: Anduril's port of L4T (Tegra) drivers + CUDA
    # toolkit to NixOS.  Used only on aarch64-linux Jetson hosts;
    # other platforms ignore this input.  The flake input lives on
    # github.com/anduril/jetpack-nixos.
    jetpack = {
      url = "github:anduril/jetpack-nixos";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, jetpack, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;  # CUDA / cuQuantum licenses
        };

        # Detect the Jetson case: aarch64-linux + jetpack input
        # provides cudaPackages.  On other platforms, fall back to
        # nixpkgs.cudaPackages (if allowUnfree allows).
        isJetson = system == "aarch64-linux";

        # Common build tooling for any platform.
        commonNativeBuildInputs = with pkgs; [
          cmake pkg-config gfortran python3
        ];

        commonBuildInputs = with pkgs; [
          openmpi openblas lapack
        ];

        # CUDA-enabled build (Jetson via jetpack-nixos cudaPackages).
        cudaPkgs = if isJetson
          then jetpack.packages.${system}.cudaPackages or null
          else (pkgs.cudaPackages or null);

        cudaBuildInputs = if cudaPkgs == null then [] else [
          cudaPkgs.cudatoolkit or cudaPkgs.cuda_cudart
        ];

      in {
        # `nix develop` -- drop into a shell with cmake/gcc/openmpi
        # and (on Jetson) the CUDA toolkit pre-arranged.  No more
        # nix-shell -p <list> boilerplate every command.
        devShells.default = pkgs.mkShell {
          name = "moonlab-dev";
          packages = commonNativeBuildInputs ++ commonBuildInputs;
          shellHook = ''
            echo "moonlab dev shell on ${system}"
            ${if isJetson then ''
              echo "  jetpack-nixos detected -- CUDA + L4T paths pre-wired"
              # Make the cuda-merged toolkit visible to cmake auto-detection.
              # Match only built trees (skip *.drv derivation source dirs)
              # AND require the result to actually contain bin/nvcc.
              for _c in /nix/store/*-cuda-merged-*; do
                case "$_c" in *.drv) continue ;; esac
                if [ -x "$_c/bin/nvcc" ]; then
                  export CUDAToolkit_ROOT="$_c"
                  break
                fi
              done
              if [ -n "$CUDAToolkit_ROOT" ]; then
                export PATH=$CUDAToolkit_ROOT/bin:$PATH
              fi
              # Driver stub for runtime libcuda.so.1.
              export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
              echo "  CUDAToolkit_ROOT=$CUDAToolkit_ROOT"
            '' else ""}
          '';
        };

        # `nix develop .#cuda` -- explicit CUDA-enabled variant, same
        # as the default on Jetson, useful name-collision-free on
        # non-Jetson aarch64 boxes.
        devShells.cuda = self.devShells.${system}.default;

        # `nix build` -- declarative build of libquantumsim.so + the
        # control-plane server.  Pulls in CUDA on Jetson automatically.
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "moonlab";
          version = builtins.readFile (./VERSION.txt);
          src = ./.;
          nativeBuildInputs = commonNativeBuildInputs;
          buildInputs = commonBuildInputs ++ cudaBuildInputs;

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DQSIM_ENABLE_MPI=ON"
            "-DQSIM_WERROR=OFF"
          ] ++ (if isJetson then [
            "-DQSIM_ENABLE_CUDA=ON"
            "-DCMAKE_CUDA_ARCHITECTURES=72-real"   # Xavier; override for Orin (87) or others
          ] else []);

          # On Jetson, point at the merged-cuda toolkit explicitly --
          # the auto-detection in CMakeLists.txt will then pick up
          # the rest via /etc/nv_tegra_release.
          preConfigure = if isJetson then ''
            export CUDAToolkit_ROOT=${cudaPkgs.cudatoolkit or cudaPkgs.cuda_cudart}
            export PATH=$CUDAToolkit_ROOT/bin:$PATH
          '' else "";

          # Don't try to run ctest at build time (we already have
          # ctest as the user-facing test surface).
          doCheck = false;
        };

        # `nix run .#bell_jetson` -- standalone Bell CUDA smoke test
        # binary, useful for `nix run` validation on a Jetson host.
        # Other apps follow the same pattern.
        apps.bell_jetson = flake-utils.lib.mkApp {
          drv = self.packages.${system}.default;
          name = "bell_jetson";
        };
      });
}
