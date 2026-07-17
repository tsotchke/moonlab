# typed: strict
# frozen_string_literal: true

# Homebrew formula for Moonlab Quantum Simulator
# This file should be copied to a homebrew-moonlab tap repository
#
# Usage:
#   brew tap tsotchke/moonlab
#   brew install moonlab
#
class Moonlab < Formula
  desc "High-performance quantum computing simulator with GPU acceleration"
  homepage "https://github.com/tsotchke/moonlab"
  url "https://github.com/tsotchke/moonlab/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"
  head "https://github.com/tsotchke/moonlab.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "libomp"
  depends_on "openssl@3"
  on_linux do
    depends_on "openblas"
  end

  def install
    libomp = Formula["libomp"]
    openssl = Formula["openssl@3"]

    args = std_cmake_args + %W[
      -G Ninja
      -DOPENSSL_ROOT_DIR=#{openssl.opt_prefix}
      -DQSIM_BUILD_TESTS=OFF
      -DQSIM_BUILD_EXAMPLES=OFF
      -DQSIM_BUILD_BENCHMARKS=OFF
      -DQSIM_NATIVE_ARCH=OFF
      -DQSIM_FAST_MATH=OFF
      -DQSIM_WERROR=ON
    ]

    ENV.append "LDFLAGS", "-L#{libomp.opt_lib}"
    ENV.append "CPPFLAGS", "-I#{libomp.opt_include}"
    unless OS.mac?
      openblas = Formula["openblas"]
      ENV.append "LDFLAGS", "-L#{openblas.opt_lib}"
      ENV.append "CPPFLAGS", "-I#{openblas.opt_include}"
    end

    system "cmake", "-S", ".", "-B", "build", *args
    system "cmake", "--build", "build", "--target", "quantumsim", "moonlab-control-server"
    system "cmake", "--install", "build"
  end

  def caveats
    <<~EOS
      Moonlab Quantum Simulator has been installed!

      CMake consumers can use:
        find_package(quantumsim CONFIG REQUIRED)
        target_link_libraries(app PRIVATE quantumsim::quantumsim)

      C ABI headers are available under:
        #{include}/moonlab

      For Python bindings:
        pip install moonlab

      Documentation: https://github.com/tsotchke/moonlab
    EOS
  end

  test do
    assert_path_exists lib/"libquantumsim.dylib" if OS.mac?
    assert_path_exists include/"moonlab/moonlab_export.h"
    assert_path_exists lib/"cmake/quantumsim/quantumsim-config.cmake"

    (testpath/"consumer.c").write <<~C
      #include <moonlab/moonlab_export.h>

      int main(void) {
          int major = -1;
          int minor = -1;
          int patch = -1;
          moonlab_abi_version(&major, &minor, &patch);
          return major < 0 || minor < 0 || patch < 0;
      }
    C

    system ENV.cc, "consumer.c", "-I#{include}", "-L#{lib}", "-lquantumsim",
           "-o", "consumer"
    ENV.prepend_path "DYLD_LIBRARY_PATH", lib.to_s if OS.mac?
    ENV.prepend_path "LD_LIBRARY_PATH", lib.to_s if OS.linux?
    system "./consumer"
  end
end
