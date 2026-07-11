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
  url "https://github.com/tsotchke/moonlab/archive/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"
  head "https://github.com/tsotchke/moonlab.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "libomp"
  depends_on "openssl@3"

  def install
    libomp = Formula["libomp"]
    openssl = Formula["openssl@3"]

    args = std_cmake_args + %W[
      -G Ninja
      -DOPENSSL_ROOT_DIR=#{openssl.opt_prefix}
      -DQSIM_BUILD_TESTS=OFF
      -DQSIM_BUILD_EXAMPLES=OFF
      -DQSIM_BUILD_BENCHMARKS=OFF
    ]

    ENV.append "LDFLAGS", "-L#{libomp.opt_lib}"
    ENV.append "CPPFLAGS", "-I#{libomp.opt_include}"

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
    assert_predicate lib/"libquantumsim.dylib", :exist? if OS.mac?
    assert_predicate include/"moonlab/moonlab_export.h", :exist?
    assert_predicate lib/"cmake/quantumsim/quantumsim-config.cmake", :exist?

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
