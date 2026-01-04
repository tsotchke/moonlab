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

  def install
    # Set OpenMP paths for Apple Silicon
    libomp = Formula["libomp"]
    ENV["LDFLAGS"] = "-L#{libomp.opt_lib} #{ENV["LDFLAGS"]}"
    ENV["CPPFLAGS"] = "-I#{libomp.opt_include} #{ENV["CPPFLAGS"]}"

    # Build using Make (primary build system)
    system "make", "clean"
    system "make"

    # Build tests
    system "make", "tests", "unit_tests"

    # Run tests to verify build
    system "make", "test_unit"

    # Install library files
    lib.install "lib/libquantum_sim.a"

    # Install headers
    include.install Dir["include/*.h"]

    # Also install to lib/moonlab for namespacing
    (lib/"moonlab").mkpath
    (lib/"moonlab").install "lib/libquantum_sim.a"
  end

  def caveats
    <<~EOS
      Moonlab Quantum Simulator has been installed!

      The static library is available at:
        #{lib}/libquantum_sim.a

      To use in your project, link with:
        -L#{lib} -lquantum_sim -lomp

      For Python bindings:
        pip install moonlab

      Documentation: https://github.com/tsotchke/moonlab
    EOS
  end

  test do
    # Test that library was built correctly
    assert_predicate lib/"libquantum_sim.a", :exist?
  end
end
