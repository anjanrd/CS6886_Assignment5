# CS6886 -- Optimising attention on a CPU
#
# Path to your C compiler. Use the LATEST clang, for example:
#   Linux:  /usr/bin/clang-23        macOS (Homebrew):  /opt/homebrew/opt/llvm/bin/clang
CC = $(or $(shell command -v clang-20),clang)

CFLAGS = -O3 -march=native
OMP    = -fopenmp
INPUTS = input/qkv_N1024_d64.bin input/qkv_N2048_d64.bin input/qkv_N4096_d64.bin input/qkv_N4096_d128.bin

# make <variant>  : compile src/attn_<variant>.c and report its time on every input
# make all        : the same for every variant
# make check      : check every variant's output against naive's output
# make clean      : delete build/ and output/

all: naive loops tiled fused best omp

naive loops tiled fused best: $(INPUTS)
	@mkdir -p build output
	$(CC) $(CFLAGS) -o build/$@ src/attn_$@.c -lm
	@set -e; for f in $(INPUTS); do ./build/$@ $$f output/$@_`basename $$f`; done

omp: $(INPUTS)
	@mkdir -p build output
	$(CC) $(CFLAGS) $(OMP) -o build/$@ src/attn_$@.c -lm
	@set -e; for f in $(INPUTS); do ./build/$@ $$f output/$@_`basename $$f`; done

# Passes if max|O - O_naive| / max|O_naive| <= 1e-4. Outputs never match exactly:
# different loop orders round differently.
check:
	@python3 -c "$$CHECK"

clean:
	rm -rf build output

.PHONY: all naive loops tiled fused best omp check clean

define CHECK
import math, struct, sys
from array import array
from pathlib import Path
def load(p):
    with open(p, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        a = array("f"); a.fromfile(f, n*d)
        if f.read(1) or not all(map(math.isfinite, a)): raise ValueError(p)
    return (n, d), a
variants = ["loops", "tiled", "fused", "best", "omp"]
cases = [(1024,64), (2048,64), (4096,64), (4096,128)]
def modified(p): return p.stat().st_mtime_ns
def fresh(variant, inp, input_path):
    binary = Path("build") / variant
    output = Path("output") / (variant + "_" + inp)
    if not binary.is_file(): raise ValueError("missing binary: " + str(binary))
    if not output.is_file(): raise ValueError("missing output: " + str(output))
    if modified(binary) < max(modified(Path("Makefile")), modified(Path("src/attn_" + variant + ".c"))):
        raise ValueError("stale binary: " + str(binary))
    if modified(output) < max(modified(binary), modified(input_path)):
        raise ValueError("stale output: " + str(output))
    return output
failed = False
for expected in cases:
    inp = "qkv_N%d_d%d.bin" % expected
    input_path = Path("input") / inp
    if not input_path.is_file(): sys.exit("Missing supplied input: " + str(input_path))
    try:
        with input_path.open("rb") as f: shape = struct.unpack("<ii",f.read(8))
        if shape != expected or input_path.stat().st_size != 8+12*math.prod(expected):
            raise ValueError("invalid input shape or size: " + str(input_path))
        shape, ref = load(fresh("naive", inp, input_path))
    except (OSError, ValueError, EOFError, struct.error) as error:
        sys.exit(str(error))
    if shape != expected: sys.exit("Wrong naive output shape for " + inp)
    scale = max(map(abs, ref))
    for v in variants:
        try:
            out_shape, out = load(fresh(v, inp, input_path))
        except (OSError, ValueError, EOFError, struct.error) as error:
            print(inp, v, "FAIL:", error)
            failed = True
            continue
        absolute = max(abs(a-b) for a,b in zip(out,ref)) if out_shape == shape else float("inf")
        err = absolute / scale if scale else (0.0 if absolute == 0 else float("inf"))
        print("%-20s %-12s %s  (error %.1e)" % (inp, v, "PASS" if err <= 1e-4 else "FAIL", err))
        failed |= err > 1e-4
sys.exit(int(failed))
endef
export CHECK
