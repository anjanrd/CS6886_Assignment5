# CS6886 -- Optimising attention on a CPU
#
# Path to your C compiler. Use the LATEST clang, for example:
#   Linux:  /usr/bin/clang-23        macOS (Homebrew):  /opt/homebrew/opt/llvm/bin/clang
CC = $(or $(shell command -v clang-20),clang)

CFLAGS = -O3 -march=native
OMP    = -fopenmp
INPUTS = $(wildcard input/*.bin)

# make <variant>  : compile src/attn_<variant>.c and report its time on every input
# make all        : the same for every variant
# make check      : check every variant's output against naive's output
# make clean      : delete build/ and output/

all: naive loops tiled fused best omp

naive loops tiled fused best:
	@mkdir -p build output
	$(CC) $(CFLAGS) -o build/$@ src/attn_$@.c -lm
	@set -e; for f in $(INPUTS); do ./build/$@ $$f output/$@_`basename $$f`; done

omp:
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
import glob, math, os, struct, sys
from array import array
def load(p):
    with open(p, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
        a = array("f"); a.fromfile(f, n*d)
        if f.read(1) or not all(map(math.isfinite, a)): raise ValueError(p)
    return (n, d), a
variants = ["loops", "tiled", "fused", "best", "omp"]
inputs = sorted(glob.glob("input/*.bin"))
if not inputs: sys.exit("No supplied inputs found.")
failed = False
for input_path in inputs:
    inp = os.path.basename(input_path)
    with open(input_path,"rb") as f: expected = struct.unpack("<ii",f.read(8))
    shape, ref = load("output/naive_" + inp)
    if shape != expected: sys.exit("Wrong naive output shape for " + inp)
    scale = max(map(abs, ref))
    for v in variants:
        path = "output/" + v + "_" + inp
        if not os.path.exists(path):
            print(inp, v, "FAIL: missing output")
            failed = True
            continue
        out_shape, out = load(path)
        absolute = max(abs(a-b) for a,b in zip(out,ref)) if out_shape == shape else float("inf")
        err = absolute / scale if scale else (0.0 if absolute == 0 else float("inf"))
        print("%-20s %-12s %s  (error %.1e)" % (inp, v, "PASS" if err <= 1e-4 else "FAIL", err))
        failed |= err > 1e-4
sys.exit(int(failed))
endef
export CHECK
