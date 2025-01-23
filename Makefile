GENCODES = -arch=sm_90 -lineinfo

CCCL_DIR=/home/scratch.christinaz_sw/ProjectTopk/christinaz/fixBlockLoad/cccl

CPPFLAGS += -DNDEBUG \
            -DBENCHMARK_DATA_TYPE=$(BENCHMARK_DATA_TYPE) \
            -I../include \
            -I${CCCL_DIR}/cub \
            -I${CCCL_DIR}/libcudacxx/include/ \
            -I${CCCL_DIR}/thrust -DTHRUST_IGNORE_CUB_VERSION_CHECK

LDFLAGS += -lcurand
NVCCFLAGS = -c -O2 -std=c++17 \
            -Xcompiler "-Wall -Wextra -Wno-unused-parameter" \
            --expt-relaxed-constexpr \
            --extended-lambda



benchmark: benchmark.o
	nvcc $(GENCODES) $(LDFLAGS) -o $@ $^

benchmark.o: benchmark.cu
	nvcc $(NVCCFLAGS) $(GENCODES) $(CPPFLAGS) -o $@ $<


.PHONY: clean
clean:
	rm benchmark benchmark.o
