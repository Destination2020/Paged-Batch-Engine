# Cache core CPU tests

This target validates the P1 page schema, logical transfer plan, binding
preflight, and CPU layout oracle without configuring or compiling CUDA.

```bash
cmake -S test/cache_core -B /tmp/Paged-Batch-Engine-cache-core
cmake --build /tmp/Paged-Batch-Engine-cache-core -j
ctest --test-dir /tmp/Paged-Batch-Engine-cache-core --output-on-failure
```

To include the real CPU BlockAllocator, PageDirectory, leases and KV manager:

```bash
cmake -S test/cache_core -B /tmp/Paged-Batch-Engine-ownership-cpu \
  -DKUIPER_TEST_CPU_OWNERSHIP=ON \
  -Dglog_DIR=/tmp/Paged-Batch-Engine-build-p1/_deps/glog-build \
  -DARMADILLO_INCLUDE_DIR=/tmp/Paged-Batch-Engine-build-p1/_deps/armadillo-src/include \
  -DARMADILLO_LIBRARY=/tmp/Paged-Batch-Engine-build-p1/_deps/armadillo-build/libarmadillo.so
cmake --build /tmp/Paged-Batch-Engine-ownership-cpu -j8
ctest --test-dir /tmp/Paged-Batch-Engine-ownership-cpu --output-on-failure
```

The explicit dependency paths above reuse this workspace's packages. With system
packages installed, omit those paths. This target requires only a C++ compiler,
GTest, glog and Armadillo; it neither enables CUDA language nor links CUDA.
`KUIPER_CPU_ONLY` explicitly rejects CUDA/pinned allocation and device operations.
The minimal schema/planner target remains available without these extra packages.
