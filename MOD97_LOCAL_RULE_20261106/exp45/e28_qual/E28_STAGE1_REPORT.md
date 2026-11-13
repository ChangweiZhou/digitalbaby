# E28 stage 1 report (brief OPUS_TASK_E28_GRU_EVERY_BYTE_20261113.md, revision 2, commit 50bd275a)

Branch `claude/charming-cori-ljzwz1`. Worker: one Opus 5.5 session. Evidence level: development checks only; no stage 2 birth has run.
**Stage 1 complete: all six items pass. Stopped here; stage 2 (the twelve births) waits for the coordinator's release.**

## Host

- The first host print found Linux x86_64 and **no torch on any interpreter**. The worker stopped and reported. The coordinator ruled: install exactly `torch==2.15.0+cpu` and continue, with item 1 as the test of the host.
- Python: **3.13.18**, the version E26's births ran under, so 3.13 was used instead of 3.12. The container has two 3.13.18 builds. The worker used `/usr/local/bin/python3.13`, `3.13.18 (main, Oct 26 2026, 14:08:19) [Clang 23.1.3]`, which is the build a plain `python3` resolves to here, as in E26's command. The other is `/usr/bin/python3.13`, built with GCC 13.3.0. E26's receipt records only `3.13.18`, not which build.
- Virtual environment: `/usr/local/bin/python3.13 -m venv /root/e28venv`.
- Install command: `/root/e28venv/bin/pip install "torch==2.15.0+cpu" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --find-links /root/e28wheel`. The wheel was first fetched with `pip download --no-deps -d /root/e28wheel "torch==2.15.0+cpu" --index-url https://download.pytorch.org/whl/cpu`.
- Wheel: `torch-2.15.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl`, 209,528,982 bytes, SHA-256 `0f6f69701b2d38ddc412364dfb92e2e13e74960d33fa3573b4930ced626e81c7`.
- **NumPy added, not in the ruling:** `numpy==2.5.4` (`/root/e28venv/bin/pip install numpy`; wheel `numpy-2.5.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`, SHA-256 `6ffa07666f8da0eef81d149934a626d0d95fbd6838432a33e66245423a9062c0`). E26's own `tensors_sha` and data digest call `.numpy()`, and the torch wheel does not bring NumPy. NumPy only turns tensors into bytes there and enters no computation.
- Host print after the install: `Linux x86_64`, `torch.__version__ = 2.15.0+cpu`, 4 cores (Intel Xeon @ 2.10GHz), `OMP_NUM_THREADS=1`, torch threads 1. The kernel string is `Linux-6.18.44-fc-v445` here, against E26's `fc-v378`.

`pip freeze`:
```
filelock==4.2.10
fsspec==2026.9.0
Jinja2==3.1.6
MarkupSafe==3.0.4
mpmath==1.3.0
networkx==3.7
numpy==2.5.4
setuptools==84.2.1
sympy==1.14.0
torch==2.15.0+cpu
typing_extensions==4.16.0
```

`torch.__config__.show()`:
```
PyTorch built with:
  - GCC 13.3
  - C++ Version: 202002
  - Intel(R) oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v3.13.3 (Git Hash b38f97eb6f2173513112f15a0481750b8a66e901)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX512
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=d0f2af8b66b20da4baa50644c0cd3f887a2eae2e, CUDA_FLAGS= -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS -Xfatbin -compress-all  -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUB_WRAPPED_NAMESPACE=at_cuda_detail -DDISABLE_CUSPARSE_DEPRECATED -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -DC10_NODEPRECATED, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TORCH_VERSION=2.15.0, USE_CUDA=0, USE_CUDNN=OFF, USE_CUSPARSELT=OFF, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_HIPSPARSELT=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF, 

```

## Code

- `exp28_everybyte.py` (new). It imports `exp26_candidate`, `arch26` and the harness unedited and has its own loop. **It does not call or patch `exp26_candidate.dev` or `exp26_candidate.step`.** Receipts carry `experiment="E28"` and `targets`, and files are named `e28_dev_<arm>[_dryN]_<seed>`. It saves the weights at every scored row (the final row's weights are the birth file) with each hash in the receipt. At every row where E26 has one, it asserts that the data digest equals E26's (both arms' receipts). It keeps a cumulative finite-loss flag with the number of updates covered, and the loss by role for each of the 48 blocks of 512 updates.
- `e28_checks.py` (new): items 1 to 6. `exp26_candidate.step` is called only here, as item 3's reference. The independent gradient uses `e25_alignment.forward_rnn`, the coordinator's written-out GRU and not arch26's.
- **Source note.** `exp16_integrated.py` hashes `c9fcedcf…` now, against `f4559971…` in E26's receipts: E21b's resume work was merged after E26's commit 02eb46e0. An AST comparison with 02eb46e0 shows that only `ckpt_check_config`, `ckpt_load`, `develop17`, `life`, `resume_gate` and `run_config` changed, plus four functions added. None of these is on E26's birth path (`adamw16`, `book_shift`, `zero_grad_report`, `sha_file`, `panels`, `score` are identical). All other E26 sources are byte-equal to 02eb46e0. Item 1 is the direct test.
- **Roles (worker's decision, stated).** An item is `code a1 a2 op d_1..d_k b1 b2 '=' | c1 c2 EOS` (`exp11_cell.render`), with `k = len(prompt) - 7`. Each target is identified by its position in that layout, and the byte class is asserted on every item of every batch. The roles: distractor = d_i; operand = a1 a2 b1 b2; structural = code, op, '='; answer = c1 c2 EOS, which equals the answer mask (asserted). The brief lists "end" among the structural bytes, but EOS is inside the answer mask that E26 trained on and the balanced loss uses, so EOS is counted under answer. The task code, which the brief does not name, is counted under structural. Per-block sums and counts are also kept for the finer roles (code, operator, equals, answer digit, EOS), so any other grouping can be rebuilt from the receipts without rerunning.

## Items

| # | Item | Result |
|---|---|---|
| 1 | Host check, answer targets, seed 73129, both arms, fresh folder `exp45/e28/host_check` | **PASS.** S-ans `tensors_sha256` d9064126… = E26; B-ans efbdf87a… = E26. All 13 tensors of each `e26_dev_{S,B}_73129.pt` are bit-equal (max diff 0.0). All six rows: sealed scores (4 tasks × 5 k × n/exact/final) equal, `data_digest` equal, row loss equal; final digest and silenced k=12 equal. The weights-file SHA differs (S 719dd7ed…, B 86631691…) only because `torch.save` embeds the file name. **This host is of E26's class.** |
| 2 | `admit_e26` refuses E28 births of every targets value | **PASS.** 12 cases (answer, all and balanced; S and B; as written and renamed to E26's file name). Each is refused with `AssertionError: <file>: not an E26 birth receipt`. admit_e26 refuses at the experiment field; it does not read `targets`. |
| 3 | The mask on the first birth batch; one answer-target step | **PASS.** The `all` mask covers exactly the 974 real bytes of prompt and answer in 64 items, with y equal to those bytes and no padding position (padding is BOS). The four roles partition the 974: distractor 334, operand 256, structural 192, answer 192 = answer mask. One step under answer targets equals `exp26_candidate.step`'s bit for bit under both paths: weights, loss and AdamW state. |
| 4 | Balanced loss against the independent weighted form (0.5/n_a, 0.5/n_o, summed) | **PASS** at relative 1e-5. Six cases: untrained, E26-born S and E26-born B, each under both paths. Loss relative difference 8.6e-8 to 1.2e-7. Every gradient tensor came out **bit-equal** here (relative difference 0), which is better than the auditor's 3e-7 to 5e-7. The brief's tolerance, not bit equality, is the criterion, and the worker draws nothing from the difference. |
| 5 | One real step per new arm against AdamW given an independent gradient | **PASS.** S-all and B-all bit for bit (e25 `grads` with the every-byte mask). S-bal and B-bal pass the tolerance and were in fact bit-equal too. Start: new_net(73129) with a fresh optimizer on the first birth batch; largest step taken 1.09e-3. |
| 6 | Dry births of the four new arms at 2,048 updates | **PASS** on the second run. All four are complete with every record present: finite flag 2048/2048, 4 role blocks, row digest, row and final weight hashes, host, torch, command, environment, git, source hashes, optimizer groups read back (equal to expected and to E26's). The **first run of the check failed through a bug in the check**, not in the births: it built the expected optimizer groups from `E.new_net` before `arch26.install()`. That record is kept as `check_dry.FAILED_first_run_check_bug.json`, and the fix is a separate commit (dcbcbb33). At 2,048 updates the digest is not asserted because E26 has no row there; the receipts say so. |

**CPU per update** (dry births, 3 processes at once): S-all 0.134, B-all 0.127, S-bal 0.134, B-bal 0.130 s. Host checks (2 at once): S-ans 3,119 CPU-s in total (3,070 updates, 49 scoring), B-ans 2,932 CPU-s. E26's were 3,278 and 3,069.

**Forecast for stage 2** (twelve births, at most three processes): about 24,576 × 0.134 + ~50 scoring ≈ 3,350 CPU-s per birth. Twelve births ≈ 40,000 CPU-s ≈ 11.2 CPU-hours, or four rounds of three ≈ 3.8 to 4 hours of wall time.

Dry-birth hard means at 2,048 updates (descriptive, not readings): S-all 2.2, B-all 10.7, S-bal 15.3, B-bal 9.7.

## Files

- Committed: both host checks' final weights (with `-f`), their receipts and logs; the dry receipts and logs; `check_*.json`; `E28_PREDICTIONS.md` (four predictions copied unchanged, verified against 50bd275a).
- On the container only, with hashes in the receipts: the host checks' ten row files and the four dry births' weights.
- Receipts of the host checks name commit fcedbcef (runner unchanged since).

## Not run

The twelve stage 2 births. Nothing has been read against the predictions.
