# Validation record

Date: 2026-10-22. Integration host: Windows; bundled Python 3.12. No PyTorch installation was available in this host runtime. The training environment targets Linux/Python 3.10 with NVIDIA GPUs.

## Executed

- Ten CPU integration tests passed using `python -m unittest discover -s tests -v`.
- All included Python sources parsed successfully as part of that suite.
- Both extraction help entry points ran without importing the GPU stack.
- Synthetic ImageNet metadata matched the DINOv2 entry schema, class ordering and metadata filenames.
- Canonical class-relative token lookup and DINOv2 split-relative lookup agreed; missing tokens and mismatched class mappings were rejected.
- Extraction settings, resume-manifest compatibility and untracked-token protection were exercised.
- The two-stage optimizer's boundary and frozen-backbone learning rate were checked without invoking CUDA.
- Checkpoint export converted chunked block keys to the expected evaluation layout.
- The PyTorch 2.1 FSDP handle adapter was checked with stand-in handles; this is an interface test, not an FSDP execution test.
- `bash -n extract_cls_token.sh distill.sh eval_linear.sh` passed using Git Bash.
- A wheel built successfully with the bundled setuptools runtime. Package contents were inspected for the generator, training modules, YAML presets and licenses.
- SHA-256 comparison confirmed all 217 recorded original input files were unchanged.
- Manuscript method pages were text-extracted, rendered and visually checked for the generation/distillation interface. The PDF was not copied into the release tree.

Windows sandbox restrictions initially prevented temporary-test-directory access and Bash initialization. Those checks were rerun with execution permission and passed. This was an execution-environment issue rather than a GPU test.

## Not executed here

- Installing or resolving the complete Linux GPU environment.
- Loading the actual REG, VAE and DINOv2 weights into PyTorch.
- Tensor-content token validation against a real generated dataset.
- Full or partial REG generation with actual tensors on one or more GPUs.
- FSDP initialization, training, gradient scaling, EMA, checkpoint resume and final checkpoint loading on GPUs.
- Full ImageNet distillation, linear evaluation or reproduction of manuscript results.

The README commands are prepared for these steps. CPU checks and successful packaging do not establish numerical equivalence or training accuracy. The manuscript-oriented preset exposes documented choices where the paper and the supplied code differ.
