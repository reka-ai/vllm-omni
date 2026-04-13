# Deploying vllm-omni

Build-and-deploy runbook for the `rekaai/vllm-omni-fork` Docker image. This is similar to how we deploy our vLLM fork, but , but this image layers on the upstream `vllm/vllm-openai` base (no multi-hour CUDA compile) and bundles two siblings: `vllm-reka` (plugin) and `moonvalley_ai` (opensora source for Marey's VAE).

References
- [Deploying our vLLM fork](https://www.notion.so/moonvalley/Deploying-our-vLLM-fork-321cef6731968073b8a4f3b4c903cf58)

## Prerequisites

- Access to Reka's Docker Hub (`rekaai` org)
- `docker` with BuildKit enabled (on Linux set `DOCKER_BUILDKIT=1`)
- Sibling repo layout on the machine you're building from:

  ```
  <parent>/
      vllm-omni/        ← this repo
      vllm-reka/        ← https://github.com/reka-ai/vllm-reka
      moonvalley_ai/    ← opensora source (private)
  ```

  The build script fails fast if any sibling is missing.

- For production deploy: access to the infra repo (`reka-code/infra`) and
  `tk` CLI set up for the target cluster.

## 1. Sync the three repos

On whatever machine you're building from, make sure all three sibling repos
are at the commits you want baked into the image.

```bash
cd <parent>
( cd vllm-omni     && git pull && git log -1 --oneline )
( cd vllm-reka     && git pull && git log -1 --oneline )
( cd moonvalley_ai && git pull && git log -1 --oneline )
```

Write down each SHA — useful for rollback later.

## 2. Build the image

Use the `build-serving.sh` wrapper. It discovers the sibling layout from its
own location, symlinks the correct `.dockerignore` into place for the build,
runs `docker build`, and cleans up.

```bash
VLLM_OMNI_SHA=$(git -C <parent>/vllm-omni rev-parse HEAD)

<parent>/vllm-omni/docker/build-serving.sh \
    -t rekaai/vllm-omni:${VLLM_OMNI_SHA} \
    -t rekaai/vllm-omni:latest
```

### Notes

- **CPU build is fine.** Unlike the full vLLM fork build, this layers on top of
  `vllm/vllm-openai:v0.17.0` which already has vLLM + CUDA kernels compiled.
  No GPU needed at build time; expect a few minutes, not hours.
- **If building on macOS**, add `--platform linux/amd64` so the resulting
  image runs on Linux/amd64 GPU nodes. Builds via QEMU will be noticeably
  slower — prefer a Linux amd64 build host if you have one.
- **Git SHA as the tag** is the `vllm-omni` commit only. The image also
  depends on `vllm-reka` and `moonvalley_ai`, so you should **always pin by
  `@sha256:` when deploying** (step 5) — the content digest covers all three.

## 3. Smoke test on a GPU node

Don't push an image you haven't run. SSH into an H100 node (or any CUDA
node), pull the image, and run a quick `/health` check.

```bash
# On the GPU node
docker pull rekaai/vllm-omni:${VLLM_OMNI_SHA}

# Plain-vLLM mode (e.g. reka-edge-2603)
docker run --gpus all --rm -p 8000:8000 \
    -v /path/to/weights:/model \
    rekaai/vllm-omni:${VLLM_OMNI_SHA} \
    vllm serve /model --tokenizer-mode yasa

# In another shell
curl http://localhost:8000/health
# → 200 OK

# vllm-omni diffusion mode (marey)
docker run --gpus all --rm -p 8000:8000 \
    -v /path/to/marey-weights:/model \
    -v /path/to/stage_configs.yaml:/config/stage_configs.yaml \
    rekaai/vllm-omni:${VLLM_OMNI_SHA} \
    vllm serve /model --omni --stage-configs-path /config/stage_configs.yaml
```

If `/health` returns 200 in both modes, the image is good.

## 4. Push to Docker Hub

```bash
docker login                                           # Reka Docker Hub creds
docker push rekaai/vllm-omni:${VLLM_OMNI_SHA}
docker push rekaai/vllm-omni:latest
```

After the push, grab the image's content digest — this is what you'll pin
in tanka.

```bash
docker inspect --format='{{index .RepoDigests 0}}' rekaai/vllm-omni:${VLLM_OMNI_SHA}
# → rekaai/vllm-omni@sha256:abcdef012345...
```

## 5. Update tanka and deploy

Edit the relevant jsonnet environment (e.g.
`infra/tanka/environments/oracle-ashburn-h100/inference.jsonnet`) and pin
the image by `@sha256:` — not by tag — for reproducibility.

```jsonnet
local inference = import 'inference.libsonnet';

{
  world_model_marey: inference.vllm_omni_inference_server(
    name='world-model-marey',
    // Pin by content digest — covers vllm-omni + vllm-reka + moonvalley_ai.
    image='rekaai/vllm-omni@sha256:abcdef012345...',
    gpu_count=8,
    replicas=1,
    model='marey/distilled-0100',
    stage_configs_yaml=importstr 'stage_configs/marey.yaml',
    override_entrypoint=[
      'vllm', 'serve', '/model',
      '--omni',
      '--port', '8000',
      '--stage-configs-path', '/config/stage_configs.yaml',
    ],
    env=[
      { name: 'PYTORCH_CUDA_ALLOC_CONF', value: 'expandable_segments:True' },
    ],
    instance_type=null,
    instance_product_type='NVIDIA-H100-80GB',
    shm_size='64Gi',
  ),
}
```

Then deploy through the normal tanka flow (see the infra repo's deploy
guide for cluster-specific steps):

```bash
cd infra/tanka
tk apply environments/oracle-ashburn-h100
```

## Rollback

Rollbacks are a one-line jsonnet revert to a previous `@sha256:` digest,
followed by `tk apply`. Keep the previous digest in commit history so this
is a pure `git revert` + apply.

## FAQ

**Why not just use `:latest`?**
Tag mutation is the thing that ruins reproducibility. `@sha256:` pins the
exact image bytes that served traffic when the deployment was last applied.
Every prod pod should reference a digest.

**Why does the image bundle `moonvalley_ai`?**
Marey's VAE loader (`vllm_omni/diffusion/models/marey/pipeline_marey.py`)
imports `opensora.models.vae.vae_adapters`. `opensora` isn't pip-installable
cleanly (its `__init__` pulls in heavy optional deps), so the code stubs
parts of it at runtime and loads the VAE submodule from a sys.path entry.
The image sets `MOONVALLEY_AI_ROOT=/workspace/moonvalley_ai` so the lookup
succeeds.

**How do I build for a different GPU architecture (e.g. L40S)?**
No change — the base image covers modern NVIDIA archs (sm_70 through sm_90).
Just deploy to a node with that GPU type.

**Which file do I edit to change the build?**
- `docker/Dockerfile.serving` — the layers.
- `docker/build-context.dockerignore` — what gets sent to the daemon.
- `docker/build-serving.sh` — the wrapper that enforces sibling layout.
