# ComfyUI Gaudi2 adaptation

Read `comfyui-hpu/README.md`, `comfyui-hpu/PROGRESS.md`, and `comfyui-hpu/ENVIRONMENT.md` before continuing this task.

## Default optimization delivery workflow

Unless the user explicitly specifies otherwise, apply this workflow to optimization tasks across this repository and the sibling ComfyUI-Gaudi repository.

1. Complete the implementation and relevant automated validation.
2. Stage this task's changes in both repositories with `git add`, including code, tests, scripts, documentation, and investigation records as applicable. Keep the changes staged pending human acceptance.
3. Deploy the validated staged version to the configured environment, verify the service and normal workflow, and provide the user with the deployment details and review results. Deployment is part of the default optimization task.
4. Wait for the user's explicit confirmation that human acceptance has passed. If revisions are requested, update, validate, stage, and redeploy the changes for another review.
5. Commit the accepted task changes in both repositories only after that confirmation. Creating or amending commits before human acceptance requires an explicit exception from the user.

Identify a pending deployment using its baseline commit, staged diff or snapshot, and deployed file hashes. Record its status as deployed and awaiting human acceptance. The detailed workflow is in `comfyui-hpu/optimization/MEASUREMENT.md`.

## Repository and server boundaries

All adaptation plans, investigation diaries, failed/successful paths, server details, and progress belong in this repository. Do not put them in ComfyUI-Gaudi or its commit history. The sibling ComfyUI-Gaudi repository should contain only product code, portable scripts and tests.

Never modify the server's original `/home/sonny/ComfyUI` directory. Use the independent `/home/sonny/ComfyUI-Gaudi` test checkout. Never record tokens or credentials.

For model downloads, first use the server's existing `HF_ENDPOINT` and `HF_TOKEN`. Do not override the endpoint by default. Use the existing shell proxy or an alternate source only when needed and record the observed failure that justified it. Reuse existing model components, especially VAEs, instead of downloading duplicates.
