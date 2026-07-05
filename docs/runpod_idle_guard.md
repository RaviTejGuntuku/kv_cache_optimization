# RunPod Lifecycle Guard

This is a Mac-side cron job and lifecycle helper for avoiding accidental RunPod spend.

It uses the RunPod REST API as the source of truth. You do not hard-code SSH host or SSH port because those change every time a new pod is created. The script resolves `publicIp` and `portMappings["22"]` from the API before every SSH or rsync operation.

## What It Does

Default guard mode:

- Finds the active pod by `RUNPOD_POD_ID`, a locally saved pod ID, or `RUNPOD_POD_NAME`.
- Resolves SSH endpoint from RunPod API.
- SSH-probes the pod for active experiment/GPU processes and recent file changes.
- After `RUNPOD_IDLE_STOP_SECONDS`, default `3600`, syncs remote data down to your Mac and stops the pod.
- After `RUNPOD_IDLE_EMAIL_SECONDS`, default `7200`, sends one email reminding you to terminate the pod manually.

Create mode:

- Builds a RunPod creation payload for one A100/H100-class GPU and `300 GB` persistent pod volume.
- Prints the payload and requires you to type `CREATE`.
- Creates the pod through RunPod API.
- Polls RunPod API until `publicIp` and SSH port mapping exist.
- Waits for SSH.
- Runs remote bootstrap:

```bash
apt-get update
apt-get upgrade -y
apt-get install -y rsync sudo
```

- Syncs relevant local project directories to the GPU.

The script stops pods but does not terminate pods. Termination is intentionally manual because it can delete pod-attached data.

## Exact Placeholders To Fill

Create a local env file outside git:

```bash
mkdir -p ~/.runpod_idle_guard
chmod 700 ~/.runpod_idle_guard
nano ~/.runpod_idle_guard/kv_cache.env
chmod 600 ~/.runpod_idle_guard/kv_cache.env
```

Template:

```bash
export RUNPOD_API_KEY="<YOUR_RUNPOD_API_KEY>"
export RUNPOD_POD_NAME="kv-cache-headroom"

export RUNPOD_SSH_KEY="$HOME/.ssh/id_ed25519"
export RUNPOD_LOCAL_REPO_ROOT="$HOME/TEJ/CS_Independent_Research/kv_cache_research"
export RUNPOD_REMOTE_ROOT="/root/kv_cache_research"
export RUNPOD_LOCAL_SYNC_ROOT="$HOME/TEJ/CS_Independent_Research/kv_cache_research/studies/results/runpod_guard_sync"

export RUNPOD_NOTIFY_EMAIL="<YOUR_EMAIL_ADDRESS>"
export RUNPOD_EMAIL_COMMAND="$HOME/.runpod_idle_guard/send_email.sh"

export RUNPOD_CREATE_GPU_TYPES="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB,NVIDIA H100 PCIe,NVIDIA H100 80GB HBM3"
export RUNPOD_CREATE_IMAGE="<RUNPOD_PYTORCH_IMAGE>"
export RUNPOD_CREATE_VOLUME_GB="300"
export RUNPOD_CREATE_CONTAINER_DISK_GB="80"
export RUNPOD_CREATE_CLOUD_TYPE="SECURE"
export RUNPOD_CREATE_INTERRUPTIBLE="false"

export RUNPOD_REMOTE_WATCH_PATHS="."
export RUNPOD_IDLE_STOP_SECONDS="3600"
export RUNPOD_IDLE_EMAIL_SECONDS="7200"
```

You must fill:

```text
<YOUR_RUNPOD_API_KEY>
<YOUR_EMAIL_ADDRESS>
<RUNPOD_PYTORCH_IMAGE>
```

Use a RunPod PyTorch image you trust. The script default is:

```text
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

If RunPod rejects that image tag, replace `<RUNPOD_PYTORCH_IMAGE>` with the exact image tag from your RunPod template.

You do not fill these anymore:

```text
RUNPOD_SSH_HOST
RUNPOD_SSH_PORT
```

The script queries them from the API.

Optional placeholders:

```bash
export RUNPOD_NETWORK_VOLUME_ID="<EXISTING_NETWORK_VOLUME_ID>"
```

Only set this if you want to attach an existing network volume instead of using the 300GB pod volume.

## Email Command

The guard calls `RUNPOD_EMAIL_COMMAND "$subject"` and passes the email body on stdin.

Example using `msmtp`:

```bash
nano ~/.runpod_idle_guard/send_email.sh
chmod +x ~/.runpod_idle_guard/send_email.sh
```

```bash
#!/usr/bin/env bash
set -euo pipefail

subject="$1"
to="<YOUR_EMAIL_ADDRESS>"
from="<YOUR_GMAIL_ADDRESS>"

{
  echo "From: $from"
  echo "To: $to"
  echo "Subject: $subject"
  echo
  cat
} | msmtp "$to"
```

Install and configure:

```bash
brew install msmtp
nano ~/.msmtprc
chmod 600 ~/.msmtprc
```

`~/.msmtprc`:

```text
defaults
auth           on
tls            on
tls_trust_file /opt/homebrew/etc/openssl@3/cert.pem
logfile        ~/.msmtp.log

account        gmail
host           smtp.gmail.com
port           587
from           <YOUR_GMAIL_ADDRESS>
user           <YOUR_GMAIL_ADDRESS>
password       <YOUR_GMAIL_APP_PASSWORD>

account default : gmail
```

Email placeholders:

```text
<YOUR_EMAIL_ADDRESS>
<YOUR_GMAIL_ADDRESS>
<YOUR_GMAIL_APP_PASSWORD>
```

For Gmail, use an App Password, not your normal password.

## Commands

Resolve the active pod SSH endpoint from the API:

```bash
cd /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research
scripts/runpod_idle_guard.sh --env ~/.runpod_idle_guard/kv_cache.env --resolve-ssh
```

Create a new pod, bootstrap apt, and sync local code up:

```bash
scripts/runpod_idle_guard.sh --env ~/.runpod_idle_guard/kv_cache.env --create
```

This asks for confirmation. You must type:

```text
CREATE
```

Sync local code up to an existing active pod:

```bash
scripts/runpod_idle_guard.sh --env ~/.runpod_idle_guard/kv_cache.env --sync-up
```

Run remote apt bootstrap on an existing active pod:

```bash
scripts/runpod_idle_guard.sh --env ~/.runpod_idle_guard/kv_cache.env --bootstrap
```

Dry-run the idle guard:

```bash
scripts/runpod_idle_guard.sh --env ~/.runpod_idle_guard/kv_cache.env --dry-run
```

Run idle guard for real:

```bash
scripts/runpod_idle_guard.sh --env ~/.runpod_idle_guard/kv_cache.env
```

Logs and state are written to:

```text
~/.runpod_idle_guard/
```

## Cron

Edit crontab:

```bash
crontab -e
```

Run every 5 minutes:

```cron
*/5 * * * * cd /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research && /bin/bash scripts/runpod_idle_guard.sh --env /Users/tejguntuku/.runpod_idle_guard/kv_cache.env >> /Users/tejguntuku/.runpod_idle_guard/cron.log 2>&1
```

Do not run `--create` from cron. Creation is intentionally interactive.

## Sync Scope

Default local-to-remote paths:

```text
AGENTS.md
CLAUDE.md
README.md
benchmarking
docs
scripts
studies/specs
datasets/synthetic
datasets/processed/empirical_headroom
```

Override with:

```bash
export RUNPOD_SYNC_UP_PATHS="AGENTS.md CLAUDE.md README.md benchmarking docs scripts studies/specs datasets/processed/empirical_headroom"
```

## Practical Notes

- Keep `RUNPOD_API_KEY` out of git.
- The script stores the latest created pod ID in `~/.runpod_idle_guard/<pod-name>.state`.
- SSH host and port are resolved from `GET /pods` or `GET /pods/{podId}` each run.
- If you terminate a pod manually, `--create` will create a new one and update local state.
- The guard stops pods, but it does not delete network volumes or terminate pods.
