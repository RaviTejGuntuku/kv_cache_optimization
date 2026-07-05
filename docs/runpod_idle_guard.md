# RunPod Idle Guard

This is a Mac-side cron job for avoiding accidental RunPod spend.

It runs from your laptop, checks a RunPod pod over SSH, and uses the RunPod REST API to stop the pod after a quiet period. It does not terminate the pod automatically; after a longer quiet period it emails you to terminate it manually.

## Design

`scripts/runpod_idle_guard.sh` treats a pod as active if either condition is true:

- A remote process matches `RUNPOD_ACTIVE_PROCESS_REGEX`.
- A watched remote file changed recently under `RUNPOD_REMOTE_WATCH_PATHS`.

If both are false, it starts a local inactivity timer in `$HOME/.runpod_idle_guard`.

Actions:

- After `RUNPOD_IDLE_STOP_SECONDS`, default `3600`, it runs `rsync` from the pod to your Mac and then calls `POST https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop`.
- After `RUNPOD_IDLE_EMAIL_SECONDS`, default `7200`, it sends one email reminding you to terminate the pod.

RunPod stop behavior follows the RunPod REST API documentation: stopping releases the GPU but persistent volume/network storage can still accrue charges. Termination is intentionally manual because it can delete pod-attached data.

## Local Env File

Create a local env file outside git, for example:

```bash
mkdir -p ~/.runpod_idle_guard
chmod 700 ~/.runpod_idle_guard
nano ~/.runpod_idle_guard/kv_cache.env
chmod 600 ~/.runpod_idle_guard/kv_cache.env
```

Example contents:

```bash
export RUNPOD_API_KEY="..."
export RUNPOD_POD_ID="your-pod-id"
export RUNPOD_SSH_HOST="64.247.201.41"
export RUNPOD_SSH_PORT="18945"
export RUNPOD_SSH_KEY="$HOME/.ssh/id_ed25519"
export RUNPOD_REMOTE_ROOT="/root/kv_cache_research"
export RUNPOD_LOCAL_SYNC_ROOT="$HOME/TEJ/CS_Independent_Research/kv_cache_research/studies/results/runpod_guard_sync"
export RUNPOD_NOTIFY_EMAIL="your_email@example.com"

  export RUNPOD_REMOTE_WATCH_PATHS="."
export RUNPOD_IDLE_STOP_SECONDS="3600"
export RUNPOD_IDLE_EMAIL_SECONDS="7200"
```

If `/usr/bin/mail` is not configured on your Mac, set a custom email command:

```bash
export RUNPOD_EMAIL_COMMAND="$HOME/bin/send-runpod-email"
```

The command receives the email subject as `$1` and the body on stdin.

## Test Manually

```bash
cd /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research
scripts/runpod_idle_guard.sh --env ~/.runpod_idle_guard/kv_cache.env --dry-run
```

Then run once for real:

```bash
scripts/runpod_idle_guard.sh --env ~/.runpod_idle_guard/kv_cache.env
```

Logs and state are written to:

```text
~/.runpod_idle_guard/
```

## Cron

Edit your crontab:

```bash
crontab -e
```

Run every 5 minutes:

```cron
*/5 * * * * cd /Users/tejguntuku/TEJ/CS_Independent_Research/kv_cache_research && /bin/bash scripts/runpod_idle_guard.sh --env /Users/tejguntuku/.runpod_idle_guard/kv_cache.env >> /Users/tejguntuku/.runpod_idle_guard/cron.log 2>&1
```

## Practical Notes

- Keep `RUNPOD_API_KEY` out of git.
- Use a pod-specific env file because SSH host/port changes when the pod changes.
- If you want the guard to watch a different project root, change `RUNPOD_REMOTE_ROOT`.
- If you run long jobs that do not write files often, make sure their process name matches `RUNPOD_ACTIVE_PROCESS_REGEX`.
- Do not watch `/tmp` by default. System temp files can change even when your experiment is idle.
- The guard stops pods, but it does not delete network volumes or terminate pods.
