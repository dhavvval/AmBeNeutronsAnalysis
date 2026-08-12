#!/usr/bin/env bash
# ============================================================
# tmux launcher for the Lucho 100k AmBe pipeline.
#
# Starts run_lucho_100k.sh inside a DETACHED tmux session so it
# survives SSH disconnects / closing your laptop. Everything lives
# in the app area (/exp/annie/app) -- nothing touches /nashome.
#
# Usage:
#   bash tmux_run_lucho_100k.sh           # start the run (detached)
#   bash tmux_run_lucho_100k.sh attach    # attach to watch live  (Ctrl-b d to detach)
#   bash tmux_run_lucho_100k.sh status    # quick status (tail log + done marker)
#   bash tmux_run_lucho_100k.sh stop      # kill the run + session
#
# After it finishes, check:   cat logs/lucho_100k.DONE
# ============================================================
set -euo pipefail

REPO="/exp/annie/app/users/dajana/AmBeNeutronsAnalysis"
SESSION="lucho100k"
LOG="$REPO/logs/lucho_100k.log"
DONE="$REPO/logs/lucho_100k.DONE"
RUNNER="$REPO/run_lucho_100k.sh"

cd "$REPO"
mkdir -p logs

cmd="${1:-start}"

case "$cmd" in
  attach)
    exec tmux attach -t "$SESSION"
    ;;

  status)
    echo "=== tmux sessions ==="
    tmux ls 2>/dev/null || echo "(no tmux server running)"
    echo ""
    if [[ -f "$DONE" ]]; then
      echo "=== DONE marker ($DONE) ==="
      cat "$DONE"
    else
      echo "=== not finished yet (no DONE marker) ==="
    fi
    echo ""
    echo "=== last 30 log lines ==="
    [[ -f "$LOG" ]] && tail -n 30 "$LOG" || echo "(no log yet)"
    exit 0
    ;;

  stop)
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "killed session '$SESSION'" \
      || echo "no session '$SESSION' to kill"
    exit 0
    ;;

  start)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "ERROR: tmux session '$SESSION' already exists."
      echo "  Attach:  bash $0 attach"
      echo "  Status:  bash $0 status"
      echo "  Stop:    bash $0 stop"
      exit 1
    fi

    # The command run INSIDE tmux: run the pipeline, then write a DONE marker
    # with the exit code + timestamp so you can confirm success after the fact.
    inner="cd '$REPO'; \
rm -f '$DONE'; \
echo \"[tmux_run] started \$(date) on \$(hostname), \$(nproc) cores\" | tee -a '$LOG'; \
bash '$RUNNER' >> '$LOG' 2>&1; \
rc=\$?; \
{ echo \"exit_code=\$rc\"; echo \"finished=\$(date)\"; echo \"host=\$(hostname)\"; \
  if [ \$rc -eq 0 ]; then echo 'status=SUCCESS'; else echo 'status=FAILED'; fi; } > '$DONE'; \
echo \"[tmux_run] finished rc=\$rc \$(date)\" | tee -a '$LOG'; \
echo; echo '=== run complete (rc='\$rc') -- this pane stays open; press Ctrl-b d to detach or type exit ==='; \
exec bash"

    tmux new-session -d -s "$SESSION" "$inner"

    echo "Started pipeline in detached tmux session '$SESSION'."
    echo "  Repo:   $REPO"
    echo "  Log:    $LOG"
    echo "  Done:   $DONE   (written when the run finishes)"
    echo ""
    echo "Safe to close your laptop. To check back later:"
    echo "  tmux attach -t $SESSION         # watch live (Ctrl-b d to leave it running)"
    echo "  bash $0 status                  # quick status without attaching"
    echo "  tail -f $LOG                    # follow the log"
    echo "  cat $DONE                       # success/fail after it ends"
    exit 0
    ;;

  *)
    echo "usage: bash $0 [start|attach|status|stop]"
    exit 2
    ;;
esac
