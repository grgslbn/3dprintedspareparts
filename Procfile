# ─────────────────────────────────────────────────────────────────────────────
# Procfile — Railway / Heroku process definition
# ─────────────────────────────────────────────────────────────────────────────
# This file tells Railway how to run the pipeline.
#
# We declare it as a "worker" (not a "web") process because it doesn't
# listen on a port — it just runs, does its job, and exits.
#
# The actual scheduling (daily at 8 AM UTC) is configured in railway.toml.
# ─────────────────────────────────────────────────────────────────────────────

worker: python pipeline.py
