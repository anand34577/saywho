# Security policy

saywho handles recordings of people's voices, which can be sensitive. Security reports are taken seriously and
handled before feature work.

## Reporting a vulnerability

**Please don't open a public issue.** Report it privately through GitHub: *Security > Report a vulnerability* on
<https://github.com/anand34577/saywho>.

Include what you found, how to reproduce it, the version (shown at `/api/v1/info`) and the impact you expect. You'll
get a reply within 7 days, and a fix or a plan within 30 days for confirmed issues. You'll be credited in the release
notes unless you'd rather not be.

## Supported versions

Only the latest release gets security fixes.

## Scope

In scope: the server and web app in this repository, the install scripts, and the Docker images published at
`ghcr.io/anand34577/saywho`.

Out of scope: the model itself, Rockchip's NPU runtime, third-party packages (report those upstream), and denial of
service through uploads within the configured `MAX_UPLOAD_MB` / `MAX_AUDIO_MINUTES` limits.

## Running saywho safely

- By default saywho listens only on the machine itself (`HOST=127.0.0.1`). If you open it to a network, set
  `API_KEY` so not everyone on that network can use it.
- Anyone who can reach the server can upload files to it. Put it behind HTTPS (see
  [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)) before exposing it to the internet, and set `MAX_UPLOAD_MB` and
  `MAX_AUDIO_MINUTES` to sensible limits.
- Results of the job queue stay in memory for `JOB_TTL_MINUTES` (default 60). Uploads are deleted as soon as they
  are processed.
