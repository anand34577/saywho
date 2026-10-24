# Install with Docker

Docker runs saywho in a ready-made container: nothing to install besides Docker itself, and the speaker model is
already inside, so it starts without internet access. This is the best choice for servers, NAS boxes (Synology,
Unraid, TrueNAS), Proxmox and anyone who already uses Docker.

| Image | For |
|---|---|
| `ghcr.io/anand34577/saywho:cpu` | any machine, Intel/AMD or ARM |
| `ghcr.io/anand34577/saywho:cuda` | machines with an NVIDIA graphics card (a much bigger download) |
| `ghcr.io/anand34577/saywho:rknn` | Rockchip boards, see [Install on a Rockchip board](Install-on-a-Rockchip-board) |

## Quick start: one command

```bash
docker run -d --name saywho -p 7860:7860 --restart unless-stopped ghcr.io/anand34577/saywho:cpu
```

Open `http://<your-server>:7860` (on the same machine: <http://localhost:7860>). The first start takes up to a
minute while the model loads; `docker ps` shows `(healthy)` once it's ready.

With an NVIDIA card, use the `cuda` image and add `--gpus all`:

```bash
docker run -d --name saywho -p 7860:7860 --restart unless-stopped --gpus all ghcr.io/anand34577/saywho:cuda
```

This needs a recent NVIDIA driver (575 or newer) and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on Linux. On Windows, Docker Desktop with WSL 2 only needs the normal NVIDIA driver. Check it worked: the page header
shows your card's name, not "CPU".

## With Docker Compose (easier to manage)

```bash
mkdir saywho && cd saywho
curl -fsSLO https://raw.githubusercontent.com/anand34577/saywho/main/docker-compose.yml
curl -fsSL  https://raw.githubusercontent.com/anand34577/saywho/main/.env.example -o .env
docker compose up -d
```

That runs the `cpu` image. To use another one, open `.env` in a text editor and change `COMPOSE_PROFILES=cpu` to
`cuda` or `rknn`, then run `docker compose up -d` again. Every other setting in `.env` is explained in the file and
on the [Settings](Settings) page.

Useful commands, run in the same folder:

```bash
docker compose logs -f      # watch what saywho is doing (Ctrl + C to stop watching)
docker compose ps           # is it running and healthy?
docker compose down         # stop it
docker compose pull && docker compose up -d   # update to the newest version
```

## Portainer, Synology, Unraid and other web panels

Create a container (or a "stack" from the compose file above) with:

- **Image:** `ghcr.io/anand34577/saywho:cpu`
- **Port:** host `7860` to container `7860`
- **Restart policy:** unless stopped
- **Environment** (optional): any setting from [Settings](Settings), for example `API_KEY`

No volumes are needed: saywho keeps no data between restarts.

## Good to know

- Pin a version with a tag such as `cpu-2.3.0` instead of `cpu` if you don't want updates to arrive by surprise.
- Anyone who can reach port 7860 can use saywho. On a shared network, set an `API_KEY`
  ([Settings](Settings)).
- The full reference (every variable, GPU details, building the images yourself) is
  [docs/DOCKER.md](https://github.com/anand34577/saywho/blob/main/docs/DOCKER.md).
