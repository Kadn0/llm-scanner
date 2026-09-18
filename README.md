# LLM Scanner

A local app for downloading AI models, chatting with them, generating images, and testing chat models for
security weaknesses with [garak](https://github.com/NVIDIA/garak). It runs entirely on your own machine at
**http://127.0.0.1:7861** and installs into your home folder without sudo.

## Features

- **Models**: search Hugging Face or the Ollama library, with automatic version choice for your GPU. Downloads
  resume after interruptions and can be prioritized.
- **Chat**: saved conversations with search, attachments (images for vision models, text/PDF/Word for any
  model), and time and token stats for every reply.
- **Images**: text-to-image with Z-Image, FLUX.1, and Stable Diffusion 1.5 / XL through stable-diffusion.cpp,
  with a gallery. Image models can be found through Hugging Face search too.
- **Scanning**: garak probes, saved probe groups, a live progress bar, and a security-analyst report with OWASP
  LLM Top 10 mapping, evidence, remediation, and the raw garak data.
- **Housekeeping**: models load only while in use, a live GPU/RAM status bar, start at boot, and one-click
  verified updates for LLM Scanner, Ollama, and garak.

## Install

On a Linux machine (x86_64), run:

```bash
curl -fsSL https://raw.githubusercontent.com/Kadn0/llm-scanner/main/install.sh | bash
```

Or clone the repository and run the installer:

```bash
git clone https://github.com/Kadn0/llm-scanner.git
cd llm-scanner
./install.sh
```

The installer sets up:

| Part | Purpose | Location |
|---|---|---|
| LLM Scanner | The app | `~/llm-scanner` |
| Ollama | Runs chat models | `~/.local/ollama` (models in `~/.ollama/models`) |
| stable-diffusion.cpp | Runs image models | `~/.local/sd-cpp` |
| garak, Gradio, PyTorch | Scanner and app packages (exact pinned versions) | `~/llm-scanner/.venv` |
| uv | Python and package installer | `~/.local/bin/uv` |

It also adds LLM Scanner to your app menu, starts it at boot, and opens it in your browser when you log in. The
first install downloads about 6 GB of Python packages (mostly PyTorch, which garak needs), so it takes a while.

Options:

| Option | Effect |
|---|---|
| `--data FILE` | Restore data exported from another machine (see below) |
| `--pull-models` | With `--data`, also re-download that machine's Ollama chat models |
| `--no-autostart` | Don't open the app in your browser at login |
| `--no-start` | Install, but don't start it yet |

## Move to a new machine

Models aren't copied, since they're large and easy to download again. Your data is copied separately:

1. On the old machine, export your chats, images, probe groups, scan reports, and model list:

   ```bash
   ~/llm-scanner/export-data.sh
   ```

   This creates `~/llm-scanner-data-DATE.tar.gz`.

2. Copy that file to the new machine, then install with it:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/Kadn0/llm-scanner/main/install.sh -o install.sh
   bash install.sh --data llm-scanner-data-DATE.tar.gz --pull-models
   ```

3. Image models you used appear in the Images tab with a **Download** button.

## Updates

Every installed copy checks GitHub for a new release at startup and every 6 hours. When one is available, the
update notice at the top of the app shows what's new and an **Update LLM Scanner** button. Clicking it:

1. Downloads the release and installs any new Python packages.
2. Starts the new version on a spare port to confirm it works, without touching your models or data.
3. Only if that check passes, backs up the current version to `data/backups/`, switches over, and restarts.

If the check fails, nothing changes. Ollama and garak updates work the same way, with automatic rollback.

### Publishing a new version (maintainer)

1. Make your changes and test them locally.
2. Bump the number in `VERSION` (for example `1.0.0` to `1.1.0`).
3. If you changed Python packages, refresh the pins:
   `~/.local/bin/uv pip freeze --python .venv/bin/python > requirements.txt`
4. Commit, push, and create a release whose tag matches the version:

   ```bash
   git commit -am "Describe the change" && git push
   gh release create v1.1.0 --title "v1.1.0" --notes "What changed"
   ```

Every machine sees the update within 6 hours, or immediately after a restart.

## Requirements

- Linux on 64-bit Intel/AMD (x86_64). Built and tested on Pop!_OS 24.04 (Ubuntu based).
- `curl` and `tar`. `zstd` is optional and makes the Ollama download smaller.
- A systemd desktop session (standard on Ubuntu, Pop!_OS, Fedora, Debian, and most others).
- About 10 GB of disk for the install, plus room for models.
- GPU, optional but recommended:
  - **NVIDIA** with the proprietary driver: chat and image models use the GPU.
  - **Other GPUs** with Vulkan drivers: image models use the GPU; chat models may use the CPU.
  - **No GPU**: everything works on the CPU, much more slowly.

The app detects your video memory and recommends model versions that fit it.

## Troubleshooting

| Problem | Try |
|---|---|
| The page won't load | `systemctl --user status llm-scanner` and `journalctl --user -u llm-scanner -n 50` |
| Chat models don't respond | `systemctl --user status ollama` and `~/.local/bin/ollama list` |
| Image generation fails | Check `~/.local/sd-cpp/sd-cli` exists; re-run `install.sh` to reinstall it |
| It doesn't start at boot | Your system didn't allow start-at-boot without sudo. Run `sudo loginctl enable-linger $USER` once |
| The page looks wrong after an update | Reload the page |
| Anything else | `systemctl --user restart ollama llm-scanner` |

## Uninstall

```bash
~/llm-scanner/uninstall.sh                 # removes everything, including your data
~/llm-scanner/uninstall.sh --keep-data     # keeps chats, images, probe groups and reports
~/llm-scanner/uninstall.sh --remove-models # also deletes downloaded chat models
```

It lists what will be deleted and asks before doing anything.

## Project files

| File | Contents |
|---|---|
| `app.py` | The app (Gradio) |
| `analyst_report.py` | Builds the security-analyst summary of a garak run |
| `garak_runner.py` | Runs garak with the app's Ollama settings (thinking control, keep-alive) |
| `static/` | Styling and browser scripts |
| `install.sh`, `uninstall.sh`, `export-data.sh` | Setup, removal, and data export |
| `VERSION` | The version number the update check compares against |
| `requirements.txt` | Exact Python package versions |
