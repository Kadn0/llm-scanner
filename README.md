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

## Set up on a fresh Pop!_OS install

LLM Scanner is made for **Pop!_OS 24.04** (COSMIC desktop). Starting from a freshly installed system:

### 1. Install Pop!_OS with the NVIDIA driver

- If the machine has an **NVIDIA** graphics card, install Pop!_OS from the **NVIDIA** download on
  [system76.com/pop](https://system76.com/pop). The driver comes preinstalled.
- If you already installed the standard download on an NVIDIA machine, add the driver and restart:

  ```bash
  sudo apt install -y system76-driver-nvidia
  sudo reboot
  ```

- Check it works with `nvidia-smi` in a terminal; it should list your graphics card. (Machines without NVIDIA
  graphics skip this step; image models use Vulkan and everything else runs on the CPU.)

### 2. Update the system

Open the **COSMIC Store** and install all updates, or run this, then restart:

```bash
sudo apt update && sudo apt full-upgrade -y
```

### 3. Install LLM Scanner

Open **Terminal** and run:

```bash
curl -fsSL https://raw.githubusercontent.com/Kadn0/llm-scanner/main/install.sh | bash
```

No sudo is needed; everything goes into your home folder. It downloads about 8 GB (mostly PyTorch, which garak
needs, plus Ollama), so expect 10 to 20 minutes. When it finishes, LLM Scanner opens in Firefox.

### 4. Add models

- **Chat model**: on the **Models** tab, search (for example `gemma` on Hugging Face or `qwen3` on Ollama),
  choose a repository, and click **Download**. The recommended version is picked to fit your graphics memory.
- **Image model**: on the **Images** tab, click **Download** next to Z-Image Turbo, or search for another.

### 5. That's it

- LLM Scanner starts automatically at boot and opens in Firefox when you log in.
- It's also in your app launcher as **LLM Scanner**, and at http://127.0.0.1:7861.
- New versions appear as an **Update LLM Scanner** button at the top of the app.

### What gets installed

| Part | Purpose | Location |
|---|---|---|
| LLM Scanner | The app | `~/llm-scanner` |
| Ollama | Runs chat models | `~/.local/ollama` (models in `~/.ollama/models`) |
| stable-diffusion.cpp | Runs image models | `~/.local/sd-cpp` |
| garak, Gradio, PyTorch | Scanner and app packages (exact pinned versions) | `~/llm-scanner/.venv` |
| uv | Python and package installer | `~/.local/bin/uv` |

Installer options:

| Option | Effect |
|---|---|
| `--data FILE` | Restore data exported from another machine (see below) |
| `--pull-models` | With `--data`, also re-download that machine's Ollama chat models |
| `--no-autostart` | Don't open the app in Firefox at login |
| `--no-start` | Install, but don't start it yet |

To install from a clone instead: `git clone https://github.com/Kadn0/llm-scanner.git && cd llm-scanner && ./install.sh`

## Move to a new machine (optional)

A fresh install starts empty: no models, chats, images, or reports. If you do want to bring your chats, images,
probe groups, and scan reports along (models are never copied), export them on the old machine:

```bash
~/llm-scanner/export-data.sh
```

Copy the file it creates to the new machine and install with it:

```bash
curl -fsSL https://raw.githubusercontent.com/Kadn0/llm-scanner/main/install.sh -o install.sh
bash install.sh --data llm-scanner-data-DATE.tar.gz --pull-models
```

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

- **Pop!_OS 24.04** (x86_64). Other Ubuntu-based systems may work, but aren't tested.
- About 10 GB of free disk for the install, plus room for models (each is typically 2 to 20 GB).
- An NVIDIA graphics card is strongly recommended. With 8 GB of graphics memory, models up to about 7.5 GB run
  fully on the GPU; bigger ones run partly in system memory, which is slower. 32 GB or more of system memory helps.
- An internet connection for the install, model downloads, and update checks. Once models are downloaded,
  chatting, image generation, and scanning all run offline.

The app detects your graphics memory and recommends model versions that fit it.

## Troubleshooting

| Problem | Try |
|---|---|
| The page won't load | `systemctl --user status llm-scanner` and `journalctl --user -u llm-scanner -n 50` |
| Chat models don't respond | `systemctl --user status ollama` and `~/.local/bin/ollama list` |
| Image generation fails | Check `~/.local/sd-cpp/sd-cli` exists; re-run `install.sh` to reinstall it |
| The installer says the NVIDIA driver isn't working | `sudo apt install -y system76-driver-nvidia`, reboot, run the installer again |
| It doesn't start at boot | Run `sudo loginctl enable-linger $USER` once |
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

## Changelog

- **1.0.11**: Show byte-level Hugging Face progress and total size while importing GGUF files into Ollama.
- **1.0.10**: Fall back to a Hugging Face Hub download and Ollama import when Xet/CDN redirects prevent Ollama's native `hf.co` pull.
- **1.0.9**: Include all Python modules in in-app updates so new releases start correctly after installation.
- **1.0.8**: Fix release packaging so in-app updates install the version shown by the release, and reject mismatched release archives.
- **1.0.7**: Check for LLM Scanner, Ollama and garak updates every 5 minutes and show the current LLM Scanner version in the page header.
- **1.0.6**: Check for LLM Scanner, Ollama and garak updates every hour and show the current LLM Scanner version in the page header.
- **1.0.5**: Model downloads now report permanent Ollama and Hugging Face HTTP errors instead of retrying them as interrupted connections.
- **1.0.3**: App icon for the desktop, app launcher, dock and browser tab. The installer adds a desktop shortcut, and existing installs get the icon after updating.
- **1.0.2**: After an update is installed, the notice no longer keeps offering that same update during the
  few seconds before the app restarts (applies to LLM Scanner and garak updates).
- **1.0.1**: Test release to confirm the in-app updater works end to end. No functional changes.
- **1.0.0**: First release.
