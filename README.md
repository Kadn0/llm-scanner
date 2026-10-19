# LLM Scanner

A local app for downloading AI models, chatting with them, generating images, and testing chat models for
security weaknesses with [garak](https://github.com/NVIDIA/garak). It runs entirely on your own machine at
**http://127.0.0.1:7861** and installs into your home folder without sudo.

## Features

- **Models**: search Hugging Face or the Ollama library, with automatic version choice for your GPU. Downloads
  resume after interruptions and can be prioritized. The gear on the Add a model card holds a Hugging Face
  access token, for gated and private models.
- **Chat**: saved conversations with search, attachments (images for vision models, text/PDF/Word for any
  model), and time and token stats for every reply.
- **Images**: text-to-image with Z-Image, FLUX.1, and Stable Diffusion 1.5 / XL through stable-diffusion.cpp,
  with a gallery. Image models can be found through Hugging Face search too.
- **Scanning**: garak probes, saved probe groups, a live progress bar, and a security-analyst report with OWASP
  LLM Top 10 mapping, evidence, remediation, and the raw garak data, also as a spreadsheet (CSV) with every value
  of every record in its own column. Scans can be paused, a stuck probe is
  detected and restarted, and finished probes are never lost.
- **Red team tools**: start and stop the own web interfaces of promptfoo (evals and red-team runs) and PyRIT from
  the app, on this computer only, with your installed models listed as promptfoo targets.
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
| Node.js, promptfoo, PyRIT | Red team tools (installed by the app when it starts and they're missing, about 2.4 GB) | `~/.local/node`, `~/.local/promptfoo`, `~/.local/pyrit` |

Installer options:

| Option | Effect |
|---|---|
| `--data FILE` | Restore data exported from another machine (see below) |
| `--pull-models` | With `--data`, also re-download that machine's Ollama chat models |
| `--no-data` | Don't restore an export found in your home folder (or the folder you run the installer from) |
| `--no-autostart` | Don't open the app in Firefox at login |
| `--no-start` | Install, but don't start it yet |

To install from a clone instead: `git clone https://github.com/Kadn0/llm-scanner.git && cd llm-scanner && ./install.sh`

## Move to a new machine (optional)

A fresh install starts empty: no models, chats, images, or reports. If you do want to bring your chats, images,
probe groups, scan reports and the red team tools' setup (promptfoo's data, and PyRIT's settings in `~/.pyrit` and
`~/pyrit`) along (models and your Hugging Face token are never copied), export them on the old machine:

```bash
~/llm-scanner/export-data.sh
```

Copy the file it creates to the new machine's home folder and run the installer there; it finds the newest export in
your home folder (or the folder you run it from) and restores it, or name one with `--data`:

```bash
curl -fsSL https://raw.githubusercontent.com/Kadn0/llm-scanner/main/install.sh -o install.sh
bash install.sh --data llm-scanner-data-DATE.tar.gz --pull-models
```

## Updates

LLM Scanner checks for new versions of itself (on GitHub), Ollama and garak once when it starts, and whenever you
click **Check for updates** in the status bar. Nothing is installed on its own: when something can be updated, a small
box beside the title lists it (for example "Ollama 0.34.2 → 0.35.0") with an **Update** button, plus **Update all**
when there is more than one. The grey x closes the box until a newer version appears or you check again. Updating
LLM Scanner:

1. Downloads the release and installs any new Python packages.
2. Starts the new version on a spare port to confirm it works, without touching your models or data.
3. Only if that check passes, backs up the current version to `data/backups/`, switches over, and restarts.

If the check fails, nothing changes. Ollama and garak updates work the same way, with automatic rollback.

### Publishing a new version (maintainer)

1. Make your changes on `main` and test them locally, including the automated tests:
   `.venv/bin/python -m unittest discover -s tests`
2. Bump the number in `VERSION` (for example `1.0.0` to `1.1.0`).
3. If you changed Python packages, refresh the pins:
   `~/.local/bin/uv pip freeze --python .venv/bin/python > requirements.txt`
4. Commit, push, and create a release whose tag matches the version:

   ```bash
   git commit -am "Describe the change" && git push
   gh release create v1.1.0 --title "v1.1.0" --notes "What changed"
   ```

Copies on 1.0.14 or later see it the next time they start or someone clicks Check for updates (1.0.7 to 1.0.13
check every 5 minutes; older versions every 6 hours or after a restart).

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
| `tests/` | Automated tests: `test_app.py` (functions), `test_ui.py` (every page, against a fake Ollama), `test_updates.py` (updates, including upgrading earlier releases). No network, models or real data needed |
| `install.sh`, `uninstall.sh`, `export-data.sh` | Setup, removal, and data export |
| `VERSION` | The version number the update check compares against |
| `requirements.txt` | Exact Python package versions |

## Changelog

- **1.1.10**: The raw garak report is also saved as a spreadsheet (CSV): one row per record, every value in its own
  column named by its path, offered in Download files and as a view under Raw data. Text that a spreadsheet would
  run as a formula is kept as plain text. Clearer setting explanations, primary button hover and Extreme tags, a
  hover title showing a saved chat's full name, and a collapsed chat sidebar no longer takes keyboard focus.
  Fixes: chat search finds words in any script; an Ollama server error shows in the chat instead of a JSON
  error or an empty reply; a GitHub rate limit no longer reads as "up to date"; a saved group whose probes are
  all gone no longer starts garak's full scan; settings files holding the wrong kind of JSON no longer crash
  saving a group or hide the built-in image models; Realistic Vision "no VAE" files are recognised as SD 1.5.
- **1.1.9**: Probe time tags have five levels instead of three: Low, Medium, High, Very high and Extreme, so a probe
  of 151 prompts is no longer tagged the same as one of 6463. Probes whose prompts an attacker model writes during the
  scan are Very high. The hover now says when a probe has more prompts than garak sends (at most 256 per probe, picked
  at random), for example badchars.BadCharacters has 36063.
- **1.1.8**: A data export copied to another machine now sets it up without any options: the installer restores the
  newest `llm-scanner-data-*.tar.gz` in the home folder (or the folder it is run from) unless `--no-data` is given,
  and a copy that updates to this version restores PyRIT's settings from such an export if it has none yet. Exports
  leave out Python environments, so they stay small.
- **1.1.7**: New installs and updated copies end up with the same red team tools: PyRIT is now installed where it's
  missing, like promptfoo and Node.js (an Install button on the Red team tools page does it right away), and
  `uninstall.sh` removes all three. Moving to a new machine now carries the red team tools' setup too: `export-data.sh`
  includes promptfoo's data and PyRIT's settings (`~/.pyrit`, `~/pyrit`), and `install.sh --data` restores them with
  their paths changed to the new home folder. The export file is now readable only by you.
- **1.1.6**: The Red team tools page can start and stop PyRIT's own web interface (when PyRIT is installed in
  `~/.local/pyrit`), on this computer only and with PyRIT's settings in `~/.pyrit`, so it is used without the command
  line. It shows PyRIT starting, links to it once it runs, and stops with LLM Scanner.
- **1.1.5**: Hugging Face no longer holds up a scan. Scans run offline from Hugging Face first, using the files earlier
  scans downloaded, and only a probe that needs a file not downloaded yet goes online for it, so most scans make no
  Hugging Face requests at all. A token Hugging Face rejects is found before going online and simply not sent; "too
  many requests" pauses and retries; and a file in a gated or private repository skips just that probe, with the reason
  in the summary, instead of halting. The app's own Hugging Face requests (search, versions, repository details) reuse
  answers for a few minutes, run at most four at once and wait out "too many requests", and downloads retry it instead
  of failing. Probe time tags now count everything that makes a probe slow: prompts, how long replies may run, how
  long prompts are to read, turns, and AI judging of replies, and the hover says which apply and when a scoring
  model is downloaded the first time. A new Red team tools page starts promptfoo's own web interface on this computer
  only, and promptfoo with Node.js is installed where it's missing, including on copies that update.
- **1.1.4**: Every probe on the Probes tab has a time tag (Low, Medium or High) with the number of prompts it sends;
  hovering it estimates how many replies to wait for. Probes whose prompts an attacker model writes during the scan
  are High, and so are probes that let each reply run far longer than usual: divergence.Repeat sends only 36 prompts,
  but each asks for a word repeated forever, so every reply runs to 1024 tokens. Scans now send garak the Hugging Face
  token saved in the app, since garak downloads detector models and datasets from Hugging Face partway through a scan.
  If Hugging Face refuses one of those downloads (a rejected or old token, a gated repository, or too many requests),
  the scan halts with an explanation of what to fix instead of retrying. Reports can be downloaded as a readable text
  file of the raw report, and the scan's model stays loaded for the whole scan.
- **1.1.3**: A scan that hangs is now caught instead of sitting frozen. garak retries a request that times out
  or fails without limit and without printing anything, so a model too slow for the response timeout used to stall
  a scan forever. The progress card now warns as soon as requests start timing out, and after 3 timeouts in a row,
  repeated Ollama errors, 10 minutes with no progress and no work by garak or the model, or far longer than any
  prompt can take, the scan restarts from the probe it was on (unloading the model, and restarting Ollama if it no
  longer answers), keeping every finished probe. If the same probe gets stuck again, or garak itself crashes there
  twice, the scan halts and waits: **Continue** retries that probe (with a new Response timeout if you raised it),
  **Skip probe** moves on without it, and **Stop** ends the scan. A halted scan stays waiting across restarts, and
  the status bar says when one needs attention. **Restart probe** does the same restart by hand at any time while a
  scan runs. If the page itself stops receiving updates (sleep, network, a restart), the progress card says so and
  suggests reloading, so a disconnected page is no longer mistaken for a stuck scan.
- **1.1.2**: A gear in the top-right of **Add a model** opens a Hugging Face access token, so gated models (the
  ones whose conditions you accept on their Hugging Face page) and models in your private repositories can be
  downloaded, and searches and downloads are no longer limited to what Hugging Face allows anonymously. The
  repository list shows the size of the version this PC would download, and marks gated and private repositories,
  which say what is needed when you pick one. The
  token is checked with Hugging Face before it is saved, is stored on this computer only in `data/settings.json`
  where just your account can read it, is never shown in full again, and is left out of `export-data.sh` archives.
  A download Hugging Face refuses now says whether a token is missing or the account it belongs to has no access
  to that model. If you already have `HF_TOKEN` set on this machine, it is used when nothing is saved in the app.
  A scan can now be paused and resumed: the model stays loaded and the scan carries on from the same probe, and
  the time it spent paused is left out of the estimate of how long is left. Chat times over a minute are shown as
  minutes and seconds instead of a long run of seconds.
- **1.1.1**: A scan cut short by LLM Scanner stopping or the computer shutting down resumes at the next start with
  only the probes it had not finished (saved as a new part of the same scan), and the Scan tab shows a running scan
  after the page is reloaded. Deleting a download also deletes the partial data Ollama kept for it, and leftover
  partial data is cleared at startup. Version lists show how each version splits across video and system memory;
  versions Ollama can't load are greyed out with an icon that explains why, and can still be downloaded. The Scan
  tab marks models Ollama can't load and won't scan them. Choosing an option by clicking its name works again, hover
  explanations are no longer hidden behind open lists, the Models tab no longer pads cards with empty space, and in
  Reports the box selects while clicking a name shows that report.
- **1.1.0**: Downloading a Hugging Face model no longer fails on a name Ollama doesn't recognise. Versions are named
  the way the repository names them (`dspark-bf16`, `PQ2_0`, `UD-Q5_K_XL`), and if Ollama refuses a tag for any
  reason the app fetches the file itself. Versions in a format Ollama cannot load are marked and never recommended,
  and the recommended version is the largest one that fits this PC's video memory. The version list also shows what
  the model is: base model, licence, downloads and whether it can see images. Reports are now a list with a tick box
  each, a Select all box and a Delete button that counts what is ticked. Long download names no longer push the
  pause and delete buttons out of reach.
- **1.0.16**: Version names of Hugging Face models are read in full, so repository-specific quantizations such as
  `PQ2_0` are no longer offered as `Q2_0` and then fail to download. Versions in a format Ollama cannot load (they
  need the model author's own llama.cpp build) are marked in the list and never recommended.
- **1.0.15**: The models table shows a small Hugging Face or Ollama logo beside each model instead of a Source
  column, and clicking a model opens its Hugging Face or Ollama page. Probe explanations appear in a quieter pop-up
  tip, and each group lists "Hover a probe or family to see what it does" underneath.
- **1.0.14**: Updates are checked when the app starts and with a **Check for updates** button (no more background
  checks), and shown in a small box beside the title with an Update button per item and Update all. Leftover files
  (abandoned downloads, the old Hugging Face cache, failed update files, unsent attachments) are cleaned up at every
  start. New scan groups for AI risk assessments: a baseline assessment, five NIST AI 600-1 risk areas, and EU AI Act
  risk areas. Every group shows its description and probes, and hovering a probe explains what it does. Ollama's
  status sits in its Start / Stop / Restart control (Stop asks first), the status bar is one tidy line, the Models
  tab columns line up, and the Scan page shows the garak version.
- **1.0.13**: Hugging Face downloads also work when Ollama reports the Xet redirect as an HTTP error (the "HTTP 400
  ... blocked redirect" failure); vision models from Hugging Face keep their image support. Downloads can be paused
  and resumed (kept across restarts), and deleting one removes its partial data. New "OWASP Top 10 for LLMs" scan
  preset. Ollama start / stop / restart in one control. Smoother look: a sliding highlight on tabs and switches, a
  cleaner tab bar, gentle transitions, and no loading flicker when switching sources or filtering. Tests now cover
  every page and every kind of update, including upgrading from earlier releases.
- **1.0.12**: Hugging Face downloads work on every PC again: when Ollama can't follow Hugging Face's new Xet storage
  redirect, the app downloads the GGUF itself (resumable, with progress, cancel and priority) and imports it, then
  deletes its copy instead of keeping a second one. Updating from 1.0.8 or earlier no longer leaves the app unable to
  start. Scans and image generation keep running and finish cleanly if the page is closed or reloaded; a second scan
  can't start while one is running. Stopped scans appear in Reports. Scrolling a dropdown list no longer scrolls the
  page. Added automated tests.
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
