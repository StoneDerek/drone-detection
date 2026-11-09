# Browser lab on Cloudflare Pages

Public app: https://drone-audio-lab.pages.dev

The site compares seventeen frozen models, including the original eleven controls and six CNN foundations controls. It plays and seeks recorded audio, displays score timelines and CNN Grad-CAM, exposes training histories, and exports results. All five source-attributed examples load together. Multiple uploads are retained alongside them; a shared table shows each recording/model timeline, peak score and whether any alarm occurred. Switch recordings without discarding results and export the whole comparison with input hashes. Draft interval notes also remain associated with their recording and are not training labels.

Grad-CAM selection includes all fourteen trained spectrogram CNNs, independently of which models were scored over the full recording. Switching models preserves the selected context. Heatmap-only/overlay views, opacity, native-cell/interpolated rendering and a relative-attribution legend expose coarse or diffuse maps clearly. The two plots share linked rectangle/wheel zoom, Shift-drag pan, numeric time/frequency limits and reset. Their display uses a common linear Hz axis, projecting each model's actual bins without changing model inputs. The exported JSON retains the native attribution grid and original input tensor shape. New coverage/readout models have a15×22 final grid; older linear256 has8×22. Spatial gradient averaging in Grad-CAM can obscure position-dependent readouts, so a map is not a complete description of the classifier.

Inference runs in the browser with TensorFlow.js CPU kernels and a JavaScript FFT/resampler. Audio selected from the user's device is not uploaded. The SVM uses its numeric double-precision support vectors and RBF coefficients; the CNNs use their float32 frozen weights. YAMNet uses Google's official TFJS export and the same project logistic head. No model was retrained for hosting, no paid plan was activated, and the Python service is not exposed through a temporary tunnel. Cloudflare Containers were unavailable on the account's current plan, so the deployed app consists entirely of static Pages assets.

The browser supports PCM/float WAV up to 64 MB and 120 s, 1–16 channels. Supported rates are 8/11.025/16/22.05/24/32/44.1/48/88.2/96/192 kHz; non-16 kHz WAV uses exported scipy-compatible polyphase FIR coefficients. Other formats and per-window clipping/low-frequency flags remain available in the Python app. The browser comparison is offline recorded audio; no live microphone or latency claim follows. Device CPU and initial weight downloads affect responsiveness.

## Rebuild

From the project root, with the existing model artifacts and development audio available:

```bash
npm --prefix web ci --cache /workspace/.npm-cache
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/export_browser_models.py --catalog reports/cnn-foundations-v1/catalog.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/observe_real_audio.py
.venv/bin/python scripts/build_browser_page.py
.venv/bin/python scripts/build_browser_report.py
npm --prefix web run build
node web/serve.cjs
```

The observation figure script needs matplotlib 3.10.8, separate from inference dependencies. The export downloads the pinned official TFJS YAMNet archive if absent and verifies SHA-256 `604467f7ce0900a27a05f274c7e7a46effbb936788a7cee84648238de5c97c82`. It caches that file at `runs/cloudflare-deployment/yamnet-tfjs-v1.tar.gz`; its official URL and hash also appear in `web/public/models/catalog.json`. Public example licenses and dependency notices are in `web/public/THIRD_PARTY.md` and `web/public/licenses/`. Those license files are included in deployments.

Model assets are generated under ignored `web/public/models/`; they do not alter the frozen source bundles. The largest numeric arrays are sharded to4MiB for upload and loading. The browser verifies the exported whole-array SHA-256 before constructing each classifier. Build output is ignored `dist/cloudflare/`.

## Verify and deploy

```bash
bash scripts/check.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/browser_parity_fixtures.py --catalog reports/cnn-foundations-v1/catalog.json --output runs/cloudflare-deployment/parity-v3
node tests/browser-parity.cjs http://127.0.0.1:8012 reports/cloudflare-v3/parity.json runs/cloudflare-deployment/parity-v3
node tests/cloudflare-browser.cjs http://127.0.0.1:8012 reports/cloudflare-v3/local-browser.json
node tests/lab-interaction-browser.cjs http://127.0.0.1:8012 reports/cloudflare-v3/interactions.json
node web/hash-assets.cjs dist/cloudflare runs/cloudflare-deployment/assets.json
.venv/bin/python scripts/deploy_cloudflare_pages.py --account-id "$CLOUDFLARE_ACCOUNT_ID" --project drone-audio-lab --evidence-dir reports/cloudflare-v3
node tests/cloudflare-browser.cjs https://drone-audio-lab.pages.dev reports/cloudflare-v3/public-browser.json
node tests/lab-interaction-browser.cjs https://drone-audio-lab.pages.dev reports/cloudflare-v3/public-interactions.json
```

Set the account ID to the intended account. The configured `CLOUDFLARE_API_TOKEN` is read only by the deploy script and sent only to `api.cloudflare.com`. It is never included in static assets. Upload JWTs stay in memory. The script uses Pages' content-addressed asset protocol as implemented by pinned Wrangler4.65.0, creates the named direct-upload project if absent, uploads missing assets and deploys the production branch. The deployment requires passing numerical/UI evidence matching the exact JavaScript asset hash. It does not configure a custom domain, change billing, publish the cloud environment, or modify the GitHub repository remotely.

The first numerical check compared 176 model/context scores across three existing real train/development files, all eleven model identities, two input sample rates, and six target/layer Grad-CAM cases. Maximum score difference was 1.93e-6, maximum checked image difference 1.06e-5 and maximum Grad-CAM difference 2.99e-6. The two 44.1 kHz fixtures also decoded/resampled identically to Python. Alarm decisions and timestamps matched on the full 14-context Batear fixture. These are porting checks, not 176 acoustic detection trials or a new independent assessment.

The interactive check verifies playback/seek, model comparison, both Grad-CAM targets, a genuine zero map, frequency zoom, JSON downloads, real-example selection, draft-note export with audio hash, invalid input handling and no network audio uploads. Local preview applies the same CSP/response headers as Pages.

The interface explicitly explains that the matching 9/188 development false-positive counts largely result from calibrating thresholds under the same 5% budget. Those are development counts, not independently measured field false-alarm rates. Playback has no app-level gain normalization, and each model retains its existing preprocessing. See the linked level-transition diagnosis for measured uniform-gain versus within-window sensitivity.

Current deployment identity and asset checksums are in [deployment.json](../reports/cloudflare-v3/deployment.json) and [assets.json](../reports/cloudflare-v3/assets.json). Numerical evidence is in [parity.json](../reports/cloudflare-v3/parity.json); browser evidence and screenshots live alongside it. The earlier v1/v2 reports are retained. The expanded port check covers all17 model identities and36 target/layer Grad-CAM cases, including both new readout architectures. Learning curves display the actual selection loss for each experiment; older asymmetric-loss histories and repaired consistent-max histories are identified explicitly.
