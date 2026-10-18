# CashLint jury frontend

Current implementation: the approved flagship polish pass, 18 October 2026.
CashLint is an evidence-led cash-operation review tool with two pages:

| Route | Purpose |
|---|---|
| `/` or `/review` | Upload, local playback, processing, findings, original evidence and next action |
| `/review?job=<id>` | Restore an analysis; optionally reselect the original for local playback |
| `/results?job=<id>` | Existing-link compatibility alias, canonicalized to Review |
| `/evaluation` | Measured development comparisons, provenance and remaining gaps |

## Running the app

From the repository root:

```bash
uv run uvicorn service.app:app --host 127.0.0.1 --port 8018 --no-access-log
```

The existing task preview is reachable on the WSL interface at
`http://172.28.117.95:8018/review`. Include `styles/` and `service/static/` in a
service bundle. No new dependency, server route or inference provision was added.

## Review flow

The initial page has one primary file-selection action and shows the real
MOV/MP4 and 200 MiB limits. Selecting a valid file opens browser-local playback,
file metadata and the Analyze action. It does not preload any model findings.
The header compacts after selection so the working area stays in view.

Submission uses `POST /ui/analyze`. XMLHttpRequest upload events show actual
transferred bytes when the browser reports them. Accepted jobs update the URL
without navigating away; the selected local video is preserved. Processing polls
`GET /ui/jobs/{job_id}` and shows elapsed time within the original five-minute
budget. There are no fabricated analysis percentages, stage names or time estimates.
Files cannot be replaced or resubmitted while a request is active. Completed jobs
have a separate New recording action, preventing accidental duplicate submissions.

Completed results have a compact verdict summary and qualified fallback notice
when applicable. A single media viewer switches between Recording and Evidence.
The findings list expands one selected event at a time; its timestamp, explanation
and supplied source references stay together. Event selection synchronizes the
original evidence and seeks a playable local recording. Previous/next finding
and frame controls support keyboard use. Evidence enlarges in a native modal
with Escape dismissal and focus return. Hover never changes evidence pixels.

The media column stays visible during desktop scrolling when the viewport is tall
enough. Short laptops use normal flow and a smaller media stage. Mobile stacks the
viewer and findings; selecting Inspect evidence brings the viewer into view.
Controls have visible focus, readable neutral text and larger mobile arrow targets.

A failed, malformed or expired job never becomes a negative verdict. Retry checks
the same job without uploading again. Unsupported browser playback remains
separate from analysis failure. Broken evidence images have a stable visible
fallback rather than silently collapsing the viewer.

Job ID, upload-start time and optional filename/size are tab-scoped sessionStorage
metadata. Source media and result payloads are not persisted there. Refresh and
navigation can restore a result but cannot restore a browser-local File object.
The Recording tab offers original-file reselection without clearing findings.
When original metadata exists, different filename/size is rejected; matching
metadata is not claimed to prove content identity. Attached playback is labelled
user-selected; API evidence remains authoritative. With no retained metadata,
local source identity is explicitly unverified. Source reselection can occur while
a stored job is processing; it does not change that server job.

The sidebar contains Review and Evaluation. Its bottom theme control and desktop
collapse preference persist. Mobile opens as a closed drawer with focus trapping,
background inertness, outside-click and Escape dismissal. Only the title-bar
button toggles it; no sidebar X, page numbers, path breadcrumbs or status-ready
badge were reintroduced.

## Evaluation and measured data

Evaluation leads with detection rising from 2/16 to 12/16 across the three
recorded configurations. The selected recipe's verdict result, such as 20/30
correct, is a supporting stat. The data remains the frozen aggregate snapshot: 40.0%, 53.3%, 66.7%
accuracy across three previously reviewed development configurations. There is no
independent test. The ≥90% ambition is a subordinate, explicitly unachieved target.
The selector changes the recorded comparison, not the live inference engine.

A compact comparison row shows 2/16, 6/16 and 12/16 positive videos detected
next to the recorded-configuration selector.
The detection card is a 2×2 confusion matrix (12 / 4 / 6 / 8 by default), with
reference rows and detector-output columns. Count controls fill intensity; true
positives use the accent and the remaining cells use slate. Failed predictions
remain accounted for explicitly. Localization retains its recorded event counts. The
separate partial runtime checkpoint remains attributed separately: 44 jobs,
zero failures and a maximum of 76.51247401 seconds. It is not full-set qualification.
The snapshot includes source hashes and aggregates, without private per-video
results, bank recordings, labels or evidence frames. No snapshot refresh was
performed in this UI task. Explicit maintenance command:

```bash
uv run python -m scripts.build_frontend_snapshot
```

## Visual system

Light mode retains Forte #A31551. Dark mode uses the lighter accent pink #D27AA3
for accent text, the wordmark and chart marks. Its contrast is 5.69:1 on the panel
surface and 6.20:1 on the canvas, exceeding the requested 3:1. Filled actions use
dark labels in dark mode and white labels in light mode. Matrix-cell ink adapts
to the fill: all 24 sampled pairs across both themes and configurations exceed
4.5:1 (minimum 4.55:1). IBM Plex Sans and
Space Grotesk are local WOFF2 files with retained SIL Open Font Licenses. Figures
use tabular numerals. The neutral ramp, 8px spacing scale, restrained radii and 1px
borders are shared. There are no runtime CDN requests, gradients or drop shadows.
The focus ring uses a contrasting neutral. The theme button names its action,
"Switch to light" or "Switch to dark", including its accessible label. The empty
Review has a Recording / Evidence / Review strip with small SVG icons.

Initial KPI values count up and localization bars grow for 500ms with ease-out.
Motion plays once per document; selecting a configuration cancels any in-flight
count-up and renders its exact measured values immediately. It does not replay
the animation. Reduced-motion users get final numbers and bar widths immediately;
KPI announcements are held with aria-busy during the initial transition.

## HTML downloads

```bash
uv run python -m scripts.export_frontend
```

Outputs: `reports/frontend-export/fortebank-review.html`, `evaluation.html`,
`results.html` (compatibility alias), `README.txt`, and
`reports/fortebank-frontend-html.zip`. Each HTML file embeds fonts, CSS, scripts
and font licenses. Evaluation embeds the same aggregate JSON. Review supports
local file preview and explicitly identifies offline Analyze as a design preview;
real inference requires the service. No recordings or findings are bundled.

## Verification and visual critique

Commands run:

```bash
uv run python -m pytest tests/test_frontend.py tests/test_api.py tests/test_video_api.py -q
uv run python -m scripts.export_frontend
```

Latest integration/API result: **45 passed in 6.46s**.

Browser scripts use the already available Node, Playwright and Edge runtimes:

```text
node scripts/check_frontend.cjs <local-url> <output-dir> [playwright-module] [browser-exe]
node scripts/check_frontend_export.cjs <export-dir> <synthetic-mp4> [playwright-module] [browser-exe]
```

Place a clearly synthetic `fixture.jpg` in the first script's output directory.
It creates a genuinely playable synthetic portrait MP4 using the existing browser's
MediaRecorder. All analysis requests are intercepted with synthetic responses;
no paid GPU request or bank footage is used. The export script opens actual file
URLs, previews that video, tests relative page navigation and checks for unwanted
network requests.

Latest browser result: **273 checks, errors: []**. The initial flagship pass
passed 245 checks; the chart/palette refinement adds data, contrast and motion
contracts. Checks include both themes at 1440/1024/768/390/320 pixels,
short-laptop playback controls, three-event findings, real video seeking, original
frame selection/enlargement, source reselection without losing results, metadata
mismatch, legacy links, retry, negative/fallback/malformed/failed/expired jobs,
unplayable video, broken frames, missing metrics, drawer focus and reduced motion.

Offline result: **13 checks, errors: [], externalRequests: 0**. It verifies actual
local MP4 playback, preview-only analysis, unchanged file URL, theme controls,
relative navigation, embedded metrics and comparison, and the legacy HTML alias.

Two visual critique rounds were completed:

| Round | Three main issues fixed |
|---|---|
| 1, 242 browser checks | Repeated completion boxes pushed media down: combined the verdict/fallback summary and quieted redundant status. Switching back to Recording left an evidence caption: tied captions to the active pane. Empty state had two competing actions: hid the inactive Analyze row and used direct page headings. |
| 2, 243 browser checks | Result header consumed short-laptop space: compacted it after selection and checked the real video controls at 1366×768. Mobile arrow buttons were small: increased them to 44px. Broken frames collapsed the viewer: added a stable fallback stage. |

The short-laptop check exposed a 3px overflow of the recording control area; the
short-screen media limit was adjusted and the final 245-check run passes. A harness
resize check was also corrected to wait for the media-query state change.

Final screenshots were inspected at:
`/mnt/c/Users/ibrak/.codex/visualizations/2026/10/18/jury-frontend/flagship/final/`.
Earlier round1/round2 screenshots preserve the critique evidence.

## Chart/palette refinement — latest pass

Implemented the seven requested adjustments without changing the data snapshot,
measurements, outcome claims or two-page structure. Three additional issues fixed:

1. The desktop hero left a one-word final line: balanced wrapping keeps it coherent.
2. A four-cell matrix could hide missing outcomes in the original recipe: positive
   failures remain labelled missed / unprocessed, and the one unprocessed negative
   is explicitly outside its four cells. The default matrix still totals 30.
3. Switching configurations during the count-up could leave stale numbers: cancel
   motion and finalize the newly selected values synchronously. Bars do not replay.

Matrix rows also share height when mobile labels wrap, and SVG axes render at the
actual container width so text remains readable on small screens.

Commands: the existing pytest command, both browser-check scripts, and
`uv run python -m scripts.export_frontend`. Results: **45 passed in 6.46s**,
**273 live browser checks**, **13 offline checks**, zero JavaScript errors and
zero external requests from the exports. Duration checks verify the 400–600ms
range; reduced-motion checks verify zero animations and immediate final values.

Updated files: `styles/tokens.css`; `service/static/{index,results,evaluation}.html`,
`evaluation.js`, `theme.js`, `workspace.css`; both browser scripts; this document;
and regenerated standalone HTML/ZIP. The evaluation snapshot's SHA-256 remains
`f989c87ee97bf42514cc263d635bb8b67e7307d8e838891e6e2ac5f43a2890e7`.
Final screenshots and contrast samples:
`/mnt/c/Users/ibrak/.codex/visualizations/2026/10/18/jury-frontend/chart-pass/final/`.

## Scope and limits

Changed: `service/static/{index,results,evaluation}.html`, `workspace.css`,
`{review,results,workspace,evaluation}.js`, `styles/tokens.css`, browser-check and
export scripts, this document, the approved plan's completion status and generated
HTML/ZIP files. The shared shell/theme logic and API schema remain compatible.

Model/features/thresholds, frozen evaluators, raw data, aggregate measurements,
infrastructure and unrelated work are unchanged. These checks qualify the UI
behavior exercised here; they do not establish recognition accuracy, a public
production deployment or a 100-point jury result. A live jury rehearsal with the
actual deployed recognition service remains a separate operational check.

## Essential-copy pass — latest refinement

Removed repeated chart kickers, subtitles, configuration descriptions and the
visible duplicate gap-summary section. Retained the requested hero, all measured
values, comparisons, axes, matrix cells and the visible previously-reviewed /
no-independent-test scope. Target/gap details, runtime headroom and full accounting
remain available under **Methodology & sources**. Missing-result warnings still
appear directly beside the matrix when applicable. Shortened KPI labels and moved
the shared comparison reference into one caption.

Review's empty state now has the upload essentials and a compact three-icon stage
strip. Returned event descriptions, sources and evidence are untouched.

At the same default 1440×1000 viewport, rendered main-content words changed:
Evaluation **353 → 200** (43% fewer); empty Review **115 → 64** (44% fewer).
These are visible-copy counts with methodology closed, not model measurements.
Main-content heights changed 2097 → 1623px and 666 → 541px respectively.

Files changed: the three frontend HTML documents, `evaluation.js`, `results.js`,
`workspace.css`, the browser contract script, this document, generated HTML and ZIP.
Commands: `uv run python -m pytest tests/test_frontend.py -q` → **2 passed in 0.80s**;
`scripts/check_frontend.cjs` → **275 checks, errors: []**; offline export checker →
**13 checks, errors: [], externalRequests: 0**;
`uv run python -m scripts.export_frontend` rebuilt the downloads.
The browser checks open the disclosure to verify retained target/headroom and
accounting details. Both themes and mobile sizes were checked. Screenshots and
before/after measurements are in
`/mnt/c/Users/ibrak/.codex/visualizations/2026/10/18/jury-frontend/concise/`.
The evaluation JSON SHA-256 is unchanged. No model, API or numerical claim changed.


## Compact configuration comparison — latest change

After the human rejected the full-width step chart, replaced it with a small
three-value row in the existing configuration toolbar. Removed the chart card,
axes, duplicate heading and resize-rendering code. Kept the three labelled counts,
selected-configuration highlight, hero, KPIs, matrix and localization chart.
No evaluation figures, snapshot contents or recognition claims changed.

Files: evaluation HTML/JS, workspace CSS, browser contract, this document and
regenerated HTML/ZIP. Commands: frontend pytest → **2 passed in 0.31s**;
existing browser contract → **275 checks, errors: []**; offline export checker →
**13 checks, errors: [], externalRequests: 0**; export module rebuilt downloads.
Desktop and mobile screenshots inspected under
`/mnt/c/Users/ibrak/.codex/visualizations/2026/10/18/jury-frontend/compact-comparison/`.

## Five small presentation fixes

Original 8B now shows `1 unprocessed · 30 videos total` below its matrix:
the existing cells total 29, plus the unprocessed negative video gives 30.
The comparison strip is labelled **Positive videos detected**, with arrows
connecting the three configurations. Matrix axes read **Rows: human reference ·
Columns: detector**. Localization adds the selected configuration's concise
takeaway (default: **5 of 16 within ±2s, up from 1.**) and its card fits its
content. Review's footer sits at the viewport bottom on short pages and follows
content on long pages, without overlaying controls.

Files changed: `service/static/evaluation.html`, `evaluation.js`, `workspace.css`,
`scripts/check_frontend.cjs`, this document and regenerated standalone HTML/ZIP.
Commands: `uv run python -m pytest tests/test_frontend.py -q` → **2 passed in 0.28s**;
`scripts/check_frontend.cjs` → **283 checks, errors: []**;
`scripts/check_frontend_export.cjs` → **13 checks, errors: [], externalRequests: 0**;
`uv run python -m scripts.export_frontend`; `git diff --check` passed.
Both themes and 320–1440px layouts were exercised with mocked review responses;
no live inference was run. Screenshots:
`/mnt/c/Users/ibrak/.codex/visualizations/2026/10/18/jury-frontend/small-fixes/`.
All evaluation figures and claims remain unchanged. Snapshot SHA-256:
`f989c87ee97bf42514cc263d635bb8b67e7307d8e838891e6e2ac5f43a2890e7`.

## Final jury usability refinement, 18 October

Review now accepts a same-service result link or 12-character job ID to retrieve
an existing analysis without uploading again. Returning reviewers have a
**Continue last review** link, using the existing tab-scoped job metadata. No
bank footage, fabricated example or verdict is bundled. Foreign-service links
are rejected with an accessible error. The saved-result entry is hidden during
active review and in offline design exports. The three-stage strip has short
action descriptions. Review's viewport-bottom footer is preserved.

Evaluation gives detection visual emphasis within the existing KPI row; the
percentage still represents positive-video recall. Mobile KPIs changed from
four tall stacked tiles to a compact two-by-two summary. The detection matrix
has slightly more desktop width; both charts, the requested hero and all
numbers/claims are unchanged. The snapshot SHA-256 is still
`f989c87ee97bf42514cc263d635bb8b67e7307d8e838891e6e2ac5f43a2890e7`.

Files: `service/static/{index,results,evaluation}.html`, `review.js`,
`workspace.js`, `workspace.css`, `scripts/check_frontend.cjs`, this document,
`docs/frontend-polish-plan.md`, and regenerated standalone HTML/ZIP downloads.
Commands: `uv run python -m pytest tests/test_frontend.py -q` → **2 passed in 0.58s**;
`scripts/check_frontend.cjs` → **302 checks, errors: []** (previously 283);
`uv run python -m scripts.export_frontend` regenerated the bundle;
`scripts/check_frontend_export.cjs` → **13 checks, errors: [], externalRequests: 0**;
`git diff --check` passed. Browser coverage includes saved-result entry/recovery,
foreign-link rejection, no duplicate upload, both themes, 320–1440px layouts,
keyboard navigation, reduced motion and existing failure/retry/playback states.
Screenshots were visually inspected at desktop and mobile widths under
`/mnt/c/Users/ibrak/.codex/visualizations/2026/10/18/jury-frontend/final-usability/`.
Checks use synthetic media and mocked API-shaped results; actual inference and
a timed human jury rehearsal were not run. No subjective 10/10 grade is claimed.

## Saved-result panel removed after screenshot feedback

Removed the “Already have a result?” block, job-ID/link input and resume button,
including their JS/CSS. The user supplied a screenshot of this panel and rejected
it. Review remains focused on upload → findings → evidence. Existing result URLs
and tab-scoped result navigation remain functional. Kept the mobile 2×2 KPIs and
evaluation hierarchy; no numerical data changed. Regenerated offline HTML/ZIP.
Frontend pytest: **2 passed in 0.34s**; `git diff --check` passed.
Browser regression: **292 checks, errors: []**. Visually confirmed removal in
`/mnt/c/Users/ibrak/.codex/visualizations/2026/10/18/jury-frontend/clean-review/empty-dark-desktop.png`.
Changed the Review HTML/alias, review/workspace JS, shared CSS, browser contract,
this document and regenerated exports. No inference or model changes.

## Visual design refinement — no feature or copy additions

Refined the shared shell, type hierarchy, page gutters, control alignment and
spacing. Sidebar selection is quieter; its toggle and theme control no longer
look like additional cards. Review has a generous dashed upload area, a compact
icon treatment and aligned stage descriptions. Media and findings headers share
a 56px height. Existing findings use consistent type and padding.

Evaluation now uses an open metric strip bounded by fine rules and two analytical
sections rather than repeated boxed cards. Numeric baselines, configuration
controls, chart headings, matrix spacing and timing bars are aligned. Detection
keeps the sole accent emphasis. Mobile retains two KPI columns and legible labels;
no new controls, prose, features, charts, pages or data were added. Existing result
links and all upload/result states are unchanged. Both theme accents and matrix
fill/contrast calculations are preserved.

Source changes are limited to `service/static/workspace.css` and the stylesheet
cache-version reference in `service/static/{index,results,evaluation}.html`.
Regenerated the standalone HTML/ZIP. The first screenshot pass identified header
misalignment and small mobile labels; both were corrected in the final pass.
Frontend integration: `uv run python -m pytest tests/test_frontend.py -q` →
**2 passed in 0.31s**. First browser pass: **292 checks, errors: []**.
`uv run python -m scripts.export_frontend` regenerated the download;
`git diff --check` passed. Snapshot SHA-256 remains
`f989c87ee97bf42514cc263d635bb8b67e7307d8e838891e6e2ac5f43a2890e7`.
Final browser pass: **292 checks, errors: []** across both themes and 320–1440px
widths, including synthetic playable video and mocked result states. Offline
checker: **13 checks, errors: [], externalRequests: 0**. Final desktop and mobile
screenshots inspected under
`/mnt/c/Users/ibrak/.codex/visualizations/2026/10/18/jury-frontend/design-refinement/final/`.
No live inference or subjective jury score is established by these UI checks.

## Three requested design corrections

Compacted Evaluation's existing heading and configuration row. Moved the existing
checkpoint date into the scope row, reduced headline size/spacing and placed the
configuration label beside its selector. First KPI top position (1000px-high
viewport) changed **465.55 → 397.53px** at 1440px width, **601.13 → 444.55px** at
390px, and **637.13 → 521.33px** at 320px. These are layout measurements, not model
results. Secondary labels now share a 13px size and muted text token, with existing
accent and matrix contrast exceptions preserved.

Review's empty state has a solid fine border, a restrained accent edge and a
recording/inspection SVG replacing the generic upload icon. Its existing label,
file constraints and button align in a grid. No copy or feature was added.
A parser comparison confirms unchanged text across all three HTML documents;
the evaluation JSON SHA-256 remains unchanged. Existing results, model outputs,
APIs and recognition code are untouched.

Changed `service/static/workspace.css`, `service/static/{index,results,evaluation}.html`,
this document and regenerated standalone HTML/ZIP. Frontend integration:
`uv run python -m pytest tests/test_frontend.py -q` → **2 passed in 0.31s**.
`uv run python -m scripts.export_frontend` rebuilt downloads; `git diff --check`
passed. Screenshots visually inspected in both themes at desktop/mobile widths;
header measurements saved under
`/mnt/c/Users/ibrak/.codex/visualizations/2026/10/18/jury-frontend/three-fixes/`.
Browser regression: **292 checks, errors: []**. Offline export checker:
**13 checks, errors: [], externalRequests: 0**. Coverage uses synthetic playable
video and mocked result states; no live inference was run.

## CashLint visual identity

Added a custom inline inspection-bracket/check mark beside the unchanged CashLint
name. The wordmark uses neutral text with the single theme accent reserved for
the mark. A restrained bracket motif connects the existing eyebrow, active
navigation, upload boundary and detection KPI. Control corners are consistently
2px. System monospace is used only for technical labels/timecodes; the bundled
body/heading fonts and readable secondary-label size remain. Selected finding,
frame and timestamp treatments share the accent geometry. No overlay or image
filter covers original evidence.

Changes are presentation-only: `service/static/workspace.css` and
`service/static/{index,results,evaluation}.html`, this document and regenerated
standalone HTML/ZIP. No new page, feature, copy, dependency or measurement.
Parser comparison verified identical page text; the evaluation snapshot SHA-256
remains `f989c87ee97bf42514cc263d635bb8b67e7307d8e838891e6e2ac5f43a2890e7`.

Commands: `uv run python -m pytest tests/test_frontend.py -q` → **2 passed in 0.50s**;
`scripts/check_frontend.cjs` → **292 checks, errors: []**;
`uv run python -m scripts.export_frontend` regenerated the HTML/ZIP;
`scripts/check_frontend_export.cjs` → **13 checks, errors: [], externalRequests: 0**;
`git diff --check` passed. Both themes and 320–1440px widths exercised with
synthetic playable video and mocked responses. Desktop and mobile screenshots
visually inspected under
`/mnt/c/Users/ibrak/.codex/visualizations/2026/10/18/jury-frontend/identity/`.
No live inference or subjective jury-grade claim follows from these UI checks.
