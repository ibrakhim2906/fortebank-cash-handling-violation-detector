# CashLint — flagship jury experience

Status: approved by the human ("go") and implemented on 18 October 2026.
Verification and delivery details: docs/frontend.md.
Requested outcome: a professional banking review tool that jurors can test
comfortably during the live demo and on their own.

## Current audit

Reviewed the current HTML/JS/CSS, recorded brand-red screenshots, API behavior
and frozen aggregate snapshot. Findings are observations about this implementation,
not new model measurements or a claimed jury score.

| Observed friction | Proposed outcome |
|---|---|
| The review introduction occupies a large amount of screen height before the working area | Compact page heading; file selection or the recording is the main focus |
| Selected evidence is large on the left and repeated as large thumbnails on the right | One primary media viewer with Recording / Evidence controls; compact frame selection |
| Every finding expands into description, images and sources, making the page long | A compact event list with one expanded event and previous/next controls |
| The recording scrolls away while the user reads long findings | A desktop media column that stays visible when viewport height permits; ordinary flow on small screens |
| Upload, processing, completed and retry states share an almost unchanged layout | An explicit state model with a single clear primary action at each step |
| After loading a result URL, choosing a file starts a new review and clears its evidence | A separate local source-reselection path that preserves the current job and findings |
| Evaluation leads with a large unachieved 90% target | Measured results lead; development scope and future target are clearly subordinate |
| A single synthetic finding is well covered by QA; multi-event workflow and playable media need stronger coverage | Synthetic multi-event and playable-video checks, including event/video synchronization |

## Design to implement after approval

### 1. Shared shell and hierarchy

Keep exactly Review and Evaluation, CashLint, the collapsible sidebar with its
bottom theme switch, and the official Forte #A31551 accent. Keep both themes,
readable neutral text, locally bundled fonts and tabular numerals. Refine spacing,
control alignment, focus/hover states and a consistent SVG icon set. Keep the
header compact and free of breadcrumbs, page numbers and service-status badges.
Use a deliberate portrait-footage layout rather than stretching bank video to fill
an arbitrary landscape frame. Keep reduced-motion preferences and 1px borders.

### 2. Review as a working interface

Initial state: concise heading, accessible drop area, visible MOV/MP4 and 200 MiB
limits, one Analyze action and a short explanation of the returned evidence.
No preloaded bank footage or example verdicts.

After selection: show the video prominently, filename, size and readable duration
when available. Keep replacement secondary. During upload, show real transferred
bytes/progress where the browser reports them. During analysis, show elapsed time
and actual API state, without invented stage names, percentages or estimated time.

Completed: a compact verdict and event-count summary above a two-column working
area. The main viewer switches between local recording and original API evidence.
The event list expands one selected event's explanation and source context;
selecting an event synchronizes the evidence and seeks the recording when playable.
Provide clearly labelled previous/next event and frame controls with keyboard
access. Keep recommendations concise and tied to the actual response.

Direct result links remain usable. Local file metadata may be retained with the
job in tab-scoped state to help reselect the original; file bytes are not retained.
A restored job can attach a local video without clearing findings. Verify against
retained file metadata when present, label it as user-selected playback, and do
not imply that filename/size establish source identity. When the original cannot be matched, label local playback as unverified and keep
the API evidence as the authoritative source for the result.

Failures and timeouts retain the existing distinction from a negative verdict.
Retry checks the same job. Unplayable local video falls back to original evidence
with a compact explanation. New-recording selection clears old findings. Prevent
accidental repeat submissions; any reanalysis is explicit. Browser-side controls
must not claim to cancel a server job: the current API has no cancellation endpoint.

### 3. Evaluation for technical judges

Lead with the selected recipe's measured verdict accuracy, positive-video recall,
event localization and separately attributed runtime checkpoint. State the sample
size and previously reviewed development scope in the main hierarchy. Move the
90% ambition into a modest target annotation. Preserve snapshot provenance,
recipe comparison and the distinction between measured development data and
independent testing. Do not silently refresh the snapshot or imply that the
selected historical recipe is the current live engine.

Retain two takeaway-led charts with direct labels, subtle grid lines and a single
highlighted series. Make methodology and limitations available in a readable
disclosure. No invented confidence values, outcomes, savings or legal conclusions.

### 4. Reliability and accessibility

Exercise empty, selected, upload, processing, positive multi-event, negative,
fallback, unavailable, malformed and timeout states. Cover direct links, retry,
new-file resets, source reselection, latest-job navigation, long bilingual event
names/descriptions, unavailable frames and missing source links.

Verify keyboard-only upload, visible focus, modal focus/escape, event selection,
screen-reader announcements, small-label contrast and reduced motion. Check both
themes at 1440, 1024, 768, 390 and 320 pixels and a short laptop viewport. Confirm
no horizontal overflow and that active controls remain reachable.

Use synthetic media/results for automated UI QA. Recognition measurements require
real evaluation output; browser mocks are not model validation. Do not trigger
paid GPU inference as part of this frontend polish pass.

### 5. Inspect, refine and deliver

Run the existing API/frontend integration tests and update the browser contract
checks around user behavior rather than obsolete markup. Add playable synthetic
media coverage using already available browser tooling; no new dependency.

Capture empty/processing/completed/evaluation views in both themes and desktop /
mobile sizes. Perform two visual critique passes and fix the three largest
observed issues in each. Re-run affected checks after the fixes.

Rebuild self-contained HTML downloads and the ZIP. Offline exports retain layout,
local media preview, theme and embedded measured evaluation data; they clearly
identify that real analysis requires the running service. Verify zero unexpected
external requests. Update docs/frontend.md with actual commands, results and limits.

## Implementation boundaries and files

Primary edits: styles/tokens.css; service/static/index.html, results.html,
evaluation.html, workspace.css, review.js, results.js, workspace.js, shell.js and
necessary theme/evaluation JS adjustments. Keep /results as a compatibility alias.
Update scripts/check_frontend.cjs, scripts/export_frontend.py and
tests/test_frontend.py only as needed for the new behavior; regenerate exports.

Use the current FastAPI routes and result schema. No model/feature/threshold,
frozen evaluator, raw-data, infrastructure, new-library or public-hosting changes.
Preserve unrelated work from other chats. Any production deployment or real
recognition qualification remains a separate scope.

## Definition of done

- A juror can upload, analyze, select a finding, inspect the source frame and open
  its supplied reference in one coherent Review flow.
- The source recording remains selected through processing in the same document;
  result refresh has an explicit, honest path to reselect local playback.
- Multi-event results are scannable without duplicated large evidence panels.
- Failed/unconfirmed analysis cannot be mistaken for a clean recording.
- Evaluation leads with measured results, sample scope and real provenance.
- Both themes and target viewports pass the checks above; results are recorded.
- The live local preview and downloadable HTML reflect the same final design.
- No claim that visual polish guarantees 100 points or improves recognition.

## Authorized final usability refinement, 18 October

Human requests “make it 10/10” after the specific critique of first-use review,
evaluation emphasis and mobile KPI height. Keep the two pages, all figures,
measured scope, themes, matrix and localization unchanged. Add saved-result
entry/recovery using existing job APIs; do not package bank footage or invent a
demo verdict. Give detection visual priority and make mobile KPIs a compact 2×2
summary. Validate real browser interactions with synthetic fixtures, inspect
both themes/mobile, rebuild self-contained downloads, and report remaining limits.
