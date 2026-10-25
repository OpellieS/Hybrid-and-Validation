# Open Issues Register

| ID | Issue | Impact | Planned handling |
|---|---|---|---|
| OI-01 | Week 12 PDF and named `Recommendation System.pptx` are absent | Exact teaching conventions cannot be independently read | Use assignment wording and clearly declare formula variants |
| OI-02 | Companion script is absent | Exact candidate grid, CF support settings, fallback/tie details, and calculation authority cannot be reproduced | Predeclare settings before validation; compare final results honestly |
| OI-03 | Reference notebook and original reference outputs are absent | Cannot byte-for-byte or trace-for-trace reproduce instructor execution | Preserve independent outputs and compare only with numbers printed in the assignment |
| OI-04 | Student ID is absent | Required final ZIP cannot receive its final submission name | Create `961731_STUDENTID_REQUIRED_Lab11C.zip`; user must rename after inserting the actual ID |
| OI-05 | Source gives no exposure or collection mechanism | Offline ratings cannot support causal uplift claims | State this in the methodological defense and A/B proposal |
| OI-06 | Word-to-image bundled renderer is not exposed in this Windows workspace | Source DOCX and final PDF need alternate visual QA | Use content/structure extraction for source; use PDFium/Poppler-equivalent rasterization for final PDF and record the renderer |

## Final disposition

- OI-01, OI-02, and OI-03 remain unresolved external-resource limitations. The independent variant is complete and does not claim exact reproduction.
- OI-04 remains the only action required to finalize the submission filename.
- OI-05 is addressed in the defense and A/B proposal: offline metrics are not described as causal uplift.
- OI-06 is resolved for the final PDF. PDFium rendered both pages at 2x scale and both images passed visual inspection. The source DOCX was read structurally because it was an input, not a delivered edited document.
