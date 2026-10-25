from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "output" / "pdf" / "Lab11C_Methodological_Defense.pdf"


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(0.72 * inch, 0.42 * inch, "961731 Business Data Analytics - Lab 11C")
    canvas.drawRightString(7.78 * inch, 0.42 * inch, f"Page {document.page}")
    canvas.restoreState()


def paragraph(text, style):
    return Paragraph(text, style)


def main() -> None:
    test = pd.read_csv(ROOT / "artifacts" / "results" / "test_method_metrics.csv")
    freeze = json.loads((ROOT / "artifacts" / "frozen_experiment_manifest.json").read_text(encoding="utf-8"))
    bootstrap = pd.read_csv(ROOT / "artifacts" / "results" / "bootstrap_summary.csv")
    paired = bootstrap[(bootstrap.Type == "paired_difference") & (bootstrap.Metric == "NDCGAt5")].iloc[0]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=21, textColor=colors.black, spaceAfter=10, alignment=TA_CENTER)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=12, textColor=colors.HexColor("#444444"), spaceAfter=10, alignment=TA_CENTER)
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=colors.black, spaceBefore=7, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.2, textColor=colors.black, spaceAfter=5)
    small = ParagraphStyle("Small", parent=body, fontSize=8.2, leading=10.2, spaceAfter=2)

    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=letter, leftMargin=0.72 * inch, rightMargin=0.72 * inch,
        topMargin=0.62 * inch, bottomMargin=0.62 * inch,
        title="Lab 11C Methodological Defense", author="Course 961731 submission",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=footer))

    metrics = []
    for row in test.itertuples(index=False):
        metrics.append([
            row.method.replace("matrix_factorization", "factor"),
            f"{row.precision_at_5:.3f}", f"{row.recall_at_5:.3f}", f"{row.ndcg_at_5:.3f}",
            f"{row.distinct_recommended_items}/70",
        ])
    metric_table = Table(
        [["Method", "P@5", "R@5", "NDCG@5", "Coverage"]] + metrics,
        colWidths=[2.2 * inch, 0.72 * inch, 0.72 * inch, 0.85 * inch, 0.95 * inch],
        repeatRows=1,
    )
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#243B53")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8.2),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D9D9D9")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F3F6F9")]),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ]))

    story = [
        paragraph("Lab 11C Methodological Defense", title),
        paragraph("Hybrid Recommendation and Validation | Configuration " + freeze["configuration_hash"][:12], subtitle),
        paragraph("Conclusion", heading),
        paragraph("The experiment is reproducible and temporally clean under declared independent conventions, but it does not show reliable model superiority. The final test has only nine eligible users; four methods score zero on every ranking metric, while content and hybrid each obtain one hit. That is a fragile difference, not a deployment verdict.", body),
        paragraph("Modeling choices", heading),
        paragraph("The six methods share exactly 70 training-known candidates and exclude each user's training-seen items. Popularity ranks by training counts. Content uses logarithmic TF, unsmoothed IDF, L2 normalization, and rating-weighted user profiles. User and item CF use observed masks, centered co-rated cosine, minimum overlap two, shrinkage ten, positive similarities, and deterministic neighbors. Unsupported requests and unknown users use popularity. The hybrid min-max scales content and factor scores per request; a constant component becomes zero.", body),
        paragraph("Factor objective and regularization", heading),
        paragraph("Prediction is the training mean plus user bias, item bias, and the user-item factor dot product. The loss is one-half mean observed squared error plus one-half lambda times the squared norms of biases and factors. Full-batch simultaneous gradients use the old state and preserve this scaling. The required one-event check exactly gives prediction 3.24, residual 0.76, updated biases 0.2370 and -0.0615, user factors [0.2180, 0.4018], and item factors [0.5051, 0.1147].", body),
        paragraph("Validation and freeze", heading),
        paragraph("Twelve predeclared factor candidates were compared on 17 eligible validation ratings; the selected state uses 8 factors, learning rate 0.2, lambda 0.01, 300 epochs, and seed 202411. All five hybrid alphas tied on validation ranking metrics, so the predeclared balanced tie rule selected 0.50. Before test evaluation, the manifest froze policies, metrics, software versions, data hashes, state checksums, and tie rules. The test was never used to retune or refit.", body),
        paragraph("Untouched test results", heading),
        metric_table,
        Spacer(1, 3),
        paragraph("Rating error is meaningful only for the calibrated factor predictor here: 18 warm-user, known-item test events are eligible. Ranking and rating denominators are intentionally separate.", small),
        PageBreak(),
        paragraph("Cold start and reference reproduction", heading),
        paragraph("Two of nine ranking-eligible users are cold; all methods route them to popularity. Both CF methods also lack supported neighborhoods for every evaluable test user and therefore fall back for all nine. Future-only products are excluded from the proxy catalog and reported separately. The assignment reports alpha 0.75, all-zero test ranking metrics, and coverages of 5/70, 19/70, 12/70, and 22/70 for popularity/CF, content, factors, and hybrid. This implementation matches nine users, CF fallback, popularity/CF zeros, and 5/70 coverage. It differs because content and hybrid hit item 552 for user 33, cover 29/70, and factors cover 11/70. Missing Week 12 materials, the companion script, reference notebook, and original outputs prevent exact reconstruction; the differences are not hidden.", body),
        paragraph("Leakage and uncertainty", heading),
        paragraph("Popularity, means, biases, vocabulary, IDF, rating matrix, profiles, and factors end at 20 May 2024 18:42:23 UTC. Hyperparameter and alpha decisions end at 12 June 2024 15:54:30 UTC. The candidate set has zero future-only overlap. An isolated contaminated copy using a test timestamp fails the leakage assertion as intended and cannot mutate the real experiment.", body),
        paragraph(f"The 10,000-replicate user-cluster bootstrap resamples each user's complete contribution. The paired hybrid-minus-popularity NDCG@5 difference is {paired.PointEstimate:.4f}, with a percentile interval [{paired['CI2.5']:.4f}, {paired['CI97.5']:.4f}]. The interval includes zero. With only nine clusters, endpoints are discrete and unstable; recommendation positions are not independent observations.", body),
        paragraph("Deployment limits", heading),
        paragraph("The catalog text is synthetic, exposure was not logged, non-interactions are unknown rather than negatives, product availability dates are absent, and offline relevance does not estimate incremental conversion or margin. A production system would need availability filtering, monitoring, retraining policy, latency budgets, privacy controls, calibration checks, and failure-safe fallbacks. The current evidence supports a classroom comparison, not an automated rollout.", body),
        paragraph("Prespecified online test", heading),
        paragraph("Randomize pseudonymous users 1:1 with stable assignment. Control is the current policy, or frozen popularity if no policy exists; treatment is the frozen alpha-0.50 hybrid. The single primary outcome is purchase conversion within seven days of first eligible exposure. Illustratively, a 10% baseline, 2 percentage-point absolute MDE, two-sided 5% alpha, and 80% power require 3,841 users per arm; 5% inflation gives 8,086 exposed users. Run at least 14 days and two weekly cycles, and stop only after both duration and sample requirements are met unless a prespecified safety, privacy, or reliability threshold fires.", body),
        paragraph("Guardrails are service errors, p95 latency, opt-out or complaint rate, refund/cancellation rate, and gross margin per exposed user. Logs must distinguish assignment, generated rankings, rendered/viewable exposure, outcome timestamps, and model hash. Apply purpose limitation, minimization, access control, encryption, retention/deletion, pseudonymization, and documented Thai PDPA review. Only randomized exposure can estimate causal business uplift.", body),
    ]
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
