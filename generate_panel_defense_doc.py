"""Generate the comprehensive RoadSentinel Panel Defense Handbook (.docx).

Formats:
- Font: Times New Roman
- Body: 12 pt, 1.25 line spacing, 6 pt space after
- Headings: Bold, Times New Roman, styled hierarchy
- Structure: 1 to 1.5 pages per dashboard tab, separated by page breaks
- Audience: 2nd Year AIML Student level (clear analogies, accessible math, zero fluff)
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUTPUT_DOCX = Path("/home/nitin-nandakumar/Downloads/roadsentinel/RoadSentinel_Panel_Defense_Handbook.docx")

def create_document():
    doc = docx.Document()

    # Set page margins to 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Base style: Times New Roman 12pt
    normal_style = doc.styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Times New Roman'
    normal_font.size = Pt(12)
    normal_font.color.rgb = RGBColor(30, 41, 59)
    normal_style.paragraph_format.line_spacing = 1.25
    normal_style.paragraph_format.space_after = Pt(6)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(22)
        run.font.bold = True
        run.font.color.rgb = RGBColor(15, 23, 42)
        p.paragraph_format.space_after = Pt(4)

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13)
        run.font.italic = True
        run.font.color.rgb = RGBColor(100, 116, 139)
        p.paragraph_format.space_after = Pt(24)

    def add_tab_heading(tab_num, title):
        p = doc.add_paragraph()
        run = p.add_run(f"Tab {tab_num}: {title}")
        run.font.name = 'Times New Roman'
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(2, 132, 199)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(8)

    def add_subheading(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 41, 59)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)

    def add_p(text):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(6)
        # Parse simple bold markers **bold**
        parts = text.split("**")
        for i, part in enumerate(parts):
            run = p.add_run(part)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            if i % 2 == 1:
                run.font.bold = True
        return p

    def add_bullet(text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_after = Pt(4)
        parts = text.split("**")
        for i, part in enumerate(parts):
            run = p.add_run(part)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(11.5)
            if i % 2 == 1:
                run.font.bold = True
        return p

    def add_qna(q, a):
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(6)
        p_q.paragraph_format.space_after = Pt(2)
        r_q = p_q.add_run(f"Q: {q}")
        r_q.font.name = 'Times New Roman'
        r_q.font.size = Pt(12)
        r_q.font.bold = True
        r_q.font.color.rgb = RGBColor(194, 65, 12)

        p_a = doc.add_paragraph()
        p_a.paragraph_format.space_after = Pt(8)
        r_a = p_a.add_run(f"Defense Answer: {a}")
        r_a.font.name = 'Times New Roman'
        r_a.font.size = Pt(12)

    # -------------------------------------------------------------
    # COVER / HEADER
    # -------------------------------------------------------------
    add_title("RoadSentinel Examiner Defense Handbook")
    add_subtitle("A Comprehensive, Feature-by-Feature Guide for Panel Defense\nWritten for 2nd Year AI & Data Science Students\nFormat: Times New Roman 12pt, 1.25 Spacing")

    add_p("Welcome to your comprehensive speaking and defense manual for the RoadSentinel Examiner-Facing Research Dashboard. If you are preparing to present this project to an academic evaluation committee or government panel, this guide breaks down every screen, every graph, and every mathematical parameter in simple, rock-solid terms.")
    add_p("The central philosophy of RoadSentinel is simple: **we do not trust black-box AI blindly**. Instead of using a single neural network to make life-or-death decisions about road repairs, RoadSentinel divides the problem into four decoupled evidence streams (Current Detection, Optical Reliability, Multi-Day Temporal History, and Scenario Forecasting), combining them through an audited rule-based decision engine.")
    add_p("Every metric displayed on the dashboard comes from frozen, audited machine-readable JSON and CSV files stored directly on disk. The dashboard does not fabricate data, does not retrain models in the background, and does not hide failures. Below is the tab-by-tab guide to defending every single feature.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 1: SYSTEM OVERVIEW & HEADLINE RESULT CARDS
    # -------------------------------------------------------------
    add_tab_heading("1", "System Overview & Headline Result Cards")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("The System Overview tab is the executive summary of your entire project. Think of it like the dashboard of an airplane: it gives the examiners seven critical numbers right upfront so they know the exact capabilities and boundaries of your system before diving into the details.")
    add_p("Notice that each card has an audited badge: **MEASURED** (pure empirical observation from tests), **DERIVED** (calculated via statistical formula), or **PENDING** (honest engineering acknowledgment that physical testing is scheduled for the next phase).")

    add_subheading("2. Plain-English Logic of the Seven Headline Cards")
    add_bullet("**YOLO China UAV F1 (0.7104) [MEASURED]**: This is your in-domain test score. You trained a lightweight YOLOv8n detector on drone photos of Chinese highways and tested it on 480 unseen validation drone photos. F1 is the harmonic mean of precision (did it find real defects?) and recall (did it find all defects?). A score of 0.7104 means the detector strikes a solid 71% balance under familiar aerial conditions.")
    add_bullet("**YOLO India Dashcam F1 (0.0218) [MEASURED]**: This is your cross-domain stress test. You took the exact same YOLO model and tested it on 300 dashcam photos from cars driving on Indian roads without retraining. The F1 crashed by 96.94% to 0.0218! Why? Because looking down from a 30-meter drone is completely different from looking forward from a car windshield through hood reflections and perspective skew. Showing this proves to the panel that you understand model degradation.")
    add_bullet("**India Benchmark Domain Escalation (300/300, 100%) [MEASURED]**: Even though YOLO failed on Indian dashcams, your safety layer saved the day! Your DINOv2 vision transformer foundation model measured how far each Indian photo was from normal drone photos. It flagged 300 out of 300 (100%) as 'Out-of-Distribution' (OOD), routing them away from autonomous clearance.")
    add_bullet("**Familiar China Warning Rate (7/480, ~1.46%) [MEASURED]**: On familiar Chinese drone roads, did the domain gate trigger false alarms? Only 7 out of 480 photos (1.46%) were flagged. This proves your safety gate is selective and doesn't cry wolf.")
    add_bullet("**Reliability Accepted Failure Rate (9.38% @ 80% cov | 6.25% @ 50% cov) [MEASURED]**: In machine learning, this is called 'Selective Classification' (the reject option). Like a smart student who skips questions they aren't sure about rather than guessing randomly, the system rejects uncertain photos. Rejecting the bottom 20% drops errors to 9.38%; rejecting 50% drops errors to 6.25%.")
    add_bullet("**M4 Forecast MAE Skill over Persistence (+5.27%) [DERIVED]**: Pavements age over years. You compared an XGBoost machine learning model against a simple 'Persistence Baseline' (assuming tomorrow's road condition is identical to today's). XGBoost achieved a 5.27% error reduction (Mean Absolute Error = 0.0551 vs 0.0582).")
    add_bullet("**Raspberry Pi Profiling (Pending physical profiling) [PENDING]**: We refused to invent fake numbers. Embedded profiling on a physical Raspberry Pi 5 requires dedicated hardware testing scheduled for the next phase.")

    add_subheading("3. Panel Defense Talking Points")
    add_qna("Why do you show a terrible F1 score of 0.0218 right on your first page?",
            "Because real-world AI fails when camera viewpoints change. Any student can cherry-pick good results, but engineering integrity means demonstrating safety. Our contribution is showing that while the raw detector failed, our DINOv2 domain gate intercepted 100% of those failures before any government action was taken.")
    add_qna("What are the 10 Limitation Callouts at the bottom of the page?",
            "They are scientific boundary conditions. We explicitly declare that drone-to-dashcam is confounded by camera angles, our CARLA tracking data is synthetic, and rapid 30-day forecasting cannot be claimed because the US Federal Highway dataset surveys roads every 1 to 2 years.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 2: SINGLE-IMAGE ROAD ASSESSMENT
    # -------------------------------------------------------------
    add_tab_heading("2", "Single-Image Road Assessment & Defect Inspector")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("This screen is your interactive diagnostic microscope. It allows municipal road engineers or panel members to inspect any of the 40 real physical survey captures across four highway segments (SEG_001 to SEG_004, Days 01 through 10) and examine how the AI detects cracks, scores road health, and geolocates damage.")

    add_subheading("2. Core Concepts Explained for 2nd Year AIML")
    add_bullet("**YOLOv8n Architecture**: We chose YOLOv8-nano because it has only 3.2 million parameters and executes in just 2.15 milliseconds on a modern GPU. It predicts five defect classes: D00 (Longitudinal Crack), D10 (Transverse Crack), D20 (Alligator Fatigue Cracking), D40 (Pothole), and Repair Patches.")
    add_bullet("**Current Severity Score (0.0000 to 1.0000)**: How damaged is the road? If you only count bounding boxes, 10 tiny hairline cracks look worse than 1 massive pothole that destroys car tires! Our formula calculates the total defect area divided by the image area, weighted by severity: potholes have the highest weight, followed by alligator cracking, linear cracks, and repaired patches.")
    add_bullet("**DINOv2 Anomaly Distance**: For each image, Meta's DINOv2 vision transformer extracts a 384-dimensional feature vector. We measure the cosine distance to our bank of 1,921 clean asphalt drone frames. Distances below 0.4491 mean 'normal familiar road'; distances above 0.4491 trigger domain warnings.")
    add_bullet("**Live GPU Inference vs Verified Demo Mode**: In Verified Demo Mode, precomputed evaluation metrics are loaded instantly from disk. When you toggle to Live Mode, Streamlit calls PyTorch directly, running the real YOLO model on the stored image and drawing bounding boxes in real time. This proves to the panel that your model genuinely works and isn't just pre-rendered images!")

    add_subheading("3. Panel Defense Talking Points")
    add_qna("How do you handle overlapping or duplicate bounding boxes?",
            "We apply Non-Maximum Suppression (NMS) with an IoU threshold of 0.45. Furthermore, in our research analysis, we evaluated perceptual hashing (pHash) to ensure near-duplicate drone frames didn't leak between training and validation splits.")
    add_qna("Why do you have GPS geotagging on this page?",
            "Road health data is useless without spatial context. In a government deployment, every drone capture carries telemetry metadata (latitude, longitude, altitude, and solar angle) so that temporal inspections can map the exact same physical coordinates year after year.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 3: DOMAIN AWARENESS & PERCEPTION RELIABILITY
    # -------------------------------------------------------------
    add_tab_heading("3", "Perception Reliability & Selective Classification")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("Neural networks are known to be 'confidently wrong'—they will give a 95% confidence score on something they've never seen before. This tab proves how RoadSentinel measures model uncertainty and uses **Selective Prediction** to reject low-confidence predictions before they cause maintenance blunders.")

    add_subheading("2. Mathematical Parameters Explained Simply")
    add_bullet("**Risk–Coverage Curve**: Coverage (X-axis) is the percentage of inspection photos the model is allowed to clear autonomously. Risk (Y-axis) is the error rate among those cleared photos. If you force the model to classify 100% of photos, its error rate is 17.92%. If you let it reject the bottom 20% most uncertain photos (80% coverage), the accepted failure rate drops to **9.38%**. If you reject 50%, the failure rate drops to **6.25%**.")
    add_bullet("**Reconciling 8.85% vs 9.38%**: In early research notes, an 8.85% failure rate was reported. Why the difference? The 8.85% was calculated using a fixed confidence threshold (e.g. cut off at exactly probability 0.7629). But real municipal agencies don't budget by probability numbers; they budget by inspection capacity (e.g., 'we can manually review 20% of photos'). That is pure percentile coverage (80% coverage = 9.38% failure rate). Clarifying this demonstrates advanced statistical competence to the panel.")
    add_bullet("**Expected Calibration Error (ECE = 0.1075)**: If a weather forecaster says there is an 80% chance of rain on 100 different days, it should actually rain on exactly 80 of those days. ECE divides detections into confidence bins ([0.2, 0.4), [0.4, 0.6), etc.) and measures the gap between predicted confidence and actual empirical accuracy. An ECE of 0.1075 (10.75%) proves our YOLO detector is reasonably well-calibrated.")
    add_bullet("**Brier Score (0.1921)**: The mean squared error between the model's confidence probability and the binary ground truth (0 = false detection, 1 = true detection). A score of 0.1921 confirms solid probabilistic accuracy.")

    add_subheading("3. Panel Defense Talking Points")
    add_qna("What is the practical value of selective prediction for a city government?",
            "City maintenance departments cannot afford to manually inspect every kilometer of road, nor can they afford autonomous AI hallucinating repairs. By setting coverage to 80%, 80% of road photos are processed automatically with high accuracy (under 9.38% failure), while the remaining 20% ambiguous photos are routed to human engineers.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 4: XGBOOST DETERIORATION FORECASTING (PHASE 1C)
    # -------------------------------------------------------------
    add_tab_heading("4", "Pavement Deterioration Forecasting (FHWA LTPP)")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("Pavements degrade over years due to heavy truck axle loading, rain, and temperature extremes. This tab presents our Phase 1C machine learning study using the US Federal Highway Administration's Long-Term Pavement Performance (LTPP) dataset, comparing XGBoost models against authoritative persistence baselines.")

    add_subheading("2. Core Concepts & Parameters Explained")
    add_bullet("**The M0 Persistence Baseline**: In pavement engineering, persistence assumes that the road condition next year will be identical to today's condition. Because asphalt degrades slowly, persistence is surprisingly accurate: Mean Absolute Error (MAE) of **0.0582** and R² = 0.9125! Any machine learning model MUST beat persistence to justify its computational existence.")
    add_bullet("**M3 Temporal XGBoost**: Uses current condition plus past rate of deterioration (velocity). MAE drops to **0.0563** (a +3.26% skill improvement over persistence).")
    add_bullet("**M4 Combined XGBoost (Temporal + Scenario)**: Uses condition, history, AND environmental stress factors (daily heavy truck traffic AADTT, rainfall, freeze-thaw cycles, moisture exposure). MAE drops to **0.0551** (a **+5.27%** skill improvement over persistence).")
    add_bullet("**Paired Clustered 95% Confidence Interval ([-0.0106, +0.0043])**: Multiple road observations come from the same highway section, so their errors are correlated. We used cluster-robust bootstrapping to compute a 95% confidence interval for the error difference (M4 minus Persistence). Because the interval crosses zero (ranges from negative to positive), we **cannot** claim that M4 is universally superior across all highway sites. This is honest scientific rigor!")
    add_bullet("**The Horizon Split (< 365 days vs > 365 days)**: On short intervals (< 1 year, N=43), persistence actually beats XGBoost by 3.77%! Why? Because over 6 months, road changes are mostly sensor measurement noise. But on long intervals (> 1 year, N=35), XGBoost beats persistence by **+9.12% to +10.15%**!")
    add_bullet("**The Operational Router**: We don't blindly run XGBoost on everything! If the forecast horizon is under 365 days or the road has fewer than 2 prior inspections, we route to Persistence. If the horizon is over 365 days and has sufficient history, we route to M4 XGBoost.")

    add_subheading("3. Panel Defense Talking Points")
    add_qna("Why does the dashboard state 'Do not claim validated 30-90 day rapid forecasting'?",
            "Because real highway profile surveys in the LTPP database occur on average every 429.5 days (range: 31 to 728 days). Simulating 30-day or 60-day rapid forecasts is a model extrapolation, not an empirically validated rapid survey cadence.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 5: TEMPORAL ROAD MONITORING (CARLA EVALUATION)
    # -------------------------------------------------------------
    add_tab_heading("5", "CARLA/Unreal Synthetic Temporal Road Monitoring")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("To test repeated multi-day camera inspections without waiting five years for physical asphalt to crack, we built a high-fidelity synthetic benchmark in CARLA/Unreal Engine. This tab tracks how individual defect candidates evolve across repeated drone flights under shifting sun angles, overcast diffuse light, and rain puddles.")

    add_subheading("2. Core Concepts & Parameters Explained")
    add_bullet("**Greedy Bipartite Tracking Algorithm**: How does the computer know a crack on Day 4 is the same physical crack seen on Day 3? We match bounding regions in strict priority order: First, Mask IoU >= 0.50; if not met, Bounding Box IoU >= 0.30; if not met, Centroid Distance <= 75 pixels with an Area Ratio <= 3.0x.")
    add_bullet("**Summary Numbers**: 40 physical captures across 8 compatible sequences. The tracker identified **48 total defect tracks**, of which **22 were persistent** across multiple days (a **45.83% persistence rate**). There were **33 matched transitions** (14 where area increased by >15%, and 19 where area decreased by >15%).")
    add_bullet("**The Strict Semantic Rule: NOT_OBSERVED != REPAIRED**: This is your most important defense phrase! If a crack is visible on Day 5, but on Day 6 a heavy rain puddle or sun glare obscures it, the detector might miss it. A naive AI would say: 'The crack disappeared, the road repaired itself!' RoadSentinel strictly classifies this as NOT_OBSERVED. Pavements never magically heal.")
    add_bullet("**Why results are labeled 'Synthetic Functionality'**: We explicitly tell the panel that these CARLA results demonstrate tracking algorithm capability, not real-world civil engineering deterioration.")

    add_subheading("3. Panel Defense Talking Points")
    add_qna("Why did 19 defect regions exhibit area decreases if pavement only gets worse?",
            "Because in computer vision, camera lighting modulates detected contrast! On Day 10, low-angle sunset glare washed out pavement contrast, causing the edge detector to segment a smaller region. This proves that vision models detect perceived area, not ground-truth physical area.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 6: PERCEPTION BENCHMARK (YOLO vs DINO+SAM)
    # -------------------------------------------------------------
    add_tab_heading("6", "Perception Benchmark: YOLOv8n vs DINOv2 + SAM2")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("In modern AI research, there is huge hype around Foundation Models (like Meta's DINOv2 and Segment Anything SAM 2). This tab directly answers the research question: Can a zero-shot foundation model replace a supervised YOLO detector for road inspection without any training?")

    add_subheading("2. The Shocking Empirical Findings")
    add_bullet("**The Numbers**: On the RDD2022 China_Drone benchmark (480 images, 742 ground-truth defects):")
    add_bullet("  - **YOLOv8n (Supervised)**: F1 = **0.7104** (Precision 0.6568, Recall 0.7736), Latency = **3.62 ms**, Throughput = **276.2 FPS**.")
    add_bullet("  - **DINOv2 + SAM2 (Zero-shot Foundation)**: F1 = **0.0267** (Precision 0.0139, Recall 0.5000), Latency = **263.15 ms**, Throughput = **3.80 FPS**.")
    add_bullet("**Why DINOv2 + SAM2 Completely Failed (F1 = 0.0267)**:")
    add_bullet("  1. DINOv2 splits images into 14 x 14 pixel patches to extract semantic features. It is trained to spot semantic objects (like cars, trees, people). On asphalt, rough gravel aggregates, oil stains, and paint line edges are all 'different' from clean asphalt.")
    add_bullet("  2. DINOv2 produced **1,055 false-positive anomaly prompts** on ordinary gravel textures!")
    add_bullet("  3. SAM 2 then dutifully drew crisp, high-confidence segmentation masks around normal gravel! Result: Precision collapsed to 1.39%.")
    add_bullet("  4. Hairline cracks are only 2 to 5 pixels wide—they get blurred out inside a 14 x 14 patch.")
    add_bullet("**Duplicate-Filtered Validation F1 (0.7022)**: To ensure our YOLO didn't cheat by memorizing images, we used perceptual hashing (pHash) to find near-identical frames between training and validation splits. Removing 24 connected frames barely changed the F1 (0.7104 to 0.7022), proving zero data leakage.")

    add_subheading("3. Panel Defense Talking Points")
    add_qna("Does this mean DINOv2 is completely useless in your project?",
            "No! DINOv2 is terrible at micro defect localization, but as we show in Tab 7, it is phenomenal at macro scene-level domain shift detection. We use each model for its true strength: YOLO for local defect boxes, DINOv2 for global camera safety gating.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 7: CROSS-DOMAIN GENERALIZATION & DOMAIN GATE
    # -------------------------------------------------------------
    add_tab_heading("7", "Cross-Domain Generalization & DINOv2 Domain Gate")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("What happens when you deploy an AI trained in one country on roads in another country? This tab evaluates the zero-shot transfer of our Chinese drone model onto 300 Indian road dashcam captures and shows how our DINOv2 domain gate acts as a safety firewall.")

    add_subheading("2. Core Concepts & Parameters Explained")
    add_bullet("**The Cross-Domain Collapse**: YOLO F1 plummeted from **0.7104 down to 0.0218** (an absolute drop of -0.6886, a 96.94% collapse). Under strict quality criteria (F1 >= 0.50), **293 out of 300 Indian frames failed completely**.")
    add_bullet("**The Confounding Problem**: We explain to the panel that this was not just a geographic difference. It was confounded by camera viewpoint: the drone looked straight down at 90° without obstacles; the dashcam was mounted on a car hood looking forward at 15°, capturing sky glare, hood reflections, and distant perspective foreshortening.")
    add_bullet("**The DINOv2 Domain Gate Firewall**: DINOv2 extracts a global 384-dimensional CLS token from each image and measures its cosine distance to a memory bank of 1,921 clean Chinese drone photos:")
    add_bullet("  - Familiar China distribution: mean distance ≈ 0.32, maximum = 0.5247.")
    add_bullet("  - India dashcam distribution: mean distance ≈ 0.81, minimum = 0.6405.")
    add_bullet("  - Operational Threshold Line (p99 = 0.4491): Any image with distance > 0.4491 is quarantined.")
    add_bullet("  - Separation Margin gap: **+0.1158** between the highest China frame and lowest India frame.")
    add_bullet("  - **AUROC = 1.0000**: Perfect macro separation on this benchmark. 300 out of 300 Indian images were safely quarantined, with only 7 false alarms on China drone photos.")

    add_subheading("3. Panel Defense Talking Points")
    add_qna("Can you claim this DINOv2 gate detects ANY out-of-distribution road in the world?",
            "No, and we explicitly put a warning on this page stating that this demonstrates separation specifically between the evaluated China-UAV and India-dashcam benchmark. It is not a universal out-of-distribution detector for all possible road types.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 8: DECISION ENGINE & POLICY ABLATION
    # -------------------------------------------------------------
    add_tab_heading("8", "Reliability-Aware Decision Engine & Policy Ablation")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("A municipal government does not repair a road based on raw bounding boxes. You need a policy engine that takes all evidence streams (severity, reliability, domain gate, temporal history) and routes each capture into an actionable maintenance tier.")

    add_subheading("2. The Five Action Tiers & Policy Ablation")
    add_bullet("**The 5 Action Tiers**:")
    add_bullet("  1. `DOMAIN_ESCALATION` (Red): Vision domain is uncalibrated. Stop automated processing; call an engineer.")
    add_bullet("  2. `PRIORITY_REVIEW` (Orange): Severe defect or accelerating deterioration confirmed by high-reliability perception.")
    add_bullet("  3. `REINSPECT` (Yellow): Low perception reliability or rain/glare confounds. Re-survey the section.")
    add_bullet("  4. `MONITOR` (Blue): Low/moderate severity, stable temporal trend. Normal annual survey.")
    add_bullet("  5. `AUTOMATED_ACCEPT` (Green): Pristine road confirmed under high reliability with zero forecast risk.")
    add_bullet("**The Audited Policy Ablation (India Benchmark)**: We compared what happens under different system architectures on the 300 failed Indian frames:")
    add_bullet("  - **YOLO-Only Baseline**: Accepts all 300 frames autonomously, causing **293 UNSAFE AUTOMATED ACCEPTS** (giving automated clearance to broken roads!).")
    add_bullet("  - **Full RoadSentinel Policy**: Routes all 300 frames to Domain Escalation, resulting in **0 UNSAFE AUTOMATED ACCEPTS**.")
    add_bullet("**Mandatory Defense Phrase**: *'Zero unsafe automated accepts were observed on the evaluated India benchmark.'* We also clarify: *'Escalation is safe routing, not successful defect detection.'*")

    add_subheading("3. Panel Defense Talking Points")
    add_qna("Isn't routing 100% of Indian photos to human review a failure of automation?",
            "In safety-critical engineering, safe refusal is a massive success. When an autonomous car encounters heavy snow that blinds its cameras, the safe action is to disengage and alert the driver, not drive blindly off a cliff. Routing uncalibrated images to human review prevented 293 hazardous maintenance errors.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 9: FAILURE MODE & SENSITIVITY TAXONOMY
    # -------------------------------------------------------------
    add_tab_heading("9", "Failure Mode Taxonomy & Sensitivity Analysis")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("Engineering professors love testing whether students know when their models break. This tab documents the six primary physical and optical failure modes of the system.")

    add_subheading("2. The Six Failure Modes Explained")
    add_bullet("**1. Coarse Asphalt Aggregate False Positives**: In vision transformers, rough stone aggregate chips stand out against binder mastic, generating false anomaly prompts.")
    add_bullet("**2. Road Marking Over-Suppression**: In preprocessing, filters designed to eliminate bright white and yellow lane paint sometimes over-filter, accidentally erasing true longitudinal cracks that run parallel to painted lane lines.")
    add_bullet("**3. Low-Angle Sun Glare Blindness**: During Days 09–10 of our survey, sunset glare (<35° solar angle) reflected off the pavement surface, wiping out contrast and causing YOLO to miss genuine cracks.")
    add_bullet("**4. Rain Puddle Contrast Inflation**: On Days 07–08, standing rainwater filled depressions, creating dark, high-contrast reflective puddles that YOLO mistook for potholes or alligator cracking.")
    add_bullet("**5. Perspective Foreshortening**: In dashcam imagery, distant road defects occupy only 2–3 pixels and are foreshortened, making geometric bounding impossible.")
    add_bullet("**6. Resolution Boundary in Foundation Models**: DINOv2's 14 x 14 patch size cannot resolve 2-pixel hairline cracks.")

    add_subheading("3. Panel Defense Talking Points")
    add_qna("How would you fix the sun glare and rain puddle failure modes in production?",
            "By integrating camera polarizing filters to cut glare and using the metadata sidecars to flag wet pavement conditions, automatically routing rainy or sunset captures to the REINSPECT tier.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 10: RESEARCH METHODOLOGY & PROTOCOLS
    # -------------------------------------------------------------
    add_tab_heading("10", "Research Methodology & Scientific Protocols")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("This tab acts as your experimental audit trail, documenting the data splits, cross-validation groupings, and reproducibility parameters used across the study.")

    add_subheading("2. Key Methodological Highlights")
    add_bullet("**Perceptual Hashing (pHash) Deduplication**: In sequential video surveys, consecutive video frames can be 99% identical. If frame N is in the training set and frame N+1 is in the test set, your model gets an artificially inflated test score! We computed perceptual hashes (64-bit DCT hashes) and verified that excluding near-duplicate frames (pHash <= 5) did not degrade test F1 (0.7104 vs 0.7022).")
    add_bullet("**5-Fold Grouped Cross-Validation by Site ID**: In our forecasting study, multiple survey records came from the same highway section. If you do standard random train-test splitting, data from Section A appears in both training and testing. We strictly grouped folds by site_id, ensuring test folds contained only completely unseen highway sections.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 11: CONTROLLED EXPERIMENT B (PLANNED PROTOCOL)
    # -------------------------------------------------------------
    add_tab_heading("11", "Controlled Experiment B [Planned Protocol]")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("In Experiment A, multiple factors varied simultaneously (drone altitude, lighting, weather). Experiment B is your frozen scientific specification for the next research iteration (Segments SEG_005 and SEG_006) to isolate individual variables.")

    add_subheading("2. Experimental Controls Explained")
    add_bullet("**Fixed Nadir Drone Angle**: 100% top-down perpendicular survey (90° pitch) from a constant 30m altitude.")
    add_bullet("**Constant Clear Noon Lighting**: Constant 70° solar angle to completely eliminate shadow and glare confounds.")
    add_bullet("**Formal Pavement Repair Intervention**: A defect grows naturally for 5 days. On Day 06, a formal asphalt patch repair is applied. Days 07 through 10 evaluate whether the tracking algorithm correctly observes patch retention and re-deterioration.")
    add_bullet("**Honest Status Badge**: Marked as **PLANNED / DEFERRED FOR NEXT REVIEW**. You do not show fabricated outputs; you show a frozen scientific experimental design.")

    doc.add_page_break()

    # -------------------------------------------------------------
    # TAB 12: EDGE DEPLOYMENT BENCHMARKS & RASPBERRY PI
    # -------------------------------------------------------------
    add_tab_heading("12", "Edge Deployment & Raspberry Pi Profiling")
    add_subheading("1. What This Screen Is & Why It Exists")
    add_p("Can RoadSentinel run on an embedded device mounted on a drone or survey vehicle? This tab benchmarks inference latency across hardware architectures and lays out the physical testing roadmap for the Raspberry Pi 5.")

    add_subheading("2. Audited Runtime Benchmarks Explained")
    add_bullet("**YOLO-Only Detection**: **2.15 ms per frame** (throughput: **465.3 FPS**) on an NVIDIA RTX 5060 Laptop GPU (FP32 precision, 512x512 input, batch 1, 20 warmup runs, 100 timed runs, model loading excluded). This is blazing fast and easily real-time.")
    add_bullet("**Staged Perception Pipeline (YOLO + DINOv2 Gate)**: **28.41 ms per frame** (throughput: **35.2 FPS**). Because the DINOv2 domain gate runs in 25.9 ms, the combined gated pipeline runs comfortably above the standard 30 FPS video threshold on an edge GPU.")
    add_bullet("**Full Integrated Policy (with SAM 2)**: **291.65 ms per frame** (throughput: **3.43 FPS**). Foundation vision segmentation models are heavy! Full segmentation cannot run in real time on edge hardware and is designed for post-flight server processing.")
    add_bullet("**Raspberry Pi 5 Platform Target**: Quad-core ARM Cortex-A76 @ 2.4 GHz with 8GB LPDDR4X RAM. Projected to achieve 20–30 FPS on YOLOv8n using INT8 quantization with ONNX Runtime or NCNN vector acceleration.")
    add_bullet("**Pending Status Banner**: Clearly labeled as **PENDING PHYSICAL PROFILING** (expected source: artifacts/edge/raspberry_pi5_profiling.json). We present the pre-allocated schema without fabricating simulated on-device numbers.")

    add_subheading("3. Panel Defense Talking Points")
    add_qna("Can you run DINOv2 and SAM 2 on a Raspberry Pi 5 CPU?",
            "No. Vision transformers require heavy floating-point attention operations that would cause the 4-core Cortex-A76 to choke at under 0.5 FPS. In production, our architecture deploys YOLOv8n INT8 on the Pi for real-time capture, while DINOv2 and SAM 2 are executed either on an attached edge accelerator (such as a Hailo-8 M.2 module) or offloaded to a base station.")

    # Save document
    doc.save(OUTPUT_DOCX)
    print(f"Document successfully created at {OUTPUT_DOCX}")

if __name__ == "__main__":
    create_document()
