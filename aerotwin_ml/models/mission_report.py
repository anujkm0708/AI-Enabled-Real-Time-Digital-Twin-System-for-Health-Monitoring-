"""
AeroTwin - Mission Report Generator
======================================
Generates PDF and CSV mission debrief reports from stored telemetry tick history.

PDF: one-page "Mission Health Debrief" with:
  - Mission metadata (ID, duration, phase breakdown)
  - Peak sensor readings vs normal range
  - Final RUL and overall health score
  - Fault events table (when each fault was detected, type, severity)
  - Subsystem health index summary

CSV: full tick-by-tick data export for offline analysis / MATLAB import.

Requires: reportlab  (pip install reportlab)
"""

import csv
import io
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


def export_csv(mission_ticks: List[Dict[str, Any]], out_path: Optional[str] = None) -> bytes:
    """
    Export full tick history to CSV.

    Parameters
    ----------
    mission_ticks : list of dicts, each being the full JSON payload stored per tick
    out_path      : if given, also writes to disk; always returns the CSV bytes

    Returns
    -------
    bytes — UTF-8 encoded CSV content
    """
    if not mission_ticks:
        return b""

    # Flatten nested dicts from the payload
    def flatten(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
            if isinstance(v, dict):
                out.update(flatten(v, key))
            elif isinstance(v, list):
                out[key] = str(v)
            else:
                out[key] = v
        return out

    rows = [flatten(tick) for tick in mission_ticks]
    # Collect all column names in order
    all_cols = []
    seen = set()
    for row in rows:
        for col in row.keys():
            if col not in seen:
                all_cols.append(col)
                seen.add(col)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=all_cols, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in all_cols})

    content = buf.getvalue().encode("utf-8")
    if out_path:
        with open(out_path, "wb") as f:
            f.write(content)
    return content


def export_pdf(mission_id: str, mission_ticks: List[Dict[str, Any]],
               out_path: Optional[str] = None) -> bytes:
    """
    Generate a one-page Mission Health Debrief PDF.

    Parameters
    ----------
    mission_id    : str — mission identifier
    mission_ticks : list of full tick payloads (each is the WebSocket JSON object)
    out_path      : if given, also writes to disk

    Returns
    -------
    bytes — PDF content
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        raise ImportError("reportlab is required for PDF export. Run: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=18, textColor=colors.HexColor("#1e3a5f"),
                                  spaceAfter=4)
    subtitle_style = ParagraphStyle("subtitle", parent=styles["Normal"],
                                    fontSize=10, textColor=colors.HexColor("#64748b"),
                                    spaceAfter=12)
    section_style = ParagraphStyle("section", parent=styles["Heading2"],
                                    fontSize=12, textColor=colors.HexColor("#1e3a5f"),
                                    spaceBefore=12, spaceAfter=4)
    body_style = styles["Normal"]

    # ---- Collect summary statistics ----
    def _get(tick, *keys, default=None):
        d = tick
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k, default)
        return d

    n_ticks = len(mission_ticks)
    duration_s = _get(mission_ticks[-1], "telemetry", "t_s", default=n_ticks) if n_ticks else 0
    duration_str = f"{int(duration_s // 60)}m {int(duration_s % 60)}s" if duration_s else "N/A"

    # Peak telemetry values
    peak_cht   = max((_get(t, "telemetry", "cht_c", default=0) for t in mission_ticks), default=0)
    peak_egt   = max((_get(t, "telemetry", "egt_c", default=0) for t in mission_ticks), default=0)
    min_oil_p  = min((_get(t, "telemetry", "oil_press_bar", default=99) for t in mission_ticks), default=0)
    peak_vib   = max((_get(t, "telemetry", "vibration_g", default=0) for t in mission_ticks), default=0)
    max_oil_t  = max((_get(t, "telemetry", "oil_temp_c", default=0) for t in mission_ticks), default=0)

    final_rul   = _get(mission_ticks[-1], "rul_hours", default="N/A") if n_ticks else "N/A"
    final_hi    = _get(mission_ticks[-1], "health_indices", "overall_score", default="N/A") if n_ticks else "N/A"
    final_histr = _get(mission_ticks[-1], "health_indices", "overall_status", default="N/A") if n_ticks else "N/A"

    # Fault events: collect distinct fault transitions
    fault_events = []
    last_fault = "none"
    for tick in mission_ticks:
        fa = _get(tick, "telemetry", "fault_active", default="none")
        fp = _get(tick, "fault", "predicted_fault", default="none")
        ts = _get(tick, "telemetry", "t_s", default=0)
        conf = _get(tick, "fault", "confidence", default=0)
        if fp != last_fault and fp != "none":
            fault_events.append({
                "t_s": ts, "fault": fp,
                "confidence": conf,
                "actual": fa,
            })
        last_fault = fp

    # Phase durations
    phase_times: Dict[str, float] = {}
    for tick in mission_ticks:
        ph = _get(tick, "telemetry", "phase", default="unknown")
        phase_times[ph] = phase_times.get(ph, 0) + 1  # 1 tick = 1 second

    # Health indices (last tick)
    hi_last = _get(mission_ticks[-1], "health_indices") if n_ticks else {}

    # ---- Build PDF content ----
    story = []

    # Header
    story.append(Paragraph("✈  AeroTwin Mission Health Debrief", title_style))
    story.append(Paragraph(
        f"Mission ID: <b>{mission_id}</b>  |  "
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1e3a5f"), spaceAfter=8))

    # Mission summary table
    story.append(Paragraph("Mission Summary", section_style))
    summary_data = [
        ["Parameter", "Value"],
        ["Mission ID", mission_id],
        ["Total Duration", duration_str],
        ["Total Ticks", str(n_ticks)],
        ["Final RUL", f"{final_rul:.1f} hrs" if isinstance(final_rul, (int, float)) else str(final_rul)],
        ["Overall Health Score", f"{final_hi}/100 ({final_histr})" if isinstance(final_hi, (int, float)) else "N/A"],
        ["Fault Events Detected", str(len(fault_events))],
    ]
    t = Table(summary_data, colWidths=[7*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    # Peak readings
    story.append(Paragraph("Peak Sensor Readings", section_style))
    peak_data = [
        ["Parameter",         "Peak Value",    "Normal Range",     "Status"],
        ["CHT (°C)",          f"{peak_cht:.1f}","< 180°C",          "⚠ WARN" if peak_cht > 180 else "✓ OK"],
        ["EGT (°C)",          f"{peak_egt:.1f}","< 700°C",          "⚠ WARN" if peak_egt > 700 else "✓ OK"],
        ["Min Oil Pressure",  f"{min_oil_p:.2f} bar","≥ 2.5 bar",   "✗ CRIT" if min_oil_p < 1.5 else ("⚠ WARN" if min_oil_p < 2.5 else "✓ OK")],
        ["Peak Vibration",    f"{peak_vib:.2f} g","< 2.5 g",        "✗ CRIT" if peak_vib > 3.5 else ("⚠ WARN" if peak_vib > 2.5 else "✓ OK")],
        ["Peak Oil Temp",     f"{max_oil_t:.1f}°C","< 100°C",       "⚠ WARN" if max_oil_t > 100 else "✓ OK"],
    ]

    def status_color(row):
        s = row[3]
        if "CRIT" in s: return colors.HexColor("#fee2e2")
        if "WARN" in s: return colors.HexColor("#fef3c7")
        return colors.white

    t2 = Table(peak_data, colWidths=[5.5*cm, 4*cm, 4*cm, 3.5*cm])
    row_styles = [
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i, row in enumerate(peak_data[1:], 1):
        c = status_color(row)
        if c != colors.white:
            row_styles.append(("BACKGROUND", (0, i), (-1, i), c))
    t2.setStyle(TableStyle(row_styles))
    story.append(t2)

    # Subsystem health indices
    if hi_last:
        story.append(Paragraph("Subsystem Health Indices (End of Mission)", section_style))
        sub_data = [["Subsystem", "Score", "Status"]]
        for sub in ["thermal", "lubrication", "combustion", "mechanical_vibration", "electrical"]:
            if sub in hi_last:
                sub_data.append([
                    sub.replace("_", " ").title(),
                    f"{hi_last[sub]['score']}/100",
                    hi_last[sub]["status"].upper(),
                ])
        if isinstance(hi_last.get("overall_score"), (int, float)):
            sub_data.append(["OVERALL", f"{hi_last['overall_score']}/100",
                             hi_last.get("overall_status", "").upper()])

        def hi_row_color(row):
            s = row[2].lower()
            if s == "critical": return colors.HexColor("#fee2e2")
            if s == "poor":     return colors.HexColor("#ffedd5")
            if s == "degraded": return colors.HexColor("#fef3c7")
            if s == "good":     return colors.HexColor("#dcfce7")
            return colors.HexColor("#f0fdf4")

        t3 = Table(sub_data, colWidths=[7*cm, 4*cm, 6*cm])
        hi_styles = [
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for i, row in enumerate(sub_data[1:], 1):
            hi_styles.append(("BACKGROUND", (0, i), (-1, i), hi_row_color(row)))
        t3.setStyle(TableStyle(hi_styles))
        story.append(t3)

    # Fault events
    story.append(Paragraph("Fault Events", section_style))
    if fault_events:
        fe_data = [["Time (s)", "Predicted Fault", "Confidence", "Actual"]]
        for ev in fault_events[:20]:   # cap at 20 rows
            fe_data.append([
                f"{ev['t_s']:.0f}",
                ev["fault"].replace("_", " ").title(),
                f"{ev['confidence']*100:.0f}%",
                ev["actual"].replace("_", " ").title(),
            ])
        t4 = Table(fe_data, colWidths=[3*cm, 6*cm, 3.5*cm, 4.5*cm])
        t4.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#991b1b")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#fff1f2"), colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t4)
    else:
        story.append(Paragraph("No fault events detected during this mission.", body_style))

    # Phase breakdown
    story.append(Paragraph("Phase Duration Breakdown", section_style))
    ph_data = [["Phase", "Duration (s)"]]
    for ph, secs in sorted(phase_times.items()):
        ph_data.append([ph.title(), str(int(secs))])
    t5 = Table(ph_data, colWidths=[7*cm, 10*cm])
    t5.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t5)

    # Footer
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cbd5e1")))
    story.append(Paragraph(
        "<font size=8 color='#94a3b8'>AeroTwin Digital Twin System · DRDO PS 26054 · SIH 2026 · "
        "AI-generated debrief — for advisory use only. "
        "Final maintenance decisions remain with qualified personnel.</font>",
        ParagraphStyle("footer", parent=body_style, alignment=TA_CENTER)))

    doc.build(story)
    pdf_bytes = buf.getvalue()

    if out_path:
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    import json
    sample_tick = {
        "mission_id": "test_mission",
        "tick_number": 1,
        "timestamp": "2026-09-28T00:00:00",
        "telemetry": {
            "t_s": 1.0, "phase": "cruise", "throttle": 0.55,
            "rpm": 4200, "cht_c": 152.3, "egt_c": 615.0,
            "oil_temp_c": 91.0, "oil_press_bar": 3.8,
            "fuel_flow_lph": 18.1, "vibration_g": 1.4,
            "battery_v": 14.1, "degradation": 0.0003,
            "fault_active": "none", "altitude_ft": 10000, "ambient_temp_c": 15.0,
        },
        "residuals": {"cht_c": 2.1, "egt_c": -3.0, "oil_temp_c": 1.2,
                      "oil_press_bar": 0.05, "fuel_flow_lph": 0.3,
                      "vibration_g": 0.1, "battery_v": 0.02},
        "anomaly_score": 0.08,
        "is_anomaly": False,
        "fault": {"predicted_fault": "none", "confidence": 0.99,
                  "top_reasons": [], "explanation_text": "No fault detected."},
        "rul_hours": 245.5,
        "corroboration": {"verdict": "nominal", "severity_level": "none",
                          "flagged_sensors": [], "n_flagged": 0},
        "recommendation": "continue",
        "health_indices": {
            "thermal": {"score": 94.0, "status": "excellent"},
            "lubrication": {"score": 92.5, "status": "excellent"},
            "combustion": {"score": 93.0, "status": "excellent"},
            "mechanical_vibration": {"score": 96.0, "status": "excellent"},
            "electrical": {"score": 98.5, "status": "excellent"},
            "overall_score": 94.1, "overall_status": "excellent",
        },
    }
    ticks = [sample_tick] * 10
    csv_bytes = export_csv(ticks, "/tmp/test_report.csv")
    print(f"CSV: {len(csv_bytes)} bytes -> /tmp/test_report.csv")
    pdf_bytes = export_pdf("test_mission", ticks, "/tmp/test_report.pdf")
    print(f"PDF: {len(pdf_bytes)} bytes -> /tmp/test_report.pdf")
