#!/usr/bin/env python3
"""
metrics.py — Compute and report all study metrics from the joined records.

Definitions used (§2.3, §6, §7, §8):

    diff_huella(p, e)  := veredicto_delta of fingerprint e for pair p
                          (uses veredicto_desde_cero for control rows that
                           have no veredicto_delta, but NOT *_cierre fields)

    estado(p, e)       := gold.estado_PE_dense  if e.pe == PE_dense
                          gold.estado_PE_moe    if e.pe == PE_moe

    FP_PE(p,e)         := estado(p,e) ∈ {REFACTOR, NO_ALCANZADO}
                          AND diff_huella(p,e) == DISTINTA

    FP_PR(p)           := any e : FP_PE(p,e)

    FP_GATE(p)         := gold_label(p) ∈ {REFACTOR, FUERA_DE_PE}
                          AND any e : diff_huella(p,e) == DISTINTA

    detectado(p)       := any e : estado(p,e) == CHANGE
                          AND diff_huella(p,e) == DISTINTA

    spec_suficiente_PR(p) = no  iff  any PE, any side has spec_suficiente = no

Primary:   count FP_PR over the 20 hard negatives; pass only if 0.
Secondary: count detectado over the 10 positives.

Operational rate INTERVAL over the first 30 of the permutation only:
    min = |{p: gold in {REFACTOR, FUERA_DE_PE} and any diff == DISTINTA}| / 30
    max = min + |{p: gold == AMBIGUOUS and any diff == DISTINTA}| / 30
  (Never emit a single value.)

Coverage over the same 30:
    cobertura_confirmada = |{p: some PE in {CHANGE, REFACTOR}}| / 30
    cobertura_posible    = 1 − |{p: gold == FUERA_DE_PE}| / 30

E4: spec_suficiente_PR(p) = no  iff any PE, any side has spec_suficiente = no.
    Triggered at >= 7 of the 30 primary.

E6 must be computed ONLY from the two reference fingerprints.
    Raise ValueError if asked to compute it from any *_cierre field.

NO_DERIVABLE is neither FP nor detection. Report it as its own count.

Emits one markdown report matching the output list in §11.

Usage:
    python scripts/metrics.py gold_labels.csv fingerprints.csv [--output report.md]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add scripts/ to path so schema can be imported when running from any CWD.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from schema import (
    GoldLabel,
    EstadoPE,
    SiNo,
    Veredicto,
    JoinedRecord,
    FingerprintRecord,
    join,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _diff_huella(fp: FingerprintRecord) -> Optional[Veredicto]:
    """Return the diff verdict for one fingerprint row.

    Uses veredicto_delta when present.  For control rows that were escalated
    desde-cero, falls back to veredicto_desde_cero.

    NEVER reads any *_cierre field for the verdict — those counters are only
    for E6 checks on the reference fingerprints.
    """
    if fp.veredicto_delta is not None:
        return fp.veredicto_delta
    # Control or escalated row: use veredicto_desde_cero
    if fp.veredicto_desde_cero is not None:
        return fp.veredicto_desde_cero
    return None


def _estado(rec: JoinedRecord, fp: FingerprintRecord) -> EstadoPE:
    """Return estado(p, e) for one fingerprint row."""
    from schema import PE
    if fp.pe == PE.PE_DENSE:
        return rec.gold.estado_PE_dense
    return rec.gold.estado_PE_moe


def _any_distinta(rec: JoinedRecord) -> bool:
    """True iff any fingerprint of this pair has diff_huella == DISTINTA."""
    return any(
        _diff_huella(fp) == Veredicto.DISTINTA
        for fp in rec.fingerprints
    )


def _fp_pe(rec: JoinedRecord, fp: FingerprintRecord) -> bool:
    estado = _estado(rec, fp)
    diff   = _diff_huella(fp)
    return (
        estado in (EstadoPE.REFACTOR, EstadoPE.NO_ALCANZADO)
        and diff == Veredicto.DISTINTA
    )


def _fp_pr(rec: JoinedRecord) -> bool:
    return any(_fp_pe(rec, fp) for fp in rec.fingerprints)


def _fp_gate(rec: JoinedRecord) -> bool:
    return (
        rec.gold.gold_label in (GoldLabel.REFACTOR, GoldLabel.FUERA_DE_PE)
        and _any_distinta(rec)
    )


def _detectado(rec: JoinedRecord) -> bool:
    return any(
        _estado(rec, fp) == EstadoPE.CHANGE
        and _diff_huella(fp) == Veredicto.DISTINTA
        for fp in rec.fingerprints
    )


def _spec_suficiente_pr_no(rec: JoinedRecord) -> bool:
    """True iff spec_suficiente_PR(p) = no (§7 E4 aggregation)."""
    for fp in rec.fingerprints:
        if fp.spec_suficiente_lado_1 == SiNo.NO:
            return True
        if fp.spec_suficiente_lado_2 == SiNo.NO:
            return True
    return False


def _no_derivable_count(rec: JoinedRecord) -> int:
    """Count fingerprint rows for this pair where diff_huella == NO_DERIVABLE."""
    return sum(
        1 for fp in rec.fingerprints
        if _diff_huella(fp) == Veredicto.NO_DERIVABLE
    )


# ── E6 guard ───────────────────────────────────────────────────────────────────

_CIERRE_FIELDS = (
    "cierre_C1",
    "cierre_C2",
    "huella_C1",
    "huella_C2",
    "delta_cierre",
    "veredicto_delta",
    "roles_opaque_cierre",
    "plantillas_totales_cierre",
    "plantillas_con_opaque_cierre",
    "plantillas_doble_opaque_cierre",
    "axis_opaque_cierre",
    "ejes_totales_cierre",
)


def _assert_e6_not_from_cierre(fp: FingerprintRecord) -> None:
    """Raise ValueError if any *_cierre field is populated.

    E6 must be computed ONLY from the two reference fingerprints
    (huella_completa_lado_1/2, delta_completo) — never from *_cierre fields.
    This function is called whenever an E6 counter update is requested for a
    fingerprint row to enforce that invariant.
    """
    # veredicto_delta is allowed to be populated generally, but we must not
    # use cierre counters for E6.  We guard specifically the counter fields.
    counter_fields = (
        "roles_opaque_cierre",
        "plantillas_totales_cierre",
        "plantillas_con_opaque_cierre",
        "plantillas_doble_opaque_cierre",
        "axis_opaque_cierre",
        "ejes_totales_cierre",
    )
    for field in counter_fields:
        val = getattr(fp, field, "")
        if val not in ("", None):
            raise ValueError(
                f"E6 must be computed ONLY from the reference fingerprints "
                f"(huella_completa_lado_1/2, delta_completo), never from "
                f"*_cierre fields.  Field {field!r} is set to {val!r} for "
                f"par_id={fp.par_id_opaco!r} pe={fp.pe.value!r}."
            )


# ── FP_NO_ALCANZADO helper ─────────────────────────────────────────────────────


def _fp_no_alcanzado(rec: JoinedRecord) -> bool:
    """True iff any PE is NO_ALCANZADO with diff == DISTINTA (§2.3 note).

    This is a fallo de localización in a CHANGE PR and is reported separately.
    """
    return any(
        _estado(rec, fp) == EstadoPE.NO_ALCANZADO
        and _diff_huella(fp) == Veredicto.DISTINTA
        for fp in rec.fingerprints
    )


# ── main computation ───────────────────────────────────────────────────────────


def compute_metrics(records: List[JoinedRecord]) -> dict:
    """Compute all study metrics; return a structured results dict."""

    # ── partition the records ──────────────────────────────────────────────────
    hard_negatives: List[JoinedRecord] = []   # 20 hard negatives (es_negativo_dificil=sí)
    positives:      List[JoinedRecord] = []   # 10 CHANGE
    primeros_30:    List[JoinedRecord] = []   # first 30 of the permutation

    for rec in records:
        if rec.gold.es_negativo_dificil == SiNo.SI:
            hard_negatives.append(rec)
        if rec.gold.gold_label == GoldLabel.CHANGE and rec.gold.en_primario == SiNo.SI:
            positives.append(rec)
        if rec.gold.en_primeros_30 == SiNo.SI:
            primeros_30.append(rec)

    # Primary set = hard negatives ∪ positives (primario)
    primario: List[JoinedRecord] = hard_negatives + [
        r for r in positives if r not in hard_negatives
    ]

    # ── primary metric: FP_PR over hard negatives ──────────────────────────────
    fp_pr_list          = [r for r in hard_negatives if _fp_pr(r)]
    fp_pr_count         = len(fp_pr_list)
    primary_pass        = fp_pr_count == 0

    # ── secondary metric: detectado over positives ─────────────────────────────
    detectado_list      = [r for r in positives if _detectado(r)]
    detectado_count     = len(detectado_list)

    # ── additional counts (§11.6) ──────────────────────────────────────────────
    fp_gate_list        = [r for r in records if _fp_gate(r)]
    fp_gate_count       = len(fp_gate_list)

    fp_no_alcanzado_list = [r for r in records if _fp_no_alcanzado(r)]
    fp_no_alcanzado_count = len(fp_no_alcanzado_list)

    # AMBIGUOUS and FUERA_DE_PE counts (§11.6)
    ambiguous_count     = sum(1 for r in records if r.gold.gold_label == GoldLabel.AMBIGUOUS)
    fuera_de_pe_count   = sum(1 for r in records if r.gold.gold_label == GoldLabel.FUERA_DE_PE)

    # NO_DERIVABLE: total (PE, side) fingerprint verdicts that are NO_DERIVABLE
    no_derivable_total  = sum(_no_derivable_count(r) for r in records)

    # ── operational rate interval over primeros_30 ─────────────────────────────
    # min = |{p: gold ∈ {REFACTOR, FUERA_DE_PE} and any diff == DISTINTA}| / 30
    # max = min + |{p: gold == AMBIGUOUS and any diff == DISTINTA}| / 30
    n30 = len(primeros_30)
    if n30 == 0:
        op_min = op_max = None
    else:
        op_certain = sum(
            1 for r in primeros_30
            if r.gold.gold_label in (GoldLabel.REFACTOR, GoldLabel.FUERA_DE_PE)
            and _any_distinta(r)
        )
        op_ambiguous = sum(
            1 for r in primeros_30
            if r.gold.gold_label == GoldLabel.AMBIGUOUS
            and _any_distinta(r)
        )
        op_min = op_certain  / n30
        op_max = (op_certain + op_ambiguous) / n30

    # ── coverage over primeros_30 ──────────────────────────────────────────────
    if n30 == 0:
        cobertura_confirmada = cobertura_posible = None
    else:
        confirmed = sum(
            1 for r in primeros_30
            if any(
                _estado(r, fp) in (EstadoPE.CHANGE, EstadoPE.REFACTOR)
                for fp in r.fingerprints
            )
        )
        fuera_count_30 = sum(
            1 for r in primeros_30
            if r.gold.gold_label == GoldLabel.FUERA_DE_PE
        )
        cobertura_confirmada = confirmed / n30
        cobertura_posible    = 1.0 - fuera_count_30 / n30

    # ── E4: spec_suficiente_PR over primario (30) ──────────────────────────────
    spec_no_list  = [r for r in primario if _spec_suficiente_pr_no(r)]
    spec_no_count = len(spec_no_list)
    e4_triggered  = spec_no_count >= 7

    # ── E6: guard (called here so any violation raises immediately) ────────────
    # E6 values themselves come from the reference fingerprints, not from the
    # joined records.  We enforce that no *_cierre counter fields are present
    # in any row (which would indicate a coding error trying to feed cierre data
    # to E6 calculations).
    for rec in records:
        for fp in rec.fingerprints:
            _assert_e6_not_from_cierre(fp)

    # ── §11.7: anchor control stats ────────────────────────────────────────────
    all_control_fps = [
        fp for rec in records for fp in rec.fingerprints
        if fp.es_control == SiNo.SI
    ]
    n_controls                 = len(all_control_fps)
    n_discrepancia_veredicto   = sum(
        1 for fp in all_control_fps if fp.discrepancia_veredicto == SiNo.SI
    )
    n_discrepancia_estructural = sum(
        1 for fp in all_control_fps if fp.discrepancia_estructural == SiNo.SI
    )
    delta_invalidado = n_discrepancia_veredicto > 0

    # §11.8: pairs escalated desde-cero
    escaladas = [
        fp for rec in records for fp in rec.fingerprints
        if fp.escalado_desde_cero.strip().lower().startswith("sí")
        or fp.escalado_desde_cero.strip().lower().startswith("si")
    ]

    return dict(
        # populations
        n_hard_negatives   = len(hard_negatives),
        n_positives        = len(positives),
        n_primario         = len(primario),
        n_primeros_30      = n30,
        n_total            = len(records),
        # primary
        fp_pr_count        = fp_pr_count,
        fp_pr_par_ids      = [r.gold.par_id_opaco for r in fp_pr_list],
        primary_pass       = primary_pass,
        # secondary
        detectado_count    = detectado_count,
        detectado_par_ids  = [r.gold.par_id_opaco for r in detectado_list],
        # additional
        fp_gate_count      = fp_gate_count,
        fp_no_alcanzado_count = fp_no_alcanzado_count,
        ambiguous_count    = ambiguous_count,
        fuera_de_pe_count  = fuera_de_pe_count,
        no_derivable_total = no_derivable_total,
        # operational rate interval
        op_min             = op_min,
        op_max             = op_max,
        # coverage
        cobertura_confirmada = cobertura_confirmada,
        cobertura_posible    = cobertura_posible,
        # E4
        spec_no_count      = spec_no_count,
        e4_triggered       = e4_triggered,
        spec_no_par_ids    = [r.gold.par_id_opaco for r in spec_no_list],
        # anchor controls
        n_controls                 = n_controls,
        n_discrepancia_veredicto   = n_discrepancia_veredicto,
        n_discrepancia_estructural = n_discrepancia_estructural,
        delta_invalidado           = delta_invalidado,
        # escalated pairs
        n_escaladas = len(escaladas),
    )


# ── markdown report ────────────────────────────────────────────────────────────


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:.1f}%"


def _frac(n: int, d: int) -> str:
    return f"{n}/{d}"


def render_report(m: dict) -> str:
    """Render the §11 markdown report from the metrics dict."""
    lines: List[str] = []

    def h(level: int, title: str) -> None:
        lines.append("#" * level + " " + title)
        lines.append("")

    def line(text: str = "") -> None:
        lines.append(text)

    h(1, "Informe de métricas — Estudio piloto de huella estructural")

    # ── §11.2: N_q, operational rate, NO_DERIVABLE, coverage, shadow sizing ───
    h(2, "§11.2 Métricas de exploración")

    line(f"**N_q** (PRs cualificantes en la ventana): 137  _(valor preregistrado)_")
    line()

    line("**Intervalo de tasa operacional** (sobre `primeros_30`):")
    if m["op_min"] is None or m["op_max"] is None:
        line("  _No hay registros en `primeros_30`; no se puede calcular._")
    else:
        line(f"  mín = {m['op_min']:.4f}  ({_pct(m['op_min'])})")
        line(f"  máx = {m['op_max']:.4f}  ({_pct(m['op_max'])})")
        line("  _(Orientativo. La medición válida del SLO es la sombra de 409 PRs.)_")
    line()

    line(f"**NO_DERIVABLE** (total pares PE×lado): {m['no_derivable_total']}")
    line("  _(NO_DERIVABLE no cuenta como FP ni como detección; se reporta aparte.)_")
    line()

    line("**Cobertura** (sobre `primeros_30`):")
    if m["cobertura_confirmada"] is None:
        line("  _No hay registros en `primeros_30`._")
    else:
        line(f"  cobertura_confirmada = {_pct(m['cobertura_confirmada'])}  "
             f"({m['n_primeros_30']} pares)")
        line(f"  cobertura_posible    = {_pct(m['cobertura_posible'])}  "
             f"({m['n_primeros_30']} pares)")
        _cc = m["cobertura_confirmada"]
        _cp = m["cobertura_posible"]
        if _cp < 0.40:
            line("  → **Hacen falta más PEs** (`cobertura_posible < 0.40`).")
        elif _cc >= 0.40:
            line("  → **Cobertura suficiente para continuar** "
                 "(`cobertura_confirmada ≥ 0.40`).")
        else:
            line("  → **Indeterminado:** ampliar la muestra de cobertura.")
    line()

    line("**Dimensionado de sombra** (§0.2): n = ⌈log(0.05)/log(136/137)⌉ = **409 PRs**")
    line()

    # ── §11.6: counts ─────────────────────────────────────────────────────────
    h(2, "§11.6 Recuentos")

    line(f"| Métrica | Valor |")
    line(f"|---------|-------|")
    line(f"| FP_PR (negativos difíciles, n={m['n_hard_negatives']}) | "
         f"**{m['fp_pr_count']}** |")
    line(f"| FP_GATE (total) | {m['fp_gate_count']} |")
    line(f"| FP_NO_ALCANZADO (fallo de localización) | {m['fp_no_alcanzado_count']} |")
    line(f"| Positivos detectados (CHANGE, n={m['n_positives']}) | "
         f"{m['detectado_count']} |")
    line(f"| AMBIGUOUS (total etiquetados) | {m['ambiguous_count']} |")
    line(f"| FUERA_DE_PE (total etiquetados) | {m['fuera_de_pe_count']} |")
    line(f"| NO_DERIVABLE (PE×lado, total) | {m['no_derivable_total']} |")
    line(f"| spec_suficiente_PR = no (primario) | {m['spec_no_count']} |")
    line()

    # ── Primary metric ────────────────────────────────────────────────────────
    h(2, "§6 Métrica principal — FP_PR (bloqueante)")

    verdict = "✅ **PASA** (0 FP_PR sobre los negativos difíciles)" \
        if m["primary_pass"] else \
        f"❌ **FALLA** ({m['fp_pr_count']} FP_PR sobre {m['n_hard_negatives']} negativos difíciles)"
    line(verdict)
    if m["fp_pr_par_ids"]:
        line()
        line("FP_PR par_ids:")
        for pid in m["fp_pr_par_ids"]:
            line(f"  - `{pid}`")
    line()

    # ── Secondary metric ──────────────────────────────────────────────────────
    h(2, "§6 Métrica secundaria — detectado (no bloqueante)")

    line(f"Detectados: **{_frac(m['detectado_count'], m['n_positives'])}**"
         f"  (objetivo ≥ 8/10)")
    if m["detectado_par_ids"]:
        line()
        line("Pares detectados:")
        for pid in m["detectado_par_ids"]:
            line(f"  - `{pid}`")
    line()

    # ── E4 ────────────────────────────────────────────────────────────────────
    h(2, "§7 E4 — spec_suficiente_PR")

    if m["e4_triggered"]:
        line(f"⚠️  **E4 ACTIVADO**: {m['spec_no_count']} ≥ 7 PRs primarios con "
             f"`spec_suficiente_PR = no`.")
        if m["spec_no_par_ids"]:
            line()
            line("Pares afectados:")
            for pid in m["spec_no_par_ids"]:
                line(f"  - `{pid}`")
    else:
        line(f"✅ E4 no activado: {m['spec_no_count']} < 7 "
             f"PRs primarios con `spec_suficiente_PR = no`.")
    line()

    # ── E6 note ───────────────────────────────────────────────────────────────
    h(2, "§8.1 E6 — umbrales de centinela")

    line("Los cuatro sub-umbrales de E6 se calculan **exclusivamente** sobre "
         "las huellas de referencia completas de los dos PEs (`huella_completa_lado_1/2`, "
         "`delta_completo`).  Este script rechaza cualquier intento de calcularlos "
         "desde campos `*_cierre` y elevaría un `ValueError` si se detectase.")
    line()
    line("_(E6 no es computable por este script en ausencia de las huellas de referencia "
         "completas; los valores deben proceder de E0.)_")
    line()

    # ── §11.7: control de anclaje ─────────────────────────────────────────────
    h(2, "§11.7 Control de anclaje")

    line(f"Controles totales (es_control=sí): **{m['n_controls']}**")
    line(f"Discrepancias de veredicto: {m['n_discrepancia_veredicto']}")
    line(f"Discrepancias estructurales: {m['n_discrepancia_estructural']}")
    if m["delta_invalidado"]:
        line()
        line("⚠️  **Método delta INVALIDADO** (≥1 discrepancia de veredicto). "
             "Se debe rehacer todo desde cero o activar E5.")
    else:
        line("Método delta: ✅ no invalidado.")
    line()

    # ── §11.8: escalated pairs ────────────────────────────────────────────────
    h(2, "§11.8 Pares escalados a desde_cero")

    line(f"Pares (PE×lado) escalados a huella completa: **{m['n_escaladas']}**")
    line()

    # ── §11.10: decision ──────────────────────────────────────────────────────
    h(2, "§11.10 Decisión")

    if m["primary_pass"] and not m["e4_triggered"]:
        line("✅ Sin condiciones de abandono activadas en los datos analizados.  "
             "Pendiente de verificación de E6 sobre las referencias.")
    else:
        codes = []
        if not m["primary_pass"]:
            codes.append("E1/especificación (FP_PR > 0)")
        if m["e4_triggered"]:
            codes.append("E4 (spec_suficiente_PR ≥ 7)")
        line(f"⚠️  Condiciones detectadas: {', '.join(codes)}.")
        line("Véase §7 y §9 para la política de revisión.")
    line()

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────


def _main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compute and report metrics from joined records (§2.3, §6–§8).",
    )
    parser.add_argument("gold_labels",  type=Path, help="Path to gold_labels.csv")
    parser.add_argument("fingerprints", type=Path, help="Path to fingerprints.csv")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Write the markdown report to this file (default: stdout)",
    )
    args = parser.parse_args(argv)

    try:
        records = join(args.gold_labels, args.fingerprints)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR loading data: {exc}", file=sys.stderr)
        return 1

    try:
        metrics = compute_metrics(records)
    except ValueError as exc:
        print(f"ERROR computing metrics: {exc}", file=sys.stderr)
        return 1

    report = render_report(metrics)

    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    # Exit code: 0 if primary passes, 1 if not.
    return 0 if metrics["primary_pass"] else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
