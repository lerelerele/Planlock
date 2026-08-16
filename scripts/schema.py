#!/usr/bin/env python3
"""
schema.py — Dataclasses and validation for gold_labels.csv and fingerprints.csv.

Standard library only.

Public API
----------
load_gold_labels(path)   -> list[GoldLabelRecord]
load_fingerprints(path)  -> list[FingerprintRecord]
join(gold_path, fp_path) -> list[JoinedRecord]   # requires both files frozen

Frozen convention
-----------------
A file is considered frozen when its last non-empty line is exactly::

    # FROZEN

Both load_* functions strip that sentinel before CSV parsing; join() refuses
to proceed unless both files carry it.
"""

import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Enumerations ──────────────────────────────────────────────────────────────


class EstadoPE(str, Enum):
    CHANGE       = "CHANGE"
    REFACTOR     = "REFACTOR"
    AMBIGUOUS    = "AMBIGUOUS"
    NO_ALCANZADO = "NO_ALCANZADO"


class GoldLabel(str, Enum):
    CHANGE      = "CHANGE"
    REFACTOR    = "REFACTOR"
    AMBIGUOUS   = "AMBIGUOUS"
    FUERA_DE_PE = "FUERA_DE_PE"


class SiNo(str, Enum):
    SI = "sí"
    NO = "no"


class CriterioDificultad(str, Enum):
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    NA = "n/a"


class PeQueLoHaceDificil(str, Enum):
    PE_DENSE = "PE_dense"
    PE_MOE   = "PE_moe"
    AMBOS    = "ambos"
    NA       = "n/a"


class PE(str, Enum):
    PE_DENSE = "PE_dense"
    PE_MOE   = "PE_moe"


class Veredicto(str, Enum):
    IDENTICA     = "IDÉNTICA"
    DISTINTA     = "DISTINTA"
    NO_DERIVABLE = "NO_DERIVABLE"


# ── §2.2 Derivation ───────────────────────────────────────────────────────────


def derive_gold_label(dense: EstadoPE, moe: EstadoPE) -> GoldLabel:
    """Derive gold_label from the two PE states (§2.2).

    si algún PE = CHANGE                          → CHANGE
    si no, y algún PE = AMBIGUOUS                 → AMBIGUOUS
    si no, y algún PE alcanzado (todos REFACTOR)  → REFACTOR
    si no (ningún PE alcanzado)                   → FUERA_DE_PE
    """
    states = (dense, moe)
    if EstadoPE.CHANGE in states:
        return GoldLabel.CHANGE
    if EstadoPE.AMBIGUOUS in states:
        return GoldLabel.AMBIGUOUS
    if any(s == EstadoPE.REFACTOR for s in states):
        return GoldLabel.REFACTOR
    return GoldLabel.FUERA_DE_PE


# ── GoldLabelRecord ───────────────────────────────────────────────────────────


@dataclass
class GoldLabelRecord:
    """One row of gold_labels.csv (§10.1)."""

    par_id_opaco:            str
    estado_PE_dense:         EstadoPE
    estado_PE_moe:           EstadoPE
    gold_label:              GoldLabel           # stored; cross-checked vs. derivation
    en_primario:             SiNo
    en_primeros_30:          SiNo
    es_negativo_dificil:     SiNo
    criterio_dificultad:     CriterioDificultad
    pe_que_lo_hace_dificil:  PeQueLoHaceDificil
    ficheros_cualificantes:  str
    justificacion_breve:     str

    def validate(self) -> None:
        """Raise ValueError for any constraint violation."""
        # Rule: gold_label must match §2.2 derivation — it is NEVER entered directly.
        expected = derive_gold_label(self.estado_PE_dense, self.estado_PE_moe)
        if self.gold_label != expected:
            raise ValueError(
                f"par_id={self.par_id_opaco!r}: stored gold_label={self.gold_label.value!r} "
                f"disagrees with derived value {expected.value!r} "
                f"(estado_PE_dense={self.estado_PE_dense.value}, "
                f"estado_PE_moe={self.estado_PE_moe.value})"
            )

        # Rule: es_negativo_dificil=sí requires gold_label=REFACTOR and
        #       a non-n/a criterio_dificultad.
        if self.es_negativo_dificil == SiNo.SI:
            if self.gold_label != GoldLabel.REFACTOR:
                raise ValueError(
                    f"par_id={self.par_id_opaco!r}: es_negativo_dificil=sí requires "
                    f"gold_label=REFACTOR, got {self.gold_label.value!r}"
                )
            if self.criterio_dificultad == CriterioDificultad.NA:
                raise ValueError(
                    f"par_id={self.par_id_opaco!r}: es_negativo_dificil=sí requires "
                    f"a non-n/a criterio_dificultad"
                )


# ── FingerprintRecord ─────────────────────────────────────────────────────────

# Fields belonging to the "from-scratch" path.  When es_control=no these must
# all be absent (empty string or None).
_FROM_SCRATCH_FIELDS = (
    "huella_completa_lado_1",
    "huella_completa_lado_2",
    "delta_completo",
    "veredicto_desde_cero",
    "sello_desde_cero",
    "discrepancia_veredicto",
    "discrepancia_estructural",
)


@dataclass
class FingerprintRecord:
    """One row of fingerprints.csv — one (par, PE) pair (§10.2)."""

    par_id_opaco:                   str
    pe:                             PE
    es_control:                     SiNo
    escalado_desde_cero:            str                  # free text: "sí | no + causa"
    # --- delta path ---
    cierre_C1:                      str
    cierre_C2:                      str
    certificado_frontera:           str
    identidades_validadas:          str
    huella_C1:                      str
    huella_C2:                      str
    delta_cierre:                   str
    veredicto_delta:                Optional[Veredicto]
    # --- from-scratch path (controls and scaled runs) ---
    huella_completa_lado_1:         str
    huella_completa_lado_2:         str
    delta_completo:                 str
    veredicto_desde_cero:           Optional[Veredicto]
    sello_desde_cero:               str                  # ISO 8601 timestamp or empty
    # --- control-only comparison ---
    discrepancia_veredicto:         Optional[SiNo]
    discrepancia_estructural:       Optional[SiNo]
    # --- counters ---
    roles_opaque_cierre:            str
    plantillas_totales_cierre:      str
    plantillas_con_opaque_cierre:   str
    plantillas_doble_opaque_cierre: str
    axis_opaque_cierre:             str
    ejes_totales_cierre:            str
    # --- completion fields ---
    spec_suficiente_lado_1:         Optional[SiNo]
    spec_suficiente_lado_2:         Optional[SiNo]
    regla_no_cubierta:              str
    tiempo_de_aplicacion:           str                  # minutes, free text

    def validate(self, file_mtime: Optional[datetime] = None) -> None:
        """Raise ValueError for any constraint violation.

        Parameters
        ----------
        file_mtime:
            Timezone-aware mtime of fingerprints.csv.  Required to enforce the
            sello_desde_cero < file_mtime constraint when es_control=sí.
        """
        loc = f"par_id={self.par_id_opaco!r} pe={self.pe.value}"

        if self.es_control == SiNo.NO:
            # Rule: es_control=no must leave all from-scratch fields empty.
            for fname in _FROM_SCRATCH_FIELDS:
                val = getattr(self, fname)
                if val not in ("", None):
                    raise ValueError(
                        f"{loc}: es_control=no but from-scratch field "
                        f"{fname!r} is set to {val!r}"
                    )
        else:
            # Rule: es_control=sí must have sello_desde_cero present and
            #       strictly earlier than the file mtime.
            if not self.sello_desde_cero:
                raise ValueError(
                    f"{loc}: es_control=sí requires sello_desde_cero to be set"
                )
            if file_mtime is not None:
                sello = _parse_timestamp(self.sello_desde_cero)
                if sello >= file_mtime:
                    raise ValueError(
                        f"{loc}: sello_desde_cero={self.sello_desde_cero!r} must be "
                        f"strictly earlier than file mtime "
                        f"{file_mtime.isoformat()}"
                    )


def _parse_timestamp(s: str) -> datetime:
    """Parse an ISO 8601 timestamp string; return a timezone-aware datetime."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {s!r}")


# ── Frozen-marker helpers ─────────────────────────────────────────────────────

_FROZEN_MARKER = "# FROZEN"


def _is_frozen(path: Path) -> bool:
    """Return True iff the file's last non-empty line is exactly '# FROZEN'."""
    with path.open(encoding="utf-8") as fh:
        lines = [line.rstrip("\n\r") for line in fh if line.strip()]
    return bool(lines) and lines[-1] == _FROZEN_MARKER


def _read_csv_lines(path: Path) -> List[str]:
    """Read the file, stripping the '# FROZEN' sentinel if it is the last non-empty line."""
    with path.open(encoding="utf-8") as fh:
        lines = list(fh)
    # Find the last non-empty line; remove it if it is the frozen marker.
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            if lines[i].strip() == _FROZEN_MARKER:
                lines.pop(i)
            break
    return lines


# ── Enum-parsing helpers ──────────────────────────────────────────────────────


def _parse_enum(cls, value: str, field_name: str, par_id: str = "") -> "Enum":
    try:
        return cls(value)
    except ValueError:
        valid = [e.value for e in cls]
        prefix = f"par_id={par_id!r}: " if par_id else ""
        raise ValueError(
            f"{prefix}{field_name}={value!r} is not one of {valid}"
        )


def _optional_enum(cls, value: str, field_name: str, par_id: str = "") -> Optional["Enum"]:
    if not value:
        return None
    return _parse_enum(cls, value, field_name, par_id)


def _require_columns(reader: csv.DictReader, required: Set[str], filename: str) -> None:
    """Reject truncated CSV schemas before row parsing can hide omissions."""
    actual = set(reader.fieldnames or [])
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"{filename}: missing required column(s): {', '.join(missing)}")


# ── File loaders ──────────────────────────────────────────────────────────────


def load_gold_labels(path: Path) -> List[GoldLabelRecord]:
    """Parse and validate gold_labels.csv independently.

    All rows are validated; all errors are collected and reported together.
    Raises ValueError if any error is found.
    """
    errors: List[str] = []
    records: List[GoldLabelRecord] = []

    lines = _read_csv_lines(path)
    reader = csv.DictReader(lines)
    _require_columns(
        reader,
        {
            "par_id_opaco", "estado_PE_dense", "estado_PE_moe", "gold_label",
            "en_primario", "en_primeros_30", "es_negativo_dificil",
            "criterio_dificultad", "pe_que_lo_hace_dificil",
        },
        "gold_labels.csv",
    )

    for row in reader:
        pid = row.get("par_id_opaco", "")
        try:
            rec = GoldLabelRecord(
                par_id_opaco           = pid,
                estado_PE_dense        = _parse_enum(EstadoPE, row["estado_PE_dense"], "estado_PE_dense", pid),
                estado_PE_moe          = _parse_enum(EstadoPE, row["estado_PE_moe"], "estado_PE_moe", pid),
                gold_label             = _parse_enum(GoldLabel, row["gold_label"], "gold_label", pid),
                en_primario            = _parse_enum(SiNo, row["en_primario"], "en_primario", pid),
                en_primeros_30         = _parse_enum(SiNo, row["en_primeros_30"], "en_primeros_30", pid),
                es_negativo_dificil    = _parse_enum(SiNo, row["es_negativo_dificil"], "es_negativo_dificil", pid),
                criterio_dificultad    = _parse_enum(CriterioDificultad, row["criterio_dificultad"], "criterio_dificultad", pid),
                pe_que_lo_hace_dificil = _parse_enum(PeQueLoHaceDificil, row["pe_que_lo_hace_dificil"], "pe_que_lo_hace_dificil", pid),
                ficheros_cualificantes = row.get("ficheros_cualificantes", ""),
                justificacion_breve    = row.get("justificacion_breve", ""),
            )
            rec.validate()
            if not rec.par_id_opaco:
                raise ValueError("par_id_opaco must not be empty")
            if any(existing.par_id_opaco == rec.par_id_opaco for existing in records):
                raise ValueError(f"duplicate par_id_opaco={rec.par_id_opaco!r}")
            records.append(rec)
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))

    if errors:
        raise ValueError(
            f"gold_labels.csv: {len(errors)} validation error(s):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )
    return records


def load_fingerprints(path: Path) -> List[FingerprintRecord]:
    """Parse and validate fingerprints.csv independently.

    All rows are validated; all errors are collected and reported together.
    Raises ValueError if any error is found.
    """
    errors: List[str] = []
    records: List[FingerprintRecord] = []

    file_mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    lines = _read_csv_lines(path)
    reader = csv.DictReader(lines)
    _require_columns(
        reader,
        {"par_id_opaco", "pe", "es_control", "sello_desde_cero"},
        "fingerprints.csv",
    )

    seen: Set[Tuple[str, PE]] = set()

    for row in reader:
        pid = row.get("par_id_opaco", "")
        try:
            rec = FingerprintRecord(
                par_id_opaco                   = pid,
                pe                             = _parse_enum(PE, row["pe"], "pe", pid),
                es_control                     = _parse_enum(SiNo, row["es_control"], "es_control", pid),
                escalado_desde_cero            = row.get("escalado_desde_cero", ""),
                cierre_C1                      = row.get("cierre_C1", ""),
                cierre_C2                      = row.get("cierre_C2", ""),
                certificado_frontera           = row.get("certificado_frontera", ""),
                identidades_validadas          = row.get("identidades_validadas", ""),
                huella_C1                      = row.get("huella_C1", ""),
                huella_C2                      = row.get("huella_C2", ""),
                delta_cierre                   = row.get("delta_cierre", ""),
                veredicto_delta                = _optional_enum(Veredicto, row.get("veredicto_delta", ""), "veredicto_delta", pid),
                huella_completa_lado_1         = row.get("huella_completa_lado_1", ""),
                huella_completa_lado_2         = row.get("huella_completa_lado_2", ""),
                delta_completo                 = row.get("delta_completo", ""),
                veredicto_desde_cero           = _optional_enum(Veredicto, row.get("veredicto_desde_cero", ""), "veredicto_desde_cero", pid),
                sello_desde_cero               = row.get("sello_desde_cero", ""),
                discrepancia_veredicto         = _optional_enum(SiNo, row.get("discrepancia_veredicto", ""), "discrepancia_veredicto", pid),
                discrepancia_estructural       = _optional_enum(SiNo, row.get("discrepancia_estructural", ""), "discrepancia_estructural", pid),
                roles_opaque_cierre            = row.get("roles_opaque_cierre", ""),
                plantillas_totales_cierre      = row.get("plantillas_totales_cierre", ""),
                plantillas_con_opaque_cierre   = row.get("plantillas_con_opaque_cierre", ""),
                plantillas_doble_opaque_cierre = row.get("plantillas_doble_opaque_cierre", ""),
                axis_opaque_cierre             = row.get("axis_opaque_cierre", ""),
                ejes_totales_cierre            = row.get("ejes_totales_cierre", ""),
                spec_suficiente_lado_1         = _optional_enum(SiNo, row.get("spec_suficiente_lado_1", ""), "spec_suficiente_lado_1", pid),
                spec_suficiente_lado_2         = _optional_enum(SiNo, row.get("spec_suficiente_lado_2", ""), "spec_suficiente_lado_2", pid),
                regla_no_cubierta              = row.get("regla_no_cubierta", ""),
                tiempo_de_aplicacion           = row.get("tiempo_de_aplicacion", ""),
            )
            rec.validate(file_mtime=file_mtime)
            if not rec.par_id_opaco:
                raise ValueError("par_id_opaco must not be empty")
            key = (rec.par_id_opaco, rec.pe)
            if key in seen:
                raise ValueError(
                    f"duplicate fingerprint for par_id={rec.par_id_opaco!r} "
                    f"pe={rec.pe.value!r}"
                )
            seen.add(key)
            records.append(rec)
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))

    if errors:
        raise ValueError(
            f"fingerprints.csv: {len(errors)} validation error(s):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )
    return records


# ── Join ──────────────────────────────────────────────────────────────────────


@dataclass
class JoinedRecord:
    """One par_id after joining the two files."""

    gold:         GoldLabelRecord
    fingerprints: List[FingerprintRecord]


def join(
    gold_labels_path: Path,
    fingerprints_path: Path,
) -> List[JoinedRecord]:
    """Merge gold_labels.csv and fingerprints.csv on par_id_opaco.

    Refuses to run unless both files are marked frozen (last non-empty line
    is ``# FROZEN``).  Loads and validates each file independently first, then
    groups fingerprint rows by par_id.

    Returns one JoinedRecord per gold-label row; fingerprints for par_ids
    absent from fingerprints.csv are an empty list.
    """
    if not _is_frozen(gold_labels_path):
        raise RuntimeError(
            f"{gold_labels_path} is not frozen — "
            f"append '{_FROZEN_MARKER}' as the last line before joining."
        )
    if not _is_frozen(fingerprints_path):
        raise RuntimeError(
            f"{fingerprints_path} is not frozen — "
            f"append '{_FROZEN_MARKER}' as the last line before joining."
        )

    golds = load_gold_labels(gold_labels_path)
    fps   = load_fingerprints(fingerprints_path)

    fp_by_par: Dict[str, List[FingerprintRecord]] = {}
    for fp in fps:
        fp_by_par.setdefault(fp.par_id_opaco, []).append(fp)

    gold_ids = {g.par_id_opaco for g in golds}
    fp_ids = set(fp_by_par)
    unknown_fp_ids = sorted(fp_ids - gold_ids)
    missing_fp_ids = sorted(gold_ids - fp_ids)
    incomplete = sorted(
        pid for pid, rows in fp_by_par.items()
        if pid in gold_ids and {row.pe for row in rows} != {PE.PE_DENSE, PE.PE_MOE}
    )
    errors = []
    if unknown_fp_ids:
        errors.append(f"fingerprints.csv contains unknown par_id(s): {unknown_fp_ids}")
    if missing_fp_ids:
        errors.append(f"fingerprints.csv is missing par_id(s): {missing_fp_ids}")
    if incomplete:
        errors.append(
            "each par_id must have exactly one PE_dense and one PE_moe fingerprint; "
            f"incomplete par_id(s): {incomplete}"
        )
    if errors:
        raise ValueError("join integrity error(s):\n" + "\n".join(f"  • {e}" for e in errors))

    return [
        JoinedRecord(gold=g, fingerprints=fp_by_par.get(g.par_id_opaco, []))
        for g in golds
    ]


# ── CLI entry point ───────────────────────────────────────────────────────────


def _main(argv: List[str]) -> int:
    """Validate one or both CSV files from the command line.

    Usage:
        python scripts/schema.py gold_labels.csv
        python scripts/schema.py fingerprints.csv
        python scripts/schema.py gold_labels.csv fingerprints.csv  # join
    """
    if not argv:
        print(
            "Usage: schema.py <gold_labels.csv> [fingerprints.csv]",
            file=sys.stderr,
        )
        return 2

    first_path = Path(argv[0])
    fp_path    = Path(argv[1]) if len(argv) > 1 else None

    ok = True

    if fp_path is None:
        # Validate a single file; detect which one by its name.
        name = first_path.name
        try:
            if "gold" in name:
                recs = load_gold_labels(first_path)
                print(f"OK: {len(recs)} gold-label record(s) validated.")
            elif "fingerprint" in name:
                recs = load_fingerprints(first_path)
                print(f"OK: {len(recs)} fingerprint record(s) validated.")
            else:
                print(
                    f"ERROR: cannot determine file type from name {name!r}. "
                    f"Filename must contain 'gold' or 'fingerprint'.",
                    file=sys.stderr,
                )
                return 2
        except (ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            ok = False
    else:
        try:
            joined = join(first_path, fp_path)
            print(f"OK: joined {len(joined)} par_id(s).")
        except (ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
