import base64
import gzip
import io

from .schemas import FilePayload


def _append_accessions_to_text(user_text: str, accessions: list[str]) -> str:
    if accessions:
        return f"{user_text}\nDatasets of Interest: {', '.join(accessions)}"
    return user_text


def _extract_accessions_from_files(files_raw: list[dict]) -> list[str]:
    accessions: list[str] = []
    try:
        import pandas as pd
    except ImportError:
        return accessions

    ACCESSION_COLUMN_CANDIDATES = ["accession_code", "accession", "accessions", "gse", "geo_accession"]

    for f in files_raw:
        try:
            payload = FilePayload.model_validate(f)
        except Exception:
            continue

        try:
            raw = base64.b64decode(payload.content)
        except Exception:
            continue

        if payload.name.lower().endswith(".gz"):
            try:
                raw = gzip.decompress(raw)
            except Exception:
                continue

        for encoding in ("utf-8", "latin-1", "utf-8-sig"):
            try:
                text = raw.decode(encoding) if isinstance(raw, bytes) else raw
            except Exception:
                continue
            df = None
            name_lower = payload.name.lower()
            if name_lower.endswith(".tsv") or name_lower.endswith(".tsv.gz"):
                try:
                    df = pd.read_csv(io.StringIO(text), sep="\t")
                except Exception:
                    continue
            else:
                try:
                    df = pd.read_csv(io.StringIO(text))
                except Exception:
                    continue
            if df is None:
                continue
            break

        columns_lower = [c.lower().strip() for c in df.columns]  # type: ignore
        col_idx = None
        for candidate in ACCESSION_COLUMN_CANDIDATES:
            try:
                col_idx = columns_lower.index(candidate)
                break
            except ValueError:
                continue

        if col_idx is None:
            continue

        col_name = df.columns[col_idx]  # type: ignore
        for val in df[col_name].dropna().astype(str):  # type: ignore
            if val.strip().upper().startswith("GSE"):
                accessions.append(val.strip())

    return list(dict.fromkeys(accessions))
