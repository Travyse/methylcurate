import base64
import gzip
import io

import pytest
import pandas as pd


def _make_csv_data(rows: list[tuple[str, ...]], columns: list[str]) -> str:
    buf = io.StringIO()
    pd.DataFrame(rows, columns=columns).to_csv(buf, index=False)
    return buf.getvalue()


def _make_file_dict(name: str, content: str, *, gzip_compress: bool = False) -> dict:
    data = content.encode("utf-8")
    if gzip_compress:
        data = gzip.compress(data)
    return {"name": name, "content": base64.b64encode(data).decode("utf-8")}


# ----------------------------------------------------------------
# _extract_accessions_from_files
# ----------------------------------------------------------------


class TestExtractAccessionsFromFiles:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from methylcurate.api.file_parser import _extract_accessions_from_files

        self._sut = _extract_accessions_from_files

    def test_extracts_from_csv_with_accession_code_column(self):
        csv = _make_csv_data(
            [("GSE12345",), ("GSE67890",)],
            ["accession_code"],
        )
        result = self._sut([_make_file_dict("datasets.csv", csv)])
        assert result == ["GSE12345", "GSE67890"]

    def test_extracts_from_tsv_with_accession_code_column(self):
        tsv = _make_csv_data(
            [("GSE11111",), ("GSE22222",)],
            ["accession_code"],
        )
        result = self._sut([_make_file_dict("datasets.tsv", tsv)])
        assert result == ["GSE11111", "GSE22222"]

    def test_extracts_from_txt_with_accession_code_column(self):
        csv = _make_csv_data(
            [("GSE99999",)],
            ["accession_code"],
        )
        result = self._sut([_make_file_dict("datasets.txt", csv)])
        assert result == ["GSE99999"]

    def test_extracts_from_gz_compressed_file(self):
        csv = _make_csv_data(
            [("GSE33333",), ("GSE44444",)],
            ["accession_code"],
        )
        result = self._sut([_make_file_dict("datasets.csv.gz", csv, gzip_compress=True)])
        assert result == ["GSE33333", "GSE44444"]

    def test_returns_empty_when_accession_code_column_missing(self):
        csv = _make_csv_data(
            [("GSE12345",), ("GSE67890",)],
            ["not_the_right_column"],
        )
        result = self._sut([_make_file_dict("datasets.csv", csv)])
        assert result == []

    def test_returns_empty_when_no_gse_prefixed_values(self):
        csv = _make_csv_data(
            [("GSM12345",), ("GSM67890",), ("SRP00001",)],
            ["accession_code"],
        )
        result = self._sut([_make_file_dict("datasets.csv", csv)])
        assert result == []

    def test_deduplicates_across_files(self):
        csv1 = _make_csv_data([("GSE11111",), ("GSE22222",)], ["accession_code"])
        csv2 = _make_csv_data([("GSE22222",), ("GSE33333",)], ["accession_code"])
        result = self._sut(
            [
                _make_file_dict("a.csv", csv1),
                _make_file_dict("b.csv", csv2),
            ]
        )
        assert result == ["GSE11111", "GSE22222", "GSE33333"]

    def test_case_insensitive_column_name(self):
        csv = _make_csv_data([("GSE12345",)], ["Accession_Code"])
        result = self._sut([_make_file_dict("datasets.csv", csv)])
        assert result == ["GSE12345"]

    def test_case_insensitive_gse_prefix(self):
        csv = _make_csv_data([("gse12345",), ("GsE67890",)], ["accession_code"])
        result = self._sut([_make_file_dict("datasets.csv", csv)])
        assert result == ["gse12345", "GsE67890"]

    def test_filters_non_gse_values(self):
        csv = _make_csv_data(
            [("GSE12345",), ("not_gse",), ("GSM99999",), ("GSE67890",)],
            ["accession_code"],
        )
        result = self._sut([_make_file_dict("datasets.csv", csv)])
        assert result == ["GSE12345", "GSE67890"]

    def test_alternate_column_name_accession(self):
        csv = _make_csv_data([("GSE11111",)], ["accession"])
        result = self._sut([_make_file_dict("datasets.csv", csv)])
        assert result == ["GSE11111"]

    def test_alternate_column_name_accessions(self):
        csv = _make_csv_data([("GSE22222",)], ["accessions"])
        result = self._sut([_make_file_dict("datasets.csv", csv)])
        assert result == ["GSE22222"]

    def test_alternate_column_name_gse(self):
        csv = _make_csv_data([("GSE33333",)], ["gse"])
        result = self._sut([_make_file_dict("datasets.csv", csv)])
        assert result == ["GSE33333"]

    def test_alternate_column_name_geo_accession(self):
        csv = _make_csv_data([("GSE44444",)], ["geo_accession"])
        result = self._sut([_make_file_dict("datasets.csv", csv)])
        assert result == ["GSE44444"]

    def test_empty_input_returns_empty(self):
        assert self._sut([]) == []

    def test_malformed_base64_skipped(self):
        csv = _make_csv_data([("GSE12345",)], ["accession_code"])
        good = _make_file_dict("good.csv", csv)
        bad = {"name": "bad.csv", "content": "not-valid-base64!!!"}
        result = self._sut([bad, good])
        assert result == ["GSE12345"]

    def test_corrupt_gzip_skipped(self):
        csv = _make_csv_data([("GSE12345",)], ["accession_code"])
        good = _make_file_dict("good.csv", csv)
        bad = {"name": "corrupt.csv.gz", "content": base64.b64encode(b"not-gzip-data").decode("utf-8")}
        result = self._sut([bad, good])
        assert result == ["GSE12345"]

    def test_handles_encoding_variants(self):
        content = "accession_code\nGSE12345\nGSE67890\n"
        result = self._sut(
            [
                _make_file_dict("utf8.csv", content),
            ]
        )
        assert result == ["GSE12345", "GSE67890"]

    def test_skips_invalid_file_payload_keys(self):
        result = self._sut([{"wrong_key": "value"}])
        assert result == []


# ----------------------------------------------------------------
# _append_accessions_to_text
# ----------------------------------------------------------------


class TestAppendAccessionsToText:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from methylcurate.api.file_parser import _append_accessions_to_text

        self._sut = _append_accessions_to_text

    def test_appends_datasets_of_interest_line(self):
        result = self._sut("Download these datasets", ["GSE12345", "GSE67890"])
        assert result == "Download these datasets\nDatasets of Interest: GSE12345, GSE67890"

    def test_no_accessions_returns_text_unchanged(self):
        original = "Download these datasets"
        result = self._sut(original, [])
        assert result == original

    def test_single_accession_no_trailing_comma(self):
        result = self._sut("Download this dataset", ["GSE12345"])
        assert result == "Download this dataset\nDatasets of Interest: GSE12345"

    def test_empty_user_text_with_accessions(self):
        result = self._sut("", ["GSE12345"])
        assert result == "\nDatasets of Interest: GSE12345"

    def test_accessions_appear_in_order(self):
        result = self._sut("Query", ["GSE33333", "GSE11111", "GSE22222"])
        assert result == "Query\nDatasets of Interest: GSE33333, GSE11111, GSE22222"
