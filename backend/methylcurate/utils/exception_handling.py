__all__ = ["classify_geo_error"]
import ftplib
import gzip
import socket
import ssl
import urllib.error


def classify_geo_error(e: Exception) -> tuple[str, str, str]:
    """
    Returns:
        category: machine-readable error class
        user_message: end-user safe message
        dev_message: more technical guidance
    """

    if isinstance(e, urllib.error.HTTPError):
        return (
            "http_error",
            "The GEO server returned an error. Please try again later.",
            f"HTTP error {e.code}: {e.reason}",
        )

    # -------------------------
    # NETWORK / CONNECTIVITY
    # -------------------------
    if isinstance(e, (urllib.error.URLError, socket.timeout, socket.gaierror, ConnectionResetError, TimeoutError, ssl.SSLError)):
        return ("network_error", "Network connection failed. Please check your internet or firewall.", str(e))

    # -------------------------
    # FTP-SPECIFIC FAILURES
    # -------------------------
    if isinstance(e, (ftplib.all_errors)):
        return ("ftp_error", "Unable to reach GEO via FTP. Check your network or try again later.", str(e))

    # -------------------------
    # DISK / FILE SYSTEM ISSUES
    # -------------------------
    if isinstance(e, (PermissionError, FileNotFoundError, OSError)):
        # Narrow OS errors further if needed
        if hasattr(e, "errno"):
            if e.errno == 28:  # No space left on device
                return ("disk_full", "There is not enough disk space to download this dataset.", "Disk full (errno 28)")

        return ("filesystem_error", "There was a problem writing files to disk. Check permissions and space.", str(e))

    # -------------------------
    # CORRUPT / NON-STANDARD GEO DATA
    # -------------------------
    if isinstance(e, (gzip.BadGzipFile, EOFError, ValueError, KeyError)):
        return (
            "corrupt_geo_record",
            "This GEO dataset appears to be corrupted or non-standard.",
            "Parsing or decompression failed",
        )

    # GEOparse sometimes raises RuntimeError for malformed records
    if isinstance(e, RuntimeError):
        return ("geo_parse_error", "This GEO record could not be parsed correctly.", str(e))

    # -------------------------
    # FALLBACK
    # -------------------------
    return ("unknown_error", "An unexpected error occurred while downloading the dataset.", repr(e))
