__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

"""Shared MNIST, clustering, and RBF helpers."""

import csv
import struct
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42


def read_images(path):
    with open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
    if magic != 2051 or rows != 28 or cols != 28 or pixels.size != count * 784:
        raise ValueError(f"Bad MNIST images: {path}")
    return pixels.reshape(count, 784).astype(np.float32) / 255.0


def read_labels(path):
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if magic != 2049 or labels.size != count or np.any(labels > 9):
        raise ValueError(f"Bad MNIST labels: {path}")
    return labels.copy()


def dataset(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    return path


def distances_squared(x, centers):
    # ||x-c||^2 = ||x||^2 + ||c||^2 - 2(x @ c.T)
    d = (
        (x * x).sum(axis=1, keepdims=True)
        + (centers * centers).sum(axis=1)[None, :]
        - 2 * x @ centers.T
    )
    return np.maximum(d, 0)


def kmeans(x, k, runs=5, max_iterations=35, batch_size=1000, seed=SEED):
    """Full-data Lloyd's K-means, five independent starts, lowest final distortion wins."""
    best = None
    for run in range(runs):
        rng = np.random.default_rng(seed + run)
        centers = x[rng.choice(len(x), k, replace=False)].copy()
        old_assignments = None
        for iteration in range(max_iterations):
            # Calculate nearest centroid without a per-pixel or per-center loop.
            assignments = np.empty(len(x), np.int32)
            sums = np.zeros((k, x.shape[1]), np.float64)
            sizes = np.zeros(k, np.int64)
            for start in range(0, len(x), batch_size):
                stop = min(start + batch_size, len(x))
                block = x[start:stop]
                labels = distances_squared(block, centers).argmin(axis=1)
                assignments[start:stop] = labels
                np.add.at(sums, labels, block)
                sizes += np.bincount(labels, minlength=k)
            new_centers = centers.copy()
            nonempty = sizes > 0
            new_centers[nonempty] = (sums[nonempty] / sizes[nonempty, None]).astype(
                np.float32
            )
            if np.any(~nonempty):
                # Re-seed empty clusters with random training examples.
                new_centers[~nonempty] = x[
                    rng.choice(len(x), (~nonempty).sum(), replace=False)
                ]
            converged = old_assignments is not None and np.array_equal(
                assignments, old_assignments
            )
            centers = new_centers
            old_assignments = assignments
            print(
                f"K={k}, start={run + 1}/{runs}, iteration={iteration + 1}, changed={not converged}",
                flush=True,
            )
            if converged:
                break
        # Compare all runs with distortion evaluated for their final centers.
        distortion = 0.0
        for start in range(0, len(x), batch_size):
            distortion += (
                distances_squared(x[start : start + batch_size], centers)
                .min(axis=1)
                .sum(dtype=np.float64)
            )
        print(f"K={k}, start={run + 1}: total distortion={distortion:.3f}", flush=True)
        if best is None or distortion < best[0]:
            best = (float(distortion), centers.copy(), run + 1)
    return best


def write_centers(k, centers):
    # An .xlsx file is a ZIP archive of XML files. This uses Python's built-in
    # modules so Windows does not need any spreadsheet package installed.
    from zipfile import ZipFile, ZIP_DEFLATED
    from xml.sax.saxutils import escape

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if centers.shape != (k, 784):
        raise ValueError(f"Expected {k} rows and 784 columns, got {centers.shape}")

    # Save NumPy backup first, even if Excel export later encounters an issue.
    np.save(RESULTS_DIR / f"centers_{k}.npy", centers)
    path = RESULTS_DIR / f"centers_{k}.xlsx"

    def excel_column(index):
        name = ""
        while index:
            index, rem = divmod(index - 1, 26)
            name = chr(65 + rem) + name
        return name

    with ZipFile(path, "w", ZIP_DEFLATED) as xlsx:
        xlsx.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>""",
        )
        xlsx.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        xlsx.writestr(
            "xl/workbook.xml",
            f"""<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="centers_{k}" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        )
        xlsx.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        )
        pieces = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>',
        ]
        for row_number, row in enumerate(centers, 1):
            pieces.append(f'<row r="{row_number}">')
            for col_number, value in enumerate(row, 1):
                ref = f"{excel_column(col_number)}{row_number}"
                pieces.append(f'<c r="{ref}"><v>{float(value):.12g}</v></c>')
            pieces.append("</row>")
        pieces.append("</sheetData></worksheet>")
        xlsx.writestr("xl/worksheets/sheet1.xml", "".join(pieces))
    print(f"Saved {k} centroid rows to {path}")


def read_centers(k):
    # Read the Excel file itself, as required for the RBF program.
    from zipfile import ZipFile
    import xml.etree.ElementTree as ET

    path = RESULTS_DIR / f"centers_{k}.xlsx"
    with ZipFile(path) as xlsx:
        root = ET.fromstring(xlsx.read("xl/worksheets/sheet1.xml"))
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = []
    for row in root.findall(".//s:sheetData/s:row", ns):
        vals = [float(cell.find("s:v", ns).text) for cell in row.findall("s:c", ns)]
        rows.append(vals)
    centers = np.array(rows, dtype=np.float32)
    if centers.shape != (k, 784):
        raise ValueError(f"Invalid center matrix {centers.shape}; expected {(k, 784)}")
    return centers


def rbf_features(x, centers, variance, block=1000):
    # Phi_j(x) = exp(-||x-c_j||^2 / (2*variance))
    outputs = np.empty((len(x), len(centers)), np.float32)
    for start in range(0, len(x), block):
        stop = min(start + block, len(x))
        outputs[start:stop] = np.exp(
            -distances_squared(x[start:stop], centers) / (2 * variance)
        )
    return outputs


def save_rows(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
