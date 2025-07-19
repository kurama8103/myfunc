import pypdf
import glob
import os
import sys


def merge_pdfs(output_name="merge.pdf", path_dir="."):
    l = glob.glob(os.path.join(path_dir, "*.pdf"))
    path_merge = os.path.join(".", output_name)
    if os.path.exists(path_merge):
        l.remove(path_merge)
    l.sort()

    merger = pypdf.PdfWriter()
    [merger.append(p) for p in l]
    merger.write(path_merge)
    merger.close()


if __name__ == "__main__":
    if len(sys.argv[1:]) > 1:
        merge_pdfs(**sys.argv[1:])
