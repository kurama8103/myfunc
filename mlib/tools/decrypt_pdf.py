import sys
import os
import pypdf


def decrypt_pdf(path_pdf, path_decripted_pdf, password=""):
    reader = pypdf.PdfReader(path_pdf)
    if reader.is_encrypted:
        reader.decrypt(password)

    writer = pypdf.PdfWriter(clone_from=reader)
    with open(path_decripted_pdf, "wb") as f:
        writer.write(f)


if __name__ == "__main__":
    f = sys.argv[1]
    decrypt_pdf(f, os.path.splitext(f)[0] + "_.pdf")
