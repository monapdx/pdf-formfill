from __future__ import annotations

from pathlib import Path
import json
import pikepdf

from renderer_overlay_fields import build_fields_only_pdf


def make_fillable_from_background(template_json: str, output_pdf: str) -> None:
    """
    Build a temp "fields-only" PDF (AcroForm widgets), overlay it onto the background PDF,
    then copy the AcroForm so the result stays fillable.

    Fix for Android / mobile viewers (Google Pixel, Drive viewer, etc.):
      - Set /AcroForm /NeedAppearances = True
      - Remove widget /AP streams so the viewer regenerates them
        (prevents multiline caret starting vertically centered)
    """

    # Create temp fields-only PDF
    tmp = str(Path(output_pdf).with_suffix(".fields_tmp.pdf"))
    build_fields_only_pdf(template_json, tmp)

    tpl = json.loads(Path(template_json).read_text(encoding="utf-8"))
    bg = tpl["background_pdf"]

    with pikepdf.open(bg) as bg_pdf, pikepdf.open(tmp) as fields_pdf:
        if len(bg_pdf.pages) != len(fields_pdf.pages):
            raise ValueError("Background and fields PDF page counts do not match.")

        # Overlay fields onto background pages
        for i in range(len(bg_pdf.pages)):
            bg_pdf.pages[i].add_overlay(fields_pdf.pages[i])

        # Copy AcroForm so fields remain interactive
        if "/AcroForm" in fields_pdf.Root:
            bg_pdf.Root["/AcroForm"] = fields_pdf.Root["/AcroForm"]

            # ✅ Ask viewers (especially mobile) to regenerate field appearances
            bg_pdf.Root["/AcroForm"]["/NeedAppearances"] = True

            # ✅ Remove existing widget appearance streams (/AP)
            # so Android viewers don't keep using a centered baseline
            for page in bg_pdf.pages:
                if "/Annots" not in page:
                    continue

                for annot in page["/Annots"]:
                    a = annot.get_object()

                    if str(a.get("/Subtype")) != "/Widget":
                        continue

                    if "/AP" in a:
                        del a["/AP"]

        bg_pdf.save(output_pdf)

    # Cleanup temp file
    try:
        Path(tmp).unlink()
    except OSError:
        pass


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("template", help="Template JSON")
    ap.add_argument("output", help="Output fillable PDF")
    args = ap.parse_args()

    make_fillable_from_background(args.template, args.output)
    print("Wrote:", args.output)