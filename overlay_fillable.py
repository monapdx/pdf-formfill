from __future__ import annotations

from pathlib import Path
import json

import pikepdf

from renderer_overlay_fields import build_fields_only_pdf


def make_fillable_from_background(template_json: str, output_pdf: str) -> None:
    """
    Build a temp "fields-only" PDF (AcroForm widgets), overlay it onto the background PDF,
    then copy the AcroForm so the result stays fillable.

    Mobile/Pixels often render multiline text field carets "vertically centered" when an
    existing widget appearance stream (/AP) is present or odd. The fix:
      - Set /AcroForm /NeedAppearances true (ask viewer to regenerate appearances)
      - Remove widget /AP entries (force regeneration, avoids stubborn centered caret)
    """
    # Create a temp fields-only PDF
    tmp = str(Path(output_pdf).with_suffix(".fields_tmp.pdf"))
    build_fields_only_pdf(template_json, tmp)

    tpl = json.loads(Path(template_json).read_text(encoding="utf-8"))
    bg = tpl["background_pdf"]

    with pikepdf.open(bg) as bg_pdf, pikepdf.open(tmp) as fields_pdf:
        if len(bg_pdf.pages) != len(fields_pdf.pages):
            raise ValueError("Background and fields PDF page counts do not match.")

        # Overlay each page's widget annotations (and any other content from fields_pdf)
        for i in range(len(bg_pdf.pages)):
            bg_pdf.pages[i].add_overlay(fields_pdf.pages[i])

        # Copy AcroForm dictionary so fields remain interactive
        if "/AcroForm" in fields_pdf.Root:
            bg_pdf.Root["/AcroForm"] = fields_pdf.Root["/AcroForm"]

            # ✅ Mobile-friendly: request regenerated appearances
            bg_pdf.Root["/AcroForm"]["/NeedAppearances"] = pikepdf.Boolean(True)

            # ✅ Mobile-friendly: remove existing widget appearance streams
            # so viewers can't keep using a "centered" AP for multiline text.
            for page in bg_pdf.pages:
                if "/Annots" not in page:
                    continue

                annots = page["/Annots"]
                for annot in annots:
                    a = annot.get_object()

                    # Only touch widgets
                    if str(a.get("/Subtype")) != "/Widget":
                        continue

                    # Remove widget appearance stream
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