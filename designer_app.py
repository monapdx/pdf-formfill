from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog

# Preview rendering
import fitz  # PyMuPDF (GPL)
from PIL import Image, ImageTk

# PDF form generation / overlay
import pikepdf
from reportlab.pdfgen import canvas


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Field:
    type: str  # text, checkbox, dropdown, radio
    name: str
    label: str
    x: float
    y: float
    w: float
    h: float
    font_size: int = 11
    required: bool = False
    multiline: bool = False
    options: Optional[List[str]] = None   # dropdown
    default: Optional[Any] = None         # text: str, checkbox: bool, dropdown: str, radio: value
    value: Optional[str] = None           # radio export value (per button)
    group: str = ""                       # checkbox grouping (question id)


# ----------------------------
# PDF helpers (fields-only + overlay)
# ----------------------------

def get_pdf_page_sizes_points(pdf_path: str) -> List[Tuple[float, float]]:
    sizes: List[Tuple[float, float]] = []
    with pikepdf.open(pdf_path) as pdf:
        for p in pdf.pages:
            mb = p.MediaBox
            w = float(mb[2]) - float(mb[0])
            h = float(mb[3]) - float(mb[1])
            sizes.append((w, h))
    return sizes


def _draw_fields_for_page(c: canvas.Canvas, fields: List[Dict[str, Any]]) -> None:
    form = c.acroForm

    for f in fields:
        ftype = f["type"]
        name = f["name"]
        label = f.get("label", "") or name

        x = float(f["x"]); y = float(f["y"])
        w = float(f["w"]); h = float(f["h"])
        required = bool(f.get("required", False))

        if ftype == "text":
            multiline = bool(f.get("multiline", False))
            font_size = int(f.get("font_size", 11))
            value = f.get("default", "")

            flags = 0
            if multiline:
                flags |= 4096  # multiline
            if required:
                flags |= 2     # required

            form.textfield(
                name=name,
                tooltip=label,
                x=x, y=y, width=w, height=h,
                fontName="Helvetica", fontSize=font_size,
                value=value,
                borderStyle="inset", borderWidth=1,
                forceBorder=True,
                fieldFlags=flags,
            )

        elif ftype == "checkbox":
            size = min(w, h)
            form.checkbox(
                name=name,
                tooltip=label,
                x=x, y=y, size=size,
                checked=bool(f.get("default", False)),
                borderWidth=1,
                forceBorder=True,
            )

        elif ftype == "dropdown":
            options = f.get("options") or ["Option A", "Option B"]
            value = f.get("default", options[0] if options else "")
            flags = 2 if required else 0
            form.choice(
                name=name,
                tooltip=label,
                x=x, y=y, width=w, height=h,
                options=options,
                value=value,
                borderStyle="solid",
                borderWidth=1,
                forceBorder=True,
                fieldFlags=flags,
            )

        elif ftype == "radio":
            export_value = f.get("value")
            if not export_value:
                raise ValueError(f"Radio field '{name}' missing 'value' (export value).")

            size = min(w, h)
            form.radio(
                name=name,  # group name
                tooltip=label,
                value=export_value,
                x=x, y=y,
                buttonStyle="circle",
                selected=(f.get("default") == export_value),
                size=size,
                borderWidth=1,
                forceBorder=True,
            )

        else:
            raise ValueError(f"Unsupported field type: {ftype}")


def build_fields_only_pdf(template_dict: Dict[str, Any], out_pdf: str) -> None:
    bg = (template_dict.get("background_pdf") or "").strip()
    if not bg:
        raise ValueError("Template missing background_pdf")

    page_sizes = get_pdf_page_sizes_points(bg)
    pages = template_dict.get("pages") or []

    fields_by_page: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(page_sizes))}
    for p in pages:
        idx = int(p["index"])
        fields_by_page.setdefault(idx, []).extend(p.get("fields") or [])

    c: Optional[canvas.Canvas] = None
    for i, (pw, ph) in enumerate(page_sizes):
        if c is None:
            c = canvas.Canvas(out_pdf, pagesize=(pw, ph))
        else:
            c.setPageSize((pw, ph))

        _draw_fields_for_page(c, fields_by_page.get(i, []))
        c.showPage()

    if c is None:
        raise ValueError("No pages found.")
    c.save()


def make_fillable_from_background(template_dict: Dict[str, Any], output_pdf: str) -> None:
    bg = (template_dict.get("background_pdf") or "").strip()
    if not bg:
        raise ValueError("Template missing background_pdf")

    tmp = str(Path(output_pdf).with_suffix(".fields_tmp.pdf"))
    build_fields_only_pdf(template_dict, tmp)

    with pikepdf.open(bg) as bg_pdf, pikepdf.open(tmp) as fields_pdf:
        if len(bg_pdf.pages) != len(fields_pdf.pages):
            raise ValueError("Background and fields PDF page counts do not match.")

        for i in range(len(bg_pdf.pages)):
            bg_page = bg_pdf.pages[i]
            fld_page = fields_pdf.pages[i]

            # Visual overlay
            bg_page.add_overlay(fld_page)

            # Copy widget annotations
            if "/Annots" in fld_page:
                if "/Annots" not in bg_page:
                    bg_page["/Annots"] = pikepdf.Array()
                for annot in fld_page["/Annots"]:
                    bg_page["/Annots"].append(bg_pdf.copy_foreign(annot))

        # Copy/merge AcroForm
        if "/AcroForm" in fields_pdf.Root:
            foreign_af = bg_pdf.copy_foreign(fields_pdf.Root["/AcroForm"])
            if "/AcroForm" not in bg_pdf.Root:
                bg_pdf.Root["/AcroForm"] = foreign_af
            else:
                bg_af = bg_pdf.Root["/AcroForm"]
                if "/Fields" not in bg_af:
                    bg_af["/Fields"] = pikepdf.Array()
                if "/Fields" in foreign_af:
                    for fld in foreign_af["/Fields"]:
                        bg_af["/Fields"].append(fld)
                if "/NeedAppearances" in foreign_af and "/NeedAppearances" not in bg_af:
                    bg_af["/NeedAppearances"] = foreign_af["/NeedAppearances"]

        if "/AcroForm" in bg_pdf.Root:
            bg_pdf.Root["/AcroForm"]["/NeedAppearances"] = pikepdf.Boolean(True)

        bg_pdf.save(output_pdf)

    try:
        Path(tmp).unlink()
    except OSError:
        pass


# ----------------------------
# Preview-filled helper
# ----------------------------

def _first_on_state_from_widget(widget: pikepdf.Object) -> Optional[pikepdf.Name]:
    try:
        ap = widget.get("/AP", None)
        if not ap:
            return None
        n = ap.get("/N", None)
        if not n:
            return None
        keys = list(n.keys())
        for k in keys:
            if str(k) != "/Off":
                return k
    except Exception:
        return None
    return None


def create_preview_filled_pdf(fillable_pdf_path: str, out_preview_pdf_path: str) -> None:
    with pikepdf.open(fillable_pdf_path) as pdf:
        if "/AcroForm" not in pdf.Root:
            raise ValueError("No /AcroForm found in PDF.")

        af = pdf.Root["/AcroForm"]
        af["/NeedAppearances"] = pikepdf.Boolean(True)

        for page in pdf.pages:
            if "/Annots" not in page:
                continue
            for annot in page["/Annots"]:
                if str(annot.get("/Subtype", "")) != "/Widget":
                    continue
                ft = annot.get("/FT", None)
                if not ft:
                    continue

                ft_s = str(ft)
                if ft_s == "/Tx":
                    annot["/V"] = pikepdf.String("Sample text")

                elif ft_s == "/Ch":
                    opt = annot.get("/Opt", None)
                    if opt is None and "/Parent" in annot:
                        opt = annot["/Parent"].get("/Opt", None)
                    if opt and len(opt) > 0:
                        first = opt[0]
                        if isinstance(first, pikepdf.Array) and len(first) > 0:
                            annot["/V"] = first[0]
                        else:
                            annot["/V"] = first
                    else:
                        annot["/V"] = pikepdf.String("Option A")

                elif ft_s == "/Btn":
                    on_state = _first_on_state_from_widget(annot)
                    if on_state is None:
                        continue
                    annot["/V"] = on_state
                    annot["/AS"] = on_state

        pdf.save(out_preview_pdf_path)


# ----------------------------
# Designer App
# ----------------------------

class DesignerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Fillable Form Designer (Multi-page MVP)")
        self.geometry("1340x900")

        self.background_pdf: str = ""
        self.doc: Optional[fitz.Document] = None
        self.page_count: int = 0
        self.current_page: int = 0

        self.zoom: float = 1.6
        self.current_image: Optional[ImageTk.PhotoImage] = None
        self.page_points: Tuple[float, float] = (612, 792)
        self.preview_scale: float = self.zoom

        self.fields_by_page: Dict[int, List[Field]] = {}
        self._mode: str = "idle"  # idle | create | move

        self._drag_start: Optional[Tuple[int, int]] = None
        self._drag_rect_id: Optional[int] = None

        # MULTI-SELECTION: store indices (on current page)
        self.selected_indices: List[int] = []

        # move state
        self._move_start_xy_px: Optional[Tuple[int, int]] = None
        self._move_orig_xy_pt: Optional[Tuple[float, float]] = None
        self._move_active_idx: Optional[int] = None

        self._build_ui()

    # ---------------- UI ----------------

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=8, pady=8)

        ttk.Button(top, text="Open PDF…", command=self.open_pdf).pack(side="left")

        nav = ttk.Frame(top)
        nav.pack(side="left", padx=(12, 0))
        ttk.Button(nav, text="◀ Prev", command=self.prev_page).pack(side="left")
        ttk.Button(nav, text="Next ▶", command=self.next_page).pack(side="left", padx=(6, 0))

        ttk.Label(nav, text="Page:").pack(side="left", padx=(12, 4))
        self.page_var = tk.StringVar(value="0 / 0")
        ttk.Label(nav, textvariable=self.page_var).pack(side="left")

        ttk.Label(nav, text="Jump:").pack(side="left", padx=(12, 4))
        self.jump_var = tk.StringVar(value="1")
        ttk.Entry(nav, textvariable=self.jump_var, width=5).pack(side="left")
        ttk.Button(nav, text="Go", command=self.jump_to_page).pack(side="left", padx=(6, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=12)

        ttk.Button(top, text="Export Template JSON…", command=self.export_template).pack(side="left")
        ttk.Button(top, text="Generate Fillable PDF…", command=self.generate_fillable_pdf).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Preview Filled PDF…", command=self.preview_filled_pdf).pack(side="left", padx=(8, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=12)

        ttk.Label(top, text="Type:").pack(side="left", padx=(0, 4))
        self.field_type = tk.StringVar(value="text")
        ttk.Combobox(
            top, textvariable=self.field_type,
            values=["text", "textarea", "checkbox", "radio", "dropdown"],
            width=10, state="readonly"
        ).pack(side="left")

        ttk.Label(top, text="Name/Group:").pack(side="left", padx=(12, 4))
        self.field_name = tk.StringVar(value="field_1")
        ttk.Entry(top, textvariable=self.field_name, width=16).pack(side="left")

        ttk.Label(top, text="Label:").pack(side="left", padx=(12, 4))
        self.field_label = tk.StringVar(value="Label")
        ttk.Entry(top, textvariable=self.field_label, width=20).pack(side="left")

        self.required_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Required", variable=self.required_var).pack(side="left", padx=(10, 0))

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(main, bg="#ddd")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        right = ttk.Frame(main, width=420)
        right.pack(side="right", fill="y", padx=(10, 0))

        ttk.Label(right, text="Fields on this page (Ctrl/Shift to multi-select)").pack(anchor="w")
        self.listbox = tk.Listbox(right, height=18, selectmode="extended")
        self.listbox.pack(fill="x", pady=(4, 6))
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        btnrow = ttk.Frame(right)
        btnrow.pack(fill="x")
        ttk.Button(btnrow, text="Delete", command=self.delete_selected).pack(side="left", fill="x", expand=True)
        ttk.Button(btnrow, text="Rename…", command=self.rename_selected).pack(side="left", fill="x", expand=True, padx=(6, 0))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        # Group select
        grp = ttk.LabelFrame(right, text="Group tools")
        grp.pack(fill="x")

        ttk.Button(grp, text="Select Radio Group (same name)", command=self.select_radio_group).pack(fill="x", pady=(6, 4), padx=6)
        ttk.Button(grp, text="Select Checkbox Group (same group id)", command=self.select_checkbox_group).pack(fill="x", pady=(0, 6), padx=6)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        # Align/layout tools
        tools = ttk.LabelFrame(right, text="Align & Layout (applies to current selection)")
        tools.pack(fill="x")

        row1 = ttk.Frame(tools)
        row1.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(row1, text="Align Left", command=self.align_left).pack(side="left", fill="x", expand=True)
        ttk.Button(row1, text="Center X", command=self.align_center_x).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row1, text="Align Right", command=self.align_right).pack(side="left", fill="x", expand=True)

        row2 = ttk.Frame(tools)
        row2.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(row2, text="Align Top", command=self.align_top).pack(side="left", fill="x", expand=True)
        ttk.Button(row2, text="Middle Y", command=self.align_middle_y).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row2, text="Align Bottom", command=self.align_bottom).pack(side="left", fill="x", expand=True)

        row3 = ttk.Frame(tools)
        row3.pack(fill="x", padx=6, pady=(8, 0))
        ttk.Label(row3, text="Spacing (px):").pack(side="left")
        self.spacing_var = tk.StringVar(value="12")
        ttk.Entry(row3, textvariable=self.spacing_var, width=6).pack(side="left", padx=(6, 0))

        row4 = ttk.Frame(tools)
        row4.pack(fill="x", padx=6, pady=(6, 6))
        ttk.Button(row4, text="Stack Vertical", command=self.stack_vertical).pack(side="left", fill="x", expand=True)
        ttk.Button(row4, text="Stack Horizontal", command=self.stack_horizontal).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row4, text="Distribute V", command=self.distribute_vertical).pack(side="left", fill="x", expand=True)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        # Uniform size tools (carryover)
        size_tools = ttk.LabelFrame(right, text="Uniform size")
        size_tools.pack(fill="x")
        rowS = ttk.Frame(size_tools)
        rowS.pack(fill="x", padx=6, pady=6)
        ttk.Button(rowS, text="Match size → ALL on page", command=self.match_size_all).pack(side="left", fill="x", expand=True)
        ttk.Button(rowS, text="Match size → SAME TYPE", command=self.match_size_same_type).pack(side="left", fill="x", expand=True, padx=(6, 0))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        self.info = tk.Text(right, height=10, wrap="word")
        self.info.pack(fill="both", expand=True)
        self._refresh_info()

    # -------------- PDF preview --------------

    def open_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not path:
            return

        try:
            if self.doc is not None:
                self.doc.close()
        except Exception:
            pass

        self.background_pdf = path
        self.doc = fitz.open(path)
        self.page_count = self.doc.page_count
        self.current_page = 0
        self.fields_by_page = {i: [] for i in range(self.page_count)}
        self.selected_indices = []

        self._render_current_page()
        self._update_page_label()
        self._refresh_fields_list()
        messagebox.showinfo("Loaded", f"Loaded PDF with {self.page_count} pages.\n\nTip: Ctrl/Shift multi-select in the list.")

    def _render_current_page(self):
        if not self.doc:
            return
        page = self.doc.load_page(self.current_page)
        rect = page.rect
        self.page_points = (rect.width, rect.height)

        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        self.preview_scale = self.zoom
        self.current_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, img.size[0], img.size[1]))
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)
        self._redraw_field_boxes()

    def _update_page_label(self):
        self.page_var.set(f"{self.current_page + 1} / {self.page_count}")

    def prev_page(self):
        if not self.doc:
            return
        if self.current_page > 0:
            self.current_page -= 1
            self.selected_indices = []
            self._render_current_page()
            self._update_page_label()
            self._refresh_fields_list()

    def next_page(self):
        if not self.doc:
            return
        if self.current_page < self.page_count - 1:
            self.current_page += 1
            self.selected_indices = []
            self._render_current_page()
            self._update_page_label()
            self._refresh_fields_list()

    def jump_to_page(self):
        if not self.doc:
            return
        try:
            p = int(self.jump_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid", "Enter a page number (1-based).")
            return
        if p < 1 or p > self.page_count:
            messagebox.showerror("Out of range", f"Page must be 1 to {self.page_count}.")
            return
        self.current_page = p - 1
        self.selected_indices = []
        self._render_current_page()
        self._update_page_label()
        self._refresh_fields_list()

    # -------------- selection helpers --------------

    def _fields(self) -> List[Field]:
        return self.fields_by_page.get(self.current_page, [])

    def _set_selection(self, indices: List[int]):
        indices = sorted(set(i for i in indices if 0 <= i < len(self._fields())))
        self.selected_indices = indices

        # sync listbox
        self.listbox.selection_clear(0, "end")
        for i in self.selected_indices:
            self.listbox.selection_set(i)
        if self.selected_indices:
            self.listbox.activate(self.selected_indices[-1])

        self._redraw_field_boxes()

    def _primary_index(self) -> Optional[int]:
        return self.selected_indices[-1] if self.selected_indices else None

    def on_list_select(self, _evt):
        self._set_selection(list(self.listbox.curselection()))

    # -------------- canvas geometry + hit test --------------

    def _field_rect_canvas_px(self, f: Field) -> Tuple[float, float, float, float]:
        page_h_px = self.page_points[1] * self.preview_scale
        left = f.x * self.preview_scale
        right = (f.x + f.w) * self.preview_scale
        top = page_h_px - (f.y + f.h) * self.preview_scale
        bottom = page_h_px - f.y * self.preview_scale
        return left, top, right, bottom

    def _hit_test_field(self, x_px: float, y_px: float) -> Optional[int]:
        fields = self._fields()
        for idx in range(len(fields) - 1, -1, -1):
            l, t, r, b = self._field_rect_canvas_px(fields[idx])
            if l <= x_px <= r and t <= y_px <= b:
                return idx
        return None

    # -------------- mouse interactions: create vs move, with modifiers --------------

    def on_mouse_down(self, e):
        if not self.doc:
            return

        hit = self._hit_test_field(e.x, e.y)

        ctrl = bool(e.state & 0x0004)   # Control (Windows)
        shift = bool(e.state & 0x0001)  # Shift

        if hit is not None:
            # selection behavior
            if ctrl:
                # toggle
                if hit in self.selected_indices:
                    new_sel = [i for i in self.selected_indices if i != hit]
                else:
                    new_sel = self.selected_indices + [hit]
                self._set_selection(new_sel)
            elif shift and self.selected_indices:
                # range select from last primary to hit
                a = self._primary_index()
                if a is None:
                    self._set_selection([hit])
                else:
                    lo, hi = sorted([a, hit])
                    self._set_selection(list(range(lo, hi + 1)))
            else:
                self._set_selection([hit])

            # move only if single selected (keeps logic simple and predictable)
            if len(self.selected_indices) == 1:
                self._mode = "move"
                self._move_active_idx = hit
                self._move_start_xy_px = (e.x, e.y)
                f = self._fields()[hit]
                self._move_orig_xy_pt = (f.x, f.y)
            else:
                self._mode = "idle"
            return

        # click empty area: start create
        self._mode = "create"
        self._drag_start = (e.x, e.y)
        self._drag_rect_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=2)

        # if no modifiers, clear selection
        if not (ctrl or shift):
            self._set_selection([])

    def on_mouse_drag(self, e):
        if self._mode == "create":
            if not self._drag_start or not self._drag_rect_id:
                return
            x0, y0 = self._drag_start
            self.canvas.coords(self._drag_rect_id, x0, y0, e.x, e.y)
            return

        if self._mode == "move":
            if self._move_active_idx is None or self._move_start_xy_px is None or self._move_orig_xy_pt is None:
                return

            f = self._fields()[self._move_active_idx]

            dx_px = e.x - self._move_start_xy_px[0]
            dy_px = e.y - self._move_start_xy_px[1]
            dx_pt = dx_px / self.preview_scale
            dy_pt = -dy_px / self.preview_scale

            new_x = self._move_orig_xy_pt[0] + dx_pt
            new_y = self._move_orig_xy_pt[1] + dy_pt

            page_w_pt, page_h_pt = self.page_points
            new_x = max(0, min(new_x, page_w_pt - f.w))
            new_y = max(0, min(new_y, page_h_pt - f.h))

            f.x = float(new_x)
            f.y = float(new_y)
            self._redraw_field_boxes()

    def on_mouse_up(self, e):
        if not self.doc:
            return

        if self._mode == "move":
            self._mode = "idle"
            self._move_active_idx = None
            self._move_start_xy_px = None
            self._move_orig_xy_pt = None
            self._refresh_fields_list()
            return

        if self._mode != "create":
            self._mode = "idle"
            return

        if not self._drag_start or not self._drag_rect_id:
            self._mode = "idle"
            return

        x0, y0 = self._drag_start
        x1, y1 = e.x, e.y

        left, right = sorted([x0, x1])
        top, bottom = sorted([y0, y1])

        if (right - left) < 8 or (bottom - top) < 8:
            self._cancel_create()
            self._mode = "idle"
            return

        page_h_px = self.page_points[1] * self.preview_scale

        x_pt = left / self.preview_scale
        y_pt = (page_h_px - bottom) / self.preview_scale
        w_pt = (right - left) / self.preview_scale
        h_pt = (bottom - top) / self.preview_scale

        ui_type = self.field_type.get().strip()
        required = bool(self.required_var.get())
        label = self.field_label.get().strip()
        name_or_group = self.field_name.get().strip() or f"field_{self._next_field_number()}"

        field: Optional[Field] = None

        if ui_type in ("text", "textarea"):
            multiline = (ui_type == "textarea")
            font_size = simpledialog.askinteger(
                "Font size", "Font size (e.g., 10–12):",
                initialvalue=11, minvalue=6, maxvalue=36, parent=self
            )
            if font_size is None:
                self._cancel_create(); self._mode = "idle"; return

            field = Field(
                type="text", name=name_or_group, label=label,
                x=float(x_pt), y=float(y_pt), w=float(w_pt), h=float(h_pt),
                font_size=int(font_size), required=required, multiline=multiline, default=""
            )

        elif ui_type == "checkbox":
            # prompt for checkbox "question group"
            group = simpledialog.askstring(
                "Checkbox group id",
                "Enter a group id for this checkbox question.\n"
                "Example: symptoms  (leave blank for none)\n\n"
                "This is used for 'Select Checkbox Group'.",
                initialvalue="",
                parent=self
            )
            if group is None:
                self._cancel_create(); self._mode = "idle"; return

            field = Field(
                type="checkbox", name=name_or_group, label=label,
                x=float(x_pt), y=float(y_pt), w=float(w_pt), h=float(h_pt),
                required=required, default=False, group=(group.strip() if group else "")
            )

        elif ui_type == "dropdown":
            options_str = simpledialog.askstring(
                "Dropdown options",
                "Enter options separated by commas:\nExample: Red, Green, Blue",
                initialvalue="Option A, Option B",
                parent=self,
            )
            if options_str is None:
                self._cancel_create(); self._mode = "idle"; return
            options = [o.strip() for o in options_str.split(",") if o.strip()]
            if not options:
                messagebox.showerror("Invalid", "Dropdown needs at least one option.")
                self._cancel_create(); self._mode = "idle"; return

            field = Field(
                type="dropdown", name=name_or_group, label=label,
                x=float(x_pt), y=float(y_pt), w=float(w_pt), h=float(h_pt),
                required=required, options=options, default=options[0]
            )

        elif ui_type == "radio":
            export_value = simpledialog.askstring(
                "Radio option value",
                "Enter the export value for this radio button.\nExample: email",
                initialvalue="option1",
                parent=self,
            )
            if export_value is None or not export_value.strip():
                self._cancel_create(); self._mode = "idle"; return

            field = Field(
                type="radio", name=name_or_group, label=label,
                x=float(x_pt), y=float(y_pt), w=float(w_pt), h=float(h_pt),
                required=required, value=export_value.strip(), default=None
            )

        else:
            messagebox.showerror("Unsupported", f"Unknown type: {ui_type}")
            self._cancel_create(); self._mode = "idle"; return

        self._fields().append(field)

        # select newly added
        self._set_selection([len(self._fields()) - 1])
        self._refresh_fields_list()
        self._redraw_field_boxes()
        self._cancel_create()
        self.field_name.set(f"field_{self._next_field_number()}")
        self._mode = "idle"

    def _cancel_create(self):
        if self._drag_rect_id:
            try:
                self.canvas.delete(self._drag_rect_id)
            except Exception:
                pass
        self._drag_start = None
        self._drag_rect_id = None

    def _next_field_number(self) -> int:
        total = sum(len(v) for v in self.fields_by_page.values()) if self.fields_by_page else 0
        return total + 1

    # -------------- list + redraw --------------

    def _refresh_fields_list(self):
        self.listbox.delete(0, "end")
        for f in self._fields():
            if f.type == "radio":
                self.listbox.insert("end", f"radio: {f.name} = {f.value}")
            elif f.type == "checkbox" and f.group:
                self.listbox.insert("end", f"checkbox[{f.group}]: {f.name}")
            elif f.type == "text" and f.multiline:
                self.listbox.insert("end", f"textarea: {f.name}")
            else:
                self.listbox.insert("end", f"{f.type}: {f.name}")

        # reapply selection
        self.listbox.selection_clear(0, "end")
        for i in self.selected_indices:
            if 0 <= i < self.listbox.size():
                self.listbox.selection_set(i)

    def _redraw_field_boxes(self):
        self.canvas.delete("fieldbox")
        if not self.doc:
            return
        fields = self._fields()
        for idx, f in enumerate(fields):
            l, t, r, b = self._field_rect_canvas_px(f)
            selected = (idx in self.selected_indices)
            outline = "red" if selected else "blue"
            width = 3 if selected else 2
            self.canvas.create_rectangle(l, t, r, b, outline=outline, width=width, tags="fieldbox")
            if f.type == "radio":
                label = f"{f.name}={f.value}"
            elif f.type == "checkbox" and f.group:
                label = f"{f.name} ({f.group})"
            elif f.type == "text" and f.multiline:
                label = f"{f.name} (textarea)"
            else:
                label = f.name
            self.canvas.create_text(l + 4, t + 10, anchor="w", text=label, fill=outline, tags="fieldbox")

    # -------------- delete / rename --------------

    def delete_selected(self):
        if not self.selected_indices:
            return
        fields = self._fields()
        for idx in sorted(self.selected_indices, reverse=True):
            if 0 <= idx < len(fields):
                del fields[idx]
        self._set_selection([])
        self._refresh_fields_list()
        self._redraw_field_boxes()

    def rename_selected(self):
        pi = self._primary_index()
        if pi is None:
            return
        f = self._fields()[pi]

        new_name = simpledialog.askstring(
            "Rename field",
            "Enter new field name (for radio: group name):",
            initialvalue=f.name,
            parent=self,
        )
        if new_name and new_name.strip():
            f.name = new_name.strip()

        if f.type == "radio":
            new_val = simpledialog.askstring(
                "Radio option value",
                "Enter export value for this radio button:",
                initialvalue=f.value or "",
                parent=self,
            )
            if new_val and new_val.strip():
                f.value = new_val.strip()

        if f.type == "checkbox":
            new_group = simpledialog.askstring(
                "Checkbox group id",
                "Enter checkbox group id (question id).",
                initialvalue=f.group or "",
                parent=self,
            )
            if new_group is not None:
                f.group = new_group.strip()

        self._refresh_fields_list()
        self._redraw_field_boxes()

    # -------------- group selection --------------

    def select_radio_group(self):
        pi = self._primary_index()
        if pi is None:
            return
        f = self._fields()[pi]
        if f.type != "radio":
            messagebox.showinfo("Not a radio", "Select a radio button first.")
            return
        group_name = f.name
        indices = [i for i, ff in enumerate(self._fields()) if ff.type == "radio" and ff.name == group_name]
        self._set_selection(indices)
        self._refresh_fields_list()

    def select_checkbox_group(self):
        pi = self._primary_index()
        if pi is None:
            return
        f = self._fields()[pi]
        if f.type != "checkbox":
            messagebox.showinfo("Not a checkbox", "Select a checkbox first.")
            return
        if not f.group:
            messagebox.showinfo("No group id", "This checkbox has no group id.\nSet one via Rename…")
            return
        gid = f.group
        indices = [i for i, ff in enumerate(self._fields()) if ff.type == "checkbox" and ff.group == gid]
        self._set_selection(indices)
        self._refresh_fields_list()

    # -------------- spacing parsing --------------

    def _spacing_pt(self) -> float:
        try:
            px = float(self.spacing_var.get().strip())
        except Exception:
            px = 12.0
        return px / self.preview_scale  # convert px to pt

    # -------------- align actions (selection) --------------

    def _selected_fields(self) -> List[Tuple[int, Field]]:
        fields = self._fields()
        return [(i, fields[i]) for i in self.selected_indices if 0 <= i < len(fields)]

    def align_left(self):
        sf = self._selected_fields()
        if len(sf) < 2:
            return
        target = min(f.x for _, f in sf)
        for _, f in sf:
            f.x = target
        self._redraw_field_boxes()

    def align_right(self):
        sf = self._selected_fields()
        if len(sf) < 2:
            return
        target = max((f.x + f.w) for _, f in sf)
        for _, f in sf:
            f.x = target - f.w
        self._redraw_field_boxes()

    def align_center_x(self):
        sf = self._selected_fields()
        if len(sf) < 2:
            return
        centers = [(f.x + f.w / 2.0) for _, f in sf]
        target = sum(centers) / len(centers)
        for _, f in sf:
            f.x = target - f.w / 2.0
        self._redraw_field_boxes()

    def align_top(self):
        sf = self._selected_fields()
        if len(sf) < 2:
            return
        # top edge in PDF coords is y + h
        target = max((f.y + f.h) for _, f in sf)
        for _, f in sf:
            f.y = target - f.h
        self._redraw_field_boxes()

    def align_bottom(self):
        sf = self._selected_fields()
        if len(sf) < 2:
            return
        target = min(f.y for _, f in sf)
        for _, f in sf:
            f.y = target
        self._redraw_field_boxes()

    def align_middle_y(self):
        sf = self._selected_fields()
        if len(sf) < 2:
            return
        centers = [(f.y + f.h / 2.0) for _, f in sf]
        target = sum(centers) / len(centers)
        for _, f in sf:
            f.y = target - f.h / 2.0
        self._redraw_field_boxes()

    # -------------- stack / distribute --------------

    def stack_vertical(self):
        sf = self._selected_fields()
        if len(sf) < 2:
            return
        spacing = self._spacing_pt()

        # sort top-to-bottom (PDF top edge y+h descending)
        sf_sorted = sorted(sf, key=lambda t: (t[1].y + t[1].h), reverse=True)

        # keep x aligned to first element's x
        anchor_x = sf_sorted[0][1].x

        current_top = sf_sorted[0][1].y + sf_sorted[0][1].h
        for idx, f in sf_sorted:
            f.x = anchor_x
            f.y = current_top - f.h
            current_top = f.y - spacing  # next element's top becomes below this one

        self._redraw_field_boxes()

    def stack_horizontal(self):
        sf = self._selected_fields()
        if len(sf) < 2:
            return
        spacing = self._spacing_pt()

        # sort left-to-right
        sf_sorted = sorted(sf, key=lambda t: t[1].x)

        # keep y aligned to first element's y
        anchor_y = sf_sorted[0][1].y

        current_left = sf_sorted[0][1].x
        for idx, f in sf_sorted:
            f.y = anchor_y
            f.x = current_left
            current_left = f.x + f.w + spacing

        self._redraw_field_boxes()

    def distribute_vertical(self):
        sf = self._selected_fields()
        if len(sf) < 3:
            return

        # sort top-to-bottom by top edge
        sf_sorted = sorted(sf, key=lambda t: (t[1].y + t[1].h), reverse=True)

        top_edge = sf_sorted[0][1].y + sf_sorted[0][1].h
        bottom_edge = sf_sorted[-1][1].y

        total_h = sum(f.h for _, f in sf_sorted)
        gaps = len(sf_sorted) - 1
        if gaps <= 0:
            return

        available = (top_edge - bottom_edge) - total_h
        if available < 0:
            available = 0

        gap = available / gaps
        current_top = top_edge
        for idx, f in sf_sorted:
            f.y = current_top - f.h
            current_top = f.y - gap

        self._redraw_field_boxes()

    def distribute_horizontal(self):
        sf = self._selected_fields()
        if len(sf) < 3:
            return

        sf_sorted = sorted(sf, key=lambda t: t[1].x)

        left_edge = sf_sorted[0][1].x
        right_edge = max(f.x + f.w for _, f in sf_sorted)

        total_w = sum(f.w for _, f in sf_sorted)
        gaps = len(sf_sorted) - 1
        if gaps <= 0:
            return

        available = (right_edge - left_edge) - total_w
        if available < 0:
            available = 0
        gap = available / gaps

        current_left = left_edge
        for idx, f in sf_sorted:
            f.x = current_left
            current_left = f.x + f.w + gap

        self._redraw_field_boxes()

    # -------------- uniform size tools --------------

    def match_size_all(self):
        pi = self._primary_index()
        if pi is None:
            messagebox.showinfo("Select a field", "Select at least one field first.")
            return
        fields = self._fields()
        src = fields[pi]
        for f in fields:
            f.w = src.w
            f.h = src.h
        self._redraw_field_boxes()

    def match_size_same_type(self):
        pi = self._primary_index()
        if pi is None:
            messagebox.showinfo("Select a field", "Select at least one field first.")
            return
        fields = self._fields()
        src = fields[pi]
        src_type_key = ("textarea" if (src.type == "text" and src.multiline) else src.type)
        for f in fields:
            f_type_key = ("textarea" if (f.type == "text" and f.multiline) else f.type)
            if f_type_key == src_type_key:
                f.w = src.w
                f.h = src.h
        self._redraw_field_boxes()

    # -------------- export / generate / preview-filled --------------

    def _build_template_dict(self) -> Dict[str, Any]:
        pages: List[Dict[str, Any]] = []
        for idx in range(self.page_count):
            fields = [asdict(f) for f in self.fields_by_page.get(idx, [])]
            pages.append({"index": idx, "fields": fields})
        return {
            "meta": {"title": "Fillable Form", "author": "PDF Fillable Form Designer"},
            "background_pdf": self.background_pdf,
            "pages": pages,
        }

    def export_template(self):
        if not self.doc or not self.background_pdf:
            messagebox.showerror("No PDF", "Open a PDF first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        tpl = self._build_template_dict()
        Path(path).write_text(json.dumps(tpl, indent=2), encoding="utf-8")
        messagebox.showinfo("Exported", f"Template saved:\n{path}")

    def generate_fillable_pdf(self):
        if not self.doc or not self.background_pdf:
            messagebox.showerror("No PDF", "Open a PDF first.")
            return
        out_pdf = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not out_pdf:
            return
        tpl = self._build_template_dict()
        try:
            make_fillable_from_background(tpl, out_pdf)
        except Exception as ex:
            messagebox.showerror("Error", str(ex))
            return
        messagebox.showinfo("Done", f"Fillable PDF created:\n{out_pdf}")

    def preview_filled_pdf(self):
        if not self.doc or not self.background_pdf:
            messagebox.showerror("No PDF", "Open a PDF first.")
            return
        out_preview = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            title="Save Preview Filled PDF As…"
        )
        if not out_preview:
            return
        tmp_fillable = str(Path(out_preview).with_suffix(".fillable_tmp.pdf"))
        tpl = self._build_template_dict()
        try:
            make_fillable_from_background(tpl, tmp_fillable)
            create_preview_filled_pdf(tmp_fillable, out_preview)
        except Exception as ex:
            messagebox.showerror("Error", str(ex))
            return
        finally:
            try:
                Path(tmp_fillable).unlink()
            except OSError:
                pass
        messagebox.showinfo("Done", f"Preview Filled PDF created:\n{out_preview}")

    # -------------- info --------------

    def _refresh_info(self):
        self.info.delete("1.0", "end")
        self.info.insert("end", "Multi-select:\n")
        self.info.insert("end", "- Ctrl-click toggles selection\n")
        self.info.insert("end", "- Shift-click selects a range\n\n")
        self.info.insert("end", "Group tools:\n")
        self.info.insert("end", "- Radio group = same Name/Group\n")
        self.info.insert("end", "- Checkbox group = group id (prompted when creating)\n\n")
        self.info.insert("end", "Layout:\n")
        self.info.insert("end", "- Select multiple options, then Align / Stack / Distribute.\n")


if __name__ == "__main__":
    app = DesignerApp()
    app.mainloop()
