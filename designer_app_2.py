from __future__ import annotations

import json
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog

# Preview rendering
import fitz  # PyMuPDF
from PIL import Image, ImageTk

# PDF form generation / overlay
import pikepdf
from reportlab.pdfgen import canvas


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Field:
    # type: text, textarea, checkbox, dropdown, radio
    type: str
    name: str
    label: str
    x: float
    y: float
    w: float
    h: float
    font_size: int = 11
    required: bool = False
    multiline: bool = False              # backwards compat
    options: Optional[List[str]] = None  # dropdown
    default: Optional[Any] = None        # text: str, checkbox: bool, dropdown: str, radio: value
    value: Optional[str] = None          # radio export value (per button)
    group: str = ""                      # checkbox grouping (question id)


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
        ftype = (f.get("type") or "").strip()
        name = f["name"]
        label = f.get("label", "") or name

        x = float(f["x"]); y = float(f["y"])
        w = float(f["w"]); h = float(f["h"])
        required = bool(f.get("required", False))

        # ✅ FIX: accept both "text" and "textarea"
        if ftype in ("text", "textarea"):
            multiline = (ftype == "textarea") or bool(f.get("multiline", False))
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

            # Visual overlay (optional but nice)
            bg_page.add_overlay(fld_page)

            # Copy widget annotations (this is what makes fields clickable)
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
        for k in list(n.keys()):
            if str(k) != "/Off":
                return k
    except Exception:
        return None
    return None


def create_preview_filled_pdf(fillable_pdf_path: str, out_preview_pdf_path: str) -> None:
    with pikepdf.open(fillable_pdf_path) as pdf:
        if "/AcroForm" not in pdf.Root:
            raise ValueError("No /AcroForm found in PDF.")

        pdf.Root["/AcroForm"]["/NeedAppearances"] = pikepdf.Boolean(True)

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
        self.geometry("1320x880")

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

        self.selected_indices: List[int] = []

        self._move_start_xy_px: Optional[Tuple[int, int]] = None
        self._move_orig_positions_pt: Optional[Dict[int, Tuple[float, float]]] = None

        self._clipboard_fields: Optional[List[Dict[str, Any]]] = None
        self._clipboard_bbox: Optional[Tuple[float, float, float, float]] = None  # minx,miny,maxx,maxy (pt)

        self._build_ui()
        self._bind_shortcuts()

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

        ttk.Button(top, text="Copy", command=self.copy_selection).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Paste", command=self.paste_selection).pack(side="left")
        ttk.Button(top, text="Duplicate", command=self.duplicate_selection).pack(side="left", padx=(6, 0))

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

        self.info = tk.Text(right, height=18, wrap="word")
        self.info.pack(fill="both", expand=True, pady=(10, 0))
        self._refresh_info()

    def _bind_shortcuts(self):
        self.bind_all("<Control-c>", lambda _e: self.copy_selection())
        self.bind_all("<Control-C>", lambda _e: self.copy_selection())
        self.bind_all("<Control-v>", lambda _e: self.paste_selection())
        self.bind_all("<Control-V>", lambda _e: self.paste_selection())
        self.bind_all("<Control-d>", lambda _e: self.duplicate_selection())
        self.bind_all("<Control-D>", lambda _e: self.duplicate_selection())
        self.bind_all("<Delete>", lambda _e: self.delete_selected())

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
        self._clipboard_fields = None
        self._clipboard_bbox = None

        self._render_current_page()
        self._update_page_label()
        self._refresh_fields_list()

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
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)
        self._redraw_field_boxes()

    def _update_page_label(self):
        self.page_var.set(f"{self.current_page + 1} / {self.page_count}")

    def prev_page(self):
        if self.doc and self.current_page > 0:
            self.current_page -= 1
            self.selected_indices = []
            self._render_current_page()
            self._update_page_label()
            self._refresh_fields_list()

    def next_page(self):
        if self.doc and self.current_page < self.page_count - 1:
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
        self.listbox.selection_clear(0, "end")
        for i in self.selected_indices:
            self.listbox.selection_set(i)
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

    # -------------- mouse interactions --------------

    def on_mouse_down(self, e):
        if not self.doc:
            return

        hit = self._hit_test_field(e.x, e.y)
        ctrl = bool(e.state & 0x0004)
        shift = bool(e.state & 0x0001)

        if hit is not None:
            if ctrl:
                if hit in self.selected_indices:
                    self._set_selection([i for i in self.selected_indices if i != hit])
                else:
                    self._set_selection(self.selected_indices + [hit])
            elif shift and self.selected_indices:
                a = self._primary_index()
                lo, hi = sorted([a, hit]) if a is not None else (hit, hit)
                self._set_selection(list(range(lo, hi + 1)))
            else:
                self._set_selection([hit])

            self._mode = "move"
            self._move_start_xy_px = (e.x, e.y)
            self._move_orig_positions_pt = {i: (self._fields()[i].x, self._fields()[i].y) for i in self.selected_indices}
            return

        self._mode = "create"
        self._drag_start = (e.x, e.y)
        self._drag_rect_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=2)
        if not (ctrl or shift):
            self._set_selection([])

    def on_mouse_drag(self, e):
        if self._mode == "create":
            if self._drag_start and self._drag_rect_id:
                x0, y0 = self._drag_start
                self.canvas.coords(self._drag_rect_id, x0, y0, e.x, e.y)
            return

        if self._mode == "move":
            if not self._move_start_xy_px or not self._move_orig_positions_pt:
                return

            dx_px = e.x - self._move_start_xy_px[0]
            dy_px = e.y - self._move_start_xy_px[1]
            dx_pt = dx_px / self.preview_scale
            dy_pt = -dy_px / self.preview_scale

            page_w_pt, page_h_pt = self.page_points
            fields = self._fields()

            for idx, (ox, oy) in self._move_orig_positions_pt.items():
                f = fields[idx]
                nx = max(0.0, min(ox + dx_pt, page_w_pt - f.w))
                ny = max(0.0, min(oy + dy_pt, page_h_pt - f.h))
                f.x, f.y = float(nx), float(ny)

            self._redraw_field_boxes()

    def on_mouse_up(self, e):
        if not self.doc:
            return

        if self._mode == "move":
            self._mode = "idle"
            self._move_start_xy_px = None
            self._move_orig_positions_pt = None
            self._refresh_fields_list()
            return

        if self._mode != "create" or not self._drag_start or not self._drag_rect_id:
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
            font_size = simpledialog.askinteger("Font size", "Font size (e.g., 10–12):",
                                                initialvalue=11, minvalue=6, maxvalue=36, parent=self)
            if font_size is None:
                self._cancel_create(); self._mode = "idle"; return

            field = Field(
                type=("textarea" if ui_type == "textarea" else "text"),
                name=name_or_group, label=label,
                x=float(x_pt), y=float(y_pt), w=float(w_pt), h=float(h_pt),
                font_size=int(font_size), required=required,
                multiline=(ui_type == "textarea"), default=""
            )

        elif ui_type == "checkbox":
            group = simpledialog.askstring("Checkbox group id",
                                           "Enter a group id for this checkbox question (optional):",
                                           initialvalue="", parent=self)
            if group is None:
                self._cancel_create(); self._mode = "idle"; return
            field = Field(
                type="checkbox", name=name_or_group, label=label,
                x=float(x_pt), y=float(y_pt), w=float(w_pt), h=float(h_pt),
                required=required, default=False, group=(group.strip() if group else "")
            )

        elif ui_type == "dropdown":
            options_str = simpledialog.askstring("Dropdown options",
                                                 "Enter options separated by commas:",
                                                 initialvalue="Option A, Option B", parent=self)
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
            export_value = simpledialog.askstring("Radio option value",
                                                  "Enter export value for this radio button (e.g., yes):",
                                                  initialvalue="option1", parent=self)
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
        self._set_selection([len(self._fields()) - 1])
        self._refresh_fields_list()
        self._redraw_field_boxes()
        self._cancel_create()

        if ui_type != "radio":
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
            else:
                self.listbox.insert("end", f"{f.type}: {f.name}")

        self.listbox.selection_clear(0, "end")
        for i in self.selected_indices:
            if 0 <= i < self.listbox.size():
                self.listbox.selection_set(i)

    def _redraw_field_boxes(self):
        self.canvas.delete("fieldbox")
        if not self.doc:
            return
        for idx, f in enumerate(self._fields()):
            l, t, r, b = self._field_rect_canvas_px(f)
            selected = (idx in self.selected_indices)
            outline = "red" if selected else "blue"
            width = 3 if selected else 2
            self.canvas.create_rectangle(l, t, r, b, outline=outline, width=width, tags="fieldbox")
            label = f.name
            if f.type == "radio":
                label = f"{f.name}={f.value}"
            elif f.type == "checkbox" and f.group:
                label = f"{f.name} ({f.group})"
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
        new_name = simpledialog.askstring("Rename field",
                                          "Enter new field name (radio = group name):",
                                          initialvalue=f.name, parent=self)
        if new_name and new_name.strip():
            f.name = new_name.strip()

        if f.type == "radio":
            new_val = simpledialog.askstring("Radio option value",
                                             "Enter export value for this radio button:",
                                             initialvalue=f.value or "", parent=self)
            if new_val and new_val.strip():
                f.value = new_val.strip()

        if f.type == "checkbox":
            new_group = simpledialog.askstring("Checkbox group id",
                                               "Enter checkbox group id (question id).",
                                               initialvalue=f.group or "", parent=self)
            if new_group is not None:
                f.group = new_group.strip()

        self._refresh_fields_list()
        self._redraw_field_boxes()

    # -------------- copy/paste/duplicate --------------

    def _existing_names_set(self) -> set:
        s = set()
        for page_fields in self.fields_by_page.values():
            for f in page_fields:
                s.add(f.name)
        return s

    def _unique_name(self, base: str, used: set) -> str:
        if base not in used:
            used.add(base)
            return base
        n = 2
        while True:
            cand = f"{base}_{n}"
            if cand not in used:
                used.add(cand)
                return cand
            n += 1

    def copy_selection(self):
        if not self.selected_indices:
            return
        fields = self._fields()
        items = [asdict(fields[i]) for i in self.selected_indices]
        minx = min(d["x"] for d in items); miny = min(d["y"] for d in items)
        maxx = max(d["x"] + d["w"] for d in items); maxy = max(d["y"] + d["h"] for d in items)
        self._clipboard_fields = items
        self._clipboard_bbox = (minx, miny, maxx, maxy)

    def paste_selection(self):
        if not self._clipboard_fields or not self._clipboard_bbox or not self.doc:
            return

        dx_pt = 12.0 / self.preview_scale
        dy_pt = -12.0 / self.preview_scale

        used_names = self._existing_names_set()
        clip = copy.deepcopy(self._clipboard_fields)

        radio_groups = sorted({d["name"] for d in clip if d["type"] == "radio"})
        radio_group_map: Dict[str, str] = {}
        for g in radio_groups:
            radio_group_map[g] = self._unique_name(f"{g}_copy", used_names)

        checkbox_group_ids = sorted({d.get("group", "") for d in clip if d["type"] == "checkbox" and d.get("group")})
        checkbox_group_map: Dict[str, str] = {gid: f"{gid}_copy" for gid in checkbox_group_ids}

        new_fields: List[Field] = []
        for d in clip:
            d2 = dict(d)
            d2["x"] = float(d2["x"]) + dx_pt
            d2["y"] = float(d2["y"]) + dy_pt

            if d2["type"] == "radio":
                d2["name"] = radio_group_map.get(d2["name"], d2["name"])
            else:
                d2["name"] = self._unique_name(d2["name"], used_names)

            if d2["type"] == "checkbox" and d2.get("group"):
                d2["group"] = checkbox_group_map.get(d2["group"], d2["group"])

            new_fields.append(Field(**d2))

        page_w_pt, page_h_pt = self.page_points
        for f in new_fields:
            f.x = max(0.0, min(f.x, page_w_pt - f.w))
            f.y = max(0.0, min(f.y, page_h_pt - f.h))

        fields = self._fields()
        start = len(fields)
        fields.extend(new_fields)
        self._set_selection(list(range(start, start + len(new_fields))))
        self._refresh_fields_list()
        self._redraw_field_boxes()

    def duplicate_selection(self):
        self.copy_selection()
        self.paste_selection()

    # -------------- export / generate / preview-filled --------------

    def _build_template_dict(self) -> Dict[str, Any]:
        pages: List[Dict[str, Any]] = []
        for idx in range(self.page_count):
            pages.append({"index": idx, "fields": [asdict(f) for f in self.fields_by_page.get(idx, [])]})
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
        Path(path).write_text(json.dumps(self._build_template_dict(), indent=2), encoding="utf-8")
        messagebox.showinfo("Exported", f"Template saved:\n{path}")

    def generate_fillable_pdf(self):
        if not self.doc or not self.background_pdf:
            messagebox.showerror("No PDF", "Open a PDF first.")
            return
        out_pdf = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not out_pdf:
            return
        try:
            make_fillable_from_background(self._build_template_dict(), out_pdf)
        except Exception as ex:
            messagebox.showerror("Error", str(ex))
            return
        messagebox.showinfo("Done", f"Fillable PDF created:\n{out_pdf}")

    def preview_filled_pdf(self):
        if not self.doc or not self.background_pdf:
            messagebox.showerror("No PDF", "Open a PDF first.")
            return
        out_preview = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not out_preview:
            return
        tmp_fillable = str(Path(out_preview).with_suffix(".fillable_tmp.pdf"))
        try:
            make_fillable_from_background(self._build_template_dict(), tmp_fillable)
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

    def _refresh_info(self):
        self.info.delete("1.0", "end")
        self.info.insert("end", "Copy/Paste blocks:\n")
        self.info.insert("end", "- Ctrl+C / Ctrl+V\n")
        self.info.insert("end", "- Ctrl+D duplicates\n\n")
        self.info.insert("end", "Textarea fix:\n")
        self.info.insert("end", "- Renderer supports type 'textarea' + 'text' (multiline).\n")


if __name__ == "__main__":
    DesignerApp().mainloop()
