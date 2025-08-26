# app.py — web-просмотрщик (trame + PyVista off-screen)
import os, re, numpy as np, pyvista as pv
from trame.app import get_server
from trame.widgets import html, vuetify as v
from trame.ui.vuetify import SinglePageWithDrawerLayout

DATA_DIR = os.environ.get("DATA_DIR", "/srv/jaw-viewer-data")

def extract_id(name): m=re.match(r"^0*(\d+)",name); return int(m.group(1)) if m else None
def detect_role(name):
    n=name.lower()
    if any(k in n for k in ("upper","верх","verh","top","_u")): return "upper"
    if any(k in n for k in ("lower","низ","niz","bottom","_l")): return "lower"
    return None

def get_center_of_mass(m): return np.array(m.center)
def bbox_size(m): b=m.bounds; return np.array([b[1]-b[0], b[3]-b[2], b[5]-b[4]])
def normalize_scale(upper, lower):
    us, ls = bbox_size(upper), bbox_size(lower); ls = np.where(ls==0,1e-9,ls); return lower.scale(us/ls)
def prep_pair(upper, lower):
    lower = normalize_scale(upper, lower)
    t = get_center_of_mass(upper) - get_center_of_mass(lower); t[1] -= 8
    lower = lower.translate(t); upper = upper.translate([-1,+1,+3])
    return upper, lower

def mask_curv(m, q=0.65):
    try:
        curv = np.abs(m.curvature('mean')); curv = curv[np.isfinite(curv)]
        if curv.size==0: return None
        thr = np.quantile(np.abs(m.curvature('mean')), q)
        return np.abs(m.curvature('mean')) >= thr
    except: return None
def mask_height(m, q=0.60):
    z=m.points[:,2]; z=z[np.isfinite(z)];
    if z.size==0: return None
    thr = np.quantile(m.points[:,2], q)
    return m.points[:,2] >= thr

def colorize_teeth_gums(mesh, tooth_rgb=(255,255,255), gum_rgb=(242,153,153)):
    m = mesh.copy(deep=True)
    mask = mask_curv(m) or mask_height(m)
    if mask is None or mask.sum() in (0, mask.size):
        z=m.points[:,2]; order=np.argsort(z); mask=np.zeros(len(z), bool)
        if len(z)>0: mask[order[-max(1,int(0.4*len(z))) :]] = True
    colors = np.empty((m.n_points,3), np.uint8); colors[:] = gum_rgb; colors[mask]=tooth_rgb
    m.point_data["RGB"] = colors
    return m

def collect_pairs(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".stl")]
    buckets = {}
    for fn in files:
        pid = extract_id(fn); role = detect_role(fn)
        if pid is None or role is None: continue
        buckets.setdefault(pid, {"upper":[], "lower":[]})[role].append(fn)
    pairs = []
    for pid in sorted(buckets):
        g=buckets[pid]
        if not g["upper"] or not g["lower"]: continue
        up = pv.read(os.path.join(folder, g["upper"][0]))
        low= pv.read(os.path.join(folder, g["lower"][0]))
        up, low = prep_pair(up, low)
        pairs.append((colorize_teeth_gums(up), colorize_teeth_gums(low), pid))
    return pairs

server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller
pairs = collect_pairs(DATA_DIR)
state.pairs_len = len(pairs); state.idx = 0; state.pid = pairs[0][2] if pairs else -1

plotter = pv.Plotter(off_screen=True); plotter.set_background("white")
def render_idx(i):
    plotter.clear()
    if not pairs: return None
    i = max(0, min(len(pairs)-1, i))
    up, low, pid = pairs[i]
    plotter.add_mesh(up, scalars="RGB", rgb=True)
    plotter.add_mesh(low,scalars="RGB", rgb=True)
    plotter.camera_position = "xy"
    img = plotter.screenshot(return_img=True)
    state.idx, state.pid = i, pid
    return img

@ctrl.trigger("next")
def next_pair():
    img = render_idx(state.idx + 1); img is not None and ctrl.set_image("view", img)
@ctrl.trigger("prev")
def prev_pair():
    img = render_idx(state.idx - 1); img is not None and ctrl.set_image("view", img)
@ctrl.trigger("jump")
def jump_to(i):
    try: i=int(i)
    except: return
    img=render_idx(i); img is not None and ctrl.set_image("view", img)

first_img = render_idx(0)
with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("Jaw Viewer")
    with layout.toolbar:
        v.btn("⟵ Prev", click=ctrl.prev, classes="mx-2", outlined=True)
        v.btn("Next ⟶", click=ctrl.next, classes="mx-2", outlined=True)
        v.spacer()
        html.Div("Pair: {{ pid }}  [{{ idx+1 }}/{{ pairs_len }}]", classes="mx-2")
        v.text_field(v_model=("jump_to_idx", 0), label="Go to index", type="number",
                     classes="mx-2", style="max-width:120px")
        v.btn("Go", click=("jump","jump_to_idx"), classes="mx-2", outlined=True)
    with layout.content:
        html.Img(src=("view", first_img),
                 style="width:100%;max-width:1200px;border:1px solid #ddd;border-radius:8px")

if __name__ == "__main__":
    server.start(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))