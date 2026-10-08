#!/usr/bin/env python3
"""
bench.py — robust-watermark channel test harness.

Embed a payload into an image, push the marked image through simulated social /
messaging compression pipelines, and report how reliably the payload comes back.

Methods
  lsb       sequential LSB replacement — the control, not a contender. It is what every
            "hide a message in a photo" tutorial does, and what StegoSafe uses for
            *storage*, where the file is never uploaded anywhere. Run it to watch the
            channel destroy a payload; that failure is the reason the rest of this exists.
  dwt       classic dwtDctSvd, raw 256-bit payload (survives light JPEG, dies on resize)
  dwt-rs    dwtDctSvd carrying an RS(32,k) codeword  <-- the winning classic recipe
            (Reed-Solomon over bytes, NO bit-interleaving: the channel's burst errors
             stay inside a byte, so a symbol code eats them cheaply)
  rivagan   32-bit GAN watermark (built-in resize tolerance, no ECC)
  trustmark C2PA-grade DL watermark, ~100-bit + built-in BCH ECC (survives resize+JPEG)

Usage
  python bench.py photo.jpg
  python bench.py --synthetic          # no image to hand? generates a textured cover
  python bench.py photo.jpg --method trustmark --trials 6
  python bench.py photo.jpg --method dwt-rs --rs-k 8 --trials 20 --restore --csv out.csv
  python bench.py photo.jpg --channels "JPEG QF70,WhatsApp,Instagram"
"""
import argparse, io, os, sys, logging, csv, time
import numpy as np
from PIL import Image
logging.disable(logging.INFO)

EMBED_MAX = 1600   # marked image is handed back at this max side (share-ready)

# ---------- image helpers ----------
def synthetic_cover(w=1600, h=1063, seed=7):
    """A deterministic textured cover, so the harness runs with no assets at all.

    Watermarking needs somewhere to hide: a flat or smoothly-graded image gives the
    encoder no high-frequency budget, the mark lands in coefficients JPEG then throws
    away, and every method scores badly for a reason that has nothing to do with the
    method. So this is multi-octave value noise over a gradient rather than a gradient.

    It is NOT the image the published results were measured on — those used a 4288x2848
    photograph. Numbers from here are for checking that your install works; compare
    methods against each other on a real photograph of your own.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w, 3), np.float32)
    # Octaves must reach pixel scale. A first attempt stopped at 64x64 cells, which on a
    # 1600px canvas means no feature smaller than 25px: the image compressed to a quarter
    # of a real photograph's size at the same quality and JPEG ate marks the photograph
    # kept. Detail near the Nyquist limit is most of what a photograph *is*, as far as a
    # watermark is concerned.
    for octave in range(10):                     # 2x2 cells up to 1024x1024, halving weight
        n = 2 ** (octave + 1)
        small = rng.random((min(n, h), min(n, w), 3)).astype(np.float32)
        up = np.array(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                      np.float32) / 255.0
        acc += up / (1.7 ** octave)
    fine = rng.random((h, w, 3)).astype(np.float32)        # the octave BICUBIC cannot reach
    acc += 0.10 * (fine + np.roll(fine, 1, 0) + np.roll(fine, 1, 1)) / 3   # mildly correlated
    acc /= acc.max()
    grad = np.linspace(0.2, 0.8, w, dtype=np.float32)[None, :, None]
    img = np.clip(acc * 0.75 + grad * 0.25, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8), "RGB")

def load_cover(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size; s = EMBED_MAX / max(w, h)
    if s < 1: im = im.resize((round(w*s), round(h*s)), Image.LANCZOS)
    return im
def rmax(im, m):
    w, h = im.size; s = m/max(w, h)
    return im if s >= 1 else im.resize((round(w*s), round(h*s)), Image.LANCZOS)
def rshort(im, m):
    w, h = im.size; s = m/min(w, h)
    return im if s >= 1 else im.resize((round(w*s), round(h*s)), Image.LANCZOS)
def jpeg(im, q):
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=q, subsampling=2)
    data = b.getvalue(); return Image.open(io.BytesIO(data)), len(data)

# ---------- channels: marked PIL -> (received PIL, size_bytes_or_None, is_geometric) ----------
def _passthru(im): return im, None
def _jpeg(q):      return lambda im: jpeg(im, q)
def _rmax(m):      return lambda im: (rmax(im, m), None)
def CH_whatsapp(im):  r = rmax(im, 1600);  return jpeg(r, 72)
def CH_wechat(im):    r = rshort(im, 1080); return jpeg(r, 80)
def CH_instagram(im): r = rmax(im, 1080);  return jpeg(r, 80)
def CH_telegram(im):  return jpeg(im, 89)
def CH_wc_wa(im):     a,_ = CH_wechat(im);  return CH_whatsapp(a)

# The platform rows are SIMULATIONS, and the names say so on purpose.
#
# Nothing here sends a photo through WhatsApp. Each "sim" channel is a resize and
# a JPEG re-encode with hardcoded parameters, and those parameters have never been
# verified against the live platforms — they predate this harness and carry no
# provenance. A result of "sim WhatsApp 10/10" means "10/10 through ≤1600px + q72",
# and nothing stronger.
#
# Every key therefore carries its own transform, so that a table copied out of this
# tool stays honest wherever it lands. Real platform evidence comes from sending an
# actual photo through the actual app and opening it on the recipient device; that
# is a different exercise and its results do not live here.
CHANNELS = {   # name: (func, is_geometric)
    "baseline":                     (_passthru,   False),
    "JPEG QF90":                    (_jpeg(90),   False),
    "JPEG QF80":                    (_jpeg(80),   False),
    "JPEG QF70":                    (_jpeg(70),   False),
    "JPEG QF60":                    (_jpeg(60),   False),
    "JPEG QF50":                    (_jpeg(50),   False),
    "resize->1080 only":            (_rmax(1080),  True),
    "sim Telegram (q89)":           (CH_telegram, False),
    "sim WhatsApp (<=1600,q72)":    (CH_whatsapp,  True),
    "sim Instagram (<=1080,q80)":   (CH_instagram, True),
    "sim WeChat (short<=1080,q80)": (CH_wechat,    True),
    "sim WeChat->WhatsApp":         (CH_wc_wa,     True),
}
DEFAULT_ORDER = list(CHANNELS.keys())
BANNER = ("NOTE: the 'sim ...' channels are resize+JPEG approximations with unverified\n"
          "      parameters, NOT sends through the real platforms. See CHANNELS in this file.")

def resolve(name):
    """Accept a short name ('whatsapp') for a self-documenting key.

    'whatsapp' matches both the WhatsApp channel and WeChat->WhatsApp, so a name
    that leads a key wins over one that merely appears inside it.
    """
    if name in CHANNELS: return name
    n = name.lower()
    seg = lambda k: k.lower().removeprefix("sim ").split(" (")[0]   # 'sim WeChat (…)' -> 'wechat'
    exact = [k for k in CHANNELS if seg(k) == n]
    if len(exact) == 1: return exact[0]
    hits = [k for k in CHANNELS if n in k.lower()]
    if len(hits) == 1: return hits[0]
    sys.exit(f"unknown or ambiguous channel: {name!r}\navailable: {DEFAULT_ORDER}")

def _bits(b): return np.unpackbits(np.frombuffer(b, np.uint8))
def _frombits(bt): return np.packbits(bt.astype(np.uint8)).tobytes()

# ---------- methods: each yields embed() / decode() closures ----------
class Method:
    has_ber = True
    def encode(self, cover_pil): ...           # -> (stego_pil, ground_truth)
    def decode(self, recv_pil, gt): ...        # -> (success: bool, ber: float|None)

class LSB(Method):
    """sequential LSB replacement over the RGB bytes — the control.

    Deliberately the naive implementation: no permutation, no ECC, payload in the
    first NBYTES*8 subpixels. A resize moves every pixel it reads, and a JPEG
    re-encode rewrites the low bits in place, so this is expected to score 0 on
    every channel except baseline. That is the point of including it.
    """
    def __init__(self, nbytes=32):
        self.nbytes = nbytes; self.rng = np.random.default_rng(101)
    def encode(self, cover):
        p = self.rng.bytes(self.nbytes)
        arr = np.array(cover.convert("RGB")); flat = arr.reshape(-1).copy()
        bits = _bits(p); flat[:bits.size] = (flat[:bits.size] & 0xFE) | bits
        return Image.fromarray(flat.reshape(arr.shape)), p
    def decode(self, recv, gt):
        flat = np.array(recv.convert("RGB")).reshape(-1)
        n = self.nbytes * 8
        if flat.size < n: return False, 1.0
        ber = float(np.mean((flat[:n] & 1) != _bits(gt)))
        return (ber == 0.0), ber

class DWT(Method):
    """raw dwtDctSvd, NBYTES*8-bit payload, success = exact recovery"""
    def __init__(self, nbytes=32):
        import cv2; from imwatermark import WatermarkEncoder, WatermarkDecoder
        self.cv2, self.WE, self.WD = cv2, WatermarkEncoder, WatermarkDecoder
        self.nbytes = nbytes; self.rng = np.random.default_rng(101)
    def _bgr(self, im): return self.cv2.cvtColor(np.array(im.convert("RGB")), self.cv2.COLOR_RGB2BGR)
    def _pil(self, bgr): return Image.fromarray(self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB))
    def encode(self, cover):
        p = self.rng.bytes(self.nbytes)
        e = self.WE(); e.set_watermark('bytes', p)
        return self._pil(e.encode(self._bgr(cover), 'dwtDctSvd')), p
    def decode(self, recv, gt):
        out = self.WD('bytes', self.nbytes*8).decode(self._bgr(recv), 'dwtDctSvd')
        ber = float(np.mean(_bits(out) != _bits(gt)))
        return (ber == 0.0), ber

class DWT_RS(DWT):
    """dwtDctSvd carrying RS(32,k); success = RS decodes AND matches (RS failure = honest 'cannot read')"""
    has_ber = False
    def __init__(self, k=8):
        super().__init__(nbytes=32)
        from reedsolo import RSCodec, ReedSolomonError
        self.rsc = RSCodec(32 - k); self.k = k; self.RSErr = ReedSolomonError
    def encode(self, cover):
        msg = self.rng.bytes(self.k)
        coded = bytes(self.rsc.encode(bytearray(msg)))          # k -> 32 bytes
        e = self.WE(); e.set_watermark('bytes', coded)          # NO bit-interleave (see docstring)
        return self._pil(e.encode(self._bgr(cover), 'dwtDctSvd')), msg
    def decode(self, recv, gt):
        rx = _frombits(_bits(self.WD('bytes', 256).decode(self._bgr(recv), 'dwtDctSvd')))
        try: return (bytes(self.rsc.decode(bytearray(rx))[0]) == gt), None
        except self.RSErr: return False, None

class RivaGan(DWT):
    def __init__(self):
        from imwatermark import WatermarkEncoder, WatermarkDecoder
        import cv2
        self.cv2, self.WE, self.WD = cv2, WatermarkEncoder, WatermarkDecoder
        WatermarkEncoder.loadModel(); self.rng = np.random.default_rng(101)
    def encode(self, cover):
        wm = self.rng.integers(0, 2, 32).astype(np.uint8)
        e = self.WE(); e.set_watermark('bits', wm.tolist())
        return self._pil(e.encode(self._bgr(cover), 'rivaGan')), wm
    def decode(self, recv, gt):
        d = np.array(self.WD('bits', 32).decode(self._bgr(recv), 'rivaGan'))
        ber = float(np.mean(d != gt)); return (ber == 0.0), ber

class TrustMarkM(Method):
    has_ber = False
    def __init__(self):
        from trustmark import TrustMark
        self.tm = TrustMark(verbose=False, model_type='Q', encoding_type=TrustMark.Encoding.BCH_5)
        self.rng = np.random.default_rng(101)
    def _mk(self): return ''.join(self.rng.choice(list("0123456789abcdef"), 7))
    def encode(self, cover):
        s = self._mk(); return self.tm.encode(cover, s), s
    def decode(self, recv, gt):
        rs, present, _ = self.tm.decode(recv)
        return (bool(present) and str(rs).strip() == gt), None

METHODS = {"lsb": LSB, "dwt": DWT, "dwt-rs": DWT_RS, "rivagan": RivaGan, "trustmark": TrustMarkM}

def psnr(a, b):
    d = np.array(a, np.float64) - np.array(b.convert("RGB"), np.float64)
    m = np.mean(d*d); return 99.0 if m == 0 else 10*np.log10(255*255/m)

def main():
    ap = argparse.ArgumentParser(description="robust-watermark channel test harness")
    ap.add_argument("image", nargs="?", help="cover photograph; omit it and pass --synthetic")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate a textured cover instead of reading an image")
    ap.add_argument("--method", default="dwt-rs", choices=list(METHODS))
    ap.add_argument("--rs-k", type=int, default=8, help="RS message bytes for dwt-rs (8=64bit,16=128bit)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--channels", default="", help="comma list; default = all")
    ap.add_argument("--restore", action="store_true", help="also decode after resizing back (desync probe)")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    if not a.image and not a.synthetic:
        ap.error("give an image, or --synthetic to generate one")
    cover = synthetic_cover() if a.synthetic else load_cover(a.image)
    label = "synthetic" if a.synthetic else os.path.basename(a.image)
    m = METHODS[a.method](a.rs_k) if a.method == "dwt-rs" else METHODS[a.method]()
    order = [resolve(c.strip()) for c in a.channels.split(",") if c.strip()] or DEFAULT_ORDER

    stego0, _ = m.encode(cover)
    p = psnr(cover, stego0)
    tag = f"{a.method}" + (f" RS(32,{a.rs_k})" if a.method == "dwt-rs" else "")
    print(f"\n{BANNER}")
    if a.synthetic:
        print("\nNOTE: --synthetic is a smoke test, not a benchmark surface. Generated texture\n"
              "      has less fine detail than a photograph, so it compresses smaller and JPEG\n"
              "      removes marks a photograph would have kept. Expect WORSE numbers here than\n"
              "      in results/ — those were measured on a 4288x2848 photograph. To compare\n"
              "      methods, run on a real photograph of your own.")
    print(f"\nimage {label}  embed@ {cover.size[0]}x{cover.size[1]}  "
          f"method {tag}  PSNR {p:.1f} dB  trials {a.trials}")
    hdr = f"{'channel':30} {'recv':>11} {'size':>6}"
    if m.has_ber: hdr += f" {'BER':>8}"
    hdr += f" {'ok':>7}"
    if a.restore: hdr += f" {'ok(rest)':>9}"
    print(hdr); print("-"*len(hdr))

    rows = []
    for name in order:
        chan, geom = CHANNELS[name]
        okc = 0; bers = []; okr = 0; dims = ""; size = None
        for _ in range(a.trials):
            stego, gt = m.encode(cover)
            recv, size = chan(stego)
            dims = f"{recv.size[0]}x{recv.size[1]}"
            ok, ber = m.decode(recv, gt)
            okc += ok
            if ber is not None: bers.append(ber)
            if a.restore and geom:
                rest = recv.resize(cover.size, Image.LANCZOS)
                okr += m.decode(rest, gt)[0]
        szs = f"{size//1024}K" if size else "-"
        line = f"{name:30} {dims:>11} {szs:>6}"
        if m.has_ber: line += f" {np.mean(bers)*100:7.2f}%"
        line += f" {okc:>3}/{a.trials:<3}"
        if a.restore: line += (f" {okr:>4}/{a.trials:<3}" if geom else f" {'-':>9}")
        print(line)
        rows.append({"channel": name, "recv": dims, "size_bytes": size,
                     "ber": (np.mean(bers) if bers else ""), "ok": okc,
                     "ok_restore": (okr if (a.restore and geom) else ""), "trials": a.trials})

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.csv}")

if __name__ == "__main__":
    t = time.time(); main(); print(f"\n({time.time()-t:.0f}s)", file=sys.stderr)
