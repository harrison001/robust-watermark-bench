# robust-watermark-bench

How much of an invisible watermark survives being sent through a chat app?

A test harness that embeds a payload in a photograph, pushes the marked image through
resize + JPEG pipelines, and reports how often the payload comes back **exactly**. It
compares naive LSB steganography against three robust watermarking methods, including
one learned model, on identical images and identical channels.

The short version: **LSB does not degrade, it is erased.** One JPEG re-encode at
quality 90 — no resize at all — takes it to a 51% bit error rate, which is a coin flip
per bit. There is nothing left to correct, so no error-correcting code can help. That
result is the reason the rest of this exists.

## Results

Measured 2026-09-23. One 4288×2848 photograph downscaled to 1600×1063, 32-byte payload,
10 trials per cell, scored on **exact** recovery. Full write-up and method notes in
[`results/2026-09-23-channel-survival.md`](results/2026-09-23-channel-survival.md).

| Channel | LSB | dwtDctSvd | dwtDctSvd + RS(32,8) | TrustMark |
|---|---|---|---|---|
| baseline | 10/10 | 10/10 | 10/10 | 10/10 |
| JPEG QF90 | 0/10 · 51.3% BER | 10/10 | 10/10 | 10/10 |
| JPEG QF80 | 0/10 · 51.4% | 7/10 · 0.16% | 10/10 | 10/10 |
| JPEG QF70 | 0/10 · 51.4% | 0/10 · 4.22% | 9/10 | 10/10 |
| JPEG QF60 | 0/10 · 50.8% | 0/10 · 6.05% | 5/10 | 10/10 |
| JPEG QF50 | 0/10 · 50.7% | 0/10 · 8.67% | 0/10 | **10/10** |
| resize→1080, no recompression | 0/10 · 51.2% | 0/10 · 51.7% | 0/10 | **10/10** |
| ≤1600px + q89 | 0/10 · 49.5% | 9/10 · 0.04% | 10/10 | 10/10 |
| ≤1600px + q72 | 0/10 · 49.4% | 0/10 · 3.20% | **10/10** | 10/10 |
| ≤1080px + q80 | 0/10 · 51.0% | 0/10 · 50.1% | 0/10 | **10/10** |
| short side ≤1080 + q80 | 0/10 · 50.5% | 3/10 · 0.31% | 10/10 | 10/10 |
| two resizes, two re-encodes | 0/10 · 49.2% | 0/10 · 1.84% | **10/10** | 10/10 |

PSNR: LSB 94.4 dB · dwtDctSvd 39.1 dB · TrustMark 40.7 dB.

Rows are named by what they actually do, not by the service the parameters were guessed
at. In `bench.py` those same channels carry a platform name as a convenience label —
`≤1600px + q72` is `sim WhatsApp`, `≤1080px + q80` is `sim Instagram`, `short side
≤1080 + q80` is `sim WeChat`, `≤1600px + q89` is `sim Telegram`, and the last row is
`sim WeChat->WhatsApp`. The word `sim` is doing real work there; see
[What this does not measure](#what-this-does-not-measure).

### What the numbers say

**1. Reed–Solomon is decisive for recompression, and free to try.** Raw dwtDctSvd
through ≤1600px + q72 comes back at 3.2% BER and 0/10 exact — nearly intact and
completely useless. The same channel with an RS(32,8) codeword in the same 256 bits is
10/10. The cost is payload: 32 bytes becomes 8. Note there is deliberately **no
bit-interleaving** — JPEG's errors arrive in bursts, interleaving would spread each
burst across many symbols, and a byte-symbol code would rather eat one ruined byte than
eight damaged ones.

**2. Resize is a different failure, and parity cannot touch it.** Every classic method
returns ~50% BER on a bare, *lossless* rescale, while dwtDctSvd survives *lossy* QF50
at 8.67%. That is not corruption, it is desynchronisation — the decoder is reading the
wrong coefficients. The codeword is not damaged; it is not being read. Adding parity to
a signal nobody is sampling correctly changes nothing.

**3. The learned mark clears both walls and does not pay for it in visibility.**
TrustMark is 120/120 — every channel, every trial — and the expected trade-off never
appeared: at 40.7 dB it is *less* visible than the hand-designed mark at 39.1 dB while
being strictly more robust. Its real cost is torch, roughly 880 MB of dependencies and
downloaded model weights. That is a deployment and battery question, not a
signal-processing one, which is the honest reason to keep dwtDctSvd+RS as a lighter
fallback tier rather than any claim about quality.

**4. "Robustness costs imperceptibility" is only true of LSB.** 94.4 dB buys perfect
invisibility and survives nothing. Among methods that survive a channel at all, the
PSNR ranking and the robustness ranking do not conflict.

## What this does not measure

Read this before quoting a number.

**These are not the platforms.** Each channel is a resize and a JPEG re-encode with
parameters hardcoded in `bench.py`, and **those parameters have never been verified
against any live service.** Nothing here sends a photo through WhatsApp, WeChat,
Instagram or Telegram. A row reading "≤1600px + q72" means exactly that and nothing
stronger. The channel names in the source carry their own transform for this reason, so
that a table copied out of the tool stays honest wherever it lands. Real platform
evidence means sending an actual photo through the actual app and opening it on the
recipient device — a different exercise, measured separately
([one such round-trip, with results](https://stegosafe.com/blog/does-whatsapp-destroy-hidden-data-tested/)).

**This is not about detectability.** Every method here is trivially detectable by an
analyst who is looking, and the robust marks are *more* statistically conspicuous than
LSB, not less. Robustness and concealment are different properties and this harness
measures only the first.

**One image, ten trials.** Enough to separate 0/10 from 10/10 and 50% BER from 3%. Not
enough to state a survival probability to two decimal places. Cover choice matters:
fine detail is most of what a photograph is as far as a watermark is concerned, so run
it on your own images before trusting any single figure.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                  # LSB, dwtDctSvd, dwtDctSvd+RS, rivaGan
pip install -r requirements-trustmark.txt        # adds the TrustMark tier (pulls torch)
```

`numpy` is pinned below 2 so the classic and learned stacks coexist. Model weights for
rivaGan and TrustMark download on first use.

## Run

```bash
python bench.py --synthetic                              # no assets needed; smoke test only
python bench.py photo.jpg                                # default method dwt-rs, all channels
python bench.py photo.jpg --method lsb --trials 10       # watch the control get erased
python bench.py photo.jpg --method trustmark --trials 10
python bench.py photo.jpg --method dwt-rs --rs-k 16      # 128-bit payload instead of 64
python bench.py photo.jpg --channels "JPEG QF70,whatsapp" --restore --csv out.csv
```

`--synthetic` generates a textured cover so the harness runs with nothing to hand. It is
a smoke test: generated texture has less fine detail than a photograph, compresses
smaller, and loses marks a photograph keeps, so its numbers are worse than the table
above and are not comparable to it. The tool says so when you use it.

`--restore` resizes a geometrically-mangled image back to the embedding dimensions
before decoding, which separates desynchronisation from destruction: if the payload
returns after rescaling, finding 2 above is what happened.

## Methods

| `--method` | What it is | Payload | ECC |
|---|---|---|---|
| `lsb` | sequential LSB replacement, no permutation — **the control, not a contender** | 32 B | none |
| `dwt` | classic dwtDctSvd via `invisible-watermark` | 32 B | none |
| `dwt-rs` | dwtDctSvd carrying a Reed–Solomon codeword — the best classic recipe here | 8–16 B | RS(32,k) |
| `rivagan` | 32-bit GAN watermark, some built-in resize tolerance | 4 B | none |
| `trustmark` | C2PA-grade learned watermark | ~100 bit | BCH |

`lsb` is included so the failure is visible rather than asserted. It is what most "hide
a message in a photo" tutorials do, and it is appropriate for *storage*, where the file
is never re-encoded by anyone — but a chat app is not storage.

## Related measurements

- [Does WhatsApp destroy hidden data? — an actual round-trip, measured](https://stegosafe.com/blog/does-whatsapp-destroy-hidden-data-tested/)
- [Hidden data dies in chat apps: measured](https://harrisonsec.com/blog/hidden-data-dies-in-chat-apps-measured/) — the engineering write-up this harness produced the numbers for
- [Two ways to hide text in a photo, and when each one breaks](https://stegosafe.com/blog/hide-text-in-an-image-two-ways/)

## Licence

MIT, see [LICENSE](LICENSE). The watermarking implementations themselves come from
[invisible-watermark](https://github.com/ShieldMnt/invisible-watermark) and
[TrustMark](https://github.com/adobe/trustmark) and carry their own licences; this
repository is the harness and the measurements.
