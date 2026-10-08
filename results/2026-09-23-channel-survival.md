# Channel survival: LSB vs dwtDctSvd vs dwtDctSvd+RS — 2026-09-23

Run so that an engineering write-up could state numbers instead of asserting them.
Everything here came from `bench.py` on one cover photograph — 4288×2848, downscaled
by the harness to 1600×1063 — with a 32-byte payload, 10 trials per channel, scored on
**exact** recovery.

Environment: `invisible-watermark==0.2.0`, `reedsolo==1.7.0`, `numpy<2`, PyWavelets,
Pillow, opencv, plus torch and `trustmark` for the learned tier (about 880 MB once
torch is in; model weights download on first use). Re-verified on 2026-10-08: same
PSNR to 0.1 dB and the same pass/fail pattern on the four channels spot-checked.

```bash
python bench.py photo.jpg --method lsb       --trials 10
python bench.py photo.jpg --method dwt       --trials 10
python bench.py photo.jpg --method dwt-rs --rs-k 8 --trials 10
python bench.py photo.jpg --method trustmark --trials 10
```

The cover photograph is not in this repository: it is not mine to redistribute. Any
reasonably detailed photograph at a similar size behaves comparably; `--synthetic`
gives you a runnable cover but, as the tool warns, worse numbers than these.

## Results

| Channel | LSB | dwt | dwt-rs RS(32,8) | TrustMark |
|---|---|---|---|---|
| baseline | 10/10 | 10/10 | 10/10 | 10/10 |
| JPEG QF90 | 0/10 (51.29% BER) | 10/10 | 10/10 | 10/10 |
| JPEG QF80 | 0/10 (51.41%) | 7/10 (0.16%) | 10/10 | 10/10 |
| JPEG QF70 | 0/10 (51.41%) | 0/10 (4.22%) | 9/10 | 10/10 |
| JPEG QF60 | 0/10 (50.78%) | 0/10 (6.05%) | 5/10 | 10/10 |
| JPEG QF50 | 0/10 (50.74%) | 0/10 (8.67%) | 0/10 | **10/10** |
| resize→1080 only | 0/10 (51.21%) | 0/10 (51.68%) | 0/10 | **10/10** |
| sim Telegram (q89) | 0/10 (49.53%) | 9/10 (0.04%) | 10/10 | 10/10 |
| sim WhatsApp (<=1600,q72) | 0/10 (49.38%) | 0/10 (3.20%) | **10/10** | 10/10 |
| sim Instagram (<=1080,q80) | 0/10 (51.02%) | 0/10 (50.12%) | 0/10 | **10/10** |
| sim WeChat (short<=1080,q80) | 0/10 (50.51%) | 3/10 (0.31%) | 10/10 | 10/10 |
| sim WeChat->WhatsApp | 0/10 (49.22%) | 0/10 (1.84%) | **10/10** | 10/10 |

PSNR: LSB 94.4 dB, dwt/dwt-rs 39.1 dB, TrustMark 40.7 dB.

TrustMark is 120/120 — every channel, every trial.

## What the numbers say

1. **LSB is not degraded by a channel, it is erased.** ~50% BER is a coin flip per
   bit: zero information survives. JPEG QF90 with no resize at all is already
   total loss. No ECC recovers from this, because there is nothing to correct.

2. **Reed-Solomon is decisive for recompression.** Raw dwt through WhatsApp is
   3.20% BER and 0/10 exact — nearly intact, still useless. RS(32,8) on the same
   channel is 10/10, and survives WeChat→WhatsApp (two resizes, two re-encodes).
   Cost: payload drops from 32 bytes to 8.

3. **Resize is a different failure and the ECC cannot touch it.** Bare resize and
   Instagram return ~50% BER for every *classic* method. That is desynchronisation,
   not corruption — the decoder reads the wrong coefficients. Note dwt survives
   *lossy* QF50 at 8.67% BER but dies on a *lossless* rescale. Adding parity cannot
   help, because the codeword is not damaged, it is not being read.

4. **The learned mark clears the wall, and does not pay for it in visibility.**
   TrustMark is 10/10 on all twelve channels, including the two that defeat
   everything else. The expected trade-off did not appear: at **40.7 dB it is
   *less* visible than the hand-designed mark at 39.1 dB**, while being strictly
   more robust. So the real cost of the learned tier is not image quality — it is
   **torch, ~880 MB of dependencies and model weights**, which is a deployment and
   battery question, not a signal-processing one. That is the honest reason to
   keep dwtDctSvd+RS as a fallback tier rather than a quality argument.

5. **"Robustness costs visibility" is only true of LSB.** 94.4 dB buys total
   imperceptibility and survives nothing. Among the methods that actually survive
   a channel, the ranking on PSNR and the ranking on robustness do not conflict.

## Caveats

One image, ten trials — enough to separate 0/10 from 10/10 and 50% from 3%, not
enough for a survival probability to two decimals.

**The `sim` channels are not the platforms they are named after.** Each is a
resize plus a JPEG re-encode with parameters hardcoded in `bench.py` since
2026-08-21, and those parameters carry no recorded provenance — I have not
verified any of them against a real send. "sim WhatsApp 10/10" means "10/10
through ≤1600px + q72" and nothing more. The channel keys were renamed on
2026-09-23 to carry their own transform, precisely so a table copied out of this
tool cannot be mistaken for platform evidence. Real platform results come from
sending an actual photo through the actual app and opening it on the recipient
device; that matrix is measured separately and does not live here.

None of this is a claim about **detectability**. Every method above is trivially
detectable by an analyst looking for it, and the robust mark is *more*
statistically conspicuous than LSB, not less.
