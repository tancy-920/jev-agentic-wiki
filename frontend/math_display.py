"""Conservative display-only math typography; never change stored/verified answers."""

import re

from frontend.citation_display import compact_citations

_NOTICE = re.compile(
    r"\A(?:LIVE\s*[—-]\s*)?已通过模型引用验证，尚非人工核实的事实(?:\r?\n|$)\s*"
)
# Preserve code, native math, Markdown links and URLs verbatim. Unknown formulas
# remain literal rather than asking another model or guessing their meaning.
_PROTECTED = re.compile(
    r"(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`|\$\$[\s\S]*?\$\$|"
    r"(?<!\\)\$[^$\n]+\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|"
    r"\[[^\]\n]*\]\([^\n)]*\)|https?://[^\s]+)"
)
_FORMULAS = (
    (r"FFN\(x\)\s*=\s*max\(0,\s*xW[1₁]\s*\+\s*b[1₁]\)\s*W[2₂]\s*\+\s*b[2₂]",
     r"\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2"),
    (r"SwiGLU\(x\)\s*=\s*\[SiLU\(xW\)\s*⊙\s*\(xV\)\]\s*W[2₂]",
     r"\operatorname{SwiGLU}(x)=[\operatorname{SiLU}(xW)\odot(xV)]W_2"),
    (r"SiLU\(z\)\s*=\s*z\s*[·*]\s*sigmoid\(z\)",
     r"\operatorname{SiLU}(z)=z\,\operatorname{sigmoid}(z)"),
    (r"ReLU\(x\)\s*=\s*max\(0,\s*x\)", r"\operatorname{ReLU}(x)=\max(0,x)"),
    (r"GELU\(x\)\s*=\s*x\s*Φ\(x\)", r"\operatorname{GELU}(x)=x\Phi(x)"),
    (r"d_gated\s*≈\s*\(2/3\)\s*[·*]?\s*d_ff",
     r"d_{\mathrm{gated}}\approx\frac{2}{3}d_{\mathrm{ff}}"),
    (r"d_model\s*→\s*d_ff\s*→\s*d_model",
     r"d_{\mathrm{model}}\to d_{\mathrm{ff}}\to d_{\mathrm{model}}"),
    (r"2\s*[·*]\s*L\s*[·*]\s*B\s*[·*]\s*T\s*[·*]\s*H_kv\s*[·*]\s*d_head\s*[·*]\s*s",
     r"2\cdot L\cdot B\cdot T\cdot H_{\mathrm{kv}}\cdot d_{\mathrm{head}}\cdot s"),
    (r"2\s*[·*]\s*d_model\s*[·*]\s*d_ff", r"2\,d_{\mathrm{model}}d_{\mathrm{ff}}"),
    (r"3\s*[·*]\s*d_model\s*[·*]\s*d_gated", r"3\,d_{\mathrm{model}}d_{\mathrm{gated}}"),
    (r"LayerNorm\(x\s*\+\s*Sublayer\(x\)\)",
     r"\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))"),
    (r"x\s*\+\s*F\(LN\(x\)\)", r"x+F(\operatorname{LN}(x))"),
    (r"\d+(?:\s*→\s*\d+){2,}", None),
)
_RULES = [(re.compile(r"(?<![A-Za-z0-9_])" + p + r"(?![A-Za-z0-9_])"), tex)
          for p, tex in _FORMULAS]
_IDENTIFIERS = {
    "d_model": r"d_{\mathrm{model}}", "d_ff": r"d_{\mathrm{ff}}",
    "d_gated": r"d_{\mathrm{gated}}", "d_head": r"d_{\mathrm{head}}",
    "H_kv": r"H_{\mathrm{kv}}", "W₁": "W_1", "W₂": "W_2",
}
_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_/])(" + "|".join(_IDENTIFIERS) + r")(?![A-Za-z0-9_./])")


def _plain_math(text):
    for pattern, tex in _RULES:
        text = pattern.sub(lambda m: "$" + (tex if tex is not None else
                           re.sub(r"\s*→\s*", r"\\to ", m.group())) + "$", text)
    # Rules above introduce math spans; protect those before formatting remaining
    # standalone identifiers so repeated rendering remains idempotent.
    pieces = _PROTECTED.split(text)
    for i in range(0, len(pieces), 2):
        pieces[i] = _IDENTIFIER.sub(lambda m: "$" + _IDENTIFIERS[m.group()] + "$", pieces[i])
    return "".join(pieces)


def answer_markdown(answer):
    text = compact_citations(_NOTICE.sub("", answer, count=1))
    pieces = _PROTECTED.split(text)
    for i, piece in enumerate(pieces):
        if i % 2 == 0:
            pieces[i] = _plain_math(piece)
        elif piece.startswith(r"\(") and piece.endswith(r"\)"):
            pieces[i] = "$" + piece[2:-2] + "$"
        elif piece.startswith(r"\[") and piece.endswith(r"\]"):
            pieces[i] = "\n\n$$\n" + piece[2:-2] + "\n$$\n\n"
    return "".join(pieces)
