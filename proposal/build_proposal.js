// PathXDRP — Research Proposal Deck
// Generates a clean, premium 16x9 wide deck.
// Palette: Deep Navy-Teal primary, Teal mid, Coral accent, Cream/Off-white content.
// Motif: thin coral vertical bar + small coral accent dot.

const path = require("path");
const pptxgen = require(path.join(process.env.APPDATA, "npm", "node_modules", "pptxgenjs"));
const React = require(path.join(process.env.APPDATA, "npm", "node_modules", "react"));
const ReactDOMServer = require(path.join(process.env.APPDATA, "npm", "node_modules", "react-dom/server"));
const sharp = require(path.join(process.env.APPDATA, "npm", "node_modules", "sharp"));

const FA = require(path.join(process.env.APPDATA, "npm", "node_modules", "react-icons/fa"));

// ---------------------------------------------------------------------------
// Palette + typography
// ---------------------------------------------------------------------------
const C = {
  navy:    "0E2A3F",   // dominant dark
  navyAlt: "163C56",
  teal:    "1C7293",   // secondary
  tealLt:  "4FA9C7",
  coral:   "E8826A",   // accent
  coralLt: "F2B49E",
  gold:    "E8C16A",   // secondary accent for tertiary highlights
  cream:   "F7F2EB",   // light bg
  paper:   "FFFFFF",
  ink:     "1A1A1A",
  inkSoft: "404858",
  muted:   "7A8392",
  hairline:"D9D2C7",
};

const F = {
  head: "Georgia",
  body: "Calibri",
  mono: "Consolas",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderIconSvg(IconComponent, color = "#000000", size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}
async function iconPng(IconComponent, hex, size = 256) {
  // Strip "#" if accidentally provided
  const color = hex.startsWith("#") ? hex : "#" + hex;
  const svg = renderIconSvg(IconComponent, color, size);
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// Fresh shadow object factory (pptxgen mutates options in place)
const makeShadow = () => ({
  type: "outer", color: "000000", opacity: 0.10, blur: 8, offset: 3, angle: 90,
});

// Page-number footer for content slides
function addFooter(slide, pageNo, total) {
  slide.addShape("line", {
    x: 0.6, y: 7.0, w: 12.1, h: 0,
    line: { color: C.hairline, width: 0.5 },
  });
  slide.addText("PathXDRP", {
    x: 0.6, y: 7.05, w: 4, h: 0.35,
    fontFace: F.head, fontSize: 9, color: C.muted, italic: true, margin: 0,
  });
  slide.addText(`${pageNo} / ${total}`, {
    x: 11.5, y: 7.05, w: 1.2, h: 0.35,
    fontFace: F.body, fontSize: 9, color: C.muted, align: "right", margin: 0,
  });
}

// Section heading with motif (small coral dot + serif title)
function addSectionHead(slide, kicker, title, subtitle) {
  // Kicker (small uppercase teal)
  slide.addText(kicker, {
    x: 0.6, y: 0.45, w: 8, h: 0.3,
    fontFace: F.body, fontSize: 11, color: C.teal, bold: true, charSpacing: 4, margin: 0,
  });
  // Coral accent dot
  slide.addShape("ellipse", {
    x: 0.6, y: 0.95, w: 0.18, h: 0.18,
    fill: { color: C.coral }, line: { color: C.coral, width: 0 },
  });
  // Title
  slide.addText(title, {
    x: 0.92, y: 0.85, w: 11.5, h: 0.65,
    fontFace: F.head, fontSize: 30, bold: true, color: C.navy, margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.92, y: 1.42, w: 11.5, h: 0.42,
      fontFace: F.body, fontSize: 14, color: C.inkSoft, italic: true, margin: 0,
    });
  }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
  pres.author = "Aly Hesham";
  pres.title  = "PathXDRP — Research Proposal";

  const TOTAL = 14;

  // Pre-render icons
  const I = {
    cancer:  await iconPng(FA.FaDna,            C.coral),
    cost:    await iconPng(FA.FaDollarSign,     C.coral),
    flask:   await iconPng(FA.FaFlask,          C.coral),
    eye:     await iconPng(FA.FaEyeSlash,       C.coral),
    shield:  await iconPng(FA.FaShieldAlt,      C.coral),
    biology: await iconPng(FA.FaProjectDiagram, C.coral),
    network: await iconPng(FA.FaNetworkWired,   C.teal),
    layers:  await iconPng(FA.FaLayerGroup,     C.teal),
    gauge:   await iconPng(FA.FaTachometerAlt,  C.teal),
    search:  await iconPng(FA.FaSearchPlus,     C.teal),
    bolt:    await iconPng(FA.FaBolt,           C.teal),
    pill:    await iconPng(FA.FaPills,          C.coral),
    chart:   await iconPng(FA.FaChartLine,      C.teal),
    check:   await iconPng(FA.FaCheckCircle,    C.tealLt),
    gear:    await iconPng(FA.FaCogs,           C.teal),
    target:  await iconPng(FA.FaCrosshairs,     C.coral),
    book:    await iconPng(FA.FaBookOpen,       C.coral),
    map:     await iconPng(FA.FaMapSigns,       C.coral),
    brainCt: await iconPng(FA.FaBrain,          C.teal),
    quoteL:  await iconPng(FA.FaQuoteLeft,      C.coralLt),
  };

  // =========================================================================
  // SLIDE 1 — Title (dark hero)
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };

    // Decorative tealish band
    s.addShape("rect", {
      x: 0, y: 6.4, w: 13.3, h: 1.1,
      fill: { color: C.navyAlt }, line: { color: C.navyAlt, width: 0 },
    });

    // Coral motif: slim vertical bar
    s.addShape("rect", {
      x: 0.7, y: 1.1, w: 0.08, h: 5.0,
      fill: { color: C.coral }, line: { color: C.coral, width: 0 },
    });

    // Kicker
    s.addText("RESEARCH PROPOSAL  ·  Q1 BIOINFORMATICS", {
      x: 1.0, y: 1.3, w: 10, h: 0.35,
      fontFace: F.body, fontSize: 11, color: C.coralLt, bold: true,
      charSpacing: 6, margin: 0,
    });

    // Wordmark
    s.addText("PathXDRP", {
      x: 1.0, y: 1.7, w: 12, h: 1.55,
      fontFace: F.head, fontSize: 84, bold: true, color: C.paper, margin: 0,
    });

    // Subtitle
    s.addText("Pathway-masked Cross-attention Drug Response Predictor", {
      x: 1.0, y: 3.25, w: 11.5, h: 0.6,
      fontFace: F.head, fontSize: 24, italic: true, color: C.tealLt, margin: 0,
    });

    // Tagline box
    s.addText("Interpretable, calibrated, knowledge-grounded — and lightweight enough to run anywhere.", {
      x: 1.0, y: 4.05, w: 11.0, h: 0.6,
      fontFace: F.body, fontSize: 16, color: C.cream, italic: true, margin: 0,
    });

    // Rule
    s.addShape("line", {
      x: 1.0, y: 4.85, w: 2.5, h: 0,
      line: { color: C.coral, width: 1.5 },
    });

    // Author block
    s.addText("Aly Hesham   ·   GNN Drug Discovery", {
      x: 1.0, y: 5.0, w: 8, h: 0.35,
      fontFace: F.body, fontSize: 14, color: C.cream, margin: 0,
    });
    s.addText("Master's Thesis Proposal · 2026", {
      x: 1.0, y: 5.36, w: 8, h: 0.3,
      fontFace: F.body, fontSize: 12, color: C.muted, margin: 0,
    });

    // Footer band content
    s.addText("Pathway-masked Cross-attention  ·  Foundation Encoders  ·  Evidential Uncertainty  ·  Quantitative XAI  ·  Edge-Device Inference", {
      x: 0.6, y: 6.55, w: 12.1, h: 0.4,
      fontFace: F.body, fontSize: 11, color: C.tealLt, align: "center", margin: 0,
    });
  }

  // =========================================================================
  // SLIDE 2 — The Challenge
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "01  ·  THE CHALLENGE",
      "Cancer drug response is hard to predict — and the cost of being wrong is enormous.",
      "Yet the data and models meant to solve it leave most of the biology on the table.",
    );

    // Three large stat cards
    const stats = [
      { num: "$2.6B",  lab: "average cost to develop a single oncology drug",         icon: I.cost },
      { num: "10–15 y", lab: "from target identification to FDA approval",            icon: I.flask },
      { num: "≈ 5%",   lab: "of oncology candidates make it through clinical trials", icon: I.cancer },
    ];

    const top = 2.4, h = 3.7, gap = 0.35;
    const w = (13.3 - 1.2 - 2 * gap) / 3;
    stats.forEach((st, i) => {
      const x = 0.6 + i * (w + gap);
      // Card
      s.addShape("rect", {
        x, y: top, w, h,
        fill: { color: C.paper }, line: { color: C.hairline, width: 0.5 },
        shadow: makeShadow(),
      });
      // Coral accent bar (left)
      s.addShape("rect", {
        x, y: top, w: 0.09, h,
        fill: { color: C.coral }, line: { color: C.coral, width: 0 },
      });
      // Icon
      s.addImage({ data: st.icon, x: x + 0.45, y: top + 0.35, w: 0.55, h: 0.55 });
      // Big number
      s.addText(st.num, {
        x: x + 0.45, y: top + 1.05, w: w - 0.7, h: 1.4,
        fontFace: F.head, fontSize: 56, bold: true, color: C.navy, margin: 0,
      });
      // Label
      s.addText(st.lab, {
        x: x + 0.45, y: top + 2.55, w: w - 0.8, h: 1.1,
        fontFace: F.body, fontSize: 14, color: C.inkSoft, margin: 0,
      });
    });

    // Footer note
    s.addText("Better predictions in silico mean fewer dead-end trials in vivo. We need models that not only predict — but explain, and know when not to predict.", {
      x: 0.6, y: 6.3, w: 12.1, h: 0.6,
      fontFace: F.body, fontSize: 13, color: C.inkSoft, italic: true, align: "center", margin: 0,
    });

    addFooter(s, 2, TOTAL);
  }

  // =========================================================================
  // SLIDE 3 — Where current approaches fall short
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "02  ·  WHY THIS IS UNSOLVED",
      "Today's drug-response models share three fundamental limits.",
      "Each one alone is a deal-breaker for clinical translation.",
    );

    const items = [
      {
        n: "01",
        title: "Black-box predictions",
        body: "GNN-based DRP models output a number with no biological reason. Clinicians can't act on a score they can't interrogate. Existing attention maps are descriptive, not benchmarked.",
        icon: I.eye,
      },
      {
        n: "02",
        title: "Overconfident on novel inputs",
        body: "Standard MSE training gives no calibrated uncertainty. The model is just as confident on a held-out scaffold as on its training distribution — exactly when caution matters most.",
        icon: I.shield,
      },
      {
        n: "03",
        title: "Knowledge-blind architectures",
        body: "Pathway biology, gene-gene coregulation, and known drug targets exist as priors — but most models ignore them, learning correlations from scratch on a small (~150k-row) dataset.",
        icon: I.biology,
      },
    ];

    const top = 2.4, h = 4.0, gap = 0.35;
    const w = (13.3 - 1.2 - 2 * gap) / 3;
    items.forEach((it, i) => {
      const x = 0.6 + i * (w + gap);
      s.addShape("rect", {
        x, y: top, w, h,
        fill: { color: C.paper }, line: { color: C.hairline, width: 0.5 },
        shadow: makeShadow(),
      });
      // Number
      s.addText(it.n, {
        x: x + 0.4, y: top + 0.3, w: 1.5, h: 0.6,
        fontFace: F.head, fontSize: 36, bold: true, color: C.coralLt, margin: 0,
      });
      s.addImage({ data: it.icon, x: x + w - 0.95, y: top + 0.4, w: 0.5, h: 0.5 });

      // Title
      s.addText(it.title, {
        x: x + 0.4, y: top + 1.0, w: w - 0.6, h: 0.55,
        fontFace: F.head, fontSize: 19, bold: true, color: C.navy, margin: 0,
      });
      // Coral underline
      s.addShape("rect", {
        x: x + 0.4, y: top + 1.55, w: 0.5, h: 0.04,
        fill: { color: C.coral }, line: { color: C.coral, width: 0 },
      });
      // Body
      s.addText(it.body, {
        x: x + 0.4, y: top + 1.75, w: w - 0.7, h: 2.1,
        fontFace: F.body, fontSize: 13, color: C.inkSoft, valign: "top", paraSpaceAfter: 4,
      });
    });

    addFooter(s, 3, TOTAL);
  }

  // =========================================================================
  // SLIDE 4 — Our Vision
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "03  ·  OUR PROPOSAL",
      "PathXDRP — built around five mutually reinforcing ideas.",
      "Each pillar fixes a specific failure mode of today's models.",
    );

    // Big visual on left, value prop on right
    const lx = 0.6, ly = 2.4;
    s.addShape("rect", {
      x: lx, y: ly, w: 5.5, h: 4.2,
      fill: { color: C.navy }, line: { color: C.navy, width: 0 },
      shadow: makeShadow(),
    });

    // Big quote mark
    s.addImage({ data: I.quoteL, x: lx + 0.4, y: ly + 0.3, w: 0.6, h: 0.6 });

    s.addText("A drug-response model that explains its reasoning in the language biologists already speak — pathways, targets, mechanisms — and admits what it doesn't know.", {
      x: lx + 0.4, y: ly + 1.05, w: 4.7, h: 2.3,
      fontFace: F.head, fontSize: 18, italic: true, color: C.cream, margin: 0,
    });
    s.addShape("rect", {
      x: lx + 0.4, y: ly + 3.5, w: 0.5, h: 0.04,
      fill: { color: C.coral }, line: { color: C.coral, width: 0 },
    });
    s.addText("THE PATHXDRP THESIS", {
      x: lx + 0.4, y: ly + 3.6, w: 4, h: 0.3,
      fontFace: F.body, fontSize: 10, bold: true, color: C.coralLt, charSpacing: 4, margin: 0,
    });

    // Right column — five pillars
    const items = [
      { ic: I.network, t: "Pathway-masked cross-attention",     d: "Drug atoms attend over KEGG-grounded biological pathways." },
      { ic: I.layers,  t: "Foundation-model encoders",          d: "MolFormer · Graph-Mamba · GeneMamba · scGPT." },
      { ic: I.gauge,   t: "Calibrated evidential uncertainty",  d: "Aleatoric + epistemic decomposition for selective prediction." },
      { ic: I.search,  t: "Quantitative XAI benchmark",          d: "25 MoA drugs scored against ground-truth targets." },
      { ic: I.bolt,    t: "Linear-time deployment",              d: "Edge-device throughput, batch_size = 1 latency under 10 ms." },
    ];

    const rx = 6.6, ry0 = 2.4, rh = 0.78, rgap = 0.08;
    items.forEach((it, i) => {
      const y = ry0 + i * (rh + rgap);
      // Icon disc
      s.addShape("ellipse", {
        x: rx, y: y + 0.06, w: 0.65, h: 0.65,
        fill: { color: C.paper }, line: { color: C.hairline, width: 0.5 },
      });
      s.addImage({ data: it.ic, x: rx + 0.16, y: y + 0.22, w: 0.33, h: 0.33 });
      s.addText(it.t, {
        x: rx + 0.85, y: y, w: 5.7, h: 0.4,
        fontFace: F.head, fontSize: 15, bold: true, color: C.navy, margin: 0,
      });
      s.addText(it.d, {
        x: rx + 0.85, y: y + 0.38, w: 5.7, h: 0.45,
        fontFace: F.body, fontSize: 11.5, color: C.inkSoft, margin: 0,
      });
    });

    addFooter(s, 4, TOTAL);
  }

  // =========================================================================
  // SLIDE 5 — Architecture
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "04  ·  ARCHITECTURE",
      "End-to-end pipeline — and where biology enters the model.",
      "Two parallel encoders meet at a knowledge-grounded attention bridge.",
    );

    // Pipeline stages
    const y0 = 2.6;
    const stageH = 1.8;
    const stages = [
      { x: 0.6,  w: 2.3, color: C.teal, title: "DRUG",   sub: "SMILES → atom graph",     extra: "GATv2 / Graph-Mamba / MolFormer" },
      { x: 0.6,  w: 2.3, color: C.teal, title: "CELL",   sub: "Expression → pathways",   extra: "PathwaySet / GeneMamba / scGPT", row: 2 },
    ];

    // Drug branch (top row)
    s.addShape("rect", {
      x: 0.6, y: 2.5, w: 2.4, h: 1.4,
      fill: { color: C.navy }, line: { color: C.navy, width: 0 },
      shadow: makeShadow(),
    });
    s.addText("DRUG", {
      x: 0.6, y: 2.55, w: 2.4, h: 0.3,
      fontFace: F.body, fontSize: 10, bold: true, color: C.coralLt, charSpacing: 4, align: "center", margin: 0,
    });
    s.addText("SMILES → Graph", {
      x: 0.6, y: 2.85, w: 2.4, h: 0.4,
      fontFace: F.head, fontSize: 16, bold: true, color: C.paper, align: "center", margin: 0,
    });
    s.addText("GATv2  ·  Graph-Mamba  ·  MolFormer", {
      x: 0.6, y: 3.32, w: 2.4, h: 0.55,
      fontFace: F.body, fontSize: 11, color: C.tealLt, align: "center", margin: 0,
    });

    // Cell branch (bottom row)
    s.addShape("rect", {
      x: 0.6, y: 4.6, w: 2.4, h: 1.4,
      fill: { color: C.navy }, line: { color: C.navy, width: 0 },
      shadow: makeShadow(),
    });
    s.addText("CELL", {
      x: 0.6, y: 4.65, w: 2.4, h: 0.3,
      fontFace: F.body, fontSize: 10, bold: true, color: C.coralLt, charSpacing: 4, align: "center", margin: 0,
    });
    s.addText("Expression → Pathways", {
      x: 0.6, y: 4.95, w: 2.4, h: 0.4,
      fontFace: F.head, fontSize: 16, bold: true, color: C.paper, align: "center", margin: 0,
    });
    s.addText("PathwaySet  ·  GeneMamba  ·  scGPT", {
      x: 0.6, y: 5.42, w: 2.4, h: 0.55,
      fontFace: F.body, fontSize: 11, color: C.tealLt, align: "center", margin: 0,
    });

    // Cross-attention center
    s.addShape("rect", {
      x: 4.0, y: 3.4, w: 4.0, h: 1.7,
      fill: { color: C.coral }, line: { color: C.coral, width: 0 },
      shadow: makeShadow(),
    });
    s.addText("PATHWAY-MASKED", {
      x: 4.0, y: 3.45, w: 4.0, h: 0.3,
      fontFace: F.body, fontSize: 10, bold: true, color: C.cream, charSpacing: 4, align: "center", margin: 0,
    });
    s.addText("Cross-attention", {
      x: 4.0, y: 3.75, w: 4.0, h: 0.55,
      fontFace: F.head, fontSize: 22, bold: true, color: C.paper, align: "center", italic: true, margin: 0,
    });
    s.addText("KEGG-grounded mask · 370 pathway tokens", {
      x: 4.0, y: 4.4, w: 4.0, h: 0.45,
      fontFace: F.body, fontSize: 11, color: C.cream, align: "center", margin: 0,
    });

    // Connecting lines
    s.addShape("line", { x: 3.0, y: 3.2, w: 1.0, h: 1.05, line: { color: C.teal, width: 2 } });
    s.addShape("line", { x: 3.0, y: 5.3, w: 1.0, h: -1.05, line: { color: C.teal, width: 2 } });

    // Right side: head + outputs
    s.addShape("rect", {
      x: 9.0, y: 3.4, w: 3.7, h: 1.7,
      fill: { color: C.navy }, line: { color: C.navy, width: 0 },
      shadow: makeShadow(),
    });
    s.addText("EVIDENTIAL HEAD", {
      x: 9.0, y: 3.45, w: 3.7, h: 0.3,
      fontFace: F.body, fontSize: 10, bold: true, color: C.coralLt, charSpacing: 4, align: "center", margin: 0,
    });
    s.addText("IC50  +  Uncertainty", {
      x: 9.0, y: 3.78, w: 3.7, h: 0.55,
      fontFace: F.head, fontSize: 18, bold: true, color: C.paper, align: "center", margin: 0,
    });
    s.addText("μ · σ²ₐₗ · σ²ₑₚ  (NIG-NLL)", {
      x: 9.0, y: 4.4, w: 3.7, h: 0.45,
      fontFace: F.body, fontSize: 11, color: C.tealLt, align: "center", margin: 0,
    });

    s.addShape("line", { x: 8.0, y: 4.25, w: 1.0, h: 0, line: { color: C.coral, width: 2 } });

    // Caption
    s.addText("Drug atoms (Q) attend over biological pathway tokens (K, V) — the attention pattern is the explanation.", {
      x: 0.6, y: 6.3, w: 12.1, h: 0.6,
      fontFace: F.body, fontSize: 13, color: C.inkSoft, italic: true, align: "center", margin: 0,
    });

    addFooter(s, 5, TOTAL);
  }

  // =========================================================================
  // SLIDE 6 — Novelty 1: Pathway-masked Cross-attention
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "05  ·  NOVELTY 1",
      "Pathway-masked Cross-attention",
      "Biology is not a post-hoc lens — it is wired into the architecture.",
    );

    // Left: explanation list
    const items = [
      { t: "Drug atoms as Query",          d: "Each atom asks: \"which biological pathways do I act on?\"" },
      { t: "Pathway tokens as Key/Value",  d: "370 KEGG pathways form the dictionary of possible mechanisms." },
      { t: "Hard / soft / no-mask modes",   d: "Architectural ablation isolates the contribution of biological priors." },
      { t: "Memory-efficient design",      d: "to_dense_batch + head-parallel scoring → 16× less GPU memory." },
    ];

    const lx = 0.6, ly = 2.4;
    items.forEach((it, i) => {
      const y = ly + i * 1.0;
      // numbered bullet
      s.addShape("ellipse", {
        x: lx, y: y, w: 0.45, h: 0.45,
        fill: { color: C.coral }, line: { color: C.coral, width: 0 },
      });
      s.addText(String(i + 1), {
        x: lx, y: y, w: 0.45, h: 0.45,
        fontFace: F.head, fontSize: 16, bold: true, color: C.paper, align: "center", valign: "middle", margin: 0,
      });
      s.addText(it.t, {
        x: lx + 0.65, y: y - 0.05, w: 6.5, h: 0.4,
        fontFace: F.head, fontSize: 16, bold: true, color: C.navy, margin: 0,
      });
      s.addText(it.d, {
        x: lx + 0.65, y: y + 0.32, w: 6.5, h: 0.5,
        fontFace: F.body, fontSize: 12.5, color: C.inkSoft, margin: 0,
      });
    });

    // Right: pull quote / figure box
    const rx = 8.2, ry = 2.4;
    s.addShape("rect", {
      x: rx, y: ry, w: 4.5, h: 4.2,
      fill: { color: C.navy }, line: { color: C.navy, width: 0 },
      shadow: makeShadow(),
    });
    s.addText("WHY IT WORKS", {
      x: rx + 0.4, y: ry + 0.4, w: 3.5, h: 0.3,
      fontFace: F.body, fontSize: 10, bold: true, color: C.coralLt, charSpacing: 4, margin: 0,
    });
    s.addText("Drug → pathway attention is the model's reasoning trace, made native to the forward pass.", {
      x: rx + 0.4, y: ry + 0.75, w: 3.7, h: 1.6,
      fontFace: F.head, fontSize: 18, italic: true, color: C.paper, margin: 0,
    });
    s.addShape("rect", {
      x: rx + 0.4, y: ry + 2.5, w: 0.5, h: 0.04,
      fill: { color: C.coral }, line: { color: C.coral, width: 0 },
    });
    s.addText("Entropy regularisation pushes the attention to be sparse and pathway-specific — directly evaluable as an explanation.", {
      x: rx + 0.4, y: ry + 2.65, w: 3.7, h: 1.4,
      fontFace: F.body, fontSize: 12, color: C.tealLt, margin: 0,
    });

    addFooter(s, 6, TOTAL);
  }

  // =========================================================================
  // SLIDE 7 — Novelty 2: Foundation Encoders
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "06  ·  NOVELTY 2",
      "Foundation-model encoders for chemistry and biology.",
      "Stop training small encoders from scratch — borrow what's been pretrained on millions of examples.",
    );

    const cards = [
      {
        title: "MolFormer-XL",
        sub:   "1.1B SMILES strings",
        body:  "Frozen chemical-language backbone augments atom features with global molecular context.",
        side:  "DRUG ENCODER",
        ic:    I.flask,
      },
      {
        title: "Graph-Mamba",
        sub:   "Linear-time SSM over atom graph",
        body:  "Replaces quadratic attention with selective state-space modeling — a hybrid GAT + Bi-Mamba block.",
        side:  "DRUG ENCODER",
        ic:    I.bolt,
      },
      {
        title: "GeneMamba",
        sub:   "30M single cells, 65.7M params",
        body:  "Pretrained Bi-Mamba foundation model captures gene-gene coregulation we currently throw away.",
        side:  "CELL ENCODER",
        ic:    I.brainCt,
      },
      {
        title: "scGPT",
        sub:   "Single-cell foundation model",
        body:  "Symmetric ablation against GeneMamba — Transformer vs. SSM on the cell side, fair comparison.",
        side:  "CELL ENCODER",
        ic:    I.network,
      },
    ];

    const grid = { col: 2, x0: 0.6, y0: 2.4, w: 6.05, h: 2.05, gx: 0.3, gy: 0.25 };
    cards.forEach((c, i) => {
      const r = Math.floor(i / grid.col), col = i % grid.col;
      const x = grid.x0 + col * (grid.w + grid.gx);
      const y = grid.y0 + r * (grid.h + grid.gy);
      s.addShape("rect", {
        x, y, w: grid.w, h: grid.h,
        fill: { color: C.paper }, line: { color: C.hairline, width: 0.5 },
        shadow: makeShadow(),
      });
      s.addShape("rect", {
        x, y, w: 0.09, h: grid.h,
        fill: { color: C.coral }, line: { color: C.coral, width: 0 },
      });
      s.addImage({ data: c.ic, x: x + 0.35, y: y + 0.3, w: 0.5, h: 0.5 });
      s.addText(c.side, {
        x: x + 1.0, y: y + 0.3, w: 4.5, h: 0.3,
        fontFace: F.body, fontSize: 9, bold: true, color: C.coral, charSpacing: 4, margin: 0,
      });
      s.addText(c.title, {
        x: x + 1.0, y: y + 0.55, w: 4.8, h: 0.45,
        fontFace: F.head, fontSize: 20, bold: true, color: C.navy, margin: 0,
      });
      s.addText(c.sub, {
        x: x + 1.0, y: y + 1.0, w: 4.8, h: 0.35,
        fontFace: F.body, fontSize: 12, italic: true, color: C.teal, margin: 0,
      });
      s.addText(c.body, {
        x: x + 0.35, y: y + 1.4, w: grid.w - 0.55, h: 0.7,
        fontFace: F.body, fontSize: 12, color: C.inkSoft, margin: 0,
      });
    });

    addFooter(s, 7, TOTAL);
  }

  // =========================================================================
  // SLIDE 8 — Novelty 3: Calibrated Uncertainty
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "07  ·  NOVELTY 3",
      "Calibrated evidential uncertainty.",
      "A model that knows when it doesn't know — backed by a principled probabilistic loss.",
    );

    // Left: explanation
    const lx = 0.6, ly = 2.4;
    s.addText("EVIDENTIAL DEEP LEARNING", {
      x: lx, y: ly, w: 6.5, h: 0.3,
      fontFace: F.body, fontSize: 11, bold: true, color: C.coral, charSpacing: 4, margin: 0,
    });
    s.addText("Predict (μ, ν, α, β) — the parameters of a Normal-Inverse-Gamma — instead of a point estimate. The NIG-NLL loss replaces MSE; uncertainty drops out for free.", {
      x: lx, y: ly + 0.4, w: 6.5, h: 1.4,
      fontFace: F.body, fontSize: 14, color: C.inkSoft, paraSpaceAfter: 6,
    });

    s.addShape("rect", {
      x: lx, y: ly + 1.95, w: 0.5, h: 0.04,
      fill: { color: C.coral }, line: { color: C.coral, width: 0 },
    });

    s.addText("TWO KINDS OF UNCERTAINTY", {
      x: lx, y: ly + 2.1, w: 6.5, h: 0.3,
      fontFace: F.body, fontSize: 11, bold: true, color: C.coral, charSpacing: 4, margin: 0,
    });

    const dec = [
      { t: "Aleatoric — irreducible noise",  d: "β / (α − 1)  ·  intrinsic to the (drug, cell) pair." },
      { t: "Epistemic — model uncertainty",  d: "β / (ν(α − 1))  ·  shrinks with more data." },
    ];
    dec.forEach((it, i) => {
      const y = ly + 2.5 + i * 0.85;
      s.addText(it.t, {
        x: lx, y, w: 6.5, h: 0.35,
        fontFace: F.head, fontSize: 14, bold: true, color: C.navy, margin: 0,
      });
      s.addText(it.d, {
        x: lx, y: y + 0.32, w: 6.5, h: 0.5,
        fontFace: F.mono, fontSize: 11, color: C.inkSoft, margin: 0,
      });
    });

    // Right: outcomes panel
    const rx = 8.0, ry = 2.4;
    s.addShape("rect", {
      x: rx, y: ry, w: 4.7, h: 4.2,
      fill: { color: C.navy }, line: { color: C.navy, width: 0 },
      shadow: makeShadow(),
    });
    s.addText("WHAT IT BUYS US", {
      x: rx + 0.4, y: ry + 0.35, w: 4, h: 0.3,
      fontFace: F.body, fontSize: 10, bold: true, color: C.coralLt, charSpacing: 4, margin: 0,
    });
    const wins = [
      { t: "Risk-coverage curves",        d: "Trade prediction count for accuracy on the fly." },
      { t: "Selective prediction",        d: "Refuse to score (drug, cell) pairs the model is unsure about." },
      { t: "OOD detection",               d: "Flag scaffold-blind / cell-blind cases by epistemic spike." },
      { t: "Deep-ensemble decomposition", d: "5-seed ensemble splits epistemic into model + data variance." },
    ];
    wins.forEach((w, i) => {
      const y = ry + 0.85 + i * 0.78;
      s.addImage({ data: I.check, x: rx + 0.4, y: y + 0.06, w: 0.32, h: 0.32 });
      s.addText(w.t, {
        x: rx + 0.85, y, w: 3.7, h: 0.32,
        fontFace: F.head, fontSize: 14, bold: true, color: C.paper, margin: 0,
      });
      s.addText(w.d, {
        x: rx + 0.85, y: y + 0.30, w: 3.7, h: 0.45,
        fontFace: F.body, fontSize: 11.5, color: C.tealLt, margin: 0,
      });
    });

    addFooter(s, 8, TOTAL);
  }

  // =========================================================================
  // SLIDE 9 — Novelty 4: Quantitative XAI
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "08  ·  NOVELTY 4",
      "Quantitative XAI benchmark.",
      "Stop publishing pretty heatmaps — start publishing scores against ground truth.",
    );

    // Two stacked panels
    // Panel A: 25 MoA drugs
    s.addShape("rect", {
      x: 0.6, y: 2.4, w: 6.0, h: 4.2,
      fill: { color: C.paper }, line: { color: C.hairline, width: 0.5 },
      shadow: makeShadow(),
    });
    s.addShape("rect", {
      x: 0.6, y: 2.4, w: 0.09, h: 4.2,
      fill: { color: C.coral }, line: { color: C.coral, width: 0 },
    });
    s.addImage({ data: I.target, x: 1.0, y: 2.65, w: 0.55, h: 0.55 });
    s.addText("25 well-characterised oncology drugs", {
      x: 1.7, y: 2.6, w: 4.6, h: 0.45,
      fontFace: F.head, fontSize: 18, bold: true, color: C.navy, margin: 0,
    });
    s.addText("EGFR · MAPK · PI3K · CDK · BCR-ABL · HDAC · BCL-2 · DNA damage · microtubule", {
      x: 1.0, y: 3.25, w: 5.4, h: 0.45,
      fontFace: F.body, fontSize: 12, color: C.teal, italic: true, margin: 0,
    });
    const moaItems = [
      "Per drug → IC50-sensitive cells, attribution maps generated",
      "Ground truth: GDSC-curated targets + KEGG pathway label",
      "Generated dataset shipped as data/processed/moa_benchmark.json",
    ];
    moaItems.forEach((t, i) => {
      const y = 3.85 + i * 0.42;
      s.addShape("ellipse", {
        x: 1.0, y: y + 0.08, w: 0.13, h: 0.13,
        fill: { color: C.coral }, line: { color: C.coral, width: 0 },
      });
      s.addText(t, {
        x: 1.25, y, w: 5.2, h: 0.4,
        fontFace: F.body, fontSize: 12.5, color: C.inkSoft, margin: 0,
      });
    });

    // Panel B: metrics
    s.addShape("rect", {
      x: 6.85, y: 2.4, w: 5.85, h: 4.2,
      fill: { color: C.navy }, line: { color: C.navy, width: 0 },
      shadow: makeShadow(),
    });
    s.addText("FOUR QUANTITATIVE METRICS", {
      x: 7.15, y: 2.65, w: 5.5, h: 0.3,
      fontFace: F.body, fontSize: 11, bold: true, color: C.coralLt, charSpacing: 4, margin: 0,
    });
    s.addText("Beyond eyeballing.", {
      x: 7.15, y: 3.0, w: 5.5, h: 0.4,
      fontFace: F.head, fontSize: 18, italic: true, color: C.paper, margin: 0,
    });

    const metrics = [
      { name: "Target AUROC",       d: "Rank genes by attribution; AUROC vs. known target set." },
      { name: "Pathway Hit@5",      d: "Is the curated KEGG pathway in the top-5 attended?" },
      { name: "Faithfulness suff.", d: "Δ-prediction on removing low-attribution atoms." },
      { name: "Faithfulness comp.", d: "Δ-prediction on removing high-attribution atoms." },
    ];
    metrics.forEach((m, i) => {
      const y = 3.6 + i * 0.65;
      s.addShape("rect", {
        x: 7.15, y: y + 0.05, w: 0.06, h: 0.55,
        fill: { color: C.coral }, line: { color: C.coral, width: 0 },
      });
      s.addText(m.name, {
        x: 7.32, y, w: 2.2, h: 0.32,
        fontFace: F.head, fontSize: 13, bold: true, color: C.paper, margin: 0,
      });
      s.addText(m.d, {
        x: 9.55, y, w: 3.05, h: 0.6,
        fontFace: F.body, fontSize: 11, color: C.tealLt, margin: 0,
      });
    });

    addFooter(s, 9, TOTAL);
  }

  // =========================================================================
  // SLIDE 10 — Novelty 5: Edge-device deployment
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "09  ·  NOVELTY 5",
      "Linear-time inference for deployment.",
      "A drug-response model is useless if it only runs on a workstation.",
    );

    // Three big stat cards
    const stats = [
      { num: "16×",  lab: "less GPU memory than naïve cross-attention",     icon: I.gauge },
      { num: "O(n)", lab: "Mamba SSM scales linearly with sequence length", icon: I.bolt  },
      { num: "<10 ms",lab: "single-sample latency target on an RTX 3060",   icon: I.chart },
    ];

    const top = 2.4, h = 2.5, gap = 0.35;
    const w = (13.3 - 1.2 - 2 * gap) / 3;
    stats.forEach((st, i) => {
      const x = 0.6 + i * (w + gap);
      s.addShape("rect", {
        x, y: top, w, h,
        fill: { color: C.paper }, line: { color: C.hairline, width: 0.5 },
        shadow: makeShadow(),
      });
      s.addShape("rect", {
        x, y: top, w: 0.09, h,
        fill: { color: C.coral }, line: { color: C.coral, width: 0 },
      });
      s.addImage({ data: st.icon, x: x + 0.4, y: top + 0.35, w: 0.55, h: 0.55 });
      s.addText(st.num, {
        x: x + 0.4, y: top + 0.95, w: w - 0.6, h: 0.9,
        fontFace: F.head, fontSize: 44, bold: true, color: C.navy, margin: 0,
      });
      s.addText(st.lab, {
        x: x + 0.4, y: top + 1.85, w: w - 0.6, h: 0.55,
        fontFace: F.body, fontSize: 12, color: C.inkSoft, margin: 0,
      });
    });

    // Bottom panel: deliverables
    const by = 5.2;
    s.addShape("rect", {
      x: 0.6, y: by, w: 12.1, h: 1.55,
      fill: { color: C.navy }, line: { color: C.navy, width: 0 },
      shadow: makeShadow(),
    });
    s.addText("DELIVERABLE", {
      x: 0.85, y: by + 0.2, w: 4, h: 0.3,
      fontFace: F.body, fontSize: 10, bold: true, color: C.coralLt, charSpacing: 4, margin: 0,
    });
    s.addText("A reproducible inference benchmark", {
      x: 0.85, y: by + 0.5, w: 7.5, h: 0.4,
      fontFace: F.head, fontSize: 18, bold: true, color: C.paper, margin: 0,
    });
    s.addText("scripts/benchmark_inference.py — sweeps every encoder combo, reports throughput, peak GPU memory, p95 latency, CPU-only feasibility.", {
      x: 0.85, y: by + 0.95, w: 11.5, h: 0.55,
      fontFace: F.body, fontSize: 12, color: C.tealLt, margin: 0,
    });

    addFooter(s, 10, TOTAL);
  }

  // =========================================================================
  // SLIDE 11 — Validation Strategy
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "10  ·  VALIDATION",
      "A validation protocol that survives every reviewer's first complaint.",
      "Five split regimes · five seeds · external transfer · ablation matrix.",
    );

    const items = [
      {
        ic: I.gear,
        t: "Five split regimes",
        d: "random  ·  cell-blind  ·  drug-blind  ·  scaffold-blind  ·  tissue-blind. Hardness gradient included on purpose.",
      },
      {
        ic: I.layers,
        t: "Five-seed deep ensembles",
        d: "Ensembling decomposes epistemic into model + data variance — and stabilises Q1-style PCC ± std reporting.",
      },
      {
        ic: I.network,
        t: "External transfer",
        d: "CCLE × CTRPv2 evaluation with z-score alignment. Asks: does the model survive the dataset shift?",
      },
      {
        ic: I.target,
        t: "Encoder ablation matrix",
        d: "Six (drug × cell) encoder combos under identical train/eval. Isolates each contribution.",
      },
    ];

    const grid = { col: 2, x0: 0.6, y0: 2.4, w: 6.05, h: 2.0, gx: 0.3, gy: 0.3 };
    items.forEach((c, i) => {
      const r = Math.floor(i / grid.col), col = i % grid.col;
      const x = grid.x0 + col * (grid.w + grid.gx);
      const y = grid.y0 + r * (grid.h + grid.gy);

      s.addShape("rect", {
        x, y, w: grid.w, h: grid.h,
        fill: { color: C.paper }, line: { color: C.hairline, width: 0.5 },
        shadow: makeShadow(),
      });
      s.addShape("ellipse", {
        x: x + 0.35, y: y + 0.4, w: 0.85, h: 0.85,
        fill: { color: C.cream }, line: { color: C.coral, width: 1 },
      });
      s.addImage({ data: c.ic, x: x + 0.55, y: y + 0.6, w: 0.45, h: 0.45 });
      s.addText(c.t, {
        x: x + 1.4, y: y + 0.45, w: grid.w - 1.6, h: 0.45,
        fontFace: F.head, fontSize: 17, bold: true, color: C.navy, margin: 0,
      });
      s.addShape("rect", {
        x: x + 1.4, y: y + 0.92, w: 0.4, h: 0.04,
        fill: { color: C.coral }, line: { color: C.coral, width: 0 },
      });
      s.addText(c.d, {
        x: x + 1.4, y: y + 1.05, w: grid.w - 1.6, h: 0.85,
        fontFace: F.body, fontSize: 12, color: C.inkSoft, margin: 0,
      });
    });

    addFooter(s, 11, TOTAL);
  }

  // =========================================================================
  // SLIDE 12 — Expected Contributions
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "11  ·  CONTRIBUTIONS",
      "What this thesis adds to the literature.",
      "Five concrete deliverables, each independently publishable as an ablation.",
    );

    const five = [
      { ic: I.network, t: "C1. Pathway-masked cross-attention",
        d: "Architectural channel for biological priors — released as an open module." },
      { ic: I.layers,  t: "C2. Encoder-ablation matrix",
        d: "Six (drug × cell) combos benchmarked head-to-head on the same splits." },
      { ic: I.gauge,   t: "C3. Calibrated evidential PathXDRP",
        d: "First DRP model to report ECE, risk-coverage, and OOD AUROC." },
      { ic: I.search,  t: "C4. Quantitative XAI benchmark",
        d: "25-drug MoA dataset + 4 metrics — reusable by the community." },
      { ic: I.bolt,    t: "C5. Inference benchmark",
        d: "Throughput / memory / latency table including edge-device feasibility." },
    ];

    const lx = 0.6, ly = 2.4, rowH = 0.85, gap = 0.05;
    five.forEach((c, i) => {
      const y = ly + i * (rowH + gap);
      s.addShape("rect", {
        x: lx, y, w: 8.0, h: rowH,
        fill: { color: C.paper }, line: { color: C.hairline, width: 0.5 },
      });
      s.addShape("rect", {
        x: lx, y, w: 0.09, h: rowH,
        fill: { color: C.coral }, line: { color: C.coral, width: 0 },
      });
      s.addImage({ data: c.ic, x: lx + 0.32, y: y + 0.18, w: 0.5, h: 0.5 });
      s.addText(c.t, {
        x: lx + 1.0, y: y + 0.05, w: 6.9, h: 0.35,
        fontFace: F.head, fontSize: 14, bold: true, color: C.navy, margin: 0,
      });
      s.addText(c.d, {
        x: lx + 1.0, y: y + 0.4, w: 6.9, h: 0.45,
        fontFace: F.body, fontSize: 11.5, color: C.inkSoft, margin: 0,
      });
    });

    // Right panel — venue
    const rx = 9.0, ry = 2.4;
    s.addShape("rect", {
      x: rx, y: ry, w: 3.7, h: 4.2,
      fill: { color: C.navy }, line: { color: C.navy, width: 0 },
      shadow: makeShadow(),
    });
    s.addImage({ data: I.book, x: rx + 0.4, y: ry + 0.35, w: 0.55, h: 0.55 });
    s.addText("TARGET VENUE", {
      x: rx + 0.4, y: ry + 1.0, w: 3, h: 0.3,
      fontFace: F.body, fontSize: 10, bold: true, color: C.coralLt, charSpacing: 4, margin: 0,
    });
    s.addText("Briefings in Bioinformatics", {
      x: rx + 0.4, y: ry + 1.32, w: 3.0, h: 0.85,
      fontFace: F.head, fontSize: 17, bold: true, color: C.paper, margin: 0,
    });
    s.addText("Q1 · IF ≈ 9.5", {
      x: rx + 0.4, y: ry + 2.15, w: 3.0, h: 0.35,
      fontFace: F.body, fontSize: 13, color: C.coralLt, italic: true, margin: 0,
    });
    s.addShape("rect", {
      x: rx + 0.4, y: ry + 2.6, w: 0.4, h: 0.04,
      fill: { color: C.coral }, line: { color: C.coral, width: 0 },
    });
    s.addText("Backup: Bioinformatics (OUP) · NeurIPS ML4H workshop track for the XAI dataset.", {
      x: rx + 0.4, y: ry + 2.75, w: 3.0, h: 1.4,
      fontFace: F.body, fontSize: 12, color: C.tealLt, italic: true, margin: 0,
    });

    addFooter(s, 12, TOTAL);
  }

  // =========================================================================
  // SLIDE 13 — Roadmap
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    addSectionHead(
      s, "12  ·  ROADMAP",
      "Seven phases from skeleton to manuscript.",
      "Phases 0–2 complete · Phase 3 in motion · all later code already in place.",
    );

    const phases = [
      { n: "0", t: "Repo & environment", s: "done" },
      { n: "1", t: "Data pipeline",      s: "done" },
      { n: "2", t: "Baselines",          s: "done" },
      { n: "3", t: "PathXDRP v0",        s: "live" },
      { n: "4", t: "Encoder ablation",   s: "queued" },
      { n: "5", t: "Uncertainty + ext.", s: "queued" },
      { n: "6", t: "XAI benchmark",      s: "queued" },
      { n: "7", t: "Manuscript",         s: "queued" },
    ];

    // Horizontal phase strip
    const stripY = 3.0, stripH = 1.6;
    const inset = 0.6;
    const total = phases.length;
    const stripW = 13.3 - inset * 2;
    const cellW = stripW / total;

    // Background strip
    s.addShape("rect", {
      x: inset, y: stripY, w: stripW, h: stripH,
      fill: { color: C.paper }, line: { color: C.hairline, width: 0.5 },
      shadow: makeShadow(),
    });

    phases.forEach((p, i) => {
      const x = inset + i * cellW;
      // Vertical separator
      if (i > 0) {
        s.addShape("line", {
          x, y: stripY + 0.2, w: 0, h: stripH - 0.4,
          line: { color: C.hairline, width: 0.5 },
        });
      }
      // Status pill colour
      let pillColor = C.muted, label = "QUEUED";
      if (p.s === "done") { pillColor = C.teal; label = "DONE"; }
      if (p.s === "live") { pillColor = C.coral; label = "RUNNING"; }

      // Phase number
      s.addShape("ellipse", {
        x: x + cellW / 2 - 0.32, y: stripY + 0.18, w: 0.64, h: 0.64,
        fill: { color: pillColor }, line: { color: pillColor, width: 0 },
      });
      s.addText(p.n, {
        x: x + cellW / 2 - 0.32, y: stripY + 0.18, w: 0.64, h: 0.64,
        fontFace: F.head, fontSize: 22, bold: true, color: C.paper,
        align: "center", valign: "middle", margin: 0,
      });
      // Title
      s.addText(p.t, {
        x: x + 0.05, y: stripY + 0.95, w: cellW - 0.1, h: 0.32,
        fontFace: F.head, fontSize: 12, bold: true, color: C.navy, align: "center", margin: 0,
      });
      // Status
      s.addText(label, {
        x: x + 0.05, y: stripY + 1.28, w: cellW - 0.1, h: 0.25,
        fontFace: F.body, fontSize: 9, bold: true, color: pillColor,
        charSpacing: 3, align: "center", margin: 0,
      });
    });

    // Below: short narrative
    const narrY = 5.0;
    const lefts = [
      { t: "Right now", d: "PathXDRP gate run on random / seed 0 / fold 0. PCC tracking ≥ 0.93." },
      { t: "Next month", d: "Encoder ablation sweep (Phase 4) and full 5×5 seed-split matrix (Phase 5)." },
      { t: "Then", d: "XAI benchmark with the 25-drug MoA dataset (Phase 6) and write-up (Phase 7)." },
    ];

    const cw = (13.3 - 1.2 - 0.6) / 3;
    lefts.forEach((p, i) => {
      const x = 0.6 + i * (cw + 0.3);
      s.addShape("rect", {
        x, y: narrY, w: 0.06, h: 1.45,
        fill: { color: C.coral }, line: { color: C.coral, width: 0 },
      });
      s.addText(p.t, {
        x: x + 0.18, y: narrY, w: cw - 0.2, h: 0.4,
        fontFace: F.head, fontSize: 14, bold: true, color: C.navy, margin: 0,
      });
      s.addText(p.d, {
        x: x + 0.18, y: narrY + 0.4, w: cw - 0.2, h: 1.05,
        fontFace: F.body, fontSize: 12, color: C.inkSoft, margin: 0,
      });
    });

    addFooter(s, 13, TOTAL);
  }

  // =========================================================================
  // SLIDE 14 — Closing
  // =========================================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };

    s.addShape("rect", {
      x: 0, y: 0, w: 0.6, h: 7.5,
      fill: { color: C.coral }, line: { color: C.coral, width: 0 },
    });

    // Big idea
    s.addText("THANK YOU", {
      x: 1.2, y: 1.0, w: 11, h: 0.45,
      fontFace: F.body, fontSize: 12, bold: true, color: C.coralLt, charSpacing: 6, margin: 0,
    });

    s.addText("Predict it. Explain it. Calibrate it. Deploy it.", {
      x: 1.2, y: 1.55, w: 11.5, h: 1.35,
      fontFace: F.head, fontSize: 44, bold: true, color: C.paper, margin: 0,
    });

    s.addShape("rect", {
      x: 1.2, y: 3.05, w: 1.5, h: 0.06,
      fill: { color: C.coral }, line: { color: C.coral, width: 0 },
    });

    s.addText("PathXDRP brings four orthogonal advances into a single coherent system — and a fifth pillar (linear-time inference) that makes the whole thing portable.", {
      x: 1.2, y: 3.3, w: 11.5, h: 1.3,
      fontFace: F.head, fontSize: 18, italic: true, color: C.cream, margin: 0,
    });

    // Quick recap chips
    const chips = [
      "Pathway-masked cross-attention",
      "Foundation encoders (×4)",
      "Evidential uncertainty",
      "Quantitative XAI",
      "Linear-time inference",
    ];
    let x = 1.2;
    const chipY = 5.0, chipH = 0.5;
    // Calibri 11pt averages ~0.075"/char; 0.08 is a safe upper bound.
    chips.forEach(t => {
      const w = 0.35 + t.length * 0.075;
      s.addShape("rect", {
        x, y: chipY, w, h: chipH,
        fill: { color: C.navyAlt }, line: { color: C.teal, width: 0.75 },
      });
      s.addText(t, {
        x, y: chipY, w, h: chipH,
        fontFace: F.body, fontSize: 11, color: C.tealLt,
        align: "center", valign: "middle", margin: 0,
      });
      x += w + 0.15;
    });

    // Contact
    s.addText("Aly Hesham   ·   aly.hesham22197@gmail.com", {
      x: 1.2, y: 6.4, w: 8, h: 0.4,
      fontFace: F.body, fontSize: 14, color: C.cream, margin: 0,
    });
    s.addText("Master's Thesis Proposal · 2026", {
      x: 1.2, y: 6.78, w: 8, h: 0.3,
      fontFace: F.body, fontSize: 11, italic: true, color: C.muted, margin: 0,
    });
  }

  // =========================================================================
  // Write file
  // =========================================================================
  await pres.writeFile({ fileName: "PathXDRP_Proposal.pptx" });
  console.log("PathXDRP_Proposal.pptx generated.");
})();
