/**
 * cricgnaan UI — calls FastAPI /api/predict (real LightGBM + meta + collapse models).
 */
const CITIES = {
  "Mumbai Indians": "Mumbai",
  "Chennai Super Kings": "Chennai",
  "Royal Challengers Bangalore": "Bengaluru",
  "Kolkata Knight Riders": "Kolkata",
  "Rajasthan Royals": "Jaipur",
  "Delhi Capitals": "Delhi",
  "Sunrisers Hyderabad": "Hyderabad",
  "Punjab Kings": "Mohali",
  "Gujarat Titans": "Ahmedabad",
  "Lucknow Super Giants": "Lucknow",
};

let tossW = "t1",
  dec = "bat",
  playoff = false;
let debounce = null;

function sr(k) {
  const m = {
    over: "sl-over",
    s1: "sl-s1",
    s2: "sl-s2",
    wk: "sl-wk",
    bsr: "sl-bsr",
    part: "sl-part",
    eco: "sl-eco",
  };
  const rv = {
    over: "rv-over",
    s1: "rv-s1",
    s2: "rv-s2",
    wk: "rv-wk",
    bsr: "rv-bsr",
    part: "rv-part",
    eco: "rv-eco",
  };
  if (!m[k]) return;
  document.getElementById(rv[k]).textContent = document.getElementById(m[k]).value;
}
function setToss(t) {
  tossW = t;
  document.getElementById("pt1").className = "pill" + (t === "t1" ? " on" : "");
  document.getElementById("pt2").className = "pill" + (t === "t2" ? " on" : "");
  scheduleUpdate();
}
function setDec(d) {
  dec = d;
  document.getElementById("dbat").className = "pill" + (d === "bat" ? " on" : "");
  document.getElementById("dfield").className = "pill" + (d === "field" ? " on" : "");
  scheduleUpdate();
}
function togglePlayoff() {
  playoff = !playoff;
  document.getElementById("ppill").className = "pill" + (playoff ? " on" : "");
  scheduleUpdate();
}

function scheduleUpdate() {
  clearTimeout(debounce);
  debounce = setTimeout(doPredict, 100);
}

async function doPredict() {
  const t1 = document.getElementById("t1").value;
  const t2 = document.getElementById("t2").value;
  const venue = document.getElementById("venue").value;
  const over = +document.getElementById("sl-over").value;
  const s1 = +document.getElementById("sl-s1").value;
  const s2 = +document.getElementById("sl-s2").value;
  const wk = +document.getElementById("sl-wk").value;
  const batterSr = +document.getElementById("sl-bsr").value;
  const part = +document.getElementById("sl-part").value;
  const eco = +document.getElementById("sl-eco").value / 10;

  const tossTeam = tossW === "t1" ? t1 : t2;
  const other = tossW === "t1" ? t2 : t1;
  const batTeam = dec === "bat" ? tossTeam : other;
  const fieldTeam = dec === "bat" ? other : tossTeam;

  document.getElementById("pt1").textContent = t1.split(" ").pop();
  document.getElementById("pt2").textContent = t2.split(" ").pop();
  document.getElementById("inf-bat").textContent = batTeam;
  document.getElementById("inf-field").textContent = fieldTeam;

  const live = over > 0 && s1 > 0;
  ["rg-s2", "rg-wk", "rg-extra", "rg-extra2", "rg-extra3"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.className = live ? "rg" : "rg disabled-range";
  });

  const payload = {
    team_a: batTeam,
    team_b: fieldTeam,
    venue,
    toss_winner: tossTeam,
    toss_decision: dec,
    playoff,
    over,
    score_1st: s1,
    score_2nd: s2,
    wkts_2nd: wk,
    batter_sr: batterSr,
    partnership: part,
    bowler_eco: eco,
  };

  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const raw = await r.text();
    if (!r.ok) {
      document.getElementById("api-err").style.display = "block";
      document.getElementById("api-err").textContent = raw || "API error";
      return;
    }
    document.getElementById("api-err").style.display = "none";
    applyResponse(JSON.parse(raw), batTeam, fieldTeam, dec);
  } catch (e) {
    document.getElementById("api-err").style.display = "block";
    document.getElementById("api-err").textContent = String(e);
  }
}

function applyResponse(d, batTeam, fieldTeam, tossDec) {
  const pBat = d.win_a;
  const pField = d.win_b;
  const pBatPct = Math.round(pBat);
  const pFPct = Math.round(pField);

  document.getElementById("name-l").textContent = batTeam.toUpperCase();
  document.getElementById("name-r").textContent = fieldTeam.toUpperCase();
  document.getElementById("city-l").textContent = CITIES[batTeam] || "";
  document.getElementById("city-r").textContent = CITIES[fieldTeam] || "";
  document.getElementById("pct-l").textContent = pBatPct + "%";
  document.getElementById("pct-r").textContent = pFPct + "%";
  document.getElementById("vsbar").style.height = pBatPct + "%";
  document.getElementById("conf-txt").textContent = d.confidence_pct + "% confidence";

  document.getElementById("wba").style.width = pBatPct + "%";
  document.getElementById("wbb").style.width = pFPct + "%";

  const fav = d.favourite;
  const chip = document.getElementById("fav-chip");
  chip.textContent = "Favourite: " + fav;
  chip.style.cssText =
    "background:rgba(245,200,66,.1);border:1px solid rgba(245,200,66,.25);color:var(--gold)";

  document.getElementById("toss-tag").textContent = d.toss_tag_line || "";

  const te = d.toss_edge_team_a;
  document.getElementById("ic-toss").textContent = (te >= 0 ? "+" : "") + te.toFixed(2);
  document.getElementById("ic-toss").className =
    "ic-val " + (te > 0.03 ? "ct" : te < -0.03 ? "cr" : "cg");
  document.getElementById("ic-tsub").textContent =
    (tossDec === "bat" ? "Bat" : "Field") + " first · " + (d.venue_tag || "").toLowerCase() + " · modern";

  document.getElementById("ic-h2h").textContent = "—";
  document.getElementById("ic-hsub").textContent = "H2H not in ensemble; see ELO card";

  const elo = d.elo_delta;
  document.getElementById("ic-elo").textContent = (elo >= 0 ? "+" : "") + Math.round(elo);
  document.getElementById("ic-elo").className =
    "ic-val " + (elo > 20 ? "cb" : elo < -20 ? "cr" : "cm");
  document.getElementById("ic-esub").textContent =
    (elo >= 0 ? batTeam : fieldTeam).split(" ").pop() + " rated higher (ELO)";

  if (d.live && d.rrr_gap != null) {
    const g = d.rrr_gap;
    document.getElementById("ic-rrr").textContent = (g >= 0 ? "+" : "") + g.toFixed(1);
    document.getElementById("ic-rrr").className =
      "ic-val " + (g > 2 ? "cr" : g < -1 ? "ct" : "cg");
    document.getElementById("ic-rsub").textContent =
      g > 1 ? "Chase under pressure" : g < -1 ? "Chase comfortable" : "On track";
  } else {
    document.getElementById("ic-rrr").textContent = "—";
    document.getElementById("ic-rrr").className = "ic-val cm";
    document.getElementById("ic-rsub").textContent = "Activate live mode above";
  }

  document.getElementById("vtag").textContent = d.venue_tag || "";

  const eras = d.era_strip || [];
  let html = "";
  eras.forEach((cell, i) => {
    const e = cell.edge_pct / 100;
    const col = e > 0.04 ? "var(--teal)" : e < -0.04 ? "var(--red)" : "var(--muted2)";
    html += `<div class="era-cell${i === 2 ? " now" : ""}"><div class="ec-lbl">${cell.label}</div><div class="ec-edge" style="color:${col}">${cell.edge_pct >= 0 ? "+" : ""}${cell.edge_pct.toFixed(1)}%</div><div class="ec-sub">${cell.win_rate_pct}% win</div></div>`;
    if (i < eras.length - 1) html += `<div class="era-arr">›</div>`;
  });
  document.getElementById("era-strip").innerHTML = html || "";

  const e2 = eras.length > 2 ? eras[2] : null;
  const modernEdge = e2 ? e2.edge_pct / 100 : 0;
  const vc =
    Math.abs(modernEdge) > 0.04 ? "rgba(45,219,168,.08)" : "rgba(107,104,130,.1)";
  const vbc =
    Math.abs(modernEdge) > 0.04 ? "rgba(45,219,168,.2)" : "rgba(107,104,130,.2)";
  const vtc = Math.abs(modernEdge) > 0.04 ? "var(--teal)" : "var(--muted2)";
  const dn = document.getElementById("decay-note");
  dn.style.cssText = `background:${vc};border:1px solid ${vbc};color:${vtc}`;
  dn.textContent = d.decay_note || "";

  const pr = d.pressure;
  if (pr && pr.active) {
    document.getElementById("hint-area").style.display = "none";
    document.getElementById("pi-area").style.display = "block";
    const pi = pr.pressure_index || 0;
    const col =
      pi > 0.7 ? "var(--red)" : pi > 0.5 ? "var(--orange)" : pi > 0.3 ? "var(--gold)" : "var(--teal)";
    document.getElementById("pi-fill").style.width = pi * 100 + "%";
    document.getElementById("pi-fill").style.background = col;
    document.getElementById("pi-num").textContent = pi.toFixed(2);
    document.getElementById("pi-num").style.color = col;
    const tier = pr.tier || "LOW";
    const rt = document.getElementById("risk-tag");
    rt.textContent = tier;
    rt.style.color = col;
    rt.style.borderColor = col;
    rt.style.background = col + "18";

    const b = pr.bars || {};
    document.getElementById("sb1").style.width = (b.rrr || 0) + "%";
    document.getElementById("sp1").textContent = (b.rrr || 0) + "%";
    document.getElementById("sb2").style.width = (b.wkt || 0) + "%";
    document.getElementById("sp2").textContent = (b.wkt || 0) + "%";
    document.getElementById("sb3").style.width = (b.phase || 0) + "%";
    document.getElementById("sp3").textContent = (b.phase || 0) + "%";
    document.getElementById("sb4").style.width = (b.collapse || 0) + "%";
    document.getElementById("sp4").textContent = (b.collapse || 0) + "%";
  } else {
    document.getElementById("hint-area").style.display = "block";
    document.getElementById("pi-area").style.display = "none";
    const rt = document.getElementById("risk-tag");
    rt.textContent = "INACTIVE";
    rt.style.color = "var(--muted)";
    rt.style.borderColor = "var(--border)";
    rt.style.background = "transparent";
  }
}

function init() {
  ["over", "s1", "s2", "wk", "bsr", "part", "eco"].forEach(sr);
  doPredict();
}

document.addEventListener("DOMContentLoaded", init);
