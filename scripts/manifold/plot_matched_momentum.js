#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error("usage: plot_matched_momentum.js curves.json output-stem");
const data = JSON.parse(fs.readFileSync(input, "utf8"));
const names = ["MCSD", "MCSD-TP", "MuonH", "NorMuon-Sphere"];
const colors = ["#315DA8", "#8860A8", "#C77518", "#008A78"];
const methods = names.map((name, index) => {
  const row = data.runs.find(r => r.method === name);
  if (!row || row.val.length !== 58 || row.val.at(-1).step !== 11445 ||
      row.val.some(p => !Number.isFinite(p.loss))) throw new Error(`Incomplete curve: ${name}`);
  return {...row, color: colors[index], dashed: index === 1};
});
const out = [
  '<svg xmlns="http://www.w3.org/2000/svg" width="1360" height="870" viewBox="0 0 1360 870">',
  '<rect width="1360" height="870" fill="white"/>',
  '<g font-family="Arial, Helvetica, sans-serif" letter-spacing="0">',
];
function text(x, y, value, size = 16, fill = "#252B30", extra = "") {
  return `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" ${extra}>${value}</text>`;
}
out.push(text(100, 50, "Matched momentum: four optimizers on the Frobenius sphere", 28, "#22282D", 'font-weight="600"'));
out.push(text(100, 81, "Width 256 | 3B tokens | beta = 0.95, EMA | learning rate = 0.010 | seed 1234", 17, "#606A73"));
for (const [i, m] of methods.entries()) {
  const x = 120 + i * 305;
  out.push(`<line x1="${x}" y1="119" x2="${x+37}" y2="119" stroke="${m.color}" stroke-width="3.4" ${m.dashed ? 'stroke-dasharray="7 4"' : ''}/>`);
  out.push(text(x+47, 125, m.method, 18));
}
function panel(id, x, y, w, h, xmin, xmax, ymin, ymax, xticks, yticks, title) {
  const sx = v => x + (v - xmin) / (xmax - xmin) * w;
  const sy = v => y + h - (v - ymin) / (ymax - ymin) * h;
  out.push(text(x, y - 19, title, 20, "#252B30", 'font-weight="600"'));
  out.push(`<clipPath id="${id}"><rect x="${x}" y="${y}" width="${w}" height="${h}"/></clipPath>`);
  for (const v of xticks) {
    out.push(`<line x1="${sx(v)}" y1="${y}" x2="${sx(v)}" y2="${y+h}" stroke="#DFE3E6" stroke-dasharray="3 5"/>`);
    out.push(text(sx(v), y+h+28, v.toFixed(1), 16, "#5A636B", 'text-anchor="middle"'));
  }
  for (const v of yticks) {
    out.push(`<line x1="${x}" y1="${sy(v)}" x2="${x+w}" y2="${sy(v)}" stroke="#DFE3E6" stroke-dasharray="3 5"/>`);
    out.push(text(x-13, sy(v)+5, v.toFixed(2), 16, "#5A636B", 'text-anchor="end"'));
  }
  out.push(`<path d="M ${x} ${y} V ${y+h} H ${x+w}" fill="none" stroke="#50575D" stroke-width="1.5"/>`);
  out.push(`<g clip-path="url(#${id})">`);
  for (const m of methods) {
    const points = m.val.map(p => ({x: p.step * data.protocol.tokens_per_step / 1e9, y: p.loss}));
    const line = points.map((p, i) => `${i ? "L" : "M"} ${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`).join(" ");
    out.push(`<path d="${line}" fill="none" stroke="${m.color}" stroke-width="2.8" ${m.dashed ? 'stroke-dasharray="7 4"' : ''}/>`);
    const last = points.at(-1);
    out.push(`<circle cx="${sx(last.x)}" cy="${sy(last.y)}" r="4" fill="${m.color}" stroke="white" stroke-width="1"/>`);
  }
  out.push('</g>');
  out.push(text(x-62, y+h/2, "Validation loss", 17, "#252B30", `text-anchor="middle" transform="rotate(-90 ${x-62} ${y+h/2})"`));
}
panel("full", 100, 185, 1160, 235, 0, 3.04, 3.42, 6.15,
  [0, .5, 1, 1.5, 2, 2.5, 3], [3.5, 4, 4.5, 5, 5.5, 6], "Full trajectory");
panel("late", 100, 505, 1160, 235, 2.3, 3.04, 3.45, 3.56,
  [2.3, 2.4, 2.6, 2.8, 3], [3.46, 3.48, 3.50, 3.52, 3.54, 3.56], "Late-stage detail");
out.push(text(680, 804, "Processed training tokens (billions)", 19, "#252B30", 'text-anchor="middle"'));
out.push(text(100, 846, "Raw validation points; no smoothing. Same LR, momentum, seed, data order and evaluation schedule. Single-seed evidence.", 15, "#606A73"));
out.push('</g></svg>');
const svg = out.join("\n") + "\n";
fs.mkdirSync(path.dirname(output), {recursive: true});
fs.writeFileSync(`${output}.svg`, svg);
const sharp = require(process.env.SHARP_MODULE || "sharp");
sharp(Buffer.from(svg)).png().toFile(`${output}.png`).then(() => console.log(`Wrote ${output}.svg/.png`));
