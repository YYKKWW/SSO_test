"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");
const folder = path.resolve(__dirname, "../../docs/experiments/frobenius_3b_6b");
const snapshot = JSON.parse(fs.readFileSync(path.join(folder, "latest_batch_20261029.json"), "utf8"));
const series = [
  ["MCSD", "4329085", "#16877D", ""],
  ["MCSD-TP", "4329086", "#7350A4", "6 3"],
  ["MuonH", "4346665", "#357AB4", ""],
  ["NorMuon-Sphere", "4346666", "#D08117", ""],
].map(([name, job, color, dash]) => {
  const run = snapshot.runs.find(r => r.job_id === job);
  assert(run && run.state === "COMPLETED" && run.exit_code === "0:0");
  assert(run.seed === 1234 && run.beta === .95 && run.nesterov && run.lr === .01);
  assert(run.val.length === 58 && run.val.at(-1).step === 11445);
  assert(run.final_val === run.val.at(-1).loss && run.val.at(-1).final);
  assert(run.val.every(v => Number.isFinite(v.loss) && Number.isFinite(v.tokens)));
  return {name, color, dash, run};
});
for (const {run} of series) {
  assert.deepStrictEqual(run.val.map(v => v.step), series[0].run.val.map(v => v.step));
}

const out = [
  '<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="610" viewBox="0 0 1320 610">',
  '<rect width="1320" height="610" fill="white"/>',
  '<g font-family="Arial, Helvetica, sans-serif" letter-spacing="0">',
];
function text(x, y, value, size=16, extra="") {
  return `<text x="${x}" y="${y}" font-size="${size}" fill="#30363B" ${extra}>${value}</text>`;
}
out.push(text(88, 36, "Frobenius sphere: matched Nesterov comparison", 25, 'font-weight="600"'));
out.push(text(88, 65, "Width 256 | 3B tokens | LR 0.010 | beta 0.95 | Nesterov | seed 1234", 16));
series.forEach(({name, color, dash, run}, i) => {
  const x = 90 + (i % 2) * 600, y = 101 + Math.floor(i / 2) * 29;
  out.push(`<line x1="${x}" x2="${x+34}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="2.6" stroke-dasharray="${dash}"/>`);
  out.push(text(x+45, y+5, `${name} | final ${run.final_val.toFixed(6)}`, 17));
});

const panels = [
  {x:88, xmin:0, xmax:3.05, ymin:3.4, ymax:6.1, xticks:[0,1,2,3], yticks:[3.5,4,4.5,5,5.5,6], title:"Full validation trajectory"},
  {x:748, xmin:2.5, xmax:3.015, ymin:3.44, ymax:3.51, xticks:[2.5,2.6,2.7,2.8,2.9,3], yticks:[3.44,3.46,3.48,3.50], title:"Late-stage detail: last 0.5B tokens"},
];
for (const [index, p] of panels.entries()) {
  const y=199, w=510, h=285;
  const sx=v=>p.x+(v-p.xmin)/(p.xmax-p.xmin)*w;
  const sy=v=>y+h-(v-p.ymin)/(p.ymax-p.ymin)*h;
  out.push(text(p.x, y-21, p.title, 20, 'font-weight="600"'));
  for(const v of p.xticks) {
    out.push(`<line x1="${sx(v)}" x2="${sx(v)}" y1="${y}" y2="${y+h}" stroke="#DFE3E6" stroke-dasharray="3 5"/>`);
    out.push(text(sx(v), y+h+26, v.toFixed(1), 15, 'text-anchor="middle"'));
  }
  for(const v of p.yticks) {
    out.push(`<line x1="${p.x}" x2="${p.x+w}" y1="${sy(v)}" y2="${sy(v)}" stroke="#DFE3E6" stroke-dasharray="3 5"/>`);
    out.push(text(p.x-12, sy(v)+5, v.toFixed(index ? 2 : 1), 15, 'text-anchor="end"'));
  }
  out.push(`<path d="M ${p.x} ${y} V ${y+h} H ${p.x+w}" fill="none" stroke="#50575D" stroke-width="1.5"/>`);
  out.push(`<clipPath id="p${index}"><rect x="${p.x}" y="${y}" width="${w}" height="${h}"/></clipPath><g clip-path="url(#p${index})">`);
  for(const {run, color, dash} of series) {
    const coords=run.val.map(v=>[sx(v.tokens/1e9),sy(v.loss)]);
    out.push(`<path d="${coords.map(([a,b],i)=>`${i?'L':'M'} ${a.toFixed(3)} ${b.toFixed(3)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.1" stroke-dasharray="${dash}"/>`);
    if(index) for(const [a,b] of coords) out.push(`<circle cx="${a}" cy="${b}" r="2.4" fill="${color}"/>`);
  }
  out.push('</g>');
  out.push(text(p.x+w/2, y+h+57, "Processed training tokens (billions)", 16, 'text-anchor="middle"'));
  out.push(text(p.x-60, y+h/2, "Validation loss", 16, `text-anchor="middle" transform="rotate(-90 ${p.x-60} ${y+h/2})"`));
}
out.push(text(88, 578, "58 raw validation points per run; no smoothing. Reported endpoints are at step 11,445, not curve minima.", 14));
out.push(text(88, 598, "The validation iterator advances between evaluations. Compare matched steps within the same seed.", 14));
out.push('</g></svg>');
const svg=out.join('\n')+'\n';
const stem=path.join(folder,'nesterov095_matched_20261029');
fs.writeFileSync(stem+'.svg',svg);
const sharp=require(process.env.SHARP_MODULE || 'sharp');
sharp(Buffer.from(svg)).png().toFile(stem+'.png')
  .then(()=>console.log(`Wrote ${stem}.svg/.png`))
  .catch(error=>{console.error(error);process.exitCode=1;});
