"use strict";

const fs = require("fs");
const path = require("path");
const folder = path.resolve(__dirname, "../../docs/experiments/frobenius_3b_6b");
const snapshot = JSON.parse(fs.readFileSync(path.join(folder, "latest_batch_20261026.json"), "utf8"));
const controls = JSON.parse(fs.readFileSync(path.join(folder, "matched_ema095_curves.json"), "utf8"));
const out = [
  '<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="550" viewBox="0 0 1260 550">',
  '<rect width="1260" height="550" fill="white"/>',
  '<g font-family="Arial, Helvetica, sans-serif" letter-spacing="0">',
];
function text(x, y, value, size=16, extra="") {
  return `<text x="${x}" y="${y}" font-size="${size}" fill="#30363B" ${extra}>${value}</text>`;
}
out.push(text(72, 40, "Frobenius sphere: the new 0.008 learning-rate test", 26, 'font-weight="600"'));
out.push(text(72, 70, "Width 256 | 3B tokens | beta 0.95 EMA | seed 1234 | same initialization and data order", 16));
for (const [index, name, job] of [[0,"MCSD","4329081"],[1,"NorMuon-Sphere","4329082"]]) {
  const x=82+index*610, y=195, width=490, height=255;
  const sx=v=>x+(v-1.5)/(3.04-1.5)*width;
  const sy=v=>y+height-(v-3.455)/(3.69-3.455)*height;
  out.push(text(x,y-90,name,21,'font-weight="600"'));
  const rows=[
    [snapshot.runs.find(r=>r.job_id===job), "0.008", "#8656A3", true],
    [controls.runs.find(r=>r.method===name), "0.010", "#188B83", false],
  ];
  rows.forEach(([r,lr,color,dashed],i)=>{
    if (!r || r.val.length!==58 || r.val.at(-1).step!==11445 || r.final_val!==r.val.at(-1).loss) {
      throw Error(`Incomplete validation curve: ${name}, LR ${lr}`);
    }
    const ly=y-61+i*27;
    out.push(`<line x1="${x}" y1="${ly}" x2="${x+35}" y2="${ly}" stroke="${color}" stroke-width="3" ${dashed?'stroke-dasharray="7 4"':''}/>`);
    out.push(text(x+46,ly+5,`LR ${lr} | final ${r.val.at(-1).loss.toFixed(6)}`,16));
  });
  for (const v of [1.5,2,2.5,3]) {
    out.push(`<line x1="${sx(v)}" x2="${sx(v)}" y1="${y}" y2="${y+height}" stroke="#DFE3E6" stroke-dasharray="3 5"/>`);
    out.push(text(sx(v),y+height+27,v.toFixed(1),15,'text-anchor="middle"'));
  }
  for (const v of [3.46,3.5,3.54,3.58,3.62,3.66]) {
    out.push(`<line x1="${x}" x2="${x+width}" y1="${sy(v)}" y2="${sy(v)}" stroke="#DFE3E6" stroke-dasharray="3 5"/>`);
    out.push(text(x-12,sy(v)+5,v.toFixed(2),15,'text-anchor="end"'));
  }
  out.push(`<path d="M ${x} ${y} V ${y+height} H ${x+width}" fill="none" stroke="#50575D" stroke-width="1.5"/>`);
  out.push(`<clipPath id="panel${index}"><rect x="${x}" y="${y}" width="${width}" height="${height}"/></clipPath><g clip-path="url(#panel${index})">`);
  for (const [r,lr,color,dashed] of rows) {
    const points=r.val.map(v=>[sx(v.step*262144/1e9),sy(v.loss)]);
    out.push(`<path d="${points.map(([a,b],i)=>`${i?'L':'M'} ${a.toFixed(2)} ${b.toFixed(2)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.6" ${dashed?'stroke-dasharray="7 4"':''}/>`);
    const last=points.at(-1);
    out.push(`<circle cx="${last[0]}" cy="${last[1]}" r="4" fill="${color}"/>`);
  }
  out.push('</g>');
  out.push(text(x+width/2,y+height+53,"Processed training tokens (billions)",16,'text-anchor="middle"'));
}
out.push(text(21,320,"Validation loss",16,'text-anchor="middle" transform="rotate(-90 21 320)"'));
out.push(text(72,535,"Late-stage raw validation points; no smoothing. Final losses are measured at step 11,445. Single-seed evidence.",14));
out.push('</g></svg>');
const svg=out.join('\n')+'\n';
const stem=path.join(folder,'refinement_ema095_20261026');
fs.writeFileSync(stem+'.svg',svg);
const sharp=require(process.env.SHARP_MODULE || 'sharp');
sharp(Buffer.from(svg)).png().toFile(stem+'.png').then(()=>console.log(`Wrote ${stem}.svg/.png`));
