// Shared visualization helpers for the SF commute maps (the live server and the static
// explorer both inline this), so the time->color ramp, the Google Maps deep link, and
// the transit-mode palette can never drift between the two UIs.

const rgb = c => `rgb(${c[0]},${c[1]},${c[2]})`;

// Time gradient anchored at a configurable "sweet spot" (green->pale-yellow pivot at
// `ideal`), warming to red by ideal+25 — independent of the max-commute filter.
function ramp(ideal){
  const hi = ideal + 25;
  return {hi, S:[[0,[0,104,55]],[ideal*0.45,[26,152,80]],[ideal*0.72,[102,189,99]],
    [ideal*0.9,[166,217,106]],[ideal,[255,255,191]],
    [ideal+(hi-ideal)*0.35,[253,174,97]],[ideal+(hi-ideal)*0.7,[244,109,67]],[hi,[215,48,39]]]};
}

// Interpolate a value to a css color given a precomputed ramp scale S (= ramp(ideal).S).
// Precomputing S once per redraw and passing it in keeps recoloring O(cells) per frame
// instead of rebuilding the 8-stop scale for every cell.
function colorScale(v, S){
  if(v==null) return null;
  if(v<=0) return rgb(S[0][1]);
  for(let i=1;i<S.length;i++){ if(v<=S[i][0]){
    const a=S[i-1], b=S[i], t=(v-a[0])/((b[0]-a[0])||1);
    return rgb(a[1].map((c,j)=>Math.round(c+(b[1][j]-c)*t))); }}
  return rgb(S[S.length-1][1]);
}

// Google Maps transit deep link, departing the upcoming Monday ~8:30am (matches the model).
// Uses Google's UNOFFICIAL data= format because the documented ?api=1 form can't set a
// departure time: !7e2=depart-at, !8j<epoch>=time, !3e3=transit. Google reads the epoch's
// UTC wall-clock as local, so build it with Date.UTC (no timezone conversion) — verified to
// render "Depart 8:30 AM" with morning trips. (Brittle: relies on an undocumented format.)
function _nextMon830(){
  const d = new Date();
  d.setDate(d.getDate() + (((1 - d.getDay() + 7) % 7) || 7));   // upcoming Monday (not today)
  return Math.floor(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate(), 8, 30, 0) / 1000);
}
function gmapsURL(olat,olon,dlat,dlon){
  const t = _nextMon830();
  return `https://www.google.com/maps/dir/${olat},${olon}/${dlat},${dlon}/@${dlat},${dlon},14z/data=!4m6!4m5!2m3!6e0!7e2!8j${t}!3e3`;
}

// Transit-mode colors — used by both the overlay lines and the breakdown chips.
const MODECOLOR = {bart:"#6f8cff", metro:"#ff6b6b", bus:"#f6a04d", cable:"#5cd65c"};
