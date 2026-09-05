/* Северный Звук: Трасса. Анализатор трека.
   Вход: моно Float32Array + sampleRate. Выход: биты, бочки, снейры, хэты, огибающая, секции.
   Чистый JS, без библиотек. Работает и в Worker, и в главном потоке (см. низ файла). */
(function (root) {
  'use strict';

  var FFT_N = 1024, HOP = 256, TARGET_SR = 22050, FRAME_LAG = 0.03; // пик flux опережает удар (замер на синтетике)

  /* ---------- FFT radix-2 (in-place, real вход через re/im) ---------- */
  function makeFFT(n) {
    var levels = Math.round(Math.log(n) / Math.LN2);
    var cosT = new Float32Array(n / 2), sinT = new Float32Array(n / 2);
    for (var i = 0; i < n / 2; i++) { cosT[i] = Math.cos(2 * Math.PI * i / n); sinT[i] = Math.sin(2 * Math.PI * i / n); }
    var rev = new Uint32Array(n);
    for (i = 0; i < n; i++) { var x = i, y = 0; for (var j = 0; j < levels; j++) { y = (y << 1) | (x & 1); x >>>= 1; } rev[i] = y; }
    return function (re, im) {
      for (var i = 0; i < n; i++) { var j = rev[i]; if (j > i) { var t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
      for (var size = 2; size <= n; size *= 2) {
        var half = size / 2, step = n / size;
        for (var s = 0; s < n; s += size) {
          for (var k = s, t2 = 0; k < s + half; k++, t2 += step) {
            var l = k + half, tre = re[l] * cosT[t2] + im[l] * sinT[t2], tim = -re[l] * sinT[t2] + im[l] * cosT[t2];
            re[l] = re[k] - tre; im[l] = im[k] - tim; re[k] += tre; im[k] += tim;
          }
        }
      }
    };
  }

  /* ---------- утилиты ---------- */
  function percentile(arr, p) {
    var a = Float32Array.from(arr).sort(); if (!a.length) return 1;
    return a[Math.min(a.length - 1, Math.floor(a.length * p))] || 1;
  }
  function movingMean(src, radius) {
    var n = src.length, out = new Float32Array(n), sum = 0, cnt = 0;
    var pre = new Float64Array(n + 1);
    for (var i = 0; i < n; i++) pre[i + 1] = pre[i] + src[i];
    for (i = 0; i < n; i++) { var a = Math.max(0, i - radius), b = Math.min(n, i + radius + 1); out[i] = (pre[b] - pre[a]) / (b - a); }
    return out;
  }
  function pickPeaks(flux, frameSec, opts) {
    // адаптивный порог: скользящее среднее ± win * mult + delta, локальный максимум ± 3 кадра, мин. дистанция
    var radius = Math.round(opts.win / frameSec), mean = movingMean(flux, radius);
    var minDist = Math.round(opts.minDist / frameSec), peaks = [], last = -1e9;
    for (var i = 3; i < flux.length - 3; i++) {
      var v = flux[i], thr = mean[i] * opts.mult + opts.delta;
      if (v < thr) continue;
      var isMax = true;
      for (var k = -3; k <= 3; k++) if (k && flux[i + k] > v) { isMax = false; break; }
      if (!isMax) continue;
      if (i - last < minDist) { if (v > flux[last]) { peaks[peaks.length - 1] = i; last = i; } continue; }
      peaks.push(i); last = i;
    }
    return peaks.map(function (i) { return { t: i * frameSec + FRAME_LAG, s: Math.min(1, flux[i] / (mean[i] * opts.mult + opts.delta + 1e-6) / 2.5), i: i }; });
  }

  /* ---------- даунсэмпл ---------- */
  function downsample(pcm, sr) {
    var f = Math.max(1, Math.round(sr / TARGET_SR));
    if (f === 1) return { pcm: pcm, sr: sr };
    var n = Math.floor(pcm.length / f), out = new Float32Array(n);
    for (var i = 0, j = 0; i < n; i++, j += f) { var s = 0; for (var k = 0; k < f; k++) s += pcm[j + k]; out[i] = s / f; }
    return { pcm: out, sr: sr / f };
  }

  /* ---------- STFT + spectral flux по 3 полосам (отдельная функция: горячий цикл без замыканий) ---------- */
  function computeFlux(pcm, sr, nFrames, progress) {
    var fft = makeFFT(FFT_N), win = new Float32Array(FFT_N), i, f;
    for (i = 0; i < FFT_N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / FFT_N);
    var binHz = sr / FFT_N, half = FFT_N / 2;
    var bands = [[30, 150], [150, 2500], [2500, Math.min(10000, sr / 2 - 1)]];
    var b0 = new Int32Array(3), b1 = new Int32Array(3);
    for (i = 0; i < 3; i++) { b0[i] = Math.max(1, Math.floor(bands[i][0] / binHz)); b1[i] = Math.min(half - 1, Math.ceil(bands[i][1] / binHz)); }
    var fluxL = new Float32Array(nFrames), fluxM = new Float32Array(nFrames), fluxH = new Float32Array(nFrames), fluxAll = new Float32Array(nFrames);
    var re = new Float32Array(FFT_N), im = new Float32Array(FFT_N), prev = new Float32Array(half), cur = new Float32Array(half), magB = new Float32Array(half);
    // два вещественных кадра в одном комплексном FFT: кадр A в re, кадр B в im, потом разделяем по симметрии
    for (f = 0; f < nFrames; f += 2) {
      var offA = f * HOP, offB = (f + 1) * HOP, hasB = f + 1 < nFrames;
      if (hasB) for (i = 0; i < FFT_N; i++) { re[i] = pcm[offA + i] * win[i]; im[i] = pcm[offB + i] * win[i]; }
      else for (i = 0; i < FFT_N; i++) { re[i] = pcm[offA + i] * win[i]; im[i] = 0; }
      fft(re, im);
      // X_A[k] = (Z[k] + conj(Z[N-k]))/2 ; X_B[k] = (Z[k] - conj(Z[N-k]))/(2i)
      cur[0] = Math.log(1 + 20 * Math.abs(re[0])); magB[0] = Math.log(1 + 20 * Math.abs(im[0]));
      for (i = 1; i < half; i++) {
        var zr = re[i], zi = im[i], wr = re[FFT_N - i], wi = im[FFT_N - i];
        var ar = 0.5 * (zr + wr), ai = 0.5 * (zi - wi);       // спектр A
        var br = 0.5 * (zi + wi), bi = -0.5 * (zr - wr);      // спектр B
        cur[i] = Math.log(1 + 20 * Math.sqrt(ar * ar + ai * ai));
        magB[i] = Math.log(1 + 20 * Math.sqrt(br * br + bi * bi));
      }
      var sL = 0, sM = 0, sH = 0, d;
      for (i = b0[0]; i <= b1[0]; i++) { d = cur[i] - prev[i]; if (d > 0) sL += d; }
      for (i = b0[1]; i <= b1[1]; i++) { d = cur[i] - prev[i]; if (d > 0) sM += d; }
      for (i = b0[2]; i <= b1[2]; i++) { d = cur[i] - prev[i]; if (d > 0) sH += d; }
      fluxL[f] = sL; fluxM[f] = sM; fluxH[f] = sH; fluxAll[f] = sL + sM + sH;
      if (hasB) {
        sL = 0; sM = 0; sH = 0;
        for (i = b0[0]; i <= b1[0]; i++) { d = magB[i] - cur[i]; if (d > 0) sL += d; }
        for (i = b0[1]; i <= b1[1]; i++) { d = magB[i] - cur[i]; if (d > 0) sM += d; }
        for (i = b0[2]; i <= b1[2]; i++) { d = magB[i] - cur[i]; if (d > 0) sH += d; }
        fluxL[f + 1] = sL; fluxM[f + 1] = sM; fluxH[f + 1] = sH; fluxAll[f + 1] = sL + sM + sH;
        var tmp = prev; prev = magB; magB = cur; cur = tmp;   // prev := B, буферы по кругу
      } else { var tmp2 = prev; prev = cur; cur = tmp2; }
      if ((f & 1023) === 0) progress(0.1 + 0.6 * f / nFrames);
    }
    return [fluxL, fluxM, fluxH, fluxAll];
  }

  /* ---------- главная функция ---------- */
  function analyze(pcmIn, srIn, progress) {
    progress = progress || function () {};
    var ds = downsample(pcmIn, srIn), pcm = ds.pcm, sr = ds.sr;
    var frameSec = HOP / sr, nFrames = Math.max(1, Math.floor((pcm.length - FFT_N) / HOP));
    var i, fl = computeFlux(pcm, sr, nFrames, progress), fluxL = fl[0], fluxM = fl[1], fluxH = fl[2], fluxAll = fl[3];
    // нормировка на 95-й перцентиль полосы
    [fluxL, fluxM, fluxH, fluxAll].forEach(function (a) { var p = percentile(a, 0.95); for (var i = 0; i < a.length; i++) a[i] /= p; });

    /* онсеты по полосам */
    var kicksRaw = pickPeaks(fluxL, frameSec, { win: 0.5, mult: 1.35, delta: 0.08, minDist: 0.09 });
    var midRaw = pickPeaks(fluxM, frameSec, { win: 0.5, mult: 1.4, delta: 0.1, minDist: 0.09 });
    var hatsRaw = pickPeaks(fluxH, frameSec, { win: 0.4, mult: 1.3, delta: 0.08, minDist: 0.06 });
    progress(0.75);

    /* BPM: автокорреляция низкой полосы (60..200) */
    var lowC = movingMean(fluxL, Math.round(0.5 / frameSec));
    var sig = new Float32Array(nFrames);
    for (i = 0; i < nFrames; i++) sig[i] = Math.max(0, fluxL[i] - lowC[i]);
    var minLag = Math.floor(60 / 200 / frameSec), maxLag = Math.ceil(60 / 60 / frameSec);
    var ac = new Float32Array(maxLag + 1);
    var stride = nFrames > 40000 ? 2 : 1; // на длинных треках каждый второй кадр, точности хватает
    for (var lag = minLag; lag <= maxLag; lag++) { var acc = 0; for (i = 0; i + lag < nFrames; i += stride) acc += sig[i] * sig[i + lag]; ac[lag] = acc / (nFrames - lag); }
    // усиливаем кратные периоды (2 и 4 бита), чтобы четвертные попадали
    var score = new Float32Array(maxLag + 1);
    for (lag = minLag; lag <= maxLag; lag++) {
      var v = ac[lag];
      if (lag * 2 <= maxLag) v += 0.5 * ac[lag * 2]; else v += 0.5 * ac[lag];
      score[lag] = v;
    }
    var bestLag = minLag; for (lag = minLag; lag <= maxLag; lag++) if (score[lag] > score[bestLag]) bestLag = lag;
    // параболическая интерполяция
    var y0 = score[bestLag - 1] || 0, y1 = score[bestLag], y2 = score[bestLag + 1] || 0;
    var denom = (y0 - 2 * y1 + y2), lagF = bestLag + (denom ? 0.5 * (y0 - y2) / denom : 0);
    var bpm = 60 / (lagF * frameSec);
    // октавная неоднозначность: электронщина живёт в 118..180
    function scoreAt(b) { var l = 60 / b / frameSec; var i0 = Math.floor(l), fr = l - i0; if (i0 < minLag || i0 + 1 > maxLag) return 0; return ac[i0] * (1 - fr) + ac[i0 + 1] * fr; }
    var cands = [bpm, bpm * 2, bpm / 2, bpm * 3 / 2, bpm * 2 / 3].filter(function (b) { return b >= 100 && b <= 190; });
    if (cands.length) {
      var bestB = cands[0], bestS = -1;
      cands.forEach(function (b) { var s = scoreAt(b) * (b === bpm ? 1.15 : 1); if (s > bestS) { bestS = s; bestB = b; } });
      bpm = bestB;
    }
    // уточнение гребёнкой: период, при котором сумма fluxL на сетке (с лучшей фазой) максимальна
    var dur = pcm.length / sr;
    function fluxAt(arr, t) { var fi = (t - FRAME_LAG) / frameSec; var i0 = Math.floor(fi); if (i0 < 0 || i0 + 1 >= nFrames) return 0; var fr = fi - i0; return arr[i0] * (1 - fr) + arr[i0 + 1] * fr; }
    function fluxAtT(t) { return fluxAt(fluxL, t); }
    function combScore(per) {
      var phases = 32, best = 0;
      for (var s = 0; s < phases; s++) { var v = 0, n = 0; for (var t = per * s / phases; t < dur; t += per) { v += fluxAtT(t); n++; } v /= n; if (v > best) best = v; }
      return best;
    }
    function refine(per, span, step) { var bestP = per, bestV = -1; for (var m = 1 - span; m <= 1 + span; m += step) { var v = combScore(per * m); if (v > bestV) { bestV = v; bestP = per * m; } } return bestP; }
    var period = 60 / bpm;
    period = refine(period, 0.04, 0.001);
    period = refine(period, 0.0015, 0.00005);
    bpm = 60 / period;
    progress(0.85);

    /* фаза сетки: сдвиг, при котором сумма fluxL на битах максимальна; переоценка каждые 30 с (защита от дрейфа mp3) */
    function bestPhase(t0, t1) {
      var steps = 48, best = 0, bestV = -1;
      for (var s = 0; s < steps; s++) { var ph = period * s / steps, v = 0; for (var t = t0 + ph; t < t1; t += period) v += fluxAtT(t); if (v > bestV) { bestV = v; best = ph; } }
      // уточнение вокруг лучшего
      var fine = best, fineV = bestV;
      for (var d = -period / steps; d <= period / steps; d += period / steps / 8) { var ph2 = best + d, v2 = 0; for (var t2 = t0 + ph2; t2 < t1; t2 += period) v2 += fluxAtT(t2); if (v2 > fineV) { fineV = v2; fine = ph2; } }
      return fine;
    }
    var grid = [], seg = 30, t = 0;
    while (t < dur) {
      var t1 = Math.min(dur, t + seg), ph = bestPhase(t, t1);
      // стыкуем с предыдущим сегментом: сдвиг не больше полпериода от продолжения
      if (grid.length) { var cont = grid[grid.length - 1] + period, g0 = t + ph; while (g0 < cont - period / 2) g0 += period; while (g0 > cont + period / 2) g0 -= period; ph = g0 - t; }
      for (var g = t + ph; g < t1; g += period) if (g >= 0) grid.push(g);
      t = t1;
    }
    // индекс сильной доли (1 из 4): фаза, где бочек больше всего
    var downbeatPhase = 0, dbBest = -1;
    for (var q = 0; q < 4; q++) { var v3 = 0; for (i = q; i < grid.length; i += 4) v3 += fluxAtT(grid[i]); if (v3 > dbBest) { dbBest = v3; downbeatPhase = q; } }

    /* квантование бочек к полубитам, снейры отдельно от бочек */
    function snap(tt, div) {
      var step = period / div, gi = Math.floor((tt - grid[0]) / period); if (gi < 0) gi = 0; if (gi >= grid.length) gi = grid.length - 1;
      var base = grid[gi], k = Math.round((tt - base) / step), st = base + k * step, e = st - tt;
      return Math.abs(e) <= 0.045 ? st : null;
    }
    var kicks = [], lastK = -1;
    kicksRaw.forEach(function (k) { var st = snap(k.t, 2); var tt = st !== null ? st : k.t; if (tt - lastK < 0.08) return; lastK = tt; kicks.push({ t: tt, s: k.s, q: st !== null }); });
    // снейр/клэп в электронике живёт на 2 и 4: ищем фазу (чёт/нечет бита), где MID flux выше
    var snarePhase = 0, spBest = -1;
    for (var q2 = 0; q2 < 2; q2++) { var v5 = 0; for (i = q2; i < grid.length; i += 2) v5 += fluxAt(fluxM, grid[i]); if (v5 > spBest) { spBest = v5; snarePhase = q2; } }
    var snares = [], lastS = -1;
    midRaw.forEach(function (m) {
      var st = snap(m.t, 1), onSnareBeat = false;
      if (st !== null) { var gi = Math.round((st - grid[0]) / period); onSnareBeat = ((gi % 2) + 2) % 2 === snarePhase; }
      // не на снейрной доле: только явная сбивка (сильный пик с телом в середине, не хэт и не бочка)
      if (!onSnareBeat && !(m.s >= 0.95 && fluxM[m.i] > 1.5 * fluxL[m.i] && fluxM[m.i] > fluxH[m.i])) return;
      var tt = st !== null ? st : m.t; if (tt - lastS < 0.1) return; lastS = tt;
      snares.push({ t: tt, s: m.s, fill: !onSnareBeat });
    });
    var hats = [], si = 0;
    hatsRaw.forEach(function (h) {
      while (si < snares.length - 1 && snares[si + 1].t < h.t) si++;
      if (snares.length && (Math.abs(snares[si].t - h.t) < 0.04 || (si + 1 < snares.length && Math.abs(snares[si + 1].t - h.t) < 0.04))) return;
      hats.push({ t: h.t, s: h.s });
    });

    /* огибающая: RMS по 0,1 с; сглаживание 0,25 и 2 с; нормировка по 95-му перцентилю */
    var envStep = 0.1, envN = Math.ceil(dur / envStep), envRaw = new Float32Array(envN), spp = Math.round(envStep * sr);
    for (i = 0; i < envN; i++) { var a = i * spp, b2 = Math.min(pcm.length, a + spp), ss = 0; for (var j = a; j < b2; j++) ss += pcm[j] * pcm[j]; envRaw[i] = Math.sqrt(ss / Math.max(1, b2 - a)); }
    var envNorm = percentile(envRaw, 0.95);
    for (i = 0; i < envN; i++) envRaw[i] = Math.min(1.2, envRaw[i] / envNorm);
    var envFast = movingMean(envRaw, 1), envSlow = movingMean(envRaw, 10);
    var slowNorm = percentile(envSlow, 0.95); for (i = 0; i < envN; i++) envSlow[i] = Math.min(1.2, envSlow[i] / slowNorm);
    // плотность онсетов (событий в секунду, сглаженная 2 с) как вторая ось интенсивности
    var dens = new Float32Array(envN);
    kicks.concat(snares).forEach(function (o) { var idx = Math.floor(o.t / envStep); if (idx < envN) dens[idx] += 1; });
    var densS = movingMean(dens, 10); for (i = 0; i < envN; i++) densS[i] *= 10; // событий в секунду
    var densNorm = percentile(densS, 0.95);
    var intensity = new Float32Array(envN);
    for (i = 0; i < envN; i++) intensity[i] = Math.min(1, 0.75 * envSlow[i] + 0.25 * densS[i] / densNorm);

    /* секции: drop > 0,7 дольше 4 с, breakdown < 0,35 дольше 3 с */
    var sections = [], cur2 = null, state = 'mid';
    function label(v) { // гистерезис, чтобы граница не дребезжала
      if (state === 'drop') { if (v < 0.6) state = v < 0.32 ? 'break' : 'mid'; }
      else if (state === 'break') { if (v > 0.42) state = v > 0.72 ? 'drop' : 'mid'; }
      else { if (v > 0.72) state = 'drop'; else if (v < 0.32) state = 'break'; }
      return state;
    }
    for (i = 0; i < envN; i++) { var lb = label(intensity[i]); if (!cur2 || cur2.type !== lb) { if (cur2) sections.push(cur2); cur2 = { type: lb, from: i * envStep, to: (i + 1) * envStep }; } else cur2.to = (i + 1) * envStep; }
    if (cur2) sections.push(cur2);
    // склейка коротких
    var minLen = { drop: 4, break: 3, mid: 2 }, merged = [];
    sections.forEach(function (s) { if (merged.length && (s.to - s.from < minLen[s.type])) { merged[merged.length - 1].to = s.to; } else merged.push(s); });
    var co = []; merged.forEach(function (s) { if (co.length && co[co.length - 1].type === s.type) co[co.length - 1].to = s.to; else co.push(s); });
    sections = co;

    /* лучшее окно 60 с: максимум суммы (событий × интенсивность); старт округляем до сильной доли */
    var winLen = 60, best60 = { from: 0, to: Math.min(dur, winLen) }, bestV60 = -1;
    if (dur > winLen + 2) {
      var evts = kicks.concat(snares), cum = new Float64Array(envN + 1);
      var perBin = new Float32Array(envN);
      evts.forEach(function (o) { var idx = Math.floor(o.t / envStep); if (idx < envN) perBin[idx] += 1 + intensity[idx]; });
      for (i = 0; i < envN; i++) cum[i + 1] = cum[i] + perBin[i];
      var wBins = Math.round(winLen / envStep);
      for (i = 0; i + wBins <= envN; i++) { var v4 = cum[i + wBins] - cum[i]; if (v4 > bestV60) { bestV60 = v4; best60.from = i * envStep; } }
      // к ближайшей сильной доле (бар начинается на downbeat)
      var gi0 = 0; for (i = 0; i < grid.length; i++) if (grid[i] >= best60.from) { gi0 = i; break; }
      gi0 -= ((gi0 - downbeatPhase) % 4 + 4) % 4; if (gi0 < 0) gi0 = 0;
      best60.from = grid[gi0] || best60.from; best60.to = Math.min(dur, best60.from + winLen);
    }
    progress(1);

    return {
      bpm: Math.round(bpm * 10) / 10, period: period, duration: dur,
      grid: grid, downbeatPhase: downbeatPhase,
      kicks: kicks, snares: snares, hats: hats,
      env: { step: envStep, fast: Array.from(envFast), slow: Array.from(envSlow), intensity: Array.from(intensity) },
      sections: sections, best60: best60,
      _debug: { frameSec: frameSec, fluxL: fluxL, fluxM: fluxM, fluxH: fluxH }
    };
  }

  root.SZAnalyzer = { analyze: analyze, version: 1 };

  /* режим Worker: postMessage({pcm, sr}) → {type:'progress'|'done'} */
  if (typeof WorkerGlobalScope !== 'undefined' && root instanceof WorkerGlobalScope) {
    root.onmessage = function (e) {
      var d = e.data;
      try {
        var res = analyze(d.pcm, d.sr, function (p) { root.postMessage({ type: 'progress', p: p }); });
        if (!d.debug) delete res._debug;
        root.postMessage({ type: 'done', result: res });
      } catch (err) { root.postMessage({ type: 'error', message: String(err && err.stack || err) }); }
    };
  }
})(typeof self !== 'undefined' ? self : this);
