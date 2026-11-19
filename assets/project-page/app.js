'use strict';

const $ = selector => document.querySelector(selector);
const $$ = selector => Array.from(document.querySelectorAll(selector));
const root = document.documentElement;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');
let pausedByUser = false;
let updateFieldMotion = () => {};

function syncMotion() {
  const paused = pausedByUser || reducedMotion.matches || document.hidden;
  root.dataset.motion = paused ? 'paused' : 'playing';
  const button = $('#motion-toggle');
  button.setAttribute('aria-pressed', String(paused));
  button.disabled = reducedMotion.matches;
  const label = reducedMotion.matches ? 'Visual effects follow your reduced-motion preference' : pausedByUser ? 'Resume visual effects' : 'Pause visual effects';
  button.setAttribute('aria-label', label);
  button.title = label;
  updateFieldMotion();
}
$('#motion-toggle').addEventListener('click', () => { pausedByUser = !pausedByUser; syncMotion(); });
reducedMotion.addEventListener('change', syncMotion);
document.addEventListener('visibilitychange', syncMotion);
syncMotion();

const menuButton = $('.menu-toggle');
const navigation = $('#navigation');
function closeMenu() {
  menuButton.setAttribute('aria-expanded', 'false');
  navigation.classList.remove('is-open');
}
menuButton.addEventListener('click', () => {
  const open = menuButton.getAttribute('aria-expanded') !== 'true';
  menuButton.setAttribute('aria-expanded', String(open));
  navigation.classList.toggle('is-open', open);
});
navigation.addEventListener('click', event => { if (event.target.closest('a')) closeMenu(); });
document.addEventListener('click', event => { if (!event.target.closest('.site-header')) closeMenu(); });
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && menuButton.getAttribute('aria-expanded') === 'true') { closeMenu(); menuButton.focus(); }
});
window.matchMedia('(min-width: 901px)').addEventListener('change', event => { if (event.matches) closeMenu(); });
$$('[data-paper-link]').forEach(link => {
  const isPlaceholder = () => !link.getAttribute('href') || link.getAttribute('href') === '#';
  if (!isPlaceholder()) link.removeAttribute('aria-disabled');
  link.addEventListener('click', event => { if (isPlaceholder()) event.preventDefault(); });
});

if ('IntersectionObserver' in window) {
  const revealObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.classList.add('is-visible'); revealObserver.unobserve(entry.target); }
    });
  }, { threshold: 0.07, rootMargin: '0px 0px -20px 0px' });
  $$('.reveal').forEach(element => revealObserver.observe(element));
  root.classList.add('motion-ready');
}

const sectionLinks = $$('#navigation a[href^="#"]');
const sections = sectionLinks.map(link => $(link.getAttribute('href')));
let scrollFrame = 0;
let snapTimer = 0;
let snapping = false;
let lastScrollY = window.scrollY;
let scrollDirection = 1;
function updateScroll() {
  scrollFrame = 0;
  const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
  $('.reading-progress').style.transform = 'scaleX(' + (maxScroll > 0 ? window.scrollY / maxScroll : 0) + ')';
  let current = null;
  sections.forEach(section => { if (section.getBoundingClientRect().top <= window.innerHeight * .35) current = section.id; });
  sectionLinks.forEach(link => {
    const active = link.getAttribute('href') === '#' + current;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
/* Strengthen proximity snapping only near a section boundary, without
   intercepting wheel, touch, keyboard, or nested figure scrolling. */
function settleSection() {
  if (reducedMotion.matches || snapping || $('#figure-dialog').open ||
      menuButton.getAttribute('aria-expanded') === 'true' ||
      document.activeElement?.closest('button,a,input,textarea,select,[role="tabpanel"]')) return;
  const headerHeight = $('.site-header').offsetHeight;
  const targets = $$('#main>section').map(section => Math.max(0, section.offsetTop - headerHeight));
  const nearest = targets.reduce((best, target) => Math.abs(target - window.scrollY) < Math.abs(best - window.scrollY) ? target : best, targets[0]);
  const delta = nearest - window.scrollY;
  if (Math.abs(delta) > 6 && Math.abs(delta) < 84 && delta * scrollDirection > 0) {
    snapping = true;
    window.scrollTo({ top: nearest, behavior: 'smooth' });
    window.setTimeout(() => { snapping = false; }, 600);
  }
}
window.addEventListener('scroll', () => {
  scrollDirection = window.scrollY >= lastScrollY ? 1 : -1;
  lastScrollY = window.scrollY;
  if (!scrollFrame) scrollFrame = requestAnimationFrame(updateScroll);
  clearTimeout(snapTimer);
  if (!snapping) snapTimer = window.setTimeout(settleSection, 180);
}, { passive: true });
window.addEventListener('resize', () => { if (!scrollFrame) scrollFrame = requestAnimationFrame(updateScroll); }, { passive: true });
updateScroll();

const steps = {
  estimate: { kicker: 'Policy-relative understanding', title: 'Know what the policy is ready to learn.', description: 'Sequential scene-level evaluations separate policy capability from scene difficulty. Estimated success probabilities reveal the current frontier.', expression: 'p = σ(θ − β)', note: 'Capability meets scene difficulty.' },
  retrieve: { kicker: 'From evaluation evidence to training experience', title: 'Find related scenes in the training world.', description: 'Frontier prototypes and adaptive neighborhoods retrieve structurally related training scenes in a shared 12-dimensional scenario space.', expression: 'Scenes → Regions → Support', note: '21 descriptors. One shared scenario space.' },
  allocate: { kicker: 'An agent-guided learning strategy', title: 'Focus the next update. Retain what is learned.', description: 'Four agent roles use current evidence and cross-stage memory to allocate training effort across frontier regions and retention replay.', expression: 'q = αqF + (1 − α)qR', note: 'Frontier learning, balanced with replay.' },
  evolve: { kicker: 'The next frontier begins with the next policy', title: 'Learn. Re-evaluate. Move the frontier.', description: 'A verified curriculum drives the next policy update. New evaluation outcomes enter learning memory and reshape the frontier for the following stage.', expression: 'Policy → Frontier → Policy', note: 'Each update reshapes what to learn next.' }
};
const stepButtons = $$('[data-step]');
function selectStep(button) {
  const value = steps[button.dataset.step];
  stepButtons.forEach(item => { item.setAttribute('aria-selected', String(item === button)); item.tabIndex = item === button ? 0 : -1; });
  $('#loop-panel').setAttribute('aria-labelledby', button.id);
  $('#step-kicker').textContent = value.kicker;
  $('#step-title').textContent = value.title;
  $('#step-description').textContent = value.description;
  $('#step-expression').textContent = value.expression;
  $('#step-expression-note').textContent = value.note;
}
stepButtons.forEach((button, index) => {
  button.addEventListener('click', () => selectStep(button));
  button.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % stepButtons.length;
    else if (event.key === 'ArrowLeft') next = (index + stepButtons.length - 1) % stepButtons.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = stepButtons.length - 1;
    else return;
    event.preventDefault();
    selectStep(stepButtons[next]);
    stepButtons[next].focus();
  });
});

/* Manuscript scores, with a shared 0–100 EPDMS scale for both benchmarks. */
const benchmarkScores = {
  navtest: { title: 'NavTest', scores: [[79.1,84.7], [84.5,87.3], [84.2,87.5]] },
  navhard: { title: 'NavHard', scores: [[23.0,24.6], [27.8,29.5], [24.6,28.1]] }
};
$$('[data-benchmark]').forEach(button => button.addEventListener('click', () => {
  const benchmark = benchmarkScores[button.dataset.benchmark];
  $$('[data-benchmark]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  $('#benchmark-title').textContent = benchmark.title;
  const announcements = [];
  $$('.result-row').forEach((row, index) => {
    const [base, frontier] = benchmark.scores[index];
    row.querySelector('.base-score').textContent = base.toFixed(1);
    row.querySelector('.frontier-score').textContent = frontier.toFixed(1);
    row.querySelector('.bar.base').style.setProperty('--value', base + '%');
    row.querySelector('.bar.frontier').style.setProperty('--value', frontier + '%');
    row.querySelector('.gain strong').textContent = '+' + (frontier - base).toFixed(1);
    announcements.push(row.querySelector('h4').textContent + ': ' + base.toFixed(1) + ' to ' + frontier.toFixed(1));
  });
  $('#results-announcement').textContent = benchmark.title + '. ' + announcements.join('. ') + ' EPDMS.';
}));

const stages = {
  1: { capability: '−0.81', frontier: '9,918', support: '7,433', title: 'An initial frontier.', text: 'The first curriculum identifies broad frontier regions and retrieves their related training support.' },
  5: { capability: '0.18', frontier: '1,824', support: '2,102', title: 'A different learning landscape.', text: 'With greater capability, the frontier relocates and the curriculum draws support from a new configuration of scene regions.' },
  9: { capability: '0.72', frontier: '1,574', support: '1,203', title: 'New capability. New frontiers.', text: 'Later-stage learning follows a reshaped frontier, adapting the training focus to the improved policy.' }
};
$$('[data-stage]').forEach(button => button.addEventListener('click', () => {
  const key = button.dataset.stage, stage = stages[key];
  $$('[data-stage]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  $('#capability-value').textContent = stage.capability;
  $('#frontier-count').textContent = stage.frontier;
  $('#support-count').textContent = stage.support;
  $('#stage-title').textContent = stage.title;
  $('#stage-description').textContent = stage.text;
  $('#stage-image').src = 'media/stage-' + key + '.png';
  const description = 'Stage ' + key + ': estimated capability ' + stage.capability + ', ' + stage.frontier + ' frontier scenes, and ' + stage.support + ' training support scenes.';
  $('#stage-image').alt = description;
  $('#stage-announcement').textContent = description;
}));

const scenarios = {
  junction: { title: 'A path through the junction.', text: 'The policy adapts its planned path through the junction and continues along the outgoing road.', alt: 'Four successive views of FrontierDrive driving through a T-intersection.' },
  emergence: { title: 'Respond to what emerges.', text: 'As a non-motorized road user enters the path, the policy brakes and reduces speed to respond to the developing interaction.', alt: 'Four successive views of FrontierDrive responding to an emerging road user.' },
  intersection: { title: 'Navigate the crossing flows.', text: 'The policy adjusts its trajectory around pedestrians and non-motorized road users while progressing through the intersection.', alt: 'Four successive views of FrontierDrive driving through an intersection with crossing traffic.' },
  interaction: { title: 'Make room for others.', text: 'Through a densely occupied road section, the policy yields to nearby road users and adjusts its passage as available space narrows.', alt: 'Four successive views of FrontierDrive moving through dense pedestrian and non-motorized traffic.' },
  night: { title: 'Follow the road after dark.', text: 'Under nighttime illumination, the policy follows the road and continues along the campus route.', alt: 'Four successive views of FrontierDrive following the campus route at night.' }
};
$$('[data-scenario]').forEach(button => button.addEventListener('click', () => {
  const key = button.dataset.scenario, scene = scenarios[key];
  $$('[data-scenario]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  $('#scenario-title').textContent = scene.title;
  $('#scenario-description').textContent = scene.text;
  $('#scenario-image').src = 'media/drive-' + key + '.webp';
  $('#scenario-image').alt = scene.alt;
  $('#scenario-zoom').href = 'media/drive-' + key + '.webp';
  $('#scenario-zoom').dataset.caption = scene.text;
  $('#scenario-zoom').scrollLeft = 0;
}));

const figureDialog = $('#figure-dialog');
let figureTrigger = null;
$$('a.zoomable').forEach(link => link.addEventListener('click', event => {
  if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || typeof figureDialog.showModal !== 'function') return;
  event.preventDefault();
  figureTrigger = link;
  const image = link.querySelector('img');
  $('#lightbox-image').src = link.href;
  $('#lightbox-image').alt = link.dataset.caption || image?.alt || 'FrontierDrive research figure';
  $('#figure-caption').textContent = link.dataset.caption || image?.alt || '';
  $('#original-image').href = link.href;
  figureDialog.showModal();
  figureDialog.scrollTop = 0;
  document.body.classList.add('modal-open');
  $('#close-figure').focus({ preventScroll: true });
  updateFieldMotion();
}));
$('#close-figure').addEventListener('click', () => figureDialog.close());
figureDialog.addEventListener('click', event => {
  if (event.target !== figureDialog) return;
  const box = figureDialog.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) figureDialog.close();
});
figureDialog.addEventListener('close', () => {
  document.body.classList.remove('modal-open');
  figureTrigger?.focus({ preventScroll: true });
  updateFieldMotion();
});

$('#copy-citation').addEventListener('click', async () => {
  const button = $('#copy-citation'), citation = $('#citation-code').textContent;
  let copied = false;
  try {
    if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(citation); copied = true; }
  } catch (_) { /* Try local-document fallback. */ }
  if (!copied) {
    const textarea = document.createElement('textarea');
    textarea.value = citation;
    textarea.style.cssText = 'position:fixed;left:-9999px;top:0;';
    document.body.append(textarea);
    textarea.select();
    try { copied = document.execCommand('copy'); } catch (_) { copied = false; }
    textarea.remove();
    button.focus({ preventScroll: true });
  }
  if (copied) {
    button.querySelector('span').textContent = 'Copied';
    $('#copy-status').textContent = 'BibTeX citation copied.';
    window.setTimeout(() => { button.querySelector('span').textContent = 'Copy BibTeX'; }, 2200);
  } else {
    const selection = window.getSelection(), range = document.createRange();
    range.selectNodeContents($('#citation-code'));
    selection.removeAllRanges(); selection.addRange(range);
    $('#copy-status').textContent = 'Citation selected. Press Control+C or Command+C to copy.';
    button.querySelector('span').textContent = 'Citation selected';
  }
});

/* Original contour artwork: a moving frontier with short traveling light
   traces and local springs. The pointer produces a narrow, fading trail. */
(function initializeFrontierField() {
  const canvas = $('#frontier-field'), ctx = canvas.getContext('2d', { alpha: true });
  if (!ctx) return;
  const hero = $('#home');
  let width = 0, height = 0, ratio = 1, rings = [], inView = true;
  let animation = 0, lastFrame = 0, phase = 0;
  let pointer = { x: -1000, y: -1000, active: false };
  const trail = [], tau = Math.PI * 2;
  const colors = ['63,114,175', '120,149,178', '183,119,122'];
  function rebuild() {
    const rect = hero.getBoundingClientRect();
    width = rect.width; height = rect.height;
    ratio = Math.min(window.devicePixelRatio || 1, 1.7);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    rings = [];
    const compact = width < 650, count = compact ? 14 : 19;
    for (let field = 0; field < 2; field++) {
      for (let i = 0; i < count; i++) {
        rings.push({
          field, index: i,
          cx: width * (field === 0 ? compact ? 1.17 : 1.04 : compact ? -.25 : -.12),
          cy: height * (field === 0 ? .34 : .88),
          rx: width * ((compact ? .34 : .23) + i * (compact ? .035 : .019)),
          ry: height * (.16 + i * .023),
          angle: field === 0 ? -.47 : .30,
          vertices: Array.from({ length: 112 }, () => ({ x: 0, y: 0, dx: 0, dy: 0, vx: 0, vy: 0 }))
        });
      }
    }
    draw(0); hero.classList.add('field-ready'); manage();
  }
  function point(ring, angle) {
    const wobble = 1 + .045 * Math.sin(angle * 3 + ring.field + phase * .23) + .022 * Math.cos(angle * 5 - phase * .14);
    const breathe = 1 + .025 * Math.sin(phase * .38 + ring.index * .13);
    const x = Math.cos(angle) * ring.rx * wobble * breathe;
    const y = Math.sin(angle) * ring.ry * wobble;
    const c = Math.cos(ring.angle), s = Math.sin(ring.angle);
    return { x: ring.cx + x * c - y * s, y: ring.cy + x * s + y * c };
  }
  function draw(delta) {
    ctx.clearRect(0, 0, width, height);
    const springStep = Math.min(delta * 45, 1.5);
    for (const ring of rings) {
      const frontier = ring.index === 7 || ring.index === 8, color = colors[frontier ? 2 : ring.field];
      ctx.lineWidth = frontier ? 1.25 : .75;
      ctx.strokeStyle = 'rgba(' + color + ',' + (frontier ? .42 : .16 + .075 * Math.sin(ring.index * .35 + 1)) + ')';
      ctx.beginPath();
      ring.vertices.forEach((vertex, index) => {
        const p = point(ring, index / ring.vertices.length * tau);
        let tx = 0, ty = 0;
        if (pointer.active && root.dataset.motion !== 'paused') {
          const dx = p.x - pointer.x, dy = p.y - pointer.y, distance = Math.hypot(dx, dy);
          if (distance < 68) {
            const force = Math.pow(1 - distance / 68, 2) * 10;
            tx = dx / (distance || 1) * force; ty = dy / (distance || 1) * force;
          }
        }
        if (springStep > 0) {
          vertex.vx = (vertex.vx + (tx - vertex.dx) * .18 * springStep) * .73;
          vertex.vy = (vertex.vy + (ty - vertex.dy) * .18 * springStep) * .73;
          vertex.dx += vertex.vx * springStep; vertex.dy += vertex.vy * springStep;
        }
        vertex.x = p.x + vertex.dx; vertex.y = p.y + vertex.dy;
        if (!index) ctx.moveTo(vertex.x, vertex.y);
        else ctx.lineTo(vertex.x, vertex.y);
      });
      ctx.closePath(); ctx.stroke();
    }
    /* Signals travel before the first pointer interaction. */
    const selectedRings = rings.filter(ring => [3,7,10,14,17].includes(ring.index));
    selectedRings.forEach((ring, index) => {
      const head = phase * (.35 + (index % 3) * .045) * (ring.field ? -1 : 1) + index * 1.75;
      const color = ring.index === 7 ? colors[2] : colors[0];
      for (let segment = 0; segment < 13; segment++) {
        const a = head - (13 - segment) * .014 * (ring.field ? -1 : 1);
        const from = point(ring, a), to = point(ring, a + .016 * (ring.field ? -1 : 1));
        ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y);
        ctx.lineWidth = 1.6;
        ctx.strokeStyle = 'rgba(' + color + ',' + segment / 13 * .72 + ')'; ctx.stroke();
      }
      const headPoint = point(ring, head);
      ctx.beginPath(); ctx.arc(headPoint.x, headPoint.y, 2.1, 0, tau);
      ctx.fillStyle = 'rgba(' + color + ',.7)'; ctx.fill();
    });
    const now = performance.now();
    while (trail.length && now - trail[0].time > 480) trail.shift();
    if (trail.length > 1 && root.dataset.motion !== 'paused') {
      for (let i = 1; i < trail.length; i++) {
        const age = Math.max(0, 1 - (now - trail[i].time) / 480);
        ctx.beginPath(); ctx.moveTo(trail[i-1].x, trail[i-1].y); ctx.lineTo(trail[i].x, trail[i].y);
        ctx.lineWidth = 1.5 * age; ctx.lineCap = 'round';
        ctx.strokeStyle = 'rgba(152,91,94,' + age * .5 + ')'; ctx.stroke();
      }
    }
  }
  function tick(time) {
    animation = requestAnimationFrame(tick);
    const elapsed = time - lastFrame;
    if (elapsed < 1000 / 30) return;
    const delta = Math.min(elapsed / 1000, .06);
    lastFrame = time; phase += delta; draw(delta);
  }
  function manage() {
    const active = root.dataset.motion === 'playing' && inView && !figureDialog.open;
    if (active && !animation) { lastFrame = performance.now(); animation = requestAnimationFrame(tick); }
    else if (!active && animation) {
      cancelAnimationFrame(animation); animation = 0; pointer.active = false; trail.length = 0; draw(0);
    }
  }
  updateFieldMotion = manage;
  hero.addEventListener('pointermove', event => {
    if (!finePointer.matches || root.dataset.motion !== 'playing') return;
    const rect = hero.getBoundingClientRect();
    pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top, active: true };
    const previous = trail[trail.length - 1];
    if (!previous || Math.hypot(pointer.x - previous.x, pointer.y - previous.y) > 2) {
      trail.push({ x: pointer.x, y: pointer.y, time: performance.now() });
      if (trail.length > 32) trail.shift();
    }
  }, { passive: true });
  hero.addEventListener('pointerleave', () => { pointer.active = false; });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => { inView = entries[0].isIntersecting; manage(); }).observe(hero);
  }
  if ('ResizeObserver' in window) new ResizeObserver(rebuild).observe(hero);
  else window.addEventListener('resize', rebuild, { passive: true });
  rebuild();
})();
