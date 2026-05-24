/* ── Scroll progress ─────────────────────── */
const progressBar = document.getElementById('progress-bar');
window.addEventListener('scroll', () => {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  progressBar.style.width = (window.scrollY / max * 100) + '%';
});

/* ── Navbar sticky ───────────────────────── */
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 50);
});

/* ── Smooth scroll ───────────────────────── */
document.querySelectorAll('a[href^="#"], [data-scroll]').forEach(el => {
  el.addEventListener('click', e => {
    const target = el.dataset.scroll || el.getAttribute('href');
    const dest = document.querySelector(target);
    if (!dest) return;
    e.preventDefault();
    dest.scrollIntoView({ behavior: 'smooth', block: 'start' });
    // close mobile menu
    document.getElementById('mobile-menu').classList.remove('open');
    document.getElementById('hamburger').classList.remove('open');
  });
});

/* ── Mobile menu ─────────────────────────── */
const hamburger = document.getElementById('hamburger');
const mobileMenu = document.getElementById('mobile-menu');
hamburger.addEventListener('click', () => {
  hamburger.classList.toggle('open');
  mobileMenu.classList.toggle('open');
});

/* ── Typewriter ──────────────────────────── */
const TITLES = [
  'Cybersecurity Professional',
  'Ethical Hacker',
  'Penetration Tester',
  'AI-Driven Security',
  'SOC Analyst',
];
let twIdx = 0, twText = '', twPhase = 'typing';
const twEl = document.getElementById('tw-text');

function typeStep() {
  const current = TITLES[twIdx];
  if (twPhase === 'typing') {
    if (twText.length < current.length) {
      twText = current.slice(0, twText.length + 1);
      twEl.textContent = twText;
      setTimeout(typeStep, 70);
    } else {
      twPhase = 'pause';
      setTimeout(typeStep, 1800);
    }
  } else if (twPhase === 'pause') {
    twPhase = 'deleting';
    setTimeout(typeStep, 400);
  } else {
    if (twText.length > 0) {
      twText = twText.slice(0, -1);
      twEl.textContent = twText;
      setTimeout(typeStep, 35);
    } else {
      twIdx = (twIdx + 1) % TITLES.length;
      twPhase = 'typing';
      setTimeout(typeStep, 200);
    }
  }
}
typeStep();

/* ── Terminal animation ──────────────────── */
const T_LINES = [
  { cls: 'cmd',  text: '$ nmap -sV --script vuln 192.168.1.0/24' },
  { cls: 'info', text: 'Starting Nmap 7.93 — CVE scan in progress...' },
  { cls: 'hit',  text: 'Discovered open port 22/tcp on 192.168.1.105' },
  { cls: 'vuln', text: 'VULNERABLE: CVE-2023-38408 — OpenSSH RCE' },
  { cls: 'cmd',  text: "$ metasploit -x 'use exploit/multi/handler'" },
  { cls: 'win',  text: '[*] Session 1 opened — shell access granted' },
];
const termBody = document.getElementById('term-body');
T_LINES.forEach((l, i) => {
  const p = document.createElement('p');
  p.className = 't-line ' + l.cls;
  p.textContent = l.text;
  termBody.appendChild(p);
  setTimeout(() => p.classList.add('shown'), 600 + i * 800);
});

/* ── Intersection observer (reveal) ─────── */
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      // stagger children if data-stagger
      const children = e.target.querySelectorAll('[data-delay]');
      children.forEach(c => {
        setTimeout(() => c.classList.add('visible'), +c.dataset.delay);
      });
    }
  });
}, { threshold: 0.12, rootMargin: '-60px 0px' });

document.querySelectorAll('.reveal, .reveal-left, .reveal-right').forEach(el => {
  observer.observe(el);
});
