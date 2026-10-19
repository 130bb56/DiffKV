"use strict";
(() => {
  const evidence = window.IMPLEMENTATION_EVIDENCE;
  document.querySelectorAll("[data-tab-group]").forEach(group => {
    const tabs = [...group.querySelectorAll('[role="tab"]')];
    function select(tab) {
      tabs.forEach(item => {
        const selected = item === tab;
        item.setAttribute("aria-selected", String(selected));
        item.tabIndex = selected ? 0 : -1;
        document.getElementById(item.getAttribute("aria-controls")).hidden = !selected;
      });
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => select(tab));
      tab.addEventListener("keydown", event => {
        let next;
        if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = tabs.length - 1;
        if (next !== undefined) { event.preventDefault(); select(tabs[next]); tabs[next].focus(); }
      });
    });
  });
  document.querySelectorAll("[data-commit]").forEach(element => {
    element.textContent = evidence.manifest[element.dataset.commit].slice(0, 12);
  });
  document.getElementById("print").addEventListener("click", () => window.print());
  const nav = [...document.querySelectorAll(".sidebar nav a")];
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) {
      nav.forEach(link => link.classList.toggle("active", link.hash === `#${entry.target.id}`));
    }
  }, {rootMargin: "-5% 0px -75% 0px"});
  document.querySelectorAll("main > .section").forEach(section => observer.observe(section));
  window.IMPLEMENTATION_DOC = {evidence};
})();
