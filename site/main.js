(function () {
  "use strict";

  // Footer year
  var y = document.getElementById("year");
  if (y) y.textContent = String(new Date().getFullYear());

  // Active nav link on scroll
  var links = Array.prototype.slice.call(document.querySelectorAll(".nav-links a"));
  var sectionMap = {};
  links.forEach(function (a) {
    var id = a.getAttribute("href");
    if (id && id.charAt(0) === "#") {
      var el = document.querySelector(id);
      if (el) sectionMap[id] = { link: a, el: el };
    }
  });
  if ("IntersectionObserver" in window) {
    var navObs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var key = "#" + e.target.id;
        if (sectionMap[key] && e.isIntersecting) {
          links.forEach(function (l) { l.classList.remove("active"); });
          sectionMap[key].link.classList.add("active");
        }
      });
    }, { rootMargin: "-35% 0px -60% 0px" });
    Object.keys(sectionMap).forEach(function (k) {
      navObs.observe(sectionMap[k].el);
    });
  }

  function fmtNum(v, digits) {
    if (v === null || v === undefined) return "—";
    if (typeof v !== "number" || isNaN(v)) return String(v);
    return v.toFixed(digits === undefined ? 3 : digits);
  }

  function tierNote(entry) {
    if (entry.probe_only || entry.backend_tier === "PROBE_ONLY") {
      return "probe-first";
    }
    if (entry.backend_tier === "MOCK" || entry.availability === "mock_fallback") {
      return "fallback/proxy";
    }
    if (entry.backend_tier === "RESTRICTED_ADAPTER") {
      return "restricted adapter";
    }
    return "";
  }

  function loadLeaderboard() {
    var tbody = document.querySelector("#leaderboard-table tbody");
    if (!tbody) return;

    var urls = [
      "data/leaderboard.json",
      "../reports/public_release/leaderboard_final.json",
      "https://raw.githubusercontent.com/utkarshg20/ExactKV/main/reports/public_release/leaderboard_final.json"
    ];

    function tryFetch(i) {
      if (i >= urls.length) return;
      fetch(urls[i])
        .then(function (r) { if (!r.ok) throw new Error("bad status"); return r.json(); })
        .then(function (data) {
          var entries = (data && data.entries) || [];
          if (!entries.length) return;
          tbody.innerHTML = "";
          entries.forEach(function (e) {
            var note = tierNote(e);
            var tr = document.createElement("tr");
            if (note) tr.className = "row-muted";
            tr.innerHTML =
              "<td>" + e.rank + "</td>" +
              "<td><code>" + e.compressor + "</code>" +
                (note ? ' <span class="tier-tag">' + note + "</span>" : "") + "</td>" +
              "<td>" + (e.model_short || e.model) + "</td>" +
              '<td class="num">' + fmtNum(e.score) + "</td>" +
              '<td class="num">' + fmtNum(e.acceptance_rate) + "</td>" +
              '<td class="num">' + fmtNum(e.divergence_score) + "</td>";
            tbody.appendChild(tr);
          });
          var cap = document.getElementById("leaderboard-caption");
          if (cap && data.score_formula) {
            cap.textContent = "Loaded from leaderboard JSON (" + entries.length + " rows). " +
              data.score_formula + ". Probe-first and fallback/proxy rows are not headline evidence.";
          }
        })
        .catch(function () { tryFetch(i + 1); });
    }
    tryFetch(0);
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function loadCaseStudies() {
    var root = document.getElementById("case-studies-root");
    if (!root) return;

    fetch("data/case_studies.json")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var cases = (data && data.case_studies) || [];
        if (!cases.length) return;
        root.innerHTML = "";
        cases.slice(0, 6).forEach(function (c) {
          var card = document.createElement("article");
          card.className = "case-card";
          var title = (c.dataset_family || "panel") + " · " + (c.task_category || c.prompt_id || "");
          var meta = [
            c.compressor_name,
            c.model_name && c.model_name.split("/").pop(),
            c.context_bucket ? c.context_bucket + " ctx" : null,
            c.first_divergence_index != null ? "fdi=" + c.first_divergence_index : null
          ].filter(Boolean).join(" · ");

          card.innerHTML =
            "<h3>" + escHtml(title) + "</h3>" +
            '<p class="case-meta"><code>' + escHtml(meta) + "</code></p>" +
            '<div class="case-cols">' +
              '<div class="case-col"><span class="case-label">Full KV</span><pre>' + escHtml(c.full_snippet || "—") + "</pre></div>" +
              '<div class="case-col case-col-lossy"><span class="case-label">Lossy draft</span><pre>' + escHtml(c.lossy_snippet || "—") + "</pre></div>" +
              '<div class="case-col"><span class="case-label">ExactKV out</span><pre>' + escHtml(c.exactkv_snippet || "—") + "</pre></div>" +
            "</div>";
          root.appendChild(card);
        });
      })
      .catch(function () {
        root.innerHTML = "<p class=\"viz-caption\">Case study snippets unavailable offline.</p>";
      });
  }

  loadLeaderboard();
  loadCaseStudies();
})();
