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

})();
