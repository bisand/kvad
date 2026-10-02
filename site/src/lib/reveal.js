// `use:reveal` on a section: it fades up the first time it is scrolled to.
// The hidden state is a class this adds, so a page with no JavaScript, or a
// reader who asked for less motion, sees everything from the start.

let observer;

export function reveal(node) {
  if (typeof IntersectionObserver === "undefined") return;
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  // Already on screen when the page arrives: leave it alone.
  if (node.getBoundingClientRect().top < innerHeight * 0.9) return;

  observer ??= new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        e.target.classList.add("seen");
        observer.unobserve(e.target);
      }
    },
    { rootMargin: "0px 0px -12% 0px" },
  );
  node.classList.add("reveal");
  observer.observe(node);
  return { destroy: () => observer.unobserve(node) };
}
