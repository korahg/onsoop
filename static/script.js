function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = el.scrollHeight + "px";
}
function lockSubmit(form) {
  const btn = form.querySelector("button[type=submit]");
  if (btn) {
    btn.disabled = true;
    const original = btn.textContent;
    btn.dataset.original = original;
    btn.textContent = "처리 중…";
    setTimeout(() => { btn.disabled = false; btn.textContent = original; }, 4000);
  }
  return true;
}
document.addEventListener("click", (e) => {
  const t = e.target;
  if (t.id === "themeToggle") {
    document.body.classList.toggle("light");
  }
});
