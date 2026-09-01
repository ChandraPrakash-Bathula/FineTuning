import { useEffect, useState } from "react";
import type { Lab } from "./lib/registry";
import { loadLab } from "./lib/registry";
import { useDarkMode } from "./lib/palette";
import WhyFineTune from "./pages/WhyFineTune";
import LoRAPage from "./pages/LoRAPage";
import QLoRAPage from "./pages/QLoRAPage";
import Playground from "./pages/Playground";

const PAGES = [
  { id: "why", label: "Why fine-tune" },
  { id: "lora", label: "LoRA" },
  { id: "qlora", label: "QLoRA" },
  { id: "playground", label: "Playground" },
] as const;

type PageId = (typeof PAGES)[number]["id"];

function currentHash(): PageId {
  const h = window.location.hash.replace("#", "") as PageId;
  return PAGES.some((p) => p.id === h) ? h : "why";
}

export default function App() {
  const dark = useDarkMode();
  const [lab, setLab] = useState<Lab | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState<PageId>(currentHash);

  useEffect(() => { loadLab().then(setLab, (e) => setError(String(e))); }, []);

  useEffect(() => {
    const onHash = () => setPage(currentHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = (id: PageId) => {
    window.location.hash = id;
    setPage(id);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  if (error) {
    return (
      <div className="app">
        <div className="card"><h2>Could not load the registry</h2><p className="sub">{error}</p></div>
      </div>
    );
  }
  if (!lab) return <div className="app"><div className="muted">Loading measured results…</div></div>;

  return (
    <>
      <nav className="nav">
        <div className="nav-inner">
          <span className="nav-brand">Fine-Tuning Lab</span>
          <div className="nav-links">
            {PAGES.map((p, i) => (
              <button key={p.id} className={`nav-link${page === p.id ? " on" : ""}`}
                      onClick={() => go(p.id)}>
                <span className="nav-num">{i + 1}</span>{p.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <div className="app">
        {page === "why" && <WhyFineTune lab={lab} onNext={() => go("lora")} />}
        {page === "lora" && <LoRAPage lab={lab} dark={dark} onNext={() => go("qlora")} />}
        {page === "qlora" && <QLoRAPage lab={lab} onNext={() => go("playground")} />}
        {page === "playground" && <Playground lab={lab} dark={dark} />}

        <footer className="foot">
          Every measured number here is read from Phase 1's <code>registry.json</code> —
          11 real training runs of <code>{lab.baseModel}</code> on <code>{lab.dataset}</code>.
          Architecture figures come from the model's own <code>config.json</code>. Rental
          prices are assumptions, and are labelled where they appear.
        </footer>
      </div>
    </>
  );
}
